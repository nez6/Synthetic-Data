"""Step 00 · Data ingestion from the cloud (Google Cloud Storage).

The Synthea patient pools are the dataset. They are built once (`python -m src.pools build A`),
packaged as one tar.gz per pool, and hosted in a GCS bucket next to a manifest with SHA-256 checksums.
The pipeline, the Docker image and the deployed API download them from there, so nobody needs Java,
Synthea or the original laptop to reproduce a run.

    python -m src.cloud_data package     # data/pools/* → dist/pools/{A,B}.tar.gz + manifest.json
    python -m src.cloud_data upload      # package, then copy to gs://<bucket>/<prefix>/ (needs gcloud)
    python -m src.cloud_data download    # fetch + verify + unpack into data/pools/ (skips pools already present)

The objects are public-read (the data is synthetic), so downloading needs no credentials:
    https://storage.googleapis.com/<bucket>/<prefix>/manifest.json
Bucket and prefix come from config.yaml (`cloud:`); DATA_BUCKET overrides the bucket.
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from pathlib import Path

import requests

from src import pools
from src.config import ROOT, load_config

DIST = ROOT / "dist" / "pools"


def _cfg() -> dict:
    c = dict(load_config()["cloud"])
    c["bucket"] = os.environ.get("DATA_BUCKET", c["bucket"])
    return c


def base_url() -> str:
    c = _cfg()
    return f"{c['public_base_url'].rstrip('/')}/{c['bucket']}/{c['prefix']}"


def gs_uri() -> str:
    c = _cfg()
    return f"gs://{c['bucket']}/{c['prefix']}"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ----------------------------------------------------------------------------- publish (maintainers)

def package() -> dict:
    """Tar each local pool and write a manifest. Returns the manifest."""
    DIST.mkdir(parents=True, exist_ok=True)
    manifest = {"pools": {}}
    for row in pools.list_pools():
        name = row["pool"]
        out = DIST / f"{name}.tar.gz"
        with tarfile.open(out, "w:gz") as tar:
            tar.add(pools.pools_dir() / name, arcname=name)
        manifest["pools"][name] = {"file": out.name, "sha256": _sha256(out), "bytes": out.stat().st_size,
                                   "people": row["people"], "ages": row["ages"], "synthea": row["synthea"]}
        print(f"packaged pool {name}: {row['people']} people, {out.stat().st_size / 1e6:.1f} MB")
    (DIST / "manifest.json").write_text(json.dumps(manifest, indent=2))
    return manifest


def upload():
    manifest = package()
    files = [str(DIST / m["file"]) for m in manifest["pools"].values()] + [str(DIST / "manifest.json")]
    subprocess.run(["gcloud", "storage", "cp", *files, gs_uri() + "/"], check=True)
    print(f"uploaded to {gs_uri()}/  (public URL {base_url()}/manifest.json)")


# ----------------------------------------------------------------------------- ingest (pipeline)

def _local_matches(name: str, entry: dict) -> bool:
    marker = pools.pools_dir() / name / ".source_sha256"
    return marker.exists() and marker.read_text().strip() == entry["sha256"]


def download(force: bool = False, progress=print) -> dict:
    """Fetch the manifest and every pool not already present locally. Returns a summary."""
    res = requests.get(f"{base_url()}/manifest.json", timeout=30)
    res.raise_for_status()
    manifest = res.json()
    dest = pools.pools_dir()
    dest.mkdir(parents=True, exist_ok=True)
    summary = {}
    for name, entry in manifest["pools"].items():
        have_local = (dest / name / "pool_info.json").exists()
        if not force and (_local_matches(name, entry) or (have_local and not (dest / name / ".source_sha256").exists())):
            summary[name] = "present"
            continue
        progress(f"downloading pool {name} ({entry['bytes'] / 1e6:.0f} MB)")
        with tempfile.NamedTemporaryFile(suffix=".tar.gz", delete=False, dir=dest) as tmp:
            with requests.get(f"{base_url()}/{entry['file']}", stream=True, timeout=60) as r:
                r.raise_for_status()
                for chunk in r.iter_content(1 << 20):
                    tmp.write(chunk)
            tmp_path = Path(tmp.name)
        try:
            if _sha256(tmp_path) != entry["sha256"]:
                raise RuntimeError(f"checksum mismatch for pool {name}")
            shutil.rmtree(dest / name, ignore_errors=True)
            with tarfile.open(tmp_path, "r:gz") as tar:
                tar.extractall(dest, filter="data")
            (dest / name / ".source_sha256").write_text(entry["sha256"])
        finally:
            tmp_path.unlink(missing_ok=True)
        summary[name] = "downloaded"
        progress(f"pool {name} ready")
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("action", choices=["package", "upload", "download"])
    ap.add_argument("--force", action="store_true", help="download even if the pool is already present")
    a = ap.parse_args()
    if a.action == "package":
        package()
    elif a.action == "upload":
        upload()
    else:
        print(download(force=a.force))


if __name__ == "__main__":
    main()
