"""Step 02 · Synthea patient pools.

A pool is one Synthea run (e.g. pool "B": 2,500 people aged 50-90 with 10 years of history).
Requests select their patients from pools, so Synthea only has to run when a pool is built or
grown — not on every request.

Synthea writes ~9 MB per older patient, mostly billing resources (Claim, ExplanationOfBenefit).
To fit on a laptop the builder runs Synthea in chunks and, after each chunk, compacts every
patient bundle to the resource types in `synthea.keep_types` (references to dropped resources
are removed with the reference project's cascade rules), gzips it, and deletes the raw output.

    python -m src.pools build A            # build pool A from config.yaml
    python -m src.pools build B --population 800
    python -m src.pools list
"""
import argparse
import gzip
import hashlib
import json
import os
import shutil
import subprocess
from collections import Counter
from datetime import date, datetime, timezone
from pathlib import Path

from src.config import ROOT, load_config, resolve
from src.fhir.parser import race_ethnicity
from src.stdc import fhir as sf
from src.stdc import synthea as ss


def _cfg():
    return load_config()["synthea"]


def pools_dir() -> Path:
    return resolve(_cfg()["pools_dir"])


def java_cmd() -> str:
    path = resolve(_cfg()["java"])
    if path.exists():
        return str(path)
    if shutil.which("java"):
        return "java"
    raise RuntimeError("Java 17+ is required for Synthea (see README: 'Installing Synthea').")


def jar_path() -> Path:
    jar = resolve(_cfg()["jar"])
    if not jar.exists():
        raise RuntimeError(f"Synthea jar not found at {jar} (see README: 'Installing Synthea').")
    return jar


def compact(bundle: dict, keep_types) -> dict:
    """Keep only `keep_types`; cascade-remove what can no longer stand. Returns removal stats."""
    keep = set(keep_types)
    drop = {sf.entry_key(e) for e in sf.entries(bundle) if e["resource"].get("resourceType") not in keep}
    before = Counter(e["resource"]["resourceType"] for e in sf.entries(bundle))
    removed = sf.remove_entries(bundle, drop) if drop else set()
    after = Counter(e["resource"]["resourceType"] for e in sf.entries(bundle))
    cascaded = {t: before[t] - after.get(t, 0) for t in keep if before[t] - after.get(t, 0)}
    return {"dropped_types": {t: n for t, n in before.items() if t not in keep},
            "cascaded": cascaded, "n_removed": len(removed)}


def summarize(bundle: dict, as_of: date, groups: dict) -> dict:
    """Index row for selection: demographics, cohort criteria and resource counts (no clinical detail)."""
    p = sf.patient(bundle) or {}
    starts = sorted(d for d in (sf.resource_date(r)[:10] for r in sf.resources(bundle, "Encounter")) if d)
    return {
        "patient_id": p.get("id"),
        "age": ss.age_on(p.get("birthDate"), as_of),
        "alive": not (p.get("deceasedDateTime") or p.get("deceasedBoolean")),
        "sex": p.get("gender"),
        "race_ethnicity": race_ethnicity(p) or "Not recorded",
        "state": ((p.get("address") or [{}])[0]).get("state"),
        "condition_groups": list(ss.condition_groups(bundle, groups)),
        "counts": dict(Counter(r.get("resourceType") for r in sf.resources(bundle))),
        "first_encounter": starts[0] if starts else None,
    }


def _synthea_command(out_dir: Path, n: int, min_age: int, max_age: int, years: int, seed: int, as_of: date):
    cfg = _cfg()
    return [java_cmd(), "-jar", str(jar_path()),
            "-p", str(n), "-a", f"{min_age}-{max_age}", "-s", str(seed), "-cs", str(seed),
            "-r", as_of.strftime("%Y%m%d"),
            f"--exporter.baseDirectory={out_dir}",
            f"--exporter.years_of_history={years}",
            "--exporter.fhir.export=true", "--exporter.hospital.fhir.export=false",
            "--exporter.practitioner.fhir.export=false", "--exporter.csv.export=false",
            cfg["state"]]


def build_pool(name: str, population: int = None, min_age: int = None, max_age: int = None,
               years_of_history: int = None, seed: int = None, as_of: date = None, progress=None) -> dict:
    """Run Synthea for pool `name` (adding to it if it exists). Returns the pool info."""
    cfg = _cfg()
    defaults = cfg["pools"].get(name, {})
    population = population or defaults["population"]
    min_age = defaults.get("min_age", 18) if min_age is None else min_age
    max_age = defaults.get("max_age", 90) if max_age is None else max_age
    years = years_of_history or defaults.get("years_of_history", 3)
    seed = seed if seed is not None else defaults.get("seed", 1)
    as_of = as_of or date.today()
    groups = load_config()["condition_groups"]

    pdir = pools_dir() / name
    (pdir / "patients").mkdir(parents=True, exist_ok=True)
    index = load_index(name)
    info = load_info(name) or {"name": name, "synthea_version": cfg["version"], "chunks": []}
    info.update(min_age=min_age, max_age=max_age, years_of_history=years, state=cfg["state"], as_of=str(as_of))
    with open(jar_path(), "rb") as f:
        info["jar_sha256"] = hashlib.sha256(f.read()).hexdigest()

    done, chunk_no = 0, len(info["chunks"])
    while done < population:
        n = min(cfg["chunk_size"], population - done)
        chunk_seed = seed + chunk_no
        tmp = pdir / "_tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        cmd = _synthea_command(tmp, n, min_age, max_age, years, chunk_seed, as_of)
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        added = 0
        for f in sorted((tmp / "fhir").glob("*.json")):
            if not ss.PATIENT_FILE.search(f.name):
                continue
            bundle = json.loads(f.read_text())
            stats = compact(bundle, cfg["keep_types"])
            row = summarize(bundle, as_of, groups)
            row.update(file=f"patients/{f.stem}.json.gz", chunk=chunk_no, compaction=stats)
            with gzip.open(pdir / row["file"], "wt", encoding="utf-8") as out:
                json.dump(bundle, out)
            index.append(row)
            added += 1
        shutil.rmtree(tmp, ignore_errors=True)
        info["chunks"].append({"chunk": chunk_no, "seed": chunk_seed, "population": n, "files": added,
                               "command": cmd[3:],
                               "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        (pdir / "index.json").write_text(json.dumps(index))
        (pdir / "pool_info.json").write_text(json.dumps(info, indent=1))
        done += n
        chunk_no += 1
        if progress:
            progress(done / population, f"Pool {name}: {done}/{population} people simulated")
    return info


def load_index(name: str) -> list[dict]:
    p = pools_dir() / name / "index.json"
    return json.loads(p.read_text()) if p.exists() else []


def load_info(name: str) -> dict | None:
    p = pools_dir() / name / "pool_info.json"
    return json.loads(p.read_text()) if p.exists() else None


def load_patient(name: str, row: dict) -> dict:
    with gzip.open(pools_dir() / name / row["file"], "rt", encoding="utf-8") as f:
        return json.load(f)


def list_pools() -> list[dict]:
    out = []
    if pools_dir().exists():
        for d in sorted(pools_dir().iterdir()):
            info = load_info(d.name)
            if info:
                idx = load_index(d.name)
                size = sum(f.stat().st_size for f in (d / "patients").glob("*.gz"))
                out.append({"pool": d.name, "people": len(idx), "alive": sum(r["alive"] for r in idx),
                            "ages": f"{info['min_age']}-{info['max_age']}", "years_of_history": info["years_of_history"],
                            "state": info["state"], "size_mb": round(size / 1e6, 1),
                            "synthea": info["synthea_version"]})
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("name")
    b.add_argument("--population", type=int)
    b.add_argument("--seed", type=int)
    sub.add_parser("list")
    a = ap.parse_args()
    if a.cmd == "build":
        build_pool(a.name, population=a.population, seed=a.seed,
                   progress=lambda f, msg: print(msg, flush=True))
    for row in list_pools():
        print(row)


if __name__ == "__main__":
    os.chdir(ROOT)
    main()
