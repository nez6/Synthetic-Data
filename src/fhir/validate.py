"""Schema validation against the official FHIR models (via the `fhir.resources` package).

R5, R4B and STU3 are validated directly. R4 is validated with the R4B models, which are
identical for the resources used here. DSTU2 has no maintained Python models, so it is
reported as "not validated".
"""
import importlib

from pydantic import ValidationError

_PACKAGES = {
    "R5": "fhir.resources",
    "R4B": "fhir.resources.R4B",
    "R4": "fhir.resources.R4B",
    "STU3": "fhir.resources.STU3",
}


def can_validate(version: str) -> bool:
    return version in _PACKAGES


def _model(version: str, resource_type: str):
    module = importlib.import_module(f"{_PACKAGES[version]}.{resource_type.lower()}")
    return getattr(module, resource_type)


def validate_bundle(bundle: dict, version: str) -> dict:
    """Return {"validated": bool, "valid": bool | None, "errors": [str], "note": str}."""
    if not can_validate(version):
        return {"validated": False, "valid": None, "errors": [],
                "note": f"{version} has no maintained schema library; output was not schema-checked."}

    errors = []
    entries = bundle.get("entry", [])
    for entry in entries:
        resource = entry.get("resource", {})
        rtype = resource.get("resourceType", "?")
        try:
            _model(version, rtype).model_validate(resource)
        except (ValidationError, ValueError) as e:
            for err in getattr(e, "errors", lambda: [{"loc": (), "msg": str(e)}])():
                loc = ".".join(str(p) for p in err.get("loc", ()))
                errors.append(f"{rtype}/{resource.get('id', '?')}: {loc} — {err.get('msg')}")
        except (ImportError, AttributeError):
            errors.append(f"Unknown resourceType {rtype!r} for {version}")

    shell = {k: v for k, v in bundle.items() if k != "entry"}
    try:
        _model(version, "Bundle").model_validate(shell)
    except (ValidationError, ValueError) as e:
        for err in getattr(e, "errors", lambda: [{"loc": (), "msg": str(e)}])():
            errors.append(f"Bundle: {'.'.join(str(p) for p in err.get('loc', ()))} — {err.get('msg')}")

    note = "Validated with R4B models (identical to R4 for these resources)." if version == "R4" else \
        f"Validated against the official FHIR {version} schema."
    return {"validated": True, "valid": not errors, "errors": errors, "note": note}
