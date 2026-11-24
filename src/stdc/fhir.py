"""FHIR R4 helpers for Synthea patient bundles.

Synthea writes one transaction bundle per patient. Resources inside the
bundle reference each other as ``urn:uuid:<id>``; Practitioner, Organization
and Location are conditional references (``Practitioner?identifier=...``)
into separate directory bundles.
"""

SNOMED = "http://snomed.info/sct"
ICD10_CM = "http://hl7.org/fhir/sid/icd-10-cm"
LOINC = "http://loinc.org"
RXNORM = "http://www.nlm.nih.gov/research/umls/rxnorm"
CVX = "http://hl7.org/fhir/sid/cvx"
CPT = "http://www.ama-assn.org/go/cpt"
CDT = "http://www.ada.org/cdt"
HCPCS = "urn:oid:2.16.840.1.113883.6.285"
ICD10_PCS = "http://www.cms.gov/Medicare/Coding/ICD10"

# Synthea stamps every patient with this identifier; real EHR exports never carry it.
SYNTHEA_ID_SYSTEM = "https://github.com/synthetichealth/synthea"
SSN_SYSTEM = "http://hl7.org/fhir/sid/us-ssn"
SYNTHETIC_SSN_PREFIX = "999-"  # never issued by the SSA; Synthea uses it for every SSN

GENDERS = ("male", "female", "other", "unknown")

# Resolved outside the patient bundle (provider directory), so not checked for integrity.
DIRECTORY_TYPES = ("Practitioner", "PractitionerRole", "Organization", "Location")

# When the encounter one of these points at is removed, drop the reference and keep the
# resource: problems, allergies, medications and immunization history outlive the visit
# that recorded them.
KEEP_ON_CONTEXT_LOSS = ("Condition", "AllergyIntolerance", "MedicationRequest",
                        "MedicationStatement", "Immunization", "CarePlan", "CareTeam",
                        # local: lab/vital history, reports and procedures also stay valid without the
                        # visit that recorded them; they are then capped by the cohort maximum (most
                        # recent first) instead of being deleted with an older encounter.
                        "Observation", "DiagnosticReport", "Procedure")
CONTEXT_FIELDS = ("encounter", "context")

# Top-level lists that must not be empty; the resource is dropped if they become empty.
REQUIRED_LISTS = {("Provenance", "target")}


def entries(bundle):
    return bundle.get("entry", [])


def resources(bundle, resource_type=None):
    for e in entries(bundle):
        r = e.get("resource", {})
        if resource_type is None or r.get("resourceType") == resource_type:
            yield r


def patient(bundle):
    ps = list(resources(bundle, "Patient"))
    return ps[0] if len(ps) == 1 else None


def label(resource):
    return f"{resource.get('resourceType')}/{resource.get('id')}"


def first_coding(concept):
    return ((concept or {}).get("coding") or [{}])[0]


def resource_date(r):
    """Best clinical date for ordering, as an ISO string ('' if none)."""
    for path in (("period", "start"), ("effectivePeriod", "start"), ("performedPeriod", "start"),
                 ("billablePeriod", "start")):
        v = r.get(path[0], {}).get(path[1]) if isinstance(r.get(path[0]), dict) else None
        if v:
            return v
    for key in ("effectiveDateTime", "performedDateTime", "occurrenceDateTime", "authoredOn",
                "onsetDateTime", "recordedDate", "issued", "date", "created", "recorded"):
        if r.get(key):
            return r[key]
    return ""


# ------------------------------------------------------------- references

class Index:
    """Resolves references in one bundle to entry keys (the entry's fullUrl)."""

    def __init__(self, bundle):
        self.by_ref = {}
        for e in entries(bundle):
            r = e.get("resource", {})
            key = e.get("fullUrl") or label(r)
            self.by_ref[key] = key
            self.by_ref[label(r)] = key

    def resolve(self, ref):
        return self.by_ref.get(ref)


def ref_kind(ref):
    if not isinstance(ref, str) or not ref:
        return "invalid"
    if ref.startswith("#"):
        return "contained"
    if "?" in ref:
        return "conditional"
    return "local"


def walk_references(obj, path=()):
    """Yield (path, reference string) for every Reference in a resource."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k == "reference" and isinstance(v, str):
                yield path, v
            elif k != "contained":
                yield from walk_references(v, path + (k,))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from walk_references(v, path + (i,))


def _prune(obj, dead):
    """Remove references to dead entries from obj in place.

    Returns True when obj itself holds a single-valued reference to a dead
    entry (so its nearest list ancestor, or the resource, must go).
    """
    if isinstance(obj, dict):
        if isinstance(obj.get("reference"), str) and dead(obj["reference"]):
            return True
        kill = False
        for k in list(obj):
            if k == "contained":
                continue
            v = obj[k]
            if isinstance(v, list):
                had = len(v)
                v[:] = [item for item in v if not _prune(item, dead)]
                if had and not v:
                    del obj[k]
            elif isinstance(v, dict) and _prune(v, dead):
                kill = True
        return kill
    return False


def remove_entries(bundle, keys):
    """Remove entries and everything that can no longer stand without them.

    Returns the set of entry keys removed (including cascaded removals). Builds a
    reverse-reference index once, so each pass only revisits resources that point
    at something just removed (Synthea bundles can hold 20,000+ resources).
    """
    index = Index(bundle)
    by_key = {entry_key(e): e["resource"] for e in entries(bundle)}
    referrers = {}
    for key, r in by_key.items():
        for _, ref in walk_references(r):
            target = index.resolve(ref)
            if target is not None and target != key:
                referrers.setdefault(target, set()).add(key)
    removed = set(keys)
    dead = lambda ref: index.resolve(ref) in removed  # noqa: E731
    frontier = set(keys)
    while frontier:
        touched = set().union(*(referrers.get(k, ()) for k in frontier)) - removed
        newly = set()
        for key in touched:
            r = by_key[key]
            if r.get("resourceType") in KEEP_ON_CONTEXT_LOSS:
                for f in CONTEXT_FIELDS:
                    if isinstance(r.get(f), dict) and dead(r[f].get("reference")):
                        del r[f]
            lists_before = {k for k, v in r.items() if isinstance(v, list) and v}
            if _prune(r, dead) or any(
                    (r.get("resourceType"), k) in REQUIRED_LISTS and k not in r for k in lists_before):
                newly.add(key)
        removed |= newly
        frontier = newly
    bundle["entry"] = [e for e in entries(bundle) if entry_key(e) not in removed]
    return removed


def entry_key(entry):
    return entry.get("fullUrl") or label(entry.get("resource", {}))
