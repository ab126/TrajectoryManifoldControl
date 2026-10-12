"""I01: read-only, standard-library OA-1 configuration and release contracts.

No scientific imports, device access, subprocesses, gain solving or execution.
Eligibility checks bindings supplied by a caller; they neither authenticate a
reviewer's identity nor grant authority. I05 must obtain independent, fresh
observations and trusted decision hashes. CLI conversion writes only new review
records beneath validation/; it never changes the descriptive proposal.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Mapping

HERE = Path(__file__).resolve().parent
SERIALIZATION = "python3.11.9-json-v1:sort_keys,ascii,compact,finite,binary64,positive-zero"
SUBMITTED_COMMIT = "325beae718ab093d3ca48b18a59ae61fd93798a3"
PROPOSAL_SHA256 = "6970c76f51624a341746e337f86b61697585bf246c1d98d78217ffebcb6b2b42"
SPEC_SHA256 = "3c871ffa94aa83c27758503266ba26f8fce9022db05313278be5edc0c2214aee"
SOURCE_HASHES = {
    "scripts/run_inverted_pendulum_mc.py": "b96f4871c282db5118cc7c08dd5f2d3a7d6a78770dbb4fc94d3efd4a05f1d2e4",
    "src/manifold_control.py": "df3d9680fce60538933fece012082a34a989b3febdc389b52de3ccb2cf5df0e4",
    "src/inverted_pendulum.py": "c2ce099a7ff020f3e0c75263035f14563e6364dfdeb018339c301be1da2a58d4",
}
STATES = ("specified", "implemented", "tested", "independently_audited", "adopted", "executed")
CHECK_STATUSES = ("pass", "fail", "unavailable", "not_checked")
REQUIRED_SCOPES = ("configuration", "gain_cost_rank", "policy", "trace_audit", "launcher",
                   "nonlinear_guards", "amended_timing")
REQUIRED_IMPLEMENTATION_FILES = tuple("validation/"+name for name in (
    "oracle_config.py", "oracle_math_v1.py", "oracle_policy_v1.py", "oracle_audit_v1.py",
    "gate_a_v1.py", "oracle_queue_v1.py"))
EVIDENCE_KEYS = frozenset(("observed_release", "artifact_sha256", "trusted_adoption_sha256",
                           "trusted_allocation_sha256", "archive_available", "amended_tau_s",
                           "used_worker_seconds", "ledger_sha256"))


class ConfigError(ValueError):
    def __init__(self, code: str, field: str, expected: Any, observed: Any):
        self.code, self.field, self.expected, self.observed = code, field, expected, observed
        super().__init__(f"{code} at {field}: expected {expected!r}, observed {observed!r}")


@dataclass(frozen=True)
class Check:
    status: str
    code: str
    field: str
    expected: Any = None
    observed: Any = None

    def __post_init__(self):
        if self.status not in CHECK_STATUSES:
            raise ValueError(f"Unknown check status: {self.status}")

    def to_dict(self) -> dict:
        return dict(status=self.status, code=self.code, field=self.field,
                    expected=self.expected, observed=self.observed)


@dataclass(frozen=True)
class Report:
    checks: tuple[Check, ...]

    @property
    def passed(self) -> bool:
        return bool(self.checks) and all(c.status == "pass" for c in self.checks)

    def to_dict(self) -> dict:
        return {"passed": self.passed, "checks": [c.to_dict() for c in self.checks]}


@dataclass(frozen=True)
class Leaf:
    kind: str
    choices: tuple = ()


@dataclass(frozen=True)
class Optional:
    schema: Any


@dataclass(frozen=True)
class Many:
    schema: Any


@dataclass(frozen=True)
class HashMap:
    """Nonempty repository-relative file -> lower-case SHA256 map."""


def _path(value: str, field: str) -> str:
    # Paths are portable relative identifiers, not evaluated locator expressions.
    if type(value) is not str:
        raise ConfigError("type", field, "relative path string", value)
    p = Path(value)
    if p.is_absolute() or ".." in p.parts or not p.parts or "\\" in value or p.as_posix() != value:
        raise ConfigError("path", field, "normalized relative path without traversal", value)
    return value


def _normalize(value: Any, schema: Any, field: str = "$") -> Any:
    if isinstance(schema, Optional):
        return None if value is None else _normalize(value, schema.schema, field)
    if isinstance(schema, Many):
        if not isinstance(value, list):
            raise ConfigError("type", field, "array", type(value).__name__)
        return [_normalize(v, schema.schema, f"{field}[{i}]") for i, v in enumerate(value)]
    if isinstance(schema, HashMap):
        if not isinstance(value, dict) or not value:
            raise ConfigError("type", field, "nonempty file/hash map", value)
        return {_path(k, field): _normalize(v, Leaf("sha256"), f"{field}.{k}") for k, v in value.items()}
    if isinstance(schema, dict):
        if not isinstance(value, dict):
            raise ConfigError("type", field, "object", type(value).__name__)
        if value.keys() != schema.keys():
            raise ConfigError("fields", field, sorted(schema), sorted(value, key=str))
        return {k: _normalize(value[k], s, f"{field}.{k}") for k, s in schema.items()}
    if isinstance(schema, list):
        if not isinstance(value, list) or len(value) != len(schema):
            raise ConfigError("shape", field, f"array of length {len(schema)}", value)
        return [_normalize(v, s, f"{field}[{i}]") for i, (v, s) in enumerate(zip(value, schema))]
    fixed = not isinstance(schema, Leaf)
    kind = ("bool" if isinstance(schema, bool) else "int" if isinstance(schema, int)
            else "real" if isinstance(schema, float) else "str") if fixed else schema.kind
    if kind == "bool":
        valid = type(value) is bool
    elif kind == "int":
        valid = type(value) is int and 0 <= value < 2**63
    elif kind == "real":
        valid = type(value) in (int, float)
        if valid:
            try:
                value = float(value)
                valid = math.isfinite(value)
            except (ValueError, OverflowError):
                valid = False
            if valid and value == 0:
                value = 0.0
    else:
        valid = type(value) is str and bool(value)
        if valid and kind in ("sha256", "commit"):
            valid = bool(re.fullmatch("[0-9a-f]{64}" if kind == "sha256" else "[0-9a-f]{40}", value))
        if valid and kind == "path":
            value = _path(value, field)
    if not valid:
        raise ConfigError("type", field, kind, value)
    if fixed and value != schema:
        raise ConfigError("profile_value", field, schema, value)
    if not fixed and schema.choices and value not in schema.choices:
        raise ConfigError("enum", field, schema.choices, value)
    return value


def _pairs(pairs):
    answer = {}
    for key, value in pairs:
        if key in answer:
            raise ConfigError("duplicate_key", key, "one occurrence", "duplicate")
        answer[key] = value
    return answer


def _decode(raw: bytes) -> dict:
    try:
        return json.loads(raw.decode("utf-8"), object_pairs_hook=_pairs,
                          parse_constant=lambda value: (_ for _ in ()).throw(
                              ConfigError("nonfinite", "$", "finite JSON numbers", value)))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ConfigError("json", "$", "valid UTF-8 JSON", str(error)) from error


def _canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, ensure_ascii=True, separators=(",", ":"),
                      allow_nan=False).encode("utf-8")


def _read(path: Path) -> bytes:
    try:
        return Path(path).read_bytes()
    except FileNotFoundError as error:
        raise ConfigError("unavailable", str(path), "available document", None) from error


def indexed_case_seed(root_seed: int, stream: str, indices: list[float]) -> int:
    # The RNG payload is the ORIGINAL list serialization, not the config format.
    raw = json.dumps([root_seed, stream, *indices], separators=(",", ":")).encode()
    return int.from_bytes(hashlib.sha256(raw).digest()[:8], "little") % (2**63 - 1)


SCIENCE_SCHEMA = {
    "schema_version": 1, "reference_id": "OA-1", "algorithm_profile": "oa1-v1",
    "plant": {"identifier": "submitted-uniform-rod-rk4", "mass_ratio": 2.0,
              "cart_mass_kg": 2.0, "rod_mass_kg": 1.0, "length_m": 1.0, "g": 9.81,
              "dt_physical_s": .02, "dt_normalized": .02/math.sqrt(1/9.81),
              "input_bounds": [-10.0, 10.0], "angle_wrapping": False},
    "execution": {"prediction_device": "cuda", "gpu_uuid": "GPU-ab0ee8e5-cf5f-5b7b-82dc-1e3f054fa136",
                  "oracle_dtype": "float64", "plant_dtype": "float64", "local_dtype": "float64",
                  "local_device": "cpu", "workers": 1, "cpu_threads": 1,
                  "deterministic_algorithms": True, "TF32": False, "cpu_fallback": False},
    "cases": {"count": 200, "root_seed": 20261006, "stream": "gate_a_oracle_v1_cases",
              "indices": [1.0], "case_seed": 4264006537072803998,
              "half_widths": [1.5, 2.5, 1.30, 3.5], "scale": 1.0},
    "endpoint": {"transitions": 1000, "states": 1001, "physical_seconds": 20.0,
                 "final_indices": [850, 1000], "rail_m": 2.4, "angle_rad": math.pi/2,
                 "final_tolerances": [.1, .05, .02, .05], "all_states_finite": True,
                 "exceptions_fail": True, "past_violations_latch": True, "incomplete_cannot_pass": True,
                 "timing_exempt": True, "minimum_successes": 196,
                 "half_fraction": [95, 100], "both_halves_nonempty": True},
    "objective": {"Q": [2.0, 30.0, 100.0, 10.0], "R": .03, "Qf": [24.0, 360.0, 1200.0, 120.0],
                  "rate_weight": .002, "state_reference": [0.0, 0.0, 0.0, 0.0],
                  "input_reference": 0.0, "dt_multiplier": False, "first_input_rate_term": False},
    "local": {"gain_construction": "rk4-upright-dare-v1", "entry_max": .75, "stay_max": 1.0,
              "entry_steps": 60, "stay_steps": 1, "decrease_factor": .5, "decrease_tolerance": 1e-10,
              "gain_atol": 1e-10, "gain_rtol": 1e-10},
    "search": {"horizon": 60, "starts": 1, "population": 256, "elites": 32, "iterations": 3,
               "smoothing": .15, "unbiased_std": True, "std_floor": .02,
               "state_scale": [1.0, 1.0, 1.0, 1.0], "case_prefix": "oracle_v1"},
    "queue": {"maximum_inputs": 5, "consistency_atol": 1e-10, "consistency_rtol": 1e-10,
              "maximum_calls_per_sample": 1},
    "fallback": {"identifier": "oa1-zero-one-sample-v1"},
}
ARTIFACT = {"path": Leaf("path"), "sha256": Leaf("sha256")}
BOUND_ARTIFACT = {**ARTIFACT, "scientific_sha256": Leaf("sha256")}
EVENT = {"state": Leaf("str", STATES), "release_sha256": Optional(Leaf("sha256")),
         "reference": ARTIFACT, "reviewer": Optional(Leaf("str")), "independent": Leaf("bool"),
         "scope": Many(Leaf("str"))}
RUNTIME_SCHEMA = {
    "prediction_device": "cuda", "gpu_uuid": SCIENCE_SCHEMA["execution"]["gpu_uuid"],
    "hardware": "NVIDIA L4", "python_version": "3.11.9",
    "driver": Leaf("str"), "torch_version": Leaf("str"),
    "cuda_runtime": Leaf("str"), "oracle_dtype": "float64", "plant_dtype": "float64",
    "local_dtype": "float64", "cpu_threads": 1, "workers": 1,
    "deterministic_algorithms": True, "TF32": False,
}
METADATA_SCHEMA = {
    "metadata_version": 1, "scientific_sha256": Leaf("sha256"), "specification": ARTIFACT,
    "submitted": {"commit": SUBMITTED_COMMIT, "files": SOURCE_HASHES},
    "implementation": Optional({"commit": Leaf("commit"), "files": HashMap()}),
    "environment": Optional({"manifests": HashMap(), "native_files": HashMap()}),
    "runtime": Optional(RUNTIME_SCHEMA), "derived_gain": Optional(BOUND_ARTIFACT),
    "trace_contract": Optional(ARTIFACT), "events": Many(EVENT),
    "archive": {"status": Leaf("str", ("available", "unavailable", "not_checked")),
                "locator": Optional(Leaf("str")), "manifest": Optional(ARTIFACT)},
    "timing": Optional({**ARTIFACT, "release_sha256": Leaf("sha256")}),
    "allocation": Optional({"status": Leaf("str", ("proposed", "adopted")),
                            "cap_seconds": Leaf("int"), "support_seconds": Leaf("int"),
                            "queue_seconds": Leaf("int"), "fixed_overhead_seconds": Leaf("int"),
                            "archive_seconds": Leaf("int"), "allowed_jobs": ["oa1_gate_a"],
                            "reference": Optional(ARTIFACT)}),
    "adoption": Optional({"reference": ARTIFACT, "release_sha256": Leaf("sha256"),
                          "approved_by": Leaf("str"), "allowed_jobs": ["oa1_gate_a"]}),
}
IDENTITY_KEYS = ("scientific_sha256", "specification", "submitted", "implementation",
                 "environment", "runtime", "derived_gain", "trace_contract")


@dataclass(frozen=True)
class ScientificConfig:
    _canonical: bytes

    def __post_init__(self):
        value = _normalize(_decode(self._canonical), SCIENCE_SCHEMA)
        object.__setattr__(self, "_canonical", _canonical(value))

    def to_dict(self) -> dict:
        return _decode(self._canonical)  # fresh copy; callers cannot mutate the contract


@dataclass(frozen=True)
class ReleaseMetadata:
    _canonical: bytes

    def __post_init__(self):
        value = _normalize(_decode(self._canonical), METADATA_SCHEMA)
        object.__setattr__(self, "_canonical", _canonical(value))

    def to_dict(self) -> dict:
        return _decode(self._canonical)


def scientific_from_dict(value: dict) -> ScientificConfig:
    return ScientificConfig(_canonical(_normalize(value, SCIENCE_SCHEMA)))


def metadata_from_dict(value: dict) -> ReleaseMetadata:
    return ReleaseMetadata(_canonical(_normalize(value, METADATA_SCHEMA)))


def load_scientific(path: Path) -> ScientificConfig:
    return ScientificConfig(_read(path))


def load_release_metadata(path: Path) -> ReleaseMetadata:
    return ReleaseMetadata(_read(path))


def canonical_bytes(config: ScientificConfig) -> bytes:
    if not isinstance(config, ScientificConfig):
        raise TypeError("validated ScientificConfig required")
    return config._canonical


def scientific_hash(config: ScientificConfig) -> str:
    return hashlib.sha256(canonical_bytes(config)).hexdigest()


def release_identity(metadata: ReleaseMetadata) -> dict:
    m = metadata.to_dict()
    return {key: m[key] for key in IDENTITY_KEYS}


def release_hash(metadata: ReleaseMetadata) -> str:
    return hashlib.sha256(_canonical(release_identity(metadata))).hexdigest()


def _equal(expected, observed, field, code="match") -> Check:
    return Check("pass" if expected == observed else "fail", code, field, expected, observed)


def validate_cross_fields(config: ScientificConfig, metadata: ReleaseMetadata) -> Report:
    c, m = config.to_dict(), metadata.to_dict()
    r = release_hash(metadata)
    checks = [
        _equal(scientific_hash(config), m["scientific_sha256"], "scientific_sha256", "config_hash"),
        _equal(SPEC_SHA256, m["specification"]["sha256"], "specification.sha256", "spec_hash"),
        _equal([12*q for q in c["objective"]["Q"]], c["objective"]["Qf"], "objective.Qf", "terminal_weight"),
        _equal(indexed_case_seed(c["cases"]["root_seed"], c["cases"]["stream"], c["cases"]["indices"]),
               c["cases"]["case_seed"], "cases.case_seed", "seed"),
        _equal(c["endpoint"]["transitions"]+1, c["endpoint"]["states"], "endpoint.states", "samples"),
        _equal(c["plant"]["dt_physical_s"]*c["endpoint"]["transitions"],
               c["endpoint"]["physical_seconds"], "endpoint.physical_seconds", "duration"),
    ]
    for name in ("derived_gain",):
        if m[name] is not None:
            checks.append(_equal(m["scientific_sha256"], m[name]["scientific_sha256"], name, "artifact_config"))
    expected_states = STATES[:len(m["events"])]
    actual_states = tuple(event["state"] for event in m["events"])
    checks.append(_equal(expected_states, actual_states, "events", "lifecycle_order"))
    checks.append(_equal(True, bool(actual_states), "events", "specified_required"))
    for event in m["events"]:
        state = event["state"]
        if state != "specified":
            checks.append(_equal(r, event["release_sha256"], f"events.{state}", "event_release"))
        if state == "implemented" and m["implementation"] is None:
            checks.append(Check("fail", "implementation_missing", "implementation"))
        if state == "independently_audited":
            checks.append(_equal(True, event["independent"] and bool(event["reviewer"]),
                                 "events.independently_audited", "independent_declaration"))
    if m["adoption"] is not None:
        checks.append(_equal(r, m["adoption"]["release_sha256"], "adoption.release_sha256", "adoption_release"))
        checks.append(_equal(True, "adopted" in actual_states, "adoption", "adoption_event"))
    elif "adopted" in actual_states:
        checks.append(Check("fail", "adoption_record_missing", "adoption"))
    if m["timing"] is not None:
        checks.append(_equal(r, m["timing"]["release_sha256"], "timing.release_sha256", "timing_release"))
    if m["archive"]["status"] == "available":
        checks.append(_equal(True, bool(m["archive"]["locator"] and m["archive"]["manifest"]),
                             "archive", "archive_manifest"))
    return Report(tuple(checks))


def compare_release(expected: ReleaseMetadata, observed: Mapping[str, Any]) -> Report:
    """Compare caller-supplied observations; absent observations are unavailable."""
    checks = []

    def walk(want, got, field):
        if want is None:
            checks.append(Check("not_checked", "unbound", field))
        elif got is None:
            checks.append(Check("unavailable", "observation_missing", field, want))
        elif isinstance(want, dict):
            if not isinstance(got, dict):
                checks.append(Check("fail", "observation_type", field, "object", got))
                return
            if set(got)-set(want):
                checks.append(Check("fail", "observation_fields", field, sorted(want), sorted(got)))
            for key, value in want.items():
                walk(value, got.get(key), f"{field}.{key}")
        else:
            # Equality alone would accept True as integer 1 in runtime observations.
            match = type(got) is type(want) and got == want
            checks.append(Check("pass" if match else "fail", "observed_match", field, want, got))

    walk(release_identity(expected), dict(observed), "release")
    return Report(tuple(checks))


def execution_eligibility(config: ScientificConfig, metadata: ReleaseMetadata,
                          evidence: Mapping[str, Any]) -> Report:
    """No authority is inferred from config booleans or self-reported adoption.

    I05 supplies observed_release, artifact_sha256, trusted_adoption_sha256,
    trusted_allocation_sha256, archive_available, amended_tau_s,
    used_worker_seconds and ledger_sha256 from separately verified sources.
    A passing report is a prerequisite, never an execution capability.
    """
    m = metadata.to_dict()
    checks = list(validate_cross_fields(config, metadata).checks)
    if not isinstance(evidence, Mapping):
        return Report(tuple(checks+[Check("fail", "evidence_type", "evidence", "mapping")]))
    if set(evidence)-EVIDENCE_KEYS:
        checks.append(Check("fail", "evidence_fields", "evidence", sorted(EVIDENCE_KEYS),
                            sorted(evidence, key=str)))
    for key in ("observed_release", "artifact_sha256"):
        if key in evidence and not isinstance(evidence[key], dict):
            return Report(tuple(checks+[Check("fail", "evidence_type", key, "object")]))
    for key in ("trusted_adoption_sha256", "trusted_allocation_sha256", "ledger_sha256"):
        value = evidence.get(key)
        if value is not None and (type(value) is not str or not re.fullmatch("[0-9a-f]{64}", value)):
            checks.append(Check("fail", "evidence_hash", key, "lowercase SHA256", value))
    if "archive_available" in evidence and type(evidence["archive_available"]) is not bool:
        checks.append(Check("fail", "evidence_type", "archive_available", "boolean"))
    checks.extend(compare_release(metadata, evidence.get("observed_release", {})).checks)
    actual_hashes = evidence.get("artifact_sha256", {})

    def artifact(ref, field):
        if ref is None:
            checks.append(Check("unavailable", "artifact_unbound", field))
        elif ref["path"] not in actual_hashes:
            checks.append(Check("unavailable", "artifact_missing", field, ref["sha256"]))
        else:
            checks.append(_equal(ref["sha256"], actual_hashes[ref["path"]], field, "artifact_hash"))

    for name in ("specification", "derived_gain", "trace_contract", "timing"):
        artifact(m[name], name)
    if m["implementation"] is None:
        checks.append(Check("unavailable", "implementation_unbound", "implementation"))
    else:
        for path in REQUIRED_IMPLEMENTATION_FILES:
            checks.append(Check("pass" if path in m["implementation"]["files"] else "unavailable",
                                "implementation_coverage", path))
    for event in m["events"]:
        artifact(event["reference"], f"events.{event['state']}")
        if event["state"] in ("tested", "independently_audited"):
            checks.append(_equal(True, set(REQUIRED_SCOPES).issubset(event["scope"]),
                                 f"events.{event['state']}.scope", "required_scope"))
    if "adopted" not in [event["state"] for event in m["events"]]:
        checks.append(Check("fail", "not_adopted", "events", "adopted"))
    if m["adoption"] is None:
        checks.append(Check("fail", "proposed_config", "adoption", "externally adopted release"))
    else:
        artifact(m["adoption"]["reference"], "adoption.reference")
        trusted = evidence.get("trusted_adoption_sha256")
        checks.append(Check("unavailable", "trusted_adoption_missing", "adoption") if trusted is None
                      else _equal(trusted, m["adoption"]["reference"]["sha256"], "adoption", "trusted_adoption"))
    artifact(m["archive"]["manifest"], "archive.manifest")
    checks.append(Check("pass" if m["archive"]["status"] == "available" and evidence.get("archive_available") is True
                        else "unavailable", "archive_availability", "archive"))
    allocation = m["allocation"]
    if allocation is None or allocation["status"] != "adopted":
        checks.append(Check("fail", "allocation_not_adopted", "allocation"))
    else:
        artifact(allocation["reference"], "allocation.reference")
        trusted = evidence.get("trusted_allocation_sha256")
        if trusted is None or allocation["reference"] is None:
            checks.append(Check("unavailable", "trusted_allocation_missing", "allocation"))
        else:
            checks.append(_equal(trusted, allocation["reference"]["sha256"], "allocation", "trusted_allocation"))
        tau, used = evidence.get("amended_tau_s"), evidence.get("used_worker_seconds")
        if not evidence.get("ledger_sha256"):
            checks.append(Check("unavailable", "ledger_missing", "allocation"))
        if tau is None or used is None:
            checks.append(Check("unavailable", "reservation_unmeasured", "allocation"))
        elif type(tau) not in (int, float) or type(used) not in (int, float):
            checks.append(Check("fail", "reservation_invalid", "allocation"))
        else:
            try:
                valid = math.isfinite(tau) and tau > 0 and math.isfinite(used) and used >= 0
            except OverflowError:
                valid = False
            if not valid:
                return Report(tuple(checks+[Check("fail", "reservation_invalid", "allocation")]))
            from decimal import Decimal
            d = lambda n: Decimal(str(n))
            planning = d(200000)*d(tau)*d("1.2") + d(allocation["fixed_overhead_seconds"])
            slots = sum(allocation[k] for k in ("support_seconds", "queue_seconds", "archive_seconds"))
            fits = planning <= d(allocation["queue_seconds"]) and d(used)+d(slots) <= d(allocation["cap_seconds"])
            checks.append(_equal(True, fits, "allocation", "reservation_fits"))
    return Report(tuple(checks))


def convert_proposal(path: Path) -> tuple[ScientificConfig, ReleaseMetadata, dict]:
    """Convert only the exact reviewed descriptive proposal; never evaluate prose.

    The byte hash is the allowlist: any change, even to discarded metadata,
    requires a reviewed converter revision rather than silent data loss.
    """
    raw = _read(path)
    actual = hashlib.sha256(raw).hexdigest()
    if actual != PROPOSAL_SHA256:
        raise ConfigError("proposal_hash", str(path), PROPOSAL_SHA256, actual)
    proposal = _decode(raw)
    config = scientific_from_dict(SCIENCE_SCHEMA)
    digest = scientific_hash(config)
    metadata = metadata_from_dict({
        "metadata_version": 1, "scientific_sha256": digest,
        "specification": {"path": "validation/oracle_amendment_proposal_v1.md", "sha256": SPEC_SHA256},
        "submitted": {"commit": SUBMITTED_COMMIT, "files": SOURCE_HASHES},
        "implementation": None, "environment": None, "runtime": None,
        "derived_gain": None, "trace_contract": None, "timing": None,
        "events": [{"state": "specified", "release_sha256": None,
                    "reference": {"path": "validation/oa1_config_proposed.json", "sha256": actual},
                    "reviewer": None, "independent": False, "scope": ["specification"]}],
        "archive": {"status": "not_checked", "locator": None, "manifest": None},
        "allocation": {"status": "proposed", "cap_seconds": 72000, "support_seconds": 3600,
                       "queue_seconds": 61200, "fixed_overhead_seconds": 1200,
                       "archive_seconds": 1800, "allowed_jobs": ["oa1_gate_a"], "reference": None},
        "adoption": None,
    })
    mapping = {
        "plant": "scientific.plant; fixed formulas -> numeric dt and profile identifier",
        "device": "scientific.execution; environment provenance remains unbound",
        "cases": "scientific.cases; existing indexed-list seed serialization preserved",
        "endpoint": "scientific.endpoint; pi/2 evaluated by math, not expression evaluation",
        "objective": "scientific.objective; indexing fixed by oa1-v1",
        "gain": "scientific.local gain method/tolerances; reference K/P retained in original proposal; I02 artifact pending",
        "local_switch": "scientific.local; remaining rules fixed by oa1-v1",
        "search": "scientific.search; slot/rank/RNG rules fixed by oa1-v1",
        "queue": "scientific.queue; shift/invalidation rules fixed by oa1-v1",
        "fallback": "scientific.fallback profile identifier",
        "logging": "trace-contract requirements retained in OA-1; I04 artifact pending",
        "allocation": "metadata.allocation; prospective slots only; no adoption or timing evidence",
    }
    report = {"source_sha256": actual, "scientific_sha256": digest, "serialization": SERIALIZATION,
              "mapping": mapping, "metadata_source_fields": [key for key in proposal if key not in mapping],
              "unmapped_fields": [], "input_policy": "exact reviewed byte-hash allowlist; no arbitrary expressions",
              "gain_verification": "not_checked; I02 required", "execution_authorized": False}
    return config, metadata, report


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--proposal", type=Path, default=HERE/"oa1_config_proposed.json")
    parser.add_argument("--output-dir", type=Path, help="New review-only directory beneath validation/")
    args = parser.parse_args(argv)
    try:
        config, metadata, mapping = convert_proposal(args.proposal)
        consistency = validate_cross_fields(config, metadata)
        eligibility = execution_eligibility(config, metadata, {})
        decision = {"batch": "I01", "oa1_lifecycle": "specified", "release_sha256": release_hash(metadata),
                    "consistency": consistency.to_dict(), "execution_eligibility": eligibility.to_dict(),
                    "execution_authorized": False, "python": sys.version,
                    "pending": ["I02-I05 implementation/testing", "independent audit", "nonlinear guards",
                                "amended timing", "archive contract", "allocation", "OA-1 adoption"]}
        if args.output_dir:
            destination = args.output_dir.resolve()
            if destination.parent != HERE/"reviews":
                raise ConfigError("output_path", str(destination), "new direct child of validation/reviews/", str(destination))
            # Directory creation is exclusive; existing records are never overwritten.
            destination.mkdir(parents=True, exist_ok=False)
            for name, raw in (("scientific.json", canonical_bytes(config)),
                              ("release_metadata.json", metadata._canonical),
                              ("mapping.json", _canonical(mapping)), ("decision.json", _canonical(decision))):
                with (destination/name).open("xb") as file:
                    file.write(raw)
            print(json.dumps({"directory": str(destination), "scientific_sha256": scientific_hash(config),
                              "consistency_passed": consistency.passed, "execution_eligible": eligibility.passed}))
        else:
            print(json.dumps(decision, indent=2))
        return 0 if consistency.passed else 2
    except (ConfigError, FileExistsError) as error:
        print(str(error), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
