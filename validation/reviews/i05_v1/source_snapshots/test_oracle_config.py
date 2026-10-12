"""I01 contract tests: stdlib/synthetic metadata only, no plant or controller."""
import ast
import copy
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
from uuid import uuid4

HERE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(HERE))
import oracle_config as oc


class OracleConfigTests(unittest.TestCase):
    def setUp(self):
        self.config, self.metadata, self.mapping = oc.convert_proposal(HERE/"oa1_config_proposed.json")

    def bad_science(self, group, key, value, code=None):
        data = self.config.to_dict()
        data[group][key] = value
        with self.assertRaises(oc.ConfigError) as raised:
            oc.scientific_from_dict(data)
        if code:
            self.assertEqual(raised.exception.code, code)

    def future_fixture(self):
        """Imaginary future release/observations, never an actual adoption."""
        m = self.metadata.to_dict()
        m["implementation"] = {"commit": "a"*40,
                               "files": {path: "1"*64 for path in oc.REQUIRED_IMPLEMENTATION_FILES}}
        m["environment"] = {"manifests": {"validation/requirements.lock": "2"*64},
                            "native_files": {"validation/.venv-gpu/lib/example.so": "3"*64}}
        m["runtime"] = dict(prediction_device="cuda", gpu_uuid=oc.SCIENCE_SCHEMA["execution"]["gpu_uuid"],
                            hardware="NVIDIA L4", python_version="3.11.9",
                            driver="synthetic-driver", torch_version="synthetic-torch",
                            cuda_runtime="synthetic-runtime", oracle_dtype="float64", plant_dtype="float64",
                            local_dtype="float64", cpu_threads=1, workers=1, deterministic_algorithms=True, TF32=False)
        def artifact(name):
            return {"path": "validation/fixtures/"+name, "sha256": hashlib.sha256(name.encode()).hexdigest()}
        m["derived_gain"] = {**artifact("gain.json"), "scientific_sha256": m["scientific_sha256"]}
        m["trace_contract"] = artifact("trace.md")
        release = oc.release_hash(oc.metadata_from_dict(m))
        m["events"] = [{"state": state, "release_sha256": release, "reference": artifact(state+".json"),
                        "reviewer": "synthetic-independent-reviewer" if state == "independently_audited" else None,
                        "independent": state == "independently_audited", "scope": list(oc.REQUIRED_SCOPES)}
                       for state in oc.STATES[:5]]
        m["archive"] = {"status": "available", "locator": "synthetic-archive-only", "manifest": artifact("archive.json")}
        m["timing"] = {**artifact("timing.json"), "release_sha256": release}
        m["allocation"]["status"] = "adopted"
        m["allocation"]["reference"] = artifact("allocation.json")
        m["adoption"] = {"reference": artifact("adoption.json"), "release_sha256": release,
                         "approved_by": "synthetic-authority", "allowed_jobs": ["oa1_gate_a"]}
        metadata = oc.metadata_from_dict(m)
        observed = oc.release_identity(metadata)
        hashes = {m[key]["path"]: m[key]["sha256"] for key in ("specification", "derived_gain", "trace_contract", "timing")}
        hashes.update({event["reference"]["path"]: event["reference"]["sha256"] for event in m["events"]})
        for ref in (m["archive"]["manifest"], m["allocation"]["reference"], m["adoption"]["reference"]):
            hashes[ref["path"]] = ref["sha256"]
        evidence = dict(observed_release=observed, artifact_sha256=hashes,
                        trusted_adoption_sha256=m["adoption"]["reference"]["sha256"],
                        trusted_allocation_sha256=m["allocation"]["reference"]["sha256"],
                        archive_available=True, amended_tau_s=.25, used_worker_seconds=4987.356492,
                        ledger_sha256="4"*64)
        return metadata, evidence

    def test_import_has_only_standard_library_dependencies(self):
        tree = ast.parse((HERE/"oracle_config.py").read_text())
        names = {node.module.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
        names |= {alias.name.split(".")[0] for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names}
        self.assertFalse(names - sys.stdlib_module_names - {"__future__"})
        self.assertFalse(names & {"common", "src", "torch", "numpy", "scipy", "subprocess"})

    def test_exact_proposal_conversion_remains_proposed(self):
        self.assertTrue(oc.validate_cross_fields(self.config, self.metadata).passed)
        self.assertEqual(self.mapping["source_sha256"], oc.PROPOSAL_SHA256)
        self.assertEqual(self.mapping["unmapped_fields"], [])
        self.assertFalse(oc.execution_eligibility(self.config, self.metadata, {}).passed)
        self.assertIsNone(self.metadata.to_dict()["adoption"])

    def test_canonical_config_matches_frozen_golden_hash(self):
        self.assertEqual(oc.scientific_hash(self.config),
                         "38d6c75362c3b333d697844c2bcbff46a5f7c9950e1c6b94d52ae8f076ea4b00")
        self.assertEqual(len(oc.canonical_bytes(self.config)), 1906)
        self.assertFalse(oc.canonical_bytes(self.config).endswith(b"\n"))

    def test_converter_does_not_silently_drop_modified_fields(self):
        with tempfile.TemporaryDirectory(dir=HERE/"tests") as directory:
            path = Path(directory)/"proposal.json"
            data = json.loads((HERE/"oa1_config_proposed.json").read_text())
            data["logging"]["selected_and_warm_full_plans"] = False
            path.write_text(json.dumps(data))
            with self.assertRaises(oc.ConfigError) as error:
                oc.convert_proposal(path)
            self.assertEqual(error.exception.code, "proposal_hash")

    def test_numeric_normalization_key_order_and_zero(self):
        data = self.config.to_dict()
        data["objective"]["Q"][0] = 2
        data["objective"]["state_reference"][0] = -0.0
        data["cases"]["indices"] = [1]
        data = dict(reversed(list(data.items())))
        equivalent = oc.scientific_from_dict(data)
        self.assertEqual(oc.canonical_bytes(equivalent), oc.canonical_bytes(self.config))
        self.assertEqual(oc.scientific_hash(equivalent), oc.scientific_hash(self.config))
        self.assertIs(type(equivalent.to_dict()["cases"]["indices"][0]), float)
        self.assertNotIn(b"-0.0", oc.canonical_bytes(equivalent))
        self.assertFalse(oc.canonical_bytes(equivalent).endswith(b"\n"))

    def test_large_seed_and_original_rng_payload(self):
        c = self.config.to_dict()["cases"]
        self.assertIs(type(c["case_seed"]), int)
        self.assertEqual(c["case_seed"], 4264006537072803998)
        self.assertEqual(oc.indexed_case_seed(c["root_seed"], c["stream"], c["indices"]), c["case_seed"])
        self.assertNotEqual(oc.indexed_case_seed(c["root_seed"], c["stream"], [1]), c["case_seed"])

    def test_objects_are_immutable_and_return_fresh_copies(self):
        data = self.config.to_dict()
        data["objective"]["Q"][0] = 99
        self.assertEqual(self.config.to_dict()["objective"]["Q"][0], 2)
        with self.assertRaises(AttributeError):
            self.config._canonical = b"{}"

    def test_reject_boolean_fractional_string_and_float_counts(self):
        for value in (True, 200.0, 200.5, "200"):
            with self.subTest(value=value):
                self.bad_science("cases", "count", value, "type")
        self.bad_science("objective", "R", True, "type")

    def test_reject_nonfinite_and_overflow(self):
        for value in (float("nan"), float("inf"), -float("inf"), 10**1000):
            with self.subTest(value=str(value)[:12]):
                self.bad_science("objective", "R", value, "type")

    def test_unknown_missing_and_bad_shape(self):
        for modify in (lambda d: d.update(execute=True), lambda d: d.pop("cases"),
                       lambda d: d["local"].update(extra=True)):
            data = self.config.to_dict()
            modify(data)
            with self.assertRaises(oc.ConfigError) as error:
                oc.scientific_from_dict(data)
            self.assertEqual(error.exception.code, "fields")
        self.bad_science("objective", "Q", [2, 30, 100], "shape")

    def test_reject_duplicate_keys_nonfinite_json_and_bad_utf8(self):
        for raw in (b'{"schema_version":1,"schema_version":1}', b'{"schema_version":NaN}', b'\xff', b'{} trailing'):
            with self.subTest(raw=raw), self.assertRaises(oc.ConfigError):
                oc.ScientificConfig(raw)

    def test_reject_all_locked_scientific_changes(self):
        mutations = [("objective", "Q", [.02,30,100,10]), ("objective", "Qf", [2,30,100,10]),
                     ("local", "gain_construction", "euler-dare"), ("cases", "case_seed", 1),
                     ("cases", "scale", .5), ("cases", "half_widths", [.75,1.25,.65,1.75]),
                     ("execution", "prediction_device", "cpu"), ("execution", "oracle_dtype", "float32"),
                     ("execution", "gpu_uuid", "another-GPU"), ("endpoint", "minimum_successes", 195),
                     ("endpoint", "final_indices", [851,1000]), ("search", "starts", 5),
                     ("plant", "dt_physical_s", .01), ("plant", "angle_wrapping", True)]
        for group, key, value in mutations:
            with self.subTest(group=group, key=key):
                self.bad_science(group, key, value, "profile_value")

    def test_source_pins_hash_format_and_traversal(self):
        for mutate in (lambda m: m["submitted"]["files"].update({"src/inverted_pendulum.py": "0"*64}),
                       lambda m: m.update(scientific_sha256="not-a-hash"),
                       lambda m: m["specification"].update(path="../spec.md")):
            m = self.metadata.to_dict()
            mutate(m)
            with self.assertRaises(oc.ConfigError):
                oc.metadata_from_dict(m)

    def test_no_full_release_pins_are_invented(self):
        m = self.metadata.to_dict()
        for key in ("implementation", "environment", "runtime", "derived_gain", "trace_contract", "timing", "adoption"):
            self.assertIsNone(m[key])

    def test_cross_field_config_hash_and_gain_binding(self):
        m = self.metadata.to_dict()
        m["scientific_sha256"] = "0"*64
        self.assertFalse(oc.validate_cross_fields(self.config, oc.metadata_from_dict(m)).passed)
        m = self.metadata.to_dict()
        m["derived_gain"] = dict(path="validation/gain.json", sha256="1"*64, scientific_sha256="0"*64)
        report = oc.validate_cross_fields(self.config, oc.metadata_from_dict(m))
        self.assertIn("artifact_config", [c.code for c in report.checks if c.status == "fail"])

    def test_release_identity_ignores_state_but_binds_implementation_runtime_gain(self):
        metadata, evidence = self.future_fixture()
        m = metadata.to_dict()
        digest = oc.release_hash(metadata)
        m["events"] = m["events"][:1]
        m["adoption"] = None
        self.assertEqual(digest, oc.release_hash(oc.metadata_from_dict(m)))
        for key in ("implementation", "runtime", "derived_gain"):
            changed = metadata.to_dict()
            changed[key] = None
            self.assertNotEqual(digest, oc.release_hash(oc.metadata_from_dict(changed)))

    def test_empty_or_out_of_order_lifecycle_is_rejected(self):
        for events in ([], [self.metadata.to_dict()["events"][0]]*2):
            m = self.metadata.to_dict()
            m["events"] = events
            self.assertFalse(oc.validate_cross_fields(self.config, oc.metadata_from_dict(m)).passed)

    def test_events_and_adoption_cannot_bind_another_release(self):
        metadata, evidence = self.future_fixture()
        for target in ("event", "adoption", "timing"):
            m = metadata.to_dict()
            if target == "event":
                m["events"][2]["release_sha256"] = "0"*64
            else:
                m[target]["release_sha256"] = "0"*64
            self.assertFalse(oc.execution_eligibility(self.config, oc.metadata_from_dict(m), evidence).passed)

    def test_independent_declaration_and_scope_are_required(self):
        metadata, evidence = self.future_fixture()
        for mutate in (lambda m: m["events"][3].update(independent=False),
                       lambda m: m["events"][3].update(reviewer=None),
                       lambda m: m["events"][3].update(scope=["configuration"])):
            m = metadata.to_dict()
            mutate(m)
            self.assertFalse(oc.execution_eligibility(self.config, oc.metadata_from_dict(m), evidence).passed)

    def test_unavailable_observations_are_not_failed_matches(self):
        metadata, evidence = self.future_fixture()
        report = oc.compare_release(metadata, {})
        self.assertFalse(report.passed)
        self.assertTrue(all(c.status == "unavailable" for c in report.checks))
        observed = copy.deepcopy(evidence["observed_release"])
        observed["runtime"]["driver"] = "different"
        changed = oc.compare_release(metadata, observed)
        self.assertTrue(any(c.status == "fail" and c.field.endswith("driver") for c in changed.checks))

    def test_runtime_boolean_cannot_masquerade_as_integer(self):
        metadata, evidence = self.future_fixture()
        observed = copy.deepcopy(evidence["observed_release"])
        observed["runtime"]["cpu_threads"] = True
        self.assertFalse(oc.compare_release(metadata, observed).passed)

    def test_synthetic_future_eligibility_can_pass_without_launching(self):
        metadata, evidence = self.future_fixture()
        report = oc.execution_eligibility(self.config, metadata, evidence)
        self.assertTrue(report.passed, report.to_dict())

    def test_eligibility_requires_all_planned_implementation_files(self):
        metadata, evidence = self.future_fixture()
        m = metadata.to_dict()
        del m["implementation"]["files"]["validation/oracle_policy_v1.py"]
        report = oc.execution_eligibility(self.config, oc.metadata_from_dict(m), evidence)
        self.assertTrue(any(c.code == "implementation_coverage" and c.status == "unavailable"
                            for c in report.checks))

    def test_malformed_external_evidence_is_rejected(self):
        metadata, evidence = self.future_fixture()
        for values in ({"unexpected": True}, {"observed_release": None}, {"artifact_sha256": []},
                       {"ledger_sha256": "not-a-hash"}, {"trusted_adoption_sha256": True},
                       {"archive_available": 1}, {"amended_tau_s": 10**1000}):
            with self.subTest(values=values):
                report = oc.execution_eligibility(self.config, metadata, {**evidence, **values})
                self.assertFalse(report.passed)
                self.assertTrue(any(c.status == "fail" for c in report.checks))

    def test_self_reported_adoption_cannot_replace_trusted_decisions(self):
        metadata, evidence = self.future_fixture()
        for name in ("trusted_adoption_sha256", "trusted_allocation_sha256"):
            e = copy.deepcopy(evidence)
            e.pop(name)
            self.assertFalse(oc.execution_eligibility(self.config, metadata, e).passed)
            e[name] = "0"*64
            self.assertFalse(oc.execution_eligibility(self.config, metadata, e).passed)
        self.assertFalse(oc.execution_eligibility(self.config, self.metadata, {"execution_authorized": True}).passed)

    def test_missing_and_wrong_artifact_hashes_have_distinct_status(self):
        metadata, evidence = self.future_fixture()
        path = metadata.to_dict()["timing"]["path"]
        for available, status in ((False, "unavailable"), (True, "fail")):
            e = copy.deepcopy(evidence)
            if available:
                e["artifact_sha256"][path] = "0"*64
            else:
                del e["artifact_sha256"][path]
            report = oc.execution_eligibility(self.config, metadata, e)
            self.assertTrue(any(c.field == "timing" and c.status == status for c in report.checks))

    def test_archive_and_allocation_gate(self):
        metadata, evidence = self.future_fixture()
        e = copy.deepcopy(evidence)
        e["archive_available"] = False
        self.assertFalse(oc.execution_eligibility(self.config, metadata, e).passed)
        m = metadata.to_dict()
        m["allocation"]["status"] = "proposed"
        self.assertFalse(oc.execution_eligibility(self.config, oc.metadata_from_dict(m), evidence).passed)

    def test_reservation_boundary_cumulative_cap_and_no_old_timing_assumption(self):
        metadata, evidence = self.future_fixture()
        self.assertTrue(oc.execution_eligibility(self.config, metadata, evidence).passed)
        for values in ({"amended_tau_s": .2500000001}, {"used_worker_seconds": 5400.0001},
                       {"amended_tau_s": None}, {"ledger_sha256": None}, {"amended_tau_s": float("nan")},
                       {"amended_tau_s": True}):
            with self.subTest(values=values):
                self.assertFalse(oc.execution_eligibility(self.config, metadata, {**evidence, **values}).passed)

    def test_missing_document_is_unavailable(self):
        with self.assertRaises(oc.ConfigError) as error:
            oc.load_scientific(HERE/"tests/this-file-does-not-exist.json")
        self.assertEqual(error.exception.code, "unavailable")

    def test_cli_preview_does_not_write_or_authorize(self):
        stream = io.StringIO()
        with redirect_stdout(stream):
            self.assertEqual(oc.main([]), 0)
        decision = json.loads(stream.getvalue())
        self.assertFalse(decision["execution_authorized"])
        self.assertFalse(decision["execution_eligibility"]["passed"])

    def test_cli_export_is_exclusive_and_rejects_historical_paths(self):
        destination = HERE/"reviews"/('.i01-fixture-'+uuid4().hex)
        try:
            with redirect_stdout(io.StringIO()):
                self.assertEqual(oc.main(["--output-dir", str(destination)]), 0)
            self.assertEqual(set(p.name for p in destination.iterdir()),
                             {"scientific.json", "release_metadata.json", "mapping.json", "decision.json"})
            before = {p.name: p.read_bytes() for p in destination.iterdir()}
            with redirect_stderr(io.StringIO()):
                self.assertEqual(oc.main(["--output-dir", str(destination)]), 2)
                self.assertEqual(oc.main(["--output-dir", str(HERE/"diagnostics/new-record")]), 2)
                self.assertEqual(oc.main(["--output-dir", str(destination/"new-record")]), 2)
            self.assertEqual(before, {p.name: p.read_bytes() for p in destination.iterdir()})
        finally:
            if destination.exists():
                shutil.rmtree(destination)

    def test_cli_has_no_execute_switch(self):
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as error:
            oc.main(["--execute"])
        self.assertEqual(error.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
