# I01 configuration contract v1

I01 implements configuration/provenance support only. OA-1 remains **specified,
unadopted and unexecuted**. The current full-release pins are intentionally
incomplete. Passing I01 tests does not establish the amended controller's
correctness, safety, recoverability or runtime performance.

## Documents and exact schema

[oracle_config.py](oracle_config.py) defines the authoritative `SCIENCE_SCHEMA`,
`METADATA_SCHEMA`, `RUNTIME_SCHEMA`, and fixed `oa1-v1` profile. Every listed key
is required; nullable fields must be present as JSON `null`. Unknown or missing
keys, duplicate JSON keys, wrong shapes, booleans used as numbers, and changed
fixed values are rejected. No formula strings are evaluated.

The original [descriptive proposal](oa1_config_proposed.json) is preserved.
Conversion accepts only its reviewed byte SHA256
`6970c76f51624a341746e337f86b61697585bf246c1d98d78217ffebcb6b2b42`.
Any changed input requires a reviewed converter/proposal revision. The
[mapping record](reviews/i01_v1/mapping.json) accounts for every input root
field; reference K/P and logging requirements remain in the hashed proposal,
pending I02/I04 artifacts. They are not claimed as verified by conversion.

The scientific document has exactly these boundaries:

| Object | Required fields |
|---|---|
| Root scalars | `schema_version`, `reference_id`, `algorithm_profile` |
| `plant` | `identifier`, `mass_ratio`, `cart_mass_kg`, `rod_mass_kg`, `length_m`, `g`, `dt_physical_s`, `dt_normalized`, `input_bounds`, `angle_wrapping` |
| `execution` | `prediction_device`, `gpu_uuid`, `oracle_dtype`, `plant_dtype`, `local_dtype`, `local_device`, `workers`, `cpu_threads`, `deterministic_algorithms`, `TF32`, `cpu_fallback` |
| `cases` | `count`, `root_seed`, `stream`, `indices`, `case_seed`, `half_widths`, `scale` |
| `endpoint` | `transitions`, `states`, `physical_seconds`, `final_indices`, `rail_m`, `angle_rad`, `final_tolerances`, `all_states_finite`, `exceptions_fail`, `past_violations_latch`, `incomplete_cannot_pass`, `timing_exempt`, `minimum_successes`, `half_fraction`, `both_halves_nonempty` |
| `objective` | `Q`, `R`, `Qf`, `rate_weight`, `state_reference`, `input_reference`, `dt_multiplier`, `first_input_rate_term` |
| `local` | `gain_construction`, `entry_max`, `stay_max`, `entry_steps`, `stay_steps`, `decrease_factor`, `decrease_tolerance`, `gain_atol`, `gain_rtol` |
| `search` | `horizon`, `starts`, `population`, `elites`, `iterations`, `smoothing`, `unbiased_std`, `std_floor`, `state_scale`, `case_prefix` |
| `queue` | `maximum_inputs`, `consistency_atol`, `consistency_rtol`, `maximum_calls_per_sample` |
| `fallback` | `identifier` |

The algorithm-profile identifier binds the remaining indexing, CEM slots/noise,
ranking, switch, queue and fallback rules in sections 2–5 of the pinned
[OA-1 specification](oracle_amendment_proposal_v1.md). These are fixed rules,
not implementation discretion or executable prose.

Metadata has exactly `metadata_version`, `scientific_sha256`, `specification`,
`submitted`, `implementation`, `environment`, `runtime`, `derived_gain`,
`trace_contract`, `events`, `archive`, `timing`, `allocation`, `adoption`.

* An artifact is `{path, sha256}`: normalized repository-relative path without
  traversal, and lowercase 64-digit SHA256. A bound gain artifact additionally
  requires `scientific_sha256`.
* `submitted` contains the exact submitted commit and three source pins.
  `implementation` contains a commit and nonempty file/hash map; its manifest
  must cover the six planned executable files plus their imported source
  closure. `environment` contains nonempty `manifests` and `native_files`
  file/hash maps. Neither a Git HEAD nor a scientific hash substitutes for
  these byte hashes.
* `runtime` binds Python 3.11.9, L4 UUID, driver, Torch/CUDA versions, devices,
  dtypes, thread/worker counts and deterministic/TF32 flags. Unknown versions
  remain unbound, rather than guessed from historical timing.
* Each event contains `state`, `release_sha256`, artifact `reference`, nullable
  `reviewer`, Boolean `independent`, and string-array `scope`.
* `archive` contains `status`, nullable `locator`, nullable artifact `manifest`.
  `timing` is a nullable artifact plus `release_sha256`.
* `allocation` contains status, integer `cap_seconds`, `support_seconds`,
  `queue_seconds`, `fixed_overhead_seconds`, `archive_seconds`, exact
  `allowed_jobs: ["oa1_gate_a"]`, and nullable artifact `reference`.
  `adoption` contains artifact `reference`, `release_sha256`, `approved_by`,
  and that same exact allowed-job list. Neither belongs in scientific values.

## Numeric and hash contracts

Counts, indices for endpoint samples, seeds, schema versions and allocation
seconds are strict JSON integers, `0 <= n < 2**63`; floating equivalents,
strings and booleans fail. Real fields accept integers or floats, normalize to
finite binary64 and positive zero, and then must equal the frozen profile.
Case-stream indices remain real `[1.0]`. The original indexed RNG payload uses
its compact **list** serialization; `[1]` would produce a different seed.

Canonical bytes use Python 3.11.9 `json.dumps(sort_keys=True,
ensure_ascii=True, separators=(",", ":"), allow_nan=False)`, UTF-8, no newline,
after normalization. This is a versioned Python JSON contract, not RFC 8785.
Scientific SHA256 is
`38d6c75362c3b333d697844c2bcbff46a5f7c9950e1c6b94d52ae8f076ea4b00`
(1906 bytes). A golden test detects byte-contract drift.

Release SHA256 hashes the canonical identity containing `scientific_sha256`,
`specification`, `submitted`, `implementation`, `environment`, `runtime`,
`derived_gain`, `trace_contract`. Events, timing evidence, allocation, archive
location and adoption are outside that identity to avoid circular hashes.
Their hashed evidence references bind decisions to the release. Changing
identity requires a new release and new bound evidence; approvals do not
transfer. The current incomplete identity hash is a review identifier, not
an executable-release certification.

## Library and caller contract

I01 imports only the standard library. It imports neither `common.py`, the SUT,
NumPy/SciPy/Torch, nor subprocess APIs, and does not solve gains, initialize a
device, sample cases, launch jobs or read raw experimental arrays.

| Interface | Result |
|---|---|
| `load_scientific(path)` / `scientific_from_dict(value)` | Immutable validated `ScientificConfig`; fresh copies from `to_dict()` |
| `canonical_bytes(config)` / `scientific_hash(config)` | Canonical bytes / SHA256, validated config only |
| `load_release_metadata(path)` / `metadata_from_dict(value)` | Immutable structurally validated `ReleaseMetadata` |
| `release_identity(metadata)` / `release_hash(metadata)` | Identity copy / SHA256 |
| `validate_cross_fields(config, metadata)` | `Report`: config/spec/gain/seed/count/duration/lifecycle/decision bindings |
| `compare_release(metadata, observed)` | `Report`: exact recursive value **and type** comparisons |
| `execution_eligibility(config, metadata, evidence)` | `Report` only; no execution or authority granted |
| `convert_proposal(path)` | Scientific config, proposed metadata, exhaustive root-field mapping record |

Malformed documents raise `ConfigError(code, field, expected, observed)`;
cross-field/eligibility mismatches accumulate structured checks. Check status
is exactly `pass`, `fail`, `unavailable`, or `not_checked`. Missing files or
observations are `unavailable`; contradictory evidence is `fail`; intentionally
unbound release identities are `not_checked`. Reports pass only when nonempty
and **every** check passes. Missing archives cannot become a passing test or
trigger regeneration. Unavailable runtime/archives here mean not established
for this future release, not that historical evidence is absent.

I05 must independently collect `observed_release` (identity-shaped object),
`artifact_sha256` (path/hash map), externally trusted `trusted_adoption_sha256`
and `trusted_allocation_sha256`, Boolean `archive_available`, measured
`amended_tau_s`, cumulative `used_worker_seconds`, and `ledger_sha256`. These
are the only evidence keys; omitted values block eligibility. I01 verifies
their bindings/types, not the truth of a caller's declarations. I05 must check
the actual file/import closure, approved external decision contents, ledger
occupancy, timing-artifact contents and archive availability afresh. It must
not copy declarations from metadata into observations.

I02 consumes the immutable scientific config and emits a gain artifact bound
to its hash. I03 consumes that config and independently verified gain artifact.
I04 may share I01 parsing/hashing, but independently implements numerical
cost/rank/predicate/endpoint scoring; it imports no I02/I03 scoring or policy
helpers. I05 consumes this interface after I02–I04; preflight itself remains
free of scientific initialization. Required filename coverage is only a floor;
independent review must check manifest completeness and dependency sharing.

## Status, decisions and eligibility

Full OA-1 lifecycle is a nonempty prefix of:
`specified -> implemented -> tested -> independently_audited -> adopted -> executed`.
Repeated, skipped or reordered events fail. Evidence after `specified` binds
the same release hash. Events are append-only review records: I01 validates a
provided history; I05 must verify its immutable external history before use.

`implemented` requires a source manifest. Execution also requires all six
planned executable files, bound gain/trace/runtime/environment, hashed event
records, full tested/audited scopes, independent reviewer declaration, external
adoption and allocation records, archive evidence, fresh ledger and amended
timing. Required tested/audited scopes are `configuration`, `gain_cost_rank`,
`policy`, `trace_audit`, `launcher`, `nonlinear_guards`, `amended_timing`.
Reviewer independence must be assessed by a separate reviewer; setting a
Boolean or passing same-author tests does not establish it.

The proposed 20-hour allocation remains unadopted. Conditional arithmetic is
`200000*tau_amended*1.2 + 1200 <= 61200`, and cumulative occupancy plus
`3600 + 61200 + 1800` must fit `72000` seconds. Fixed overhead is inside the
queue slot. With the original occupancy, residual is 412.643508 seconds;
`tau_amended <= .25` is a reservation prerequisite, not a runtime guarantee.
The current 15-hour cap and stopped stages remain effective.

The CLI has only `--proposal` and `--output-dir`. Preview writes nothing;
export creates a **new direct child** of `validation/reviews/`, refuses existing
directories, and writes `scientific.json`, `release_metadata.json`, `mapping.json`,
`decision.json` exclusively. It exposes no execution switch. Exit 0 means
configuration consistency, even when execution eligibility is false.

See [oa1_decisions.md](oa1_decisions.md) for actual component status and
[I01 review records](reviews/i01_v1/decision.json) for blocked eligibility.
