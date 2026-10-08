"""Capture source/data/config/native fingerprints after timing windows close."""
import datetime
import hashlib
import importlib.metadata as md
import json
import subprocess
import numpy as np
from pathlib import Path
HERE=Path(__file__).resolve().parent; ROOT=HERE.parent; WORK=ROOT.parent; OUT=HERE/'artifacts'
def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()
def seed(stream,*indices):
    payload=json.dumps([20261006,stream,*indices],separators=(',',':')).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:8],'little')%(2**63-1)
initial=json.loads((OUT/'source_hashes_initial.json').read_text())
current={p:digest(WORK/p) for p in initial}
amendment_path=OUT/'approved_plan_amendments.json'
amendments=json.loads(amendment_path.read_text()) if amendment_path.exists() else {'changes':{}}
for name,old_hash in initial.items():
    if name in amendments['changes']:
        change=amendments['changes'][name]
        assert name=='review/stress_test_plan.md'
        assert old_hash==change['before_SHA256'] and current[name]==change['after_SHA256'], 'Plan differs from reviewed, authorized amendment'
    else:
        assert old_hash==current[name], f'Submitted source changed: {name}'
(OUT/'source_hashes_final.json').write_text(json.dumps(current,indent=2))
head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
status=subprocess.check_output(['git','status','--porcelain=v1'],cwd=ROOT,text=True)
(OUT/'git_status_final.txt').write_text(status)
(OUT/'git_diff_final.patch').write_bytes(subprocess.check_output(['git','diff','--binary'],cwd=ROOT))
assert subprocess.run(['git','diff','--quiet','HEAD'],cwd=ROOT).returncode==0
assert head=='325beae718ab093d3ca48b18a59ae61fd93798a3'
native=[HERE/'.venv/bin/python',HERE/'.venv/lib/libstdc++.so.6',HERE/'.venv/lib/libtorch_python.so',
        HERE/'.venv/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so']
(OUT/'native_fingerprints.json').write_text(json.dumps({str(p.relative_to(HERE)):digest(p) for p in native},indent=2))
rows=[json.loads(s) for s in (OUT/'usage.jsonl').read_text().splitlines()]
conservative=sum(r['worker_seconds'] for r in rows if r.get('actual_occupancy_not_available'))
used=sum(r['worker_seconds'] for r in rows)
usage={'charged_worker_seconds':used,'charged_worker_hours':used/3600,'cap_seconds':3600,
       'measured_worker_seconds':used-conservative,'conservative_failed_attempt_charge_seconds':conservative,
       'excluded':'environment installation/copying, read-only hardware/source inspection, code editing, artifact hashing, LLM work',
       'scope':'includes warm-up, experiments, process/import overhead, failed smoke/import attempts; concurrency sums worker occupancy plus supervisor wall time'}
assert used<3600
(OUT/'usage_summary.json').write_text(json.dumps(usage,indent=2))
manifest={'root_seed':20261006,'seed_derivation':'SHA256 JSON [root, stream, indices], little-endian first 8 bytes modulo 2^63-1',
          'legacy_numpy_seed_mapping':'derived seed modulo 2^32',
          'smoke':{'dataset_seed':234,'training_seed_H':234,'case_angles_deg':[-60,-30,-15,15,30,60]},
          'equivalence':{'weights_torch_seed':20261006,'cases':list(range(50)),'samples':list(range(100)),
                         'case_sample_seed_rule':'seed(cem,case,physical_sample), no method index',
                         'tensor_shape':[3,256,30,1],'generation':'CPU float32; identical tensors replayed for each implementation'},
          'future_streams':['dataset(bundle)','training(bundle,method,member)','case(case)',
                            'cem(case,physical_sample)','observation_noise(case,physical_sample)',
                            'process_disturbance(case,physical_sample)','ensemble_bootstrap(bundle,member)'],
          'benchmark_member_seeds':[seed('performance_member',i) for i in range(1,5)],
          'performance_training_member_seeds':[seed('performance_training_member',i) for i in range(5)],
          'single_training_torch_rng':'root 20261006 advanced by warm-up model initialization; no member reseed',
          'concurrency_seeds':[seed('concurrency',i) for i in range(8)],
          'training_order_seed':234,'CEM_original_benchmark_seed':234,'CEM_concurrency_seed':'20261006 + worker_index'}
(OUT/'seed_manifest.json').write_text(json.dumps(manifest,indent=2))
order=np.random.default_rng(234).permutation(120).tolist()
(OUT/'split_manifest.json').write_text(json.dumps({'smoke_original_seed':234,
    'smoke_episode_ids':{'test':order[:18],'validation':order[18:36],'train':order[36:]},
    'future_common_split':'randomized episode IDs fixed by bundle; IDs 5*j+4 validation, other IDs training; no test data inside acquisition budget',
    'future_collection_episode_transitions':30},indent=2))
config={'SUT':'scripts/run_inverted_pendulum_mc.py + src/manifold_control.py',
    'HEAD':head,'architecture':[192,192,192],'activation':'SiLU','learned_dtype':'float32',
    'horizon':30,'CEM':{'population':256,'elite':32,'iterations':3,'smoothing':.15,'rate_weight':.002},
    'Q_diagonal':[.02,30,100,10],'R':.03,'terminal_multiplier':12,'input_bounds':[-10,10],
    'local_switch_tracking_cost':.75,'replan_interval':5,'warm_start':'cat(previous[5:], previous[-5:]); preserved through local visits',
    'physical_dt':.02,'normalized_dt':.02/np.sqrt(1/9.81),'mass_ratio':2,
    'physical_interpretation':{'cart_mass_kg':2,'rod_mass_kg':1,'full_rod_length_m':1,'g':9.81,'force_bound_N':98.1},
    'smoke':json.loads((OUT/'smoke.json').read_text())['configuration'],
    'oracle':{'horizon':60,'dtype':'float64','few_performance_solves_only':12,'calibration_executed':False},
    'benchmark_cpu_threads':1,'CPU_concurrency_workers':[1,2,4,8],'GPU_benchmarks':'verified host execution; agent session device nodes unavailable' if (OUT/'stage_estimates_cuda.json').exists() else 'pending; agent session device nodes unavailable',
    'intended_campaign_device':'CUDA on host NVIDIA L4; CPU original plant/local LQR',
    'GPU_environment':'.venv-gpu; private PyTorch 2.5.1+cu121 and CUDA 12.1',
    'GPU_readiness_command':'bash TrajectoryManifoldControl/validation/run_gpu.sh readiness'}
(OUT/'config_manifest.json').write_text(json.dumps(config,indent=2))
(OUT/'interaction_manifest.json').write_text(json.dumps({
    'CPU_smoke_attempts':2,'CUDA_smoke_attempts':int((OUT/'smoke_cuda.json').exists()),'collected_episodes_per_attempt':120,'collection_steps_per_episode':30,
    'raw_smoke_collection_plant_steps':7200+3600*int((OUT/'smoke_cuda.json').exists()),'distinct_collection_seed_configurations':1,
    'original_angle_evaluation_steps_per_attempt':600,'raw_smoke_evaluation_plant_steps':1200+600*int((OUT/'smoke_cuda.json').exists()),
    'timing_training_dataset':'repeated/bootstrap smoke triples; no new plant acquisition',
    'equivalence':'scripted observations; neural model rollouts only',
    'oracle_performance':'true-model prediction rollouts only; no closed-loop calibration cases',
    'contract_checks':'separate numerical integration/RHS calls; not a learned-method acquisition budget',
    'scientific_calibration_tuning_confirmation_transfer_cases':0,
    'overnight_preparation_fixture_plant_steps':2,
    'overnight_fixture_note':'two charged lifecycle-test executions; one zero-input fixture step each; no calibration campaign launched'},indent=2))
det=json.loads((OUT/'determinism.json').read_text())
det['model_selection']['configured_model']='inherited session; no override; precise identifier not exposed'
det['model_selection']['configured_reasoning_effort']='inherited session; not exposed'
det['legacy_numpy_seed_mapping']='derived seed modulo 2^32'
(OUT/'determinism.json').write_text(json.dumps(det,indent=2))
env={'distributions':{d.metadata['Name']:d.version for d in md.distributions()},'head':head,
     'initial_status_observed':'clean at first read before creating validation/',
     'first_saved_status_note':'saved after validation/ creation, so it already includes ?? validation/',
     'submitted_controller_source_unchanged':True,'plans_unchanged':not bool(amendments['changes']),
     'approved_plan_amendments':amendments,'captured_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}
(OUT/'provenance.json').write_text(json.dumps(env,indent=2))
# Hash after every manifest has been finalized. The redirected stdout log is
# excluded because it is still open until this process exits.
import os
files=[]
for directory,dirs,names in os.walk(HERE):
    dirs[:]=[d for d in dirs if d not in ('.venv','.venv-gpu','__pycache__') and not (Path(directory)==OUT and d=='mpl')]
    files.extend(Path(directory)/name for name in names if name not in ('artifact_hashes.json','provenance.log'))
hashes={str(p.relative_to(HERE)):digest(p) for p in sorted(files)}
(OUT/'artifact_hashes.json').write_text(json.dumps(hashes,indent=2))
print(json.dumps(usage,indent=2))
