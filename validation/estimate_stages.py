"""Read measured timings and compute plan s=1 vs observed-smoke projections."""
import argparse
import json
import math
from pathlib import Path
OUT=Path(__file__).resolve().parent/'artifacts'
parser=argparse.ArgumentParser(); parser.add_argument('--device',choices=['cpu','cuda'],default='cpu'); args=parser.parse_args()
device=args.device
def read(name): return json.loads((OUT/name).read_text())
H=read(f'benchmark_control_{device}_1.json')['planning']['mean_s']
E=read(f'benchmark_control_{device}_5.json')['planning']['mean_s']
O=read(f'benchmark_oracle_{device}_1.json')['single_start']['mean_s']
epoch=read(f'benchmark_training_{device}_1.json')['epoch_seconds']
ensemble_epoch=read(f'benchmark_training_{device}_5.json')['epoch_seconds']
smoke=read('smoke.json' if device=='cpu' else 'smoke_cuda.json'); rate=smoke['replan_rate']
collection_per_transition=smoke['component_seconds']['collection']/(120*30)
readiness=sum(json.loads(s)['worker_seconds'] for s in (OUT/'usage.jsonl').read_text().splitlines())/3600
stages=[]
def add(stage,units,cap,training_s=0,collection_transitions=0):
    worst=units*1000*(H+E)/3600
    projected=units*(math.ceil(1000*rate)*H+200*E)/3600
    overhead=(training_s+collection_transitions*collection_per_transition)/3600
    stages.append({'stage':stage,'bundle_case_budget_cells':units,'cap_worker_h':cap,
                   'worst_planning_h':worst,'projected_planning_h':projected,
                   'modeled_training_collection_h':overhead,'worst_modeled_h':worst+overhead,
                   'projected_modeled_h':projected+overhead,'worst_exceeds_cap':worst+overhead>cap,
                   'projected_exceeds_cap':projected+overhead>cap,
                   'unmeasured_overhead':'LQR evaluation, plant/scoring/logging, statistics; totals are lower bounds'})
# Stage 0 separate oracle calibration. Reuse H's smoke replan rate ONLY as a
# requested projection proxy; an oracle can have a different local-switch rate.
stages.append({'stage':0,'cap_worker_h':8,'worst_planning_h':200*1000*5*O/3600,
               'projected_planning_h':200*math.ceil(1000*rate)*5*O/3600,
               'completed_readiness_h':readiness,
               'worst_modeled_h':readiness+200*1000*5*O/3600,
               'projected_modeled_h':readiness+200*math.ceil(1000*rate)*5*O/3600,
               'worst_exceeds_cap':readiness+200*1000*5*O/3600>8,
               'projected_exceeds_cap':readiness+200*math.ceil(1000*rate)*5*O/3600>8,
               'unmeasured_overhead':'calibration plant, local feedback, logging; oracle rate may differ'})
# Three budgets 1050/10500/105000 per bundle, 100 epochs H plus five E members.
# Mandatory E tuning 12 recipes x 3 bundles x 5 members x 100 epochs, Ttrain8400.
# Width^2 scaling is a model, not measured throughput at other architectures.
tuning_s=3*5*100*.1*4*((64/192)**2+1+(256/192)**2)*ensemble_epoch
bundle_train_s=100*(epoch+5*ensemble_epoch)
stage1_train=3*(.01+.1+1)*bundle_train_s+tuning_s
add(1,3*50*3,12,stage1_train,3*(1050+10500+105000)+31500)
# Maximum four refinement budgets at upper budget: conservative training model.
coarse=[300,1050,3000,10500,30000,105000]
stage2_transitions=12*(sum(coarse)+4*105000)
add(2,12*200*10,48,stage2_transitions/105000*bundle_train_s,stage2_transitions)
add(3,3*50*3,8)
stages.append({'stage':4,'cap_worker_h':0,'status':'not authorized; no committed workload'})
result={'device':device,'performance_only':True,'tau_H_s':H,'tau_E_s':E,'tau_oracle_single_start_s':O,'full_epoch_s':epoch,
        'ensemble_member_epoch_s':ensemble_epoch,
        'H_smoke_replans_per_sample':rate,'E_replans_per_sample':.2,
        'stage1_mandatory_tuning_training_s_model':tuning_s,'rows':stages,
        'projection_warning':'2s smoke on tiny training is a proxy, not a measured nominal 20s replan rate; E has no switch and replans every five samples',
        'reservation_warning':'s=1 solve COUNT bound times measured mean latency is not a guaranteed runtime bound; reserve margins and meter actual worker time',
        'training_warning':'linear scaling with T and width squared; epoch overhead/other widths unmeasured, future E recipe not frozen',
        'wall_clock_conversion':'prefer aggregate solves/s from concurrency.json for CPU or concurrency_cuda.json for GPU; divide total solves by throughput, not worker-hours by all 64 cores'}
(OUT/('stage_estimates.json' if device=='cpu' else 'stage_estimates_cuda.json')).write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
