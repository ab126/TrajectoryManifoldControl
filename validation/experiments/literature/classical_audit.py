"""Bind frozen classical artifacts and independently replay stratified endpoints.

Migration adds a sidecar binding; existing case identities/artifacts are untouched.
Digests verify current artifacts against the prior completed-fit ledger. They do
not claim that an absent historic controller digest was saved at execution time.
"""
import json
import numpy as np
from common import *


def main():
    isolation('edmd');budget=Budget('preparation',2);budget.checkpoint()
    ledger=json.loads((OUT/'inventory/model_identity_ledger.json').read_text())
    initial=ROOT/'validation/results_v1/initial_conditions.json';states=json.loads(initial.read_text())['sets'];fresh=OUT/'inventory/fresh_200_states.json';states['F']=json.loads(fresh.read_text())['states']
    manifest=json.loads((OUT/'manifest.json').read_text())
    for path,sha in manifest['protected_source_hashes'].items():assert digest(ROOT/path)==sha,path
    bindings=[];summaries=[];replays=[]
    for method,arms in [('edmd',['QP','QP+L']),('sindy',['CEM','CEM+L','NMPC_exact','NMPC_exact+L'])]:
        for seed in range(3):
            known=next(r for r in ledger['models'] if r['method']==method and r['seed']==seed)
            assert digest(ROOT/known['model_path'])==known['model_sha256']
            assert digest(ROOT/known['fit_identity_path'])==known['fit_identity_sha256']
            fit=json.loads((ROOT/known['fit_identity_path']).read_text());assert digest(ROOT/fit['path'])==fit['sha256']
            for name,sha in fit['adapter_hashes'].items():assert digest(Path(__file__).with_name(name))==sha
            selected=json.loads((OUT/method/f'seed{seed}/selection.json').read_text())
            gainpath=ROOT/f'validation/results_v1/models/episodes3500_seed{seed}/diagnostics.json'
            binding=dict(method=method,seed=seed,selected_model_sha256=known['model_sha256'],data_sha256=fit['sha256'],selected_configuration=selected['config'],gain_artifact_sha256=digest(gainpath),common_L_gain=json.loads(gainpath.read_text())['local']['gain'],simulator_sha256=digest(ROOT/'src/inverted_pendulum.py'),endpoint_implementation_sha256=digest(Path(__file__).with_name('run_classical.py')),initial_conditions_sha256=digest(initial),fresh_states_sha256=digest(fresh),cases=[])
            for arm in arms:
                rows=[]
                for p in sorted((OUT/method/'cases').glob(f'seed{seed}_{arm}_*.json')):
                    row=json.loads(p.read_text())
                    if row['arm']!=arm:continue
                    c=row['identity']['case'];assert c['state']==states[c['set']][c['index']]
                    assert row['identity']['data_sha256']==fit['sha256'] and row['identity']['adapter_hashes']==fit['adapter_hashes'] and row['identity']['selected']==selected['config']
                    raw=ROOT/row['raw_path'];assert raw.exists();binding['cases'].append(dict(case_path=str(p.relative_to(ROOT)),case_sha256=digest(p),raw_path=row['raw_path'],raw_sha256=digest(raw)))
                    rows.append((row,p))
                # One authored, random and long case per seed/arm: independent endpoint and every-step replay.
                sample=[]
                for name,seconds in [('A',14.),('C',14.),('A',60.)]:
                    eligible=[(r,p) for r,p in rows if r['set']==name and r['seconds']==seconds]
                    if eligible:sample.append(eligible[0])
                for row,p in sample:
                    raw=np.load(ROOT/row['raw_path']);X,U,t=raw['X'],raw['U'],raw['t'];assert np.array_equal(X[0],row['identity']['case']['state']);np.testing.assert_allclose(t,np.arange(len(X))*.02,atol=1e-12,rtol=0)
                    finite=bool(np.isfinite(X).all() and np.isfinite(U).all());complete=len(U)==round(row['seconds']/.02);tail=t>=row['seconds']-3.
                    no_fall=bool(finite and np.all(np.abs(X[:,2])<np.pi/2));balanced=bool(complete and finite and np.all(np.abs(X[tail,2])<=.02) and np.all(np.abs(X[tail,3])*np.sqrt(9.81)<=.05))
                    assert row['primary_success']==(balanced and no_fall)
                    step=plant_interface();error=0.
                    for k in range(len(U)):
                        if np.isfinite(X[k:k+2]).all():error=max(error,float(np.max(np.abs(step(X[k],U[k,0])-X[k+1]))))
                    assert error<1e-8
                    replays.append(dict(case=str(p.relative_to(ROOT)),endpoint_matches=True,max_one_step_replay_error=error))
                    budget.checkpoint()
                for name,seconds in [('A',14.),('C',14.),('A',60.),('F',14.)]:
                    group=[r for r,_ in rows if r['set']==name and r['seconds']==seconds]
                    summaries.append(dict(method=method,seed=seed,arm=arm,set=name,seconds=seconds,n=len(group),successes=sum(r['primary_success'] for r in group),solver_rejections=sum(r['solver_failures'] for r in group)))
            bindings.append(binding)
    dump(OUT/'inventory/classical_reuse_bindings.json',dict(state='verified current artifacts; explicit sidecar migration',historic_model_ledger_verified=True,bindings=bindings))
    dump(OUT/'inventory/classical_endpoint_replay.json',dict(passed=True,cases=replays))
    dump(OUT/'reviews/classical_closure_packet.json',dict(question='Review artifact binding migration and stratified endpoint replay; no retroactive claim of historical digest capture',source='experiments/literature/classical_audit.py',bindings='inventory/classical_reuse_bindings.json',replay='inventory/classical_endpoint_replay.json',results=summaries,interpretation='Native classical controllers fail the strict endpoint on all minimum cases; declared L arms can rescue some SINDy cases. No plant derivatives or extra feedback were added.',accounting=budget.checkpoint()))
    dump(OUT/'preparation/audit_completion.json',dict(state='complete',sample_count=len(replays)))


if __name__=='__main__':main()
