"""Recompute descriptive endpoints from authored and historical saved runs."""
from campaign import *

def main():
    path=OUT/'E0/authored_summary.json'
    summary=json.loads(path.read_text())
    data=np.load(summary['artifacts']['results'])
    rows=[]
    for label,old in summary['closed_loop_metrics'].items():
        t,X,U=(data[f'{prefix}_{label}'] for prefix in ('t','X','U'))
        diagnostics=summary['solver_diagnostics'][label]
        rows.append(dict(case=label,seed=234,arm='H',seconds=float(t[-1]),
            **metrics(t,X,U,diagnostics['terminal_policy_used'],losses=diagnostics['loss'])))
    csv_write(OUT/'E0/authored_cases.csv',rows)
    stats=dict(n=len(rows),primary_successes=sum(r['primary_success'] for r in rows),
        balanced=sum(r['balanced'] for r in rows),cart_rest=sum(r['cart_rest'] for r in rows),
        no_fall=sum(r['no_fall'] for r in rows),
        local_fraction_range=[min(r['local_fraction'] for r in rows),max(r['local_fraction'] for r in rows)],
        cem_solves_range=[min(r['cem_solves'] for r in rows),max(r['cem_solves'] for r in rows)],
        max_cart_excursion_range=[min(r['max_abs_p'] for r in rows),max(r['max_abs_p'] for r in rows)])
    stats['wilson95']=wilson(stats['primary_successes'],stats['n'])
    dump(OUT/'E0/authored_endpoint_summary.json',stats)
    config=runner.make_config('dev',device='cuda')
    stored=np.load(summary['artifacts']['data']);splits=runner.split_episodes(stored['X'],stored['U'],234)
    gain,local=runner.fit_local_feedback(splits['train'],Q,R)
    decoder,_=runner.load_transition(Path(summary['artifacts']['checkpoint']),'cuda')
    dump(OUT/'E0/authored_model_diagnostics.json',dict(prediction=prediction_rmse(CanonicalBehaviorDecoder(decoder,30),splits['test']),local=fit_diagnostics(splits['train'],local)))
    historical=ROOT/'validation/overnight_runs/20261007T054549.695689Z/calibration'
    out=OUT/'historical_oracle';out.mkdir(exist_ok=True)
    start=time.perf_counter();rows=[]
    for path in sorted(historical.glob('case_*.npz')):
        data=np.load(path); X,U=data['X'],data['U'];t=np.arange(len(X))*.02
        local=data['branches'].astype(str)=='local'
        rows.append(dict(case=path.stem,seconds=float(t[-1]),raw_path=str(path.relative_to(ROOT)),
            **metrics(t,X,U,local)))
    csv_write(out/'summary.csv',rows)
    values=np.array([r['max_abs_p'] for r in rows])
    stats=dict(n=len(rows),balanced=sum(r['balanced'] for r in rows),
        balanced_and_cart_rest=sum(r['balanced'] and r['cart_rest'] for r in rows),
        cart_rest=sum(r['cart_rest'] for r in rows),fell=sum(r['fell'] for r in rows),
        primary_successes=sum(r['primary_success'] for r in rows),
        max_cart_excursion_quantiles={str(q):float(np.quantile(values,q)) for q in (0,.25,.5,.75,.95,1)})
    stats['wilson95']=wilson(stats['primary_successes'],stats['n'])
    dump(out/'endpoint_summary.json',stats)
    dump(out/'manifest.json',dict(git_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        config=json.loads((historical/'config.json').read_text()),seeds=[20261006],device='CPU postprocessing of prior CUDA trajectories',
        wall_seconds=time.perf_counter()-start,source=str(historical.relative_to(ROOT)),status='DONE',
        note='N=60, 20 s, historical diagnostic; not matched to new N=30, 14 s arms'))
    print('Authored',json.dumps(json.loads((OUT/'E0/authored_endpoint_summary.json').read_text())),flush=True)
    print('Historical',json.dumps(stats),flush=True)

if __name__=='__main__':main()
