"""Describe strict angular endpoint misses using already-completed E0 arrays."""
from campaign import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    out=OUT/'E0';rows=[]
    def number(value):return float(value) if np.isfinite(value) else None
    for path in sorted((out/'cases').glob('*.json')):
        item=json.loads(path.read_text())
        if item['seconds']!=14.:continue
        data=np.load(ROOT/item['raw_path']);t,X,U=data['t'],data['X'],data['U']
        final=t>=max(0.,t[-1]-min(3.,.4*t[-1]));ufinal=t[:-1]>=t[-1]-3.
        rows.append(dict(case=item['case'],seed=item['seed'],arm=item['arm'],ic_index=item['ic_index'],
            initial_theta_degrees=float(np.rad2deg(item['initial_state'][2])),primary_success=item['primary_success'],no_fall=item['no_fall'],
            tail_max_abs_theta=number(np.max(np.abs(X[final,2]))),
            tail_max_abs_angular_velocity_physical=number(np.max(np.abs(X[final,3]))*np.sqrt(9.81)),
            tail_input_rms=number(np.sqrt(np.mean(U[ufinal]**2))),
            tail_local_fraction=float(np.mean(data['local'][ufinal])),raw_path=item['raw_path']))
    csv_write(out/'terminal_behavior.csv',rows)
    fig,axes=plt.subplots(2,3,figsize=(12,7),sharex=True)
    for axrow,index in zip(axes,(2,5)):
        for arm in ('H','L','M','M1','Hstar'):
            data=np.load(out/'raw'/f'seed0_{arm}_A_{index:04d}_14s.npz');t,X,U=data['t'],data['X'],data['U']
            keep=t>=11.;u_keep=t[:-1]>=11.
            axrow[0].plot(t[keep],X[keep,2],label=arm)
            axrow[1].plot(t[keep],X[keep,3]*np.sqrt(9.81),label=arm)
            axrow[2].step(t[:-1][u_keep],U[u_keep,0],label=arm)
        for ax,limit in zip(axrow[:2],(.02,.05)):
            ax.axhline(limit,color='black',ls=':',lw=1);ax.axhline(-limit,color='black',ls=':',lw=1)
        axrow[0].set_ylabel('θ [rad]');axrow[1].set_ylabel('Physical θdot [rad/s]');axrow[2].set_ylabel('u')
        axrow[0].set_title(f'Seed 0, θ0={-15 if index==2 else 60}°; terminal window')
    for ax in axes[-1]:ax.set_xlabel('Physical time [s]')
    axes[0,2].legend(ncol=3,fontsize=8)
    fig.suptitle('E0: approaching upright can still miss the authored strict settling limits')
    fig.tight_layout();fig.savefig(out/'terminal_behavior.png',dpi=150);plt.close(fig)

def feasible_alternatives():
    from failure_diagnostics import witness
    start=time.perf_counter();results=[];out=OUT/'E0'
    for arm,index in [('M1',2),('H',5),('Hstar',5)]:
        key=f'seed0_{arm}_A_{index:04d}_14s';data=np.load(out/'raw'/(key+'.npz'))
        j=len(data['solve_steps'])-1;k=int(data['solve_steps'][j])
        inputs=data['plan_inputs'][j] if 'plan_inputs' in data else None
        result=dict(case=key,planning_time_seconds=k*.02,
            **witness(0,arm,data['X'][k],inputs,data['predictions'][j],data['loss'][j]))
        results.append(result)
    dump(out/'feasible_alternative_diagnostics.json',dict(selection='Last saved plan of three illustrative seed-0 cases; not a random sample',
        wall_seconds=time.perf_counter()-start,cases=results))

if __name__=='__main__':main();feasible_alternatives()
