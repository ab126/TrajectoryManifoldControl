"""Post hoc local-law diagnostic; never changes training or controller gains."""
from campaign import *
from scipy.linalg import solve_discrete_are
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def nominal_linearization(dt):
    # Differentiate the submitted uniform-rod mass matrix at upright, M=2.
    A=np.array([[0,1,0,0],[0,0,-1/3,0],[0,0,0,1],[0,0,2,0]],dtype=float)
    B=np.array([[0],[4/9],[0],[-2/3]],dtype=float)
    Ad=np.eye(4);Bd=np.zeros((4,1));power=np.eye(4)
    for order in range(1,5):
        Bd+=dt**order/math.factorial(order)*power@B
        power=power@A;Ad+=dt**order/math.factorial(order)*power
    return Ad,Bd


def spectral_radius(matrix):
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


def checks(dt):
    Ad,Bd=nominal_linearization(dt);zero=np.zeros(4);errors=[]
    for epsilon in (1e-5,1e-6):
        columns=[]
        for axis in np.eye(4):columns.append((runner._plant_step(epsilon*axis,[0.],dt)-runner._plant_step(-epsilon*axis,[0.],dt))/(2*epsilon))
        numerical_A=np.column_stack(columns)
        numerical_B=((runner._plant_step(zero,[epsilon],dt)-runner._plant_step(zero,[-epsilon],dt))/(2*epsilon)).reshape(4,1)
        errors.append(max(float(np.max(np.abs(numerical_A-Ad))),float(np.max(np.abs(numerical_B-Bd)))))
    if max(errors)>1e-8:raise AssertionError(errors)
    P=solve_discrete_are(Ad,Bd,Q,R);gain=np.linalg.solve(R+Bd.T@P@Bd,Bd.T@P@Ad)
    if not spectral_radius(Ad)>1 or not spectral_radius(Ad-Bd@gain)<1:raise AssertionError('Uncontrolled/DARE stability reference failed')
    return dict(central_difference_epsilons=[1e-5,1e-6],maximum_abs_errors=errors,
                uncontrolled_radius=spectral_radius(Ad),true_DARE_radius=spectral_radius(Ad-Bd@gain),
                interpretation='Jacobian of the sample-and-hold true RK4 closed loop at x=0, before input saturation; diagnostic only, not a finite-IC stability guarantee')


def main():
    dt=runner.make_config('dev').dt;Ad,Bd=nominal_linearization(dt);verification=checks(dt)
    rows=[]
    for episodes in (35,100,350,1000,3500,10000):
        for seed in range(5):
            path=OUT/'models'/f'episodes{episodes}_seed{seed}'/'diagnostics.json';item=json.loads(path.read_text())
            local=item['local'];gain=np.asarray(local['gain']);fit_A=np.asarray(local['A']);fit_B=np.asarray(local['B'])
            rows.append(dict(episodes=episodes,seed=seed,training_transitions=local['training_transitions'],
                fitted_closed_loop_radius=spectral_radius(fit_A-fit_B@gain),true_closed_loop_radius=spectral_radius(Ad-Bd@gain),
                selected_max_abs_theta=local['max_abs_theta'],theta_gain=float(gain[0,2]),omega_gain=float(gain[0,3])))
    out=OUT/'E2';csv_write(out/'local_stability.csv',rows);dump(out/'local_stability_checks.json',verification)
    fig,axes=plt.subplots(1,2,figsize=(12,4));ax=axes[0]
    for seed in range(5):
        items=[r for r in rows if r['seed']==seed]
        ax.plot([r['training_transitions'] for r in items],[r['true_closed_loop_radius'] for r in items],'o-',label=f'seed {seed}')
    ax.axhline(1,color='black',linestyle=':',label='Unit circle')
    ax.set(xscale='log',xlabel='Fitted training transitions',ylabel='Spectral radius of true local closed-loop Jacobian',title='Identified local law at upright; diagnostic, no finite-IC guarantee')
    ax.legend(fontsize=8,ncol=3)
    for seed in range(5):
        items=[r for r in rows if r['seed']==seed and r['episodes']>=1000]
        axes[1].plot([r['training_transitions'] for r in items],[r['true_closed_loop_radius'] for r in items],'o-')
    axes[1].axhline(1,color='black',linestyle=':')
    axes[1].set(xscale='log',ylim=(.983,1.005),xlabel='Fitted training transitions',ylabel='Spectral radius',title='Larger budgets: zoom near the unit circle')
    ax.set_title('All budgets: true local closed-loop Jacobian')
    fig.suptitle('Identified local law at upright; diagnostic, no finite-IC guarantee')
    fig.tight_layout();fig.savefig(out/'local_stability.png',dpi=150);plt.close(fig)
    for episodes in (35,100,350,1000,3500,10000):
        values=[r['true_closed_loop_radius'] for r in rows if r['episodes']==episodes]
        print(episodes,values,'locally unstable',sum(v>1 for v in values),flush=True)
    print('Independent analytic/finite-difference and true-DARE reference checks:',verification,flush=True)

if __name__=='__main__':main()
