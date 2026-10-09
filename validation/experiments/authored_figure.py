"""Show cart excursions as well as the authored angular demonstration."""
from campaign import *
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
summary=json.loads((OUT/'E0/authored_summary.json').read_text())
data=np.load(summary['artifacts']['results'])
fig,axes=plt.subplots(3,1,figsize=(9,8),sharex=True)
for label in summary['closed_loop_metrics']:
    t,X,U=(data[f'{prefix}_{label}'] for prefix in ('t','X','U'))
    tag=label.replace('theta0_','').replace('deg','°')
    axes[0].plot(t,np.rad2deg(X[:,2]),label=tag)
    axes[1].plot(t,X[:,0],label=tag)
    axes[2].step(t[:-1],U[:,0],where='post',label=tag)
axes[0].set_ylabel('Angle [degrees]');axes[0].legend(ncol=3,fontsize=9)
axes[1].set_ylabel('Cart position [m]');axes[2].set_ylabel('Input F/(mg)')
axes[2].set_xlabel('Physical time [s]')
fig.suptitle('Fresh authored dev reproduction, seed 234: 6/6 balanced with no fall')
fig.tight_layout();fig.savefig(OUT/'E0/authored_excursions.png',dpi=160);plt.close(fig)
print('Saved authored_excursions.png')
