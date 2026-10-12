#!/usr/bin/env python3
"""Postprocess saved CSVs only; never runs controllers or training."""
import argparse, ast, csv, json
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm

ROOT=Path(__file__).resolve().parents[2] if 'validation/experiments' in str(Path(__file__).resolve()) else Path('/home/ap60/github/data-driven-control/TrajectoryManifoldControl')
parser=argparse.ArgumentParser();parser.add_argument('--project',type=Path,default=ROOT);parser.add_argument('--output',type=Path);args=parser.parse_args()
out=args.output or args.project/'validation/results_v1/briefing_figures';out.mkdir(parents=True,exist_ok=True)
plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,'savefig.dpi':220,'pdf.fonttype':42})
cmap=ListedColormap(['#f0f0ef','#b8d6e3','#5b9cb9','#135373']);norm=BoundaryNorm([-.167,.167,.5,.833,1.167],4)
def read(name):
 rows=list(csv.DictReader(open(args.project/f'validation/results_v1/{name}/summary.csv')))
 for r in rows:r['state']=np.array(ast.literal_eval(r['initial_state']));r['ok']=r['primary_success'].lower()=='true'
 return rows
def groups(rows,cohort,arm):
 g=defaultdict(list)
 for r in rows:
  if r['ic_set']==cohort and r['arm']==arm and float(r['seconds'])==14:g[int(r['ic_index'])].append(r)
 for rs in g.values():assert sorted(int(r['seed']) for r in rs)==[0,1,2]
 return {k:(v[0]['state'],sum(r['ok'] for r in v)/3) for k,v in g.items()}
def save(fig,name):
 for ext in ['png','pdf']:fig.savefig(out/f'{name}.{ext}',bbox_inches='tight')
 plt.close(fig)
e1=read('E1');arms=['H','L'];labels=['H: hybrid','L: fitted linear feedback']
fig,axs=plt.subplots(2,2,figsize=(8.5,5.7),gridspec_kw={'height_ratios':[4,1]},layout='constrained');stats={}
for j,(arm,label) in enumerate(zip(arms,labels)):
 g=groups(e1,'D',arm);states=np.array([x[0] for x in g.values()]);v=np.array([x[1] for x in g.values()]);theta=np.rad2deg(states[:,2]);rate=states[:,3]*np.sqrt(9.81)
 xs=np.unique(theta);ys=np.unique(rate);z=np.full((len(ys),len(xs)),np.nan)
 for x,y,s in zip(theta,rate,v):z[np.searchsorted(ys,y),np.searchsorted(xs,x)]=s
 im=axs[0,j].imshow(z,origin='lower',extent=[xs[0]-(xs[1]-xs[0])/2,xs[-1]+(xs[1]-xs[0])/2,ys[0]-(ys[1]-ys[0])/2,ys[-1]+(ys[1]-ys[0])/2],aspect='auto',cmap=cmap,norm=norm,interpolation='nearest')
 axs[0,j].set(title=label,xlabel=r'Initial angle $\theta_0$ (deg)');axs[0,j].set_xticks([-60,-30,0,30,60])
 if j==0:axs[0,j].set_ylabel(r'Initial angular rate $\dot\theta_0$ (rad/s)')
 b=groups(e1,'B',arm);bs=sorted(b.values(),key=lambda x:x[0][2]);bv=np.array([s for x,s in bs]);bt=np.rad2deg([x[2] for x,s in bs]);axs[1,j].imshow(bv[None,:],extent=[-85.5,85.5,0,1],aspect='auto',cmap=cmap,norm=norm,interpolation='nearest');axs[1,j].set(xlabel='From rest: angle (deg)',yticks=[],xticks=[-80,-40,0,40,80]);
 if j==0:axs[1,j].set_ylabel('From rest')
 stats[arm]={'B_any_success_angles_deg':[round(float(x),8) for x in bt[bv>0]],'B_all3_success_angles_deg':[round(float(x),8) for x in bt[bv==1]],'D_successful_cells_of_1681':int(sum(v>0)),'D_successes_of_5043':int(round(sum(v)*3))}
fig.suptitle('Success frequency at each initial state | 14 s | three fitting/planning seeds\nAngle/rate grid and from-rest slice: initial cart position and speed are zero',fontsize=13)
cb=fig.colorbar(im,ax=axs.ravel().tolist(),fraction=.018,pad=.015,ticks=[0,1/3,2/3,1]);cb.ax.set_yticklabels(['0/3','1/3','2/3','3/3']);cb.set_label('Successful runs / 3')
save(fig,'initial_state_regimes')
e4=read('E4');e6=read('E6_SMC');basearms=['H','L','M1','E5']+[f'DeePC_{k}_{v}' for k in ['state','output'] for v in ['0.001','0.01','0.1']]+['H','L','SMC','SMC_linear'];sources=['E4']*10+['E6']*4;baselabels=['H','L','NN planner (M1)','Ensemble (E5)']+[f'DeePC {k}, {v}' for k in ['state','output'] for v in ['0.001','0.01','0.1']]+['H','L','SMC','SMC zero correction']
fig=plt.figure(figsize=(17,9),layout='constrained');outer=fig.add_gridspec(1,2,width_ratios=[4.3,1.7]);baseline_counts={}
for col,cohort in enumerate(['C50','A']):
 gs=outer[col].subgridspec(5,1,height_ratios=[7,1,1,1,1],hspace=.07);g=groups(e4,cohort,'H');order=sorted(g,key=lambda k:g[k][0][2]);states=np.array([g[k][0] for k in order]);matrix=[];totals=[]
 for source,arm in zip(sources,basearms):
  gg=groups(e6,'nominal_'+cohort,arm) if source=='E6' else groups(e4,cohort,arm)
  assert set(gg)==set(g)
  for k in order:assert np.allclose(gg[k][0],g[k][0])
  vv=[gg[k][1] for k in order];matrix.append(vv);totals.append(int(round(sum(vv)*3)))
 baseline_counts[cohort]=dict(zip([source+'_'+arm for source,arm in zip(sources,basearms)],totals));ax=fig.add_subplot(gs[0]);im=ax.imshow(matrix,aspect='auto',cmap=cmap,norm=norm,interpolation='nearest');ax.set_yticks(range(len(basearms)),[f'{source}: {label}  [{n}/{len(order)*3}]' for source,label,n in zip(sources,baselabels,totals)]);ax.axhline(9.5,color='#b34b31',lw=2);ax.set_xticks([]);ax.set_title(f'{cohort}: {len(order)} initial states, sorted by angle',fontsize=12)
 for k,(vals,label) in enumerate([(np.rad2deg(states[:,2]),r'$\theta_0$ (deg)'),(states[:,3]*np.sqrt(9.81),r'$\dot\theta_0$ (rad/s)'),(states[:,0],r'$p_0$ (m)'),(states[:,1]*np.sqrt(9.81),r'$\dot p_0$ (m/s)')]):
  aa=fig.add_subplot(gs[k+1]);aa.plot(range(len(order)),vals,'o-',markersize=2.5,lw=.7,color='#3b657a');aa.set_xlim(-.5,len(order)-.5);aa.set_ylabel(label,fontsize=9);aa.grid(axis='y',alpha=.18)
  if k<3:aa.set_xticks([])
  else:aa.set_xlabel('Initial-state rank (other coordinates vary)')
fig.suptitle('Nominal baselines: same initial states; different training collections; compare within each block\nE4: 3,281 × 32 collection | E6: original 3,500 × 30 collection | 14 s | three whole-run seeds per cell',fontsize=13)
cb=fig.colorbar(im,ax=fig.axes,fraction=.018,pad=.015,ticks=[0,1/3,2/3,1]);cb.ax.set_yticklabels(['0/3','1/3','2/3','3/3']);save(fig,'matched_baseline_regimes')
(out/'regime_statistics.json').write_text(json.dumps({'E1':stats,'baseline_success_counts':baseline_counts},indent=2)+'\n')
print(out)
