"""Audit an old PCA view and export a target-plane view with measured training rows."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from src.trajectory_visualization import Trajectories, TrajectoryManifoldVisualization
from src.trajectory_target_visualization import target_plane_view


def prepare(source, dataset):
    base, metadata = TrajectoryManifoldVisualization.from_export(source)
    checkpoint = Path(metadata['model_source'])
    companion = checkpoint.with_name('data.npz')
    if hashlib.sha256(checkpoint.read_bytes()).hexdigest() != metadata['checkpoint_sha256']:
        raise ValueError('checkpoint hash does not match source visualization')
    if hashlib.sha256(companion.read_bytes()).digest() != hashlib.sha256(Path(dataset).read_bytes()).digest():
        raise ValueError('dataset must match the checkpoint companion training artifact')
    # Reproduce split_episodes exactly, retaining original dataset row IDs.
    with np.load(dataset) as data:
        x, u = data['X'], data['U']
    seed = metadata['config']['simulation']['seed']
    order = np.random.default_rng(seed).permutation(len(x))
    count = max(1, round(.15*len(x)))
    train_ids = order[2*count:]
    training = Trajectories(x[train_ids], u[train_ids])
    view = target_plane_view(base, training, training_indices=train_ids)
    selected_indices = [i for i, p in enumerate(base.selected_xyz) if p is not None]
    old_distances = np.array([np.linalg.norm(base.selected_xyz[i]-base.reference_xyz) for i in selected_indices])
    old_steps = np.array([base.frames[i].step for i in selected_indices])
    summary = {'source': str(Path(source).resolve()), 'dataset': str(Path(dataset).resolve()),
        'dataset_sha256': hashlib.sha256(Path(dataset).read_bytes()).hexdigest(),
        'checkpoint_sha256': metadata['checkpoint_sha256'], 'training_split_seed': seed,
        'training_provenance_verified': 'checkpoint hash and companion dataset hash matched',
        'training_episodes_total': len(train_ids), 'training_episodes_drawn': len(view.training.x),
        'training_episode_ids': view.training_indices.tolist(),
        'old_reference_PCA_distance_first_last': old_distances[[0,-1]].tolist(),
        'old_background_center_distance_to_selected_first_last': [
            float(np.linalg.norm(base.selected_xyz[i]-base.background_xyz.mean(0))) for i in [selected_indices[0], selected_indices[-1]]],
        'displayed_state_error_first_last': view.state_errors[[0,-1]].tolist(),
        'target_plane_height_first_last': np.log1p(view.state_errors[[0,-1]]).tolist(),
        'final_state_SI': (view.states[-1]*view.scale).tolist(),
        'target_coordinates': ['angle', 'angular_speed', 'cart_speed'],
        'target_SI': (view.target*view.scale)[[2,3,1]].tolist(),
        'error_tolerances_SI': view.projection.tolerances.tolist(),
        'mapping': '[training-PC1(full behavior), training-PC2(full behavior), log1p(norm((x0-target)[2,3,1]*SI_scale/tolerance))]',
        'plane_meaning': 'initial state of horizon matches left displayed target; cart position and future inputs free; not a guarantee of future tracking/feasibility',
        'background': '256 newly decoded random initial states and bounded random input sequences, not training records',
        'candidates': '24 newly decoded random input sequences conditioned on measured x0, not training records or optimizer population'}
    return base, view, metadata, summary, (old_steps, old_distances)


def figures(base, view, summary, old, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fig = view.save_static(directory/'elements.png', step=view.frames[len(view.frames)//2].step)
    plt.close(fig)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    t, states = view.t, view.states*view.scale
    for idx, label in [(2,'angle [rad]'), (3,'angular speed [rad/s]'), (1,'cart speed [m/s]')]:
        axes[0].plot(t, states[:,idx], label=label)
    axes[0].axhline(0,color='red',alpha=.4);axes[0].legend(fontsize=7)
    axes[0].set(title='Displayed physical states approach target', xlabel='time [s]')
    axes[1].plot(base.t[old[0]],old[1],color='red')
    axes[1].set(title='Old constant-reference PCA distance',xlabel='time [s]',ylabel='PCA units (not state error)')
    axes[2].plot(t,np.log1p(view.state_errors),color='dodgerblue')
    axes[2].axhline(0,color='red',label='target plane');axes[2].legend(fontsize=8)
    axes[2].set(title='New distance to target plane',xlabel='time [s]',ylabel='log(1 + normalized state error)')
    fig.tight_layout();fig.savefig(directory/'progress.png',dpi=140);plt.close(fig)
    fig=plt.figure(figsize=(13,4.5))
    for column,idx in enumerate([0,len(view.frames)//2,len(view.frames)-2]):
        axis=fig.add_subplot(1,3,column+1,projection='3d')
        # Reuse exact cached drawing for report snapshots.
        p=view.selected_xyz[idx];limits=view.projected_limits
        xx,yy=np.meshgrid(limits[0],limits[1]);axis.plot_surface(xx,yy,np.zeros_like(xx),color='red',alpha=.14,shade=False)
        axis.scatter(*view.training_xyz.T,c='dimgray',s=5,alpha=.25)
        axis.scatter(*view.candidate_xyz[idx].T,c='purple',s=12,alpha=.65)
        if p is not None:
            axis.scatter(*p,c='dodgerblue',marker='D',s=65);axis.plot([p[0]]*2,[p[1]]*2,[0,p[2]],color='dodgerblue',linestyle=':')
        axis.set(xlim=limits[0],ylim=limits[1],zlim=limits[2],title=f't={view.t[view.frames[idx].step]:.1f}s',xlabel='behavior PC 1',ylabel='behavior PC 2',zlabel='state-error height')
        axis.view_init(elev=23,azim=-58)
    fig.tight_layout();fig.savefig(directory/'snapshots.png',dpi=140);plt.close(fig)
    (directory/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--dataset',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--report',type=Path,default=Path('docs/visualization/target-plane'))
    parser.add_argument('--skip-animation',action='store_true')
    args=parser.parse_args()
    if args.source.resolve()==args.output.resolve():parser.error('preserve source: use a separate output')
    base,view,metadata,summary,old=prepare(args.source,args.dataset)
    figures(base,view,summary,old,args.report)
    args.output.mkdir(parents=True,exist_ok=True)
    view.save_static(args.output/'static.png',step=view.frames[len(view.frames)//2].step)
    # Preserve the original controller trajectory; additional cache records training provenance.
    np.savez_compressed(args.output/'closed_loop.npz',t=view.t,X=view.states,U=view.inputs)
    np.savez_compressed(args.output/'target_view.npz',training_w=view.training.w,
        training_episode_ids=view.training_indices,training_xyz=view.training_xyz,
        background_xyz=view.background_xyz,candidates_xyz=np.stack(view.candidate_xyz),
        selected_xyz=np.array([p if p is not None else [np.nan]*3 for p in view.selected_xyz]),
        mean=view.projection.mean,scale=view.projection.scale,components=view.projection.components,
        state_error=view.state_errors,steps=[f.step for f in view.frames])
    (args.output/'metadata.json').write_text(json.dumps({**metadata,**summary,'projection':'target-aligned',
        'projection_diagnostics':view.projection.diagnostics, 'manifold_style':'target-plane',
        'frames':len(view.frames),'fps':15},indent=2)+'\n')
    print(json.dumps(summary,indent=2),flush=True)
    if not args.skip_animation:view.save_animation(args.output/'animation.gif',fps=15)


if __name__=='__main__':main()
