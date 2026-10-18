"""Bounded diagnostics of an unchanged upstream learned decoder and CEM.

Uses saved measured episodes; plant access occurs only in diagnostic replay,
never inside candidate optimization. No controller or model is selected here.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

from validation_benchmark import load_upstream, sha256, write_json, ROOT, PIN


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model-dir',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--device',choices=('cpu','cuda'),default='cpu')
    p.add_argument('--candidates',type=int,default=4)
    args=p.parse_args();args.output=args.output.resolve()
    if args.output.is_relative_to(ROOT/'validation'):
        p.error('output must be outside validation snapshot')
    args.output.mkdir(parents=True,exist_ok=True)
    if (args.output/'diagnostics.json').exists():
        p.error('diagnostic output already exists; use a new directory')
    c,_,_=load_upstream(args.output)
    np,torch=c.np,c.torch
    transition,payload=c.runner.load_transition(args.model_dir/'transition.pt',args.device)
    cfg=c.runner.make_config('dev',seed=payload['config']['seed'],device=args.device)
    if payload['horizon']!=30:raise ValueError('diagnostic requires original N=30')
    decoder=c.CanonicalBehaviorDecoder(transition,30).to(args.device).eval()
    for parameter in decoder.parameters():parameter.requires_grad_(False)
    with np.load(args.model_dir/'data.npz') as data:
        split=c.runner.split_episodes(data['X'],data['U'],cfg.seed)
    gain,_=c.runner.fit_local_feedback(split['train'],c.Q,c.R)
    x,u=(torch.from_numpy(a).to(args.device) for a in split['test'])
    with torch.no_grad():
        prediction=decoder.rollout(x[:,0],u)
        q=decoder.coordinates(x[:,0],u)
        reconstructed,inputs=decoder.unpack_behavior(decoder(q))
        err=(prediction[:,1:]-x[:,1:]).cpu().numpy()
        truth=x[:,1:].cpu().numpy()
        rmse=np.sqrt(np.mean(err**2,axis=(0,1)))
        scale=truth.std(axis=(0,1)).clip(1e-8)
        r2=1-np.sum(err**2,axis=(0,1))/np.sum((truth-truth.mean(axis=(0,1)))**2,axis=(0,1)).clip(1e-16)
        continuation=decoder.rollout(prediction[:,10],torch.cat((u[:,10:],u[:,-10:]),dim=1))
        prefix_error=float((continuation[:,:21]-prediction[:,10:]).abs().max())
    point=q[0].detach().requires_grad_(True)
    jac=torch.autograd.functional.jacobian(decoder,point)
    singular=torch.linalg.svdvals(jac).detach().cpu().numpy()
    epsilon=1e-3
    eye=torch.eye(len(point),device=point.device)
    with torch.no_grad():
        plus=decoder(point[None,:]+epsilon*eye)
        minus=decoder(point[None,:]-epsilon*eye)
        numerical=((plus-minus)/(2*epsilon)).T
        numerical_singular=torch.linalg.svdvals(numerical).cpu().numpy()
    z=torch.zeros(4,device=args.device);uz=torch.zeros(1,device=args.device)
    with torch.no_grad():bias=transition(z,uz).cpu().numpy()
    report=dict(upstream=PIN,checkpoint_sha256=sha256(args.model_dir/'transition.pt'),
        dataset_sha256=sha256(args.model_dir/'data.npz'),script_sha256=sha256(__file__),
        test_episodes=len(x),training_transitions=len(split['train'][1])*30,
        model_parameters=sum(p.numel() for p in transition.parameters()),
        coordinate_dimension=decoder.alpha_dim,free_input_coordinates=30,behavior_dimension=decoder.w_dim,
        rmse_by_state=rmse.tolist(),nrmse_by_state=(rmse/scale).tolist(),r2_by_state=r2.tolist(),
        p95_by_state=np.percentile(np.abs(err),95,axis=(0,1)).tolist(),
        rmse_by_horizon_state=np.sqrt(np.mean(err**2,axis=0)).tolist(),
        p95_by_horizon_state=np.percentile(np.abs(err),95,axis=0).tolist(),
        initial_residual=float((reconstructed[:,0]-x[:,0]).abs().max()),
        input_residual=float((inputs-u).abs().max()),
        predicted_prefix_residual=prefix_error,
        equilibrium_one_step_bias=bias.tolist(),
        decoder_jacobian=dict(autograd_singular_values=singular.tolist(),
            finite_difference_singular_values=numerical_singular.tolist(),
            finite_difference_rank=int(np.linalg.matrix_rank(numerical.cpu().numpy())),
            max_autograd_difference=float((numerical-jac).abs().max()),finite_difference_step=epsilon,
            interpretation='Identity coordinate projection guarantees full decoder rank; future-state sensitivity can still be weak or inaccurate.'),
        candidates=[],selection='First validation episodes, no final evaluation starts; diagnostic only, no tuning')
    Q=torch.as_tensor(c.Q,dtype=torch.float32,device=args.device)
    R=torch.as_tensor(c.R,dtype=torch.float32,device=args.device)
    for index in range(min(args.candidates,len(split['validation'][0]))):
        initial=split['validation'][0][index,0].copy()
        current=torch.as_tensor(initial,device=args.device)
        solver=c.DataDrivenCEMSolver(decoder,Q,R,Q_terminal=12*Q,u_bounds=(-10,10),
            population=256,elite=32,iterations=3,input_rate_weight=.002,seed=cfg.seed)
        original_cost=solver._cost; sampled=[]
        def record_cost(states,inputs):
            costs=original_cost(states,inputs)
            sampled.extend(costs.detach().cpu().tolist())
            return costs
        solver._cost=record_cost
        start=time.perf_counter();solution=solver.solve(current)
        if args.device=='cuda':torch.cuda.synchronize()
        elapsed=time.perf_counter()-start
        # Fixed L is diagnostic only. It never supplies an action to the manifold
        # controller; evaluate it as an independent plant-generated candidate.
        linear=c.controller(None,cfg,gain,'L')
        _,lx,lu=c.runner.simulate_plant(linear,initial,cfg,seconds=.6)
        proposed=solution.u.detach().cpu().numpy()
        def proposal(k,state):return proposed[k]
        _,actual,applied=c.runner.simulate_plant(proposal,initial,cfg,seconds=.6)
        np.testing.assert_array_equal(applied,proposed)
        with torch.no_grad():
            learned_l=decoder.rollout(current,torch.as_tensor(lu,dtype=current.dtype,device=current.device))
            lcost=float(original_cost(learned_l[None],torch.as_tensor(lu,dtype=current.dtype,device=current.device)[None])[0])
            chosen_cost=float(original_cost(solution.x[None],solution.u[None])[0])
        # Physical angle/rate response to each input coordinate, using learned
        # future states only. Plant finite-difference derivatives are not queried.
        def future(inputs):return decoder.rollout(current,inputs)[:,2:4]
        sensitivity=torch.autograd.functional.jacobian(future,solution.u).detach().cpu().numpy()
        sv=np.linalg.svd(sensitivity.reshape(62,30),compute_uv=False)
        delta=solution.x.detach().cpu().numpy()-actual
        finite=np.asarray(sampled)[np.isfinite(sampled)]
        row=dict(index=index,initial_state=initial.tolist(),predicted_cost=chosen_cost,
            reported_cost=solution.loss,best_sampled_cost=float(finite.min()) if len(finite) else None,
            solver_finite=solution.finite,solver_iterations=solution.iterations,
            sampled_candidates=len(sampled),nonfinite_costs=int(np.sum(~np.isfinite(sampled))),
            solve_seconds=elapsed,exceeds_20ms=elapsed>.02,
            l_candidate_predicted_cost=lcost,cem_cost_exceeds_l_candidate=chosen_cost>lcost+1e-5*max(1,abs(lcost)),
            replay_rmse_by_state=np.sqrt(np.mean(delta[1:]**2,axis=0)).tolist(),
            replay_terminal_error_by_state=delta[-1].tolist(),
            terminal_angle_input_gradient=sensitivity[-1,0,:,0].tolist(),
            terminal_rate_input_gradient=sensitivity[-1,1,:,0].tolist(),
            angle_rate_sensitivity_singular_values=sv.tolist(),
            input_bound_residual=float(np.maximum(np.abs(proposed)-10,0).max()))
        report['candidates'].append(row)
        np.savez_compressed(args.output/f'candidate_{index:03d}.npz',initial=initial,
            proposed=proposed,prediction=solution.x.detach().cpu().numpy(),actual=actual,
            linear_inputs=lu,linear_actual=lx,linear_prediction=learned_l.cpu().numpy())
    write_json(args.output/'diagnostics.json',report)
    print(json.dumps({key:report[key] for key in ('model_parameters','nrmse_by_state','r2_by_state','equilibrium_one_step_bias','initial_residual','input_residual','predicted_prefix_residual')},indent=2))


if __name__=='__main__':main()
