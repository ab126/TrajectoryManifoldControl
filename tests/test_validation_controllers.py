import numpy as np
import pytest
import torch
from src.manifold_control import CanonicalBehaviorDecoder, DataDrivenTransition, DataDrivenMPCSolution
from src.manifold_experiments import NativeConfig
from src.validation_controllers import validation_lqr, trajectory_controller, ControllerFailure, TrackingObjective


def decoder():
    return CanonicalBehaviorDecoder(DataDrivenTransition(4, 1, (8,)), 4)


def failed(state, initial, previous):
    return DataDrivenMPCSolution(torch.zeros(5, 4), initial, float('inf'), 0, False)


def test_reference_linear_and_hybrid_share_exact_actions():
    gain = np.array([[1.234567891, 3., 4., 5.]])
    xr, ur = np.array([2., 0., 0., 0.]), np.array([.7])
    linear = validation_lqr(gain, x_ref=xr, u_ref=ur)
    hybrid = trajectory_controller(decoder(), NativeConfig(), x_ref=xr, u_ref=ur, linear=linear)
    hybrid.solver.solve = failed
    for k, state in enumerate((xr, xr + [.1, 0, 0, 0], xr + [0, 0, 1., 0])):
        np.testing.assert_array_equal(hybrid(k, state), linear(k, state))
    assert hybrid.diagnostics['fallback_events'] == [2]
    np.testing.assert_array_equal(linear(0, xr), ur)
    np.testing.assert_array_equal(linear(0, xr+[100.,0,0,0]), [-10.])


def test_zero_reference_matches_validation_formula():
    gain = np.array([[2., 3., 4., 5.]])
    linear = validation_lqr(gain)
    for state in (np.zeros(4), np.ones(4), np.array([.1,-.3,.2,-.8])):
        np.testing.assert_array_equal(linear(0, state), np.clip(-gain@state, -10, 10))


def test_standalone_failure_never_applies_previous_or_linear_action():
    ctl = trajectory_controller(decoder(), NativeConfig())
    def valid(state, initial, previous):
        return DataDrivenMPCSolution(torch.zeros(5,4), torch.ones(4,1), 1., 1, True)
    ctl.solver.solve = valid
    np.testing.assert_array_equal(ctl(0, np.ones(4)), [1.])
    ctl.solver.solve = failed
    with pytest.raises(ControllerFailure):
        ctl(1, np.ones(4))
    assert ctl.diagnostics['fallback_events'] == []
    assert ctl.diagnostics['linear_steps'] == []
    assert ctl.diagnostics['failure_steps'] == [1]


def test_tracking_objective_is_reference_shift_invariant():
    q, r = torch.eye(4), torch.eye(1)
    xr, ur = torch.tensor([2.,1.,.3,.4]), torch.tensor([.6])
    base = TrackingObjective(q,r,torch.zeros(4),torch.zeros(1),include_previous_input=True)
    shifted = TrackingObjective(q,r,xr,ur,include_previous_input=True)
    x, u, prev = torch.randn(2,5,4), torch.randn(2,4,1), torch.ones(1)
    torch.testing.assert_close(base(x,u,prev), shifted(x+xr,u+ur,prev+ur))


def test_nonfinite_plan_is_rejected_even_if_solver_flags_finite():
    ctl = trajectory_controller(decoder(), NativeConfig())
    ctl.solver.solve = lambda *args: DataDrivenMPCSolution(torch.zeros(5,4),torch.full((4,1),float('nan')),1.,1,True)
    with pytest.raises(ControllerFailure):ctl(0,np.zeros(4))


def test_cli_records_strict_failure_continues_linear_and_resumes(tmp_path):
    import json
    import os
    from pathlib import Path
    import subprocess
    import sys
    from dataclasses import asdict
    from scripts.run_inverted_pendulum_mc import ExperimentConfig
    model = DataDrivenTransition(4, 1, (8,))
    with torch.no_grad():
        for parameter in model.parameters():parameter.fill_(float('nan'))
    checkpoint = tmp_path/'invalid.pt'
    cfg = ExperimentConfig(horizon=4, hidden_dims=(8,), device='cpu')
    torch.save(dict(format='canonical_behavior_decoder_v1', transition_state_dict=model.state_dict(),
        x_dim=4,u_dim=1,horizon=4,hidden_dims=(8,),config=asdict(cfg)), checkpoint)
    root = Path(__file__).resolve().parents[1]
    output = tmp_path/'evaluation'
    cmd = [sys.executable,str(root/'scripts/validate_tmc.py'),'--checkpoint',str(checkpoint),
        '--archived-gain-seed','0','--output',str(output),'--limit','1','--seconds','.04',
        '--iterations','1','--x-ref','1','0','0','0','--u-ref','.2']
    env = {**os.environ, 'MPLCONFIGDIR':str(tmp_path/'mpl')}
    subprocess.run(cmd, cwd=root, env=env, capture_output=True, text=True, check=True)
    row = json.loads((output/'cases/TMC_0000.json').read_text())
    assert not row['completed'] and not row['terminal_recovery']
    assert row['failure_steps']==[0] and row['fallback_events']==[] and row['linear_steps']==[]
    with np.load(output/'raw/TMC_0000.npz') as raw:assert raw['U'].shape==(0,1)
    assert json.loads((output/'cases/L_0000.json').read_text())['completed']
    before = (output/'cases/TMC_0000.json').read_bytes()
    subprocess.run(cmd,cwd=root,env=env,capture_output=True,text=True,check=True)
    assert (output/'cases/TMC_0000.json').read_bytes()==before
    assert json.loads((output/'summary.json').read_text())['complete']
    result = subprocess.run(cmd+['--learning-rate','.1'],cwd=root,env=env,capture_output=True,text=True)
    assert result.returncode!=0 and 'output protocol differs' in result.stderr
