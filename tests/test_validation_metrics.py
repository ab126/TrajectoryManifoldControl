import numpy as np
import pytest

from src.validation_metrics import score_trace


def trace():
    t = np.arange(701)*.02
    return t, np.zeros((701,4)), np.zeros((700,1))


def score(t, x, u):
    final = t >= t[-1]-3
    legacy = bool(np.isfinite(x).all() and np.all(np.abs(x[:,2])<np.pi/2)
                  and np.all(np.abs(x[final,2])<=.02)
                  and np.all(np.abs(x[final,3])*np.sqrt(9.81)<=.05))
    return score_trace(t,x,u,upstream={'primary_success': legacy})


def test_all_endpoints_agree_at_zero():
    result=score(*trace())
    assert all(result[name] for name in ('legacy_uninterrupted','terminal_recovery','periodic_terminal_recovery'))
    assert result['angle_settle_seconds']==0


def test_prior_horizontal_crossing_only_changes_legacy():
    t,x,u=trace(); x[30,2]=2
    result=score(t,x,u)
    assert not result['legacy_uninterrupted']
    assert result['terminal_recovery'] and result['periodic_terminal_recovery']
    assert result['horizontal_crossings_sampled']==2


@pytest.mark.parametrize('turns', [-2,-1,1,2])
def test_full_rotations_keep_unwrapped_state_and_physical_rate(turns):
    t,x,u=trace(); x[:,2]=2*np.pi*turns
    original=x.copy()
    result=score(t,x,u)
    assert result['periodic_terminal_recovery']
    assert not result['terminal_recovery']
    assert not result['legacy_uninterrupted']
    np.testing.assert_array_equal(x,original)
    x[-1,3]=.051/np.sqrt(9.81)
    assert not score(t,x,u)['periodic_terminal_recovery']


def test_tolerances_and_complete_window():
    t,x,u=trace(); x[-1,2]=.02001
    assert not score(t,x,u)['terminal_recovery']
    x[-1,2]=.01999; x[-1,3]=.04999/np.sqrt(9.81)
    assert score(t,x,u)['terminal_recovery']
    assert not score(t[:50],x[:50],u[:49])['terminal_recovery']


def test_nonfinite_trace_fails_all_endpoints_and_remains_json_safe():
    import json
    t,x,u=trace(); x[10,2]=np.nan
    with np.errstate(invalid='ignore'):
        result=score(t,x,u)
    assert result['numerical_failure']
    assert not any(result[k] for k in ('legacy_uninterrupted','terminal_recovery','periodic_terminal_recovery'))
    json.dumps(result,allow_nan=False)


def test_effort_and_saturation_use_sample_durations():
    t,x,u=trace(); u[:]=10
    result=score(t,x,u)
    assert result['control_energy_normalized']==pytest.approx(1400)
    assert result['saturation_seconds']==pytest.approx(14)


def test_bad_trace_shapes_rejected():
    t,x,u=trace()
    with pytest.raises(ValueError):
        score_trace(t,x.T,u,upstream={'primary_success': False})
