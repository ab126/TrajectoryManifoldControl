"""Reports must distinguish failed, missing and paired saved cases."""
import json

import numpy as np

from src.validation_report import generate_report


def test_partial_failed_trace_and_no_shared_starts(tmp_path):
    run = tmp_path/'run'
    (run/'cases').mkdir(parents=True)
    (run/'raw').mkdir()
    manifest = dict(arms=['TMC', 'L'], states=[[0., 0., 0., 0.]],
                    state_set='tuning', seconds=14., plant_config={'seed': 0},
                    x_ref=[0.]*4, u_ref=[0.], config={},
                    gain_source={'kind': 'fitted_training_data'}, checkpoint_sha256='test')
    (run/'manifest.json').write_text(json.dumps(manifest))
    row = dict(arm='TMC', ic_index=0, terminal_recovery=False, completed=False,
               controller_latency={'median': .2})
    (run/'cases'/'TMC_0000.json').write_text(json.dumps(row))
    np.savez(run/'raw'/'TMC_0000.npz', t=[0.], X=np.zeros((1, 4)), U=np.empty((0, 1)))
    output = tmp_path/'report'
    report = generate_report(run, output).read_text()
    assert 'Partial run: 1/2 cases saved' in report
    assert 'Paired starts: 0' in report
    assert '| TMC | 1 | 0 | 1 |' in report
    assert (output/'recovery_map.png').is_file()
    assert not list(output.glob('convergence_*.png'))
    evidence = json.loads((output/'evidence.json').read_text())
    assert evidence['cases'] == [row]
    # Add a matching baseline: failed/empty TMC traces must plot without padding.
    row.update(arm='L', completed=True, terminal_recovery=True)
    (run/'cases'/'L_0000.json').write_text(json.dumps(row))
    np.savez(run/'raw'/'L_0000.npz', t=[0., .02], X=np.zeros((2, 4)), U=np.zeros((1, 1)))
    report = generate_report(run, output).read_text()
    assert 'Complete: 2/2 cases saved' in report
    assert 'L-only recoveries: 1' in report
    assert (output/'convergence_001.png').is_file()
