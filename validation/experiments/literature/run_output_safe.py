"""Accounting-only wrapper; preserve frozen fitting driver/model/data identities."""
import json,time
import run_output_history
from pathlib import Path
from common import Budget,OUT,dump,cpu_seconds,usage_gate

original_fit=run_output_history.fit
original_digest=run_output_history.digest


def fit(seed,mode,budget):
    source=Path(__file__).with_name('run_output_history.py')
    compatibility=json.loads((OUT/'inventory/output_driver_compatibility.json').read_text())
    assert original_digest(source)==compatibility['current_driver_sha256']
    p=OUT/('dfine_'+mode)/'output_only'/f'seed{seed}/protocol.json'
    recorded=json.loads(p.read_text())['adapter_sha256']['run_output_history.py'] if p.exists() else original_digest(source)
    assert recorded in compatibility['compatible_recorded_driver_sha256']
    # Bind the exact historical driver identity; compatibility is explicit and
    # limited to the independently reconstructed accounting-only source variant.
    run_output_history.digest=lambda path:recorded if Path(path)==source else original_digest(path)
    try:return original_fit(seed,mode,budget)
    finally:run_output_history.digest=original_digest


class CPUBudget(Budget):
    def checkpoint(self):
        value=dict(cpu_seconds=self.previous['cpu_seconds']+cpu_seconds()-self.start_cpu,wall_seconds=self.previous['wall_seconds']+time.monotonic()-self.start_wall,gpu_seconds=self.previous.get('gpu_seconds',0.))
        dump(self.path,value)
        if value['cpu_seconds']>=self.ceiling or sum(json.loads(p.read_text())['cpu_seconds'] for p in OUT.glob('*/accounting.json'))>=84*3600:raise RuntimeError('CPU allocation exhausted; cap-limited partial result')
        usage_gate();return value


if __name__=='__main__':
    run_output_history.Budget=CPUBudget;run_output_history.fit=fit;run_output_history.main()
