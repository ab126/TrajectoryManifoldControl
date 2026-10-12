"""One-GPU leases, conservative GPU occupancy accounting and atomic torch saves."""
import fcntl
import time
import torch
from common import Budget, OUT, dump, cpu_seconds, usage_gate


def atomic_save(path,value):
    path.parent.mkdir(parents=True,exist_ok=True)
    temp=path.with_suffix(path.suffix+'.tmp');torch.save(value,temp);temp.replace(path)


class GPUBudget(Budget):
    def __init__(self,stage,cpu_hours,gpu_hours):
        super().__init__(stage,cpu_hours)
        self.gpu_ceiling=gpu_hours*3600
        self.lease=open(OUT/'.gpu.lock','a')
        fcntl.flock(self.lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        torch.set_num_threads(1);torch.set_num_interop_threads(1)
        if not torch.cuda.is_available():raise RuntimeError('GPU unavailable; use guarded host job, not another environment')

    def checkpoint(self):
        value=dict(cpu_seconds=self.previous['cpu_seconds']+cpu_seconds()-self.start_cpu,
                   wall_seconds=self.previous['wall_seconds']+time.monotonic()-self.start_wall,
                   gpu_seconds=self.previous.get('gpu_seconds',0.)+time.monotonic()-self.start_wall)
        dump(self.path,value)
        cpu_total=sum(__import__('json').loads(p.read_text())['cpu_seconds'] for p in OUT.glob('*/accounting.json'))
        total=sum(__import__('json').loads(p.read_text()).get('gpu_seconds',0.) for p in OUT.glob('*/accounting.json'))
        if value['cpu_seconds']>=self.ceiling or cpu_total>=84*3600:raise RuntimeError('CPU ceiling reached: preserve partial result')
        if value['gpu_seconds']>=self.gpu_ceiling or total>=30*3600:raise RuntimeError('GPU ceiling reached: preserve partial result')
        usage_gate()
        return value
