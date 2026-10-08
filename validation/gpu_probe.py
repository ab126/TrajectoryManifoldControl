"""Fail-fast CUDA verification. A successful import alone does not pass."""
import time
PROCESS_START=time.perf_counter()
from common import *
import glob
import subprocess

def probe():
    def command(args):
        try:
            r=subprocess.run(args,capture_output=True,text=True,timeout=15)
            return {'argv':args,'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr}
        except (OSError,subprocess.TimeoutExpired) as error:
            return {'argv':args,'error':str(error)}
    report={'torch':torch.__version__,'torch_build_cuda':torch.version.cuda,
            'python_executable':sys.executable,'system_site_packages':False,
            'device_nodes':glob.glob('/dev/nvidia*'),
            'dev_mounts':[s for s in Path('/proc/self/mountinfo').read_text().splitlines() if ' /dev ' in s],
            'nvidia_smi':command(['nvidia-smi','--query-gpu=name,uuid,memory.total,memory.used,memory.free,driver_version','--format=csv']),
            'nvidia_smi_header':command(['nvidia-smi']),
            'user_report':{'provenance':'user host shell, not independently verified in sandbox','VRAM_total_GiB':24,'VRAM_used_GiB_approx':3},
            'determinism':determinism_manifest(),'passed':False}
    try:
        if torch.version.cuda is None: raise RuntimeError('CPU-only PyTorch; use the private .venv-gpu')
        if not torch.cuda.is_available(): raise RuntimeError('CUDA unavailable in this context; NVIDIA nodes must be exposed by the execution environment. Run this launcher in the host terminal where nvidia-smi works.')
        torch.cuda.set_device(0)
        props=torch.cuda.get_device_properties(0)
        report.update(device_name=props.name,compute_capability=[props.major,props.minor],
                      VRAM_total_bytes=props.total_memory)
        x=torch.randn(32,32,device='cuda'); first=x@x.T; torch.cuda.synchronize()
        second=x@x.T; torch.cuda.synchronize()
        torch.testing.assert_close(first,second,atol=0,rtol=0)
        assert torch.isfinite(first).all()
        from models import check_oracle
        report['oracle_RK4_gpu_max_abs_error']=check_oracle('cuda')
        report['free_bytes'],report['total_bytes']=torch.cuda.mem_get_info()
        report['allocated_peak_bytes']=torch.cuda.max_memory_allocated()
        report['reserved_peak_bytes']=torch.cuda.max_memory_reserved()
        report['passed']=True
    except (RuntimeError,AssertionError) as error:
        report['failure']=str(error)
    save('gpu_probe.json',report)
    print(json.dumps(report,indent=2))
    return report['passed']

if __name__=='__main__':
    configure()
    # Includes imports and failure diagnostics; charge even an unavailable probe.
    ok=probe()
    with (OUT/'usage.jsonl').open('a') as file:
        file.write(json.dumps({'label':'CUDA probe','worker_seconds':time.perf_counter()-PROCESS_START,
                               'passed':ok})+'\n')
    raise SystemExit(0 if ok else 2)
