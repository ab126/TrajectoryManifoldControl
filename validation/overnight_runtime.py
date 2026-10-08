"""Current host context checks for the separately budgeted overnight queue."""
from common import *
import argparse
import subprocess
import hashlib
from models import check_oracle

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('destination');args=parser.parse_args()
    configure()
    if not torch.cuda.is_available():raise RuntimeError('Overnight queue requires host CUDA; no CPU fallback')
    props=torch.cuda.get_device_properties(0)
    if props.name!='NVIDIA L4':raise RuntimeError(f'Device differs from measured L4: {props.name}')
    if torch.__version__!='2.5.1+cu121':raise RuntimeError('CUDA environment differs from measured version')
    free,total=torch.cuda.mem_get_info()
    if free<2*2**30:raise RuntimeError('Less than 2 GiB free; stop rather than contend for GPU memory')
    query=subprocess.check_output(['nvidia-smi','--id='+os.environ.get('CUDA_VISIBLE_DEVICES','0').split(',')[0],
        '--query-gpu=uuid,driver_version,memory.used,memory.free','--format=csv,noheader'],text=True).strip()
    uuid,driver,*memory=[part.strip() for part in query.split(',')]
    expected=json.loads((OUT/'gpu_probe.json').read_text())['nvidia_smi']['stdout']
    if uuid not in expected or driver!='610.43.02':raise RuntimeError('GPU UUID/driver differs from measured readiness')
    native=json.loads((OUT/'gpu_native_fingerprints.json').read_text())
    verified=[]
    for name in ('.venv-gpu/lib/libstdc++.so.6','.venv-gpu/lib/python3.11/site-packages/torch/lib/libtorch_cuda.so',
                 '.venv-gpu/lib/python3.11/site-packages/torch/lib/libtorch_cpu.so'):
        value=hashlib.sha256()
        with (ROOT/'validation'/name).open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):value.update(block)
        if value.hexdigest()!=native[name]:raise RuntimeError('Native environment differs: '+name)
        verified.append(name)
    a=torch.arange(16,dtype=torch.float32,device='cuda').reshape(4,4)
    first=a@a.T;sync('cuda');second=a@a.T;sync('cuda')
    torch.testing.assert_close(first,second,atol=0,rtol=0)
    result={'passed':True,'device':props.name,'free_bytes':free,'total_bytes':total,
            'GPU_UUID':uuid,'driver':driver,'NVML_memory_fields':memory,'native_files_verified':verified,
            'oracle_RK4_error':check_oracle('cuda'),'determinism':determinism_manifest()}
    Path(args.destination).write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2))
