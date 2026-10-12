"""Lossless chunked forensic archive. No scientific imports at module load."""
import hashlib
import io
import json
import os
from pathlib import Path
import zipfile

import oracle_config as oc


class TraceIOError(ValueError):
    def __init__(self,code,detail):
        self.code=code
        self.status='unavailable' if code=='unavailable' else 'fail'
        super().__init__(f'{code}: {detail}')


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):value.update(block)
    return value.hexdigest()


def atomic_json(path,value):
    """Replace only a current-run checkpoint; callers own its new directory."""
    path=Path(path);tmp=path.with_name(path.name+'.tmp')
    with tmp.open('xb') as f:
        f.write(json.dumps(value,sort_keys=True,allow_nan=False).encode());f.flush();os.fsync(f.fileno())
    os.replace(tmp,path)


def write_tree(path,value):
    import numpy as np
    arrays={}
    def encode(v):
        if isinstance(v,np.ndarray):
            if v.dtype not in (np.dtype('float64'),np.dtype('float32'),np.dtype('int64'),np.dtype('bool')):
                raise TraceIOError('dtype',str(v.dtype))
            key=f'a{len(arrays)}';arrays[key]=v
            return ['array',key]
        if type(v) is dict:
            if any(type(k) is not str for k in v):raise TraceIOError('key','string keys required')
            return ['dict',[[k,encode(v[k])] for k in sorted(v)]]
        if type(v) in (list,tuple):return ['tuple' if type(v) is tuple else 'list',[encode(x) for x in v]]
        if v is None or type(v) in (str,int,float,bool):return ['scalar',v]
        raise TraceIOError('type',str(type(v)))
    header=json.dumps({'version':1,'tree':encode(value)},sort_keys=True,separators=(',',':'),allow_nan=False).encode()
    arrays['header']=np.frombuffer(header,dtype=np.uint8)
    with Path(path).open('xb') as f:
        np.savez(f,**arrays);f.flush();os.fsync(f.fileno())
    return digest(path)


def read_tree(path,expected_sha256):
    import numpy as np
    path=Path(path)
    if not path.exists():raise TraceIOError('unavailable',str(path))
    if path.stat().st_size>512*2**20:raise TraceIOError('archive_bounds',str(path))
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=expected_sha256:raise TraceIOError('hash',str(path))
    try:
        with zipfile.ZipFile(io.BytesIO(raw)) as z:
            infos=z.infolist()
            if len(infos)>100001 or len({i.filename for i in infos})!=len(infos) or sum(i.file_size for i in infos)>512*2**20:
                raise TraceIOError('archive_bounds',str(path))
        with np.load(io.BytesIO(raw),allow_pickle=False) as z:
            header=z['header']
            if header.dtype!=np.uint8 or header.ndim!=1 or header.size>16*2**20:
                raise TraceIOError('header',str(path))
            record=oc._decode(header.tobytes());used={'header'}
            if set(record)!={'version','tree'} or type(record['version']) is not int or record['version']!=1:
                raise TraceIOError('version',str(path))
            def decode(node,depth=0):
                if depth>64 or type(node) is not list or len(node)!=2:raise TraceIOError('tree','invalid node')
                tag,value=node
                if tag=='array':
                    if type(value) is not str or value in used or value not in z.files:raise TraceIOError('array_ref',str(value))
                    used.add(value);a=z[value]
                    if a.dtype not in (np.dtype('float64'),np.dtype('float32'),np.dtype('int64'),np.dtype('bool')):
                        raise TraceIOError('dtype',str(a.dtype))
                    return a.copy()
                if tag in ('list','tuple') and type(value) is list:
                    values=[decode(n,depth+1) for n in value];return tuple(values) if tag=='tuple' else values
                if tag=='dict' and type(value) is list:
                    d={}
                    for entry in value:
                        if type(entry) is not list or len(entry)!=2 or type(entry[0]) is not str or entry[0] in d:
                            raise TraceIOError('key','duplicate or malformed')
                        d[entry[0]]=decode(entry[1],depth+1)
                    return d
                if tag=='scalar' and (value is None or type(value) in (str,int,float,bool)):return value
                raise TraceIOError('tree','invalid tag or value')
            result=decode(record['tree'])
            if used!=set(z.files):raise TraceIOError('unused_arrays',str(path))
            return result
    except (OSError,ValueError,KeyError,zipfile.BadZipFile) as error:
        if isinstance(error,TraceIOError):raise
        raise TraceIOError('malformed',str(error)) from error


class CaseArchive:
    """32-action lossless chunks; interruption flushes the exact saved prefix."""
    def __init__(self,directory,config,case,initial,*,evidence_kind):
        self.directory=Path(directory);self.directory.mkdir(exist_ok=False)
        self.pending=[];self.transitions=0
        initial_hash=write_tree(self.directory/'initial.npz',initial)
        self.index=dict(case_version=1,case=case,scientific_sha256=oc.scientific_hash(config),
            evidence_kind=evidence_kind,status='running',transitions=0,actions=0,
            initial={'path':'initial.npz','sha256':initial_hash},chunks=[])
        atomic_json(self.directory/'index.json',self.index)

    def append(self,action,next_state):
        self.pending.append({'action':action,'next_state':next_state})
        self.transitions+=int(next_state is not None)
        if len(self.pending)>=32:self.flush()

    def flush(self):
        if not self.pending:return
        name=f'chunk_{len(self.index["chunks"]):05d}.npz'
        checksum=write_tree(self.directory/name,self.pending)
        self.index['chunks'].append({'path':name,'sha256':checksum})
        self.index['actions']+=len(self.pending);self.index['transitions']=self.transitions
        self.pending=[];atomic_json(self.directory/'index.json',self.index)

    def finish(self,status):
        if status not in ('completed','interrupted','nonfinite_stop'):raise TraceIOError('status',status)
        self.flush();self.index['status']=status;atomic_json(self.directory/'index.json',self.index)
        return digest(self.directory/'index.json')


def read_case(directory,index_sha256):
    import numpy as np
    directory=Path(directory);path=directory/'index.json'
    if not path.exists():raise TraceIOError('unavailable',str(path))
    raw=path.read_bytes()
    if hashlib.sha256(raw).hexdigest()!=index_sha256:raise TraceIOError('hash',str(path))
    index=oc._decode(raw)
    if set(index)!={'case_version','case','scientific_sha256','evidence_kind','status','transitions','actions','initial','chunks'} or type(index['case_version']) is not int or index['case_version']!=1:
        raise TraceIOError('index','invalid case header')
    if type(index['transitions']) is not int or not 0<=index['transitions']<=1000 or type(index['actions']) is not int or not 0<=index['actions']<=1001 or type(index['chunks']) is not list:
        raise TraceIOError('index','invalid prefix counts')
    seen=set()
    def load(ref):
        if type(ref) is not dict or set(ref)!={'path','sha256'} or Path(ref['path']).name!=ref['path'] or ref['path'] in seen:
            raise TraceIOError('path','duplicate/traversal reference')
        p=directory/ref['path']
        if p.is_symlink():raise TraceIOError('path','symlink archive')
        seen.add(ref['path']);return read_tree(p,ref['sha256'])
    initial=load(index['initial'])
    if not isinstance(initial,np.ndarray) or initial.dtype!=np.float64 or initial.shape!=(4,):
        raise TraceIOError('state_dtype','initial float64 shape-(4,) required')
    nodes=[initial];controls=[];actions=[]
    for j,ref in enumerate(index['chunks']):
        if ref['path']!=f'chunk_{j:05d}.npz':raise TraceIOError('chunk_order','invalid chunk sequence')
        entries=load(ref)
        if type(entries) is not list or not 1<=len(entries)<=32:raise TraceIOError('chunk_size','invalid')
        for entry in entries:
            if type(entry) is not dict or set(entry)!={'action','next_state'}:raise TraceIOError('entry','invalid')
            action=entry['action'];actions.append(action)
            if entry['next_state'] is not None:
                if not isinstance(entry['next_state'],np.ndarray) or entry['next_state'].dtype!=np.float64 or entry['next_state'].shape!=(4,):
                    raise TraceIOError('state_dtype','next float64 shape-(4,) required')
                nodes.append(entry['next_state']);controls.append([action['input']])
    if index['actions']!=len(actions) or index['transitions']!=len(controls):raise TraceIOError('counts','checkpoint mismatch')
    return dict(case=index['case'],scientific_sha256=index['scientific_sha256'],evidence_kind=index['evidence_kind'],
        status=index['status'],X=np.array(nodes,dtype=np.float64),U=np.array(controls,dtype=np.float64).reshape(-1,1),actions=actions)
