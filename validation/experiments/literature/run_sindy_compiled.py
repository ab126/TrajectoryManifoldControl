"""Same SINDYc fits/search, using a checked serial compiled evaluation backend."""
from pathlib import Path
import numpy as np
import run_classical
import sindy
from sindy_kernel import rollout
from common import digest


def predict_inputs(self,x,U):
    x=np.asarray(x,dtype=float);U=np.asarray(U,dtype=float)
    original_shape=x.shape[:-1]
    prediction=rollout(np.ascontiguousarray(x.reshape(-1,4)),np.ascontiguousarray(U.reshape(-1,U.shape[-2],1)),
                       self.library.powers_,self.coef,self.trig)
    return prediction.reshape(*original_shape,U.shape[-2]+1,4)


original_identity=run_classical.code_identity
def identity():
    value=original_identity()
    for name in ['sindy_kernel.py','run_sindy_compiled.py']:value[name]=digest(Path(__file__).with_name(name))
    return value


if __name__=='__main__':
    sindy.SINDYc.predict_inputs=predict_inputs
    run_classical.code_identity=identity
    run_classical.main()
