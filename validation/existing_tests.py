from common import *
import pytest
if __name__=='__main__':
    configure()
    with Meter('existing test suite once',reservation=300):
        code=pytest.main(['-q','-p','no:cacheprovider','--basetemp',str(OUT/'pytest_tmp'),str(ROOT/'tests')])
        save('existing_tests.json',{'exit_code':int(code),'scope':'all six actual test modules; once',
                                  'excluded':'test_lqr_dd.py is a top-level 100-episode plotting experiment, not unit tests',
                                  'cvxpy_unused_import_workaround':CVXPY_IMPORT_WORKAROUND})
        if code: raise SystemExit(code)
