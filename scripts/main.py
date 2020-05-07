import sys
sys.path.append('/home/sxchao')
from bayesmf.scripts.utils import run_kfold_xval


if __name__ == "__main__":
    # TO DO: define X
    
    errs, durs = run_kfold_xval(X)
    