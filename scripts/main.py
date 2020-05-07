import os
import sys
import pickle
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.getcwd())))
from bayesmf.scripts.utils.model import run_kfold_xval
from bayesmf.scripts.utils.dataset import make_insilico_dataset, make_downsampled_dataset


DIR_PATH = '/home/sxchao/bayesmf/output'


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--handle', type=str, required=True, help='handle of output files')
    args = parser.parse_args()
    handle = args.handle
    
    #X, _ = make_insilico_dataset()
    X, _ = make_downsampled_dataset()
    
    errs, durs = run_kfold_xval(X, kfold=10, random_state=22690, init=None, components = [10, 15, 20, 25, 30], 
                                methods = ['nmf-vanilla', 'nmf-consensus', 'lda-batch', 'lda-stochastic',
                                           'bmf-batch', 'bmf-stochastic', 'cmf-batch'])
    
    pickle.dump(errs, open(os.path.join(DIR_PATH, 'd_errs_' + handle + '.pkl'), 'wb'))
    pickle.dump(durs, open(os.path.join(DIR_PATH, 'd_durs_' + handle + '.pkl'), 'wb'))
    
    errs, durs = run_kfold_xval(X, kfold=10, random_state=22690, init='nmf', components = [10, 15, 20, 25, 30], 
                                methods = ['nmf-vanilla', 'nmf-consensus', 'lda-batch', 'lda-stochastic',
                                           'bmf-batch', 'bmf-stochastic', 'cmf-batch'])
    
    pickle.dump(errs, open(os.path.join(DIR_PATH, 'd_errs_nmf_' + handle + '.pkl'), 'wb'))
    pickle.dump(durs, open(os.path.join(DIR_PATH, 'd_durs_nmf_' + handle + '.pkl'), 'wb'))