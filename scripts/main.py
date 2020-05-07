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
    
    X1, _ = make_insilico_dataset()
    X2, _ = make_downsampled_dataset()
    
    for X, label in zip([X1, X2], ['l_', 'l_d_']):
        errs, durs = run_kfold_xval(X, kfold=10, random_state=22690, init=None, components = [10, 15, 20], 
                                    methods = ['lda-batch', 'lda-stochastic'])

        pickle.dump(errs, open(os.path.join(DIR_PATH, label + 'errs_' + handle + '.pkl'), 'wb'))
        pickle.dump(durs, open(os.path.join(DIR_PATH, label + 'durs_' + handle + '.pkl'), 'wb'))

        errs, durs = run_kfold_xval(X, kfold=10, random_state=22690, init='nmf', components = [10, 15, 20], 
                                    methods = ['lda-batch', 'lda-stochastic'])

        pickle.dump(errs, open(os.path.join(DIR_PATH, label + 'errs_nmf_' + handle + '.pkl'), 'wb'))
        pickle.dump(durs, open(os.path.join(DIR_PATH, label + 'durs_nmf_' + handle + '.pkl'), 'wb'))
    