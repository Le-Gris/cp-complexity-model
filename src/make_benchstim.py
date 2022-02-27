import os
from os import path as osp
import glob
from pathlib import Path
import numpy as np
from scipy.spatial.distance import cdist

def main(**kwargs):
    
    # Paths
    root = osp.abspath(osp.join(Path(__file__).parent, '..'))
    newdir_path = osp.join(root, 'data', 'benchmark_stimuli')
    paths = glob.glob(osp.join(root,'data', 'benchmark_stimuli_', '*'))
    
    # Make new dir to save
    os.mkdir(newdir_path)

    # Iterate through paths
    for p in paths:
        
        cur_dir = osp.split(p)[1]
        os.mkdir(osp.join(newdir_path, cur_dir))
        cur_paths = glob.glob(osp.join(p, '*'))
        
        for c in cur_paths:
            
            params = osp.split(c)[1].split('_')
            cat_code = params[1] + '-' + params[2] + '-' + params[3]
            
            catA = np.load(osp.join(c, 'cat0', 'A.npy'))
            catB = np.load(osp.join(c, 'cat1', 'B.npy'))
            save_cur = osp.join(newdir_path, cur_dir, cat_code)
            os.mkdir(save_cur)
            np.savez_compressed(osp.join(save_cur, cat_code), a=catA, b=catB)
            

if __name__ == '__main__':
    main()
