import numpy as np
import os
import argparse
from scipy.spatial.distance import cdist

def compute_sdm(catA, catB):
    # Similarities
    wA = cdist(catA, catA, metric='cosine')
    wB = cdist(catB, catB, metric='cosine')
    b = cdist(catA, catB, metric='cosine')
    
    # Make sure diag is zero and verify why behaviour is not as intended
    np.fill_diagonal(wA, 0)
    np.fill_diagonal(wB, 0)
    
    avg_w = (wA.sum()/(wA.size-len(wA)) + wB.sum()/(wB.size-len(wB)))/2
    avg_b = np.mean(b)

    sdm = avg_w/avg_b

    return avg_w, avg_b, sdm

def main(**kwargs):
    
    #TODO: standardize

    # Save dir
    path = os.path.join('.', '..', 'data','continuous', 'exp4')
    # Distribution properties
    params = [[[0,0.25],[0,-0.25],[[3,0],[0,0.01]]], [[0,0.25],[0,-0.25],[[2,0],[0,0.01]]], [[0,0.25],[0,-0.25],[[1.5,0],[0,0.01]]], [[0,0.25],[0,-0.25],[[1,0],[0,0.01]]],
              [[0,0.25],[0,-0.25],[[0.5,0],[0,0.01]]], [[0,0.25],[0,-0.25],[[0.25,0],[0,0.01]]], [[0,0.25],[0,-0.25],[[0.10,0],[0,0.01]]], [[0,0.25],[0,-0.25],[[0.01,0],[0,0.01]]],
              [[0,0.5],[0,-0.5],[[3,0],[0,0.01]]], [[0,0.75],[0,-0.75],[[3,0],[0,0.01]]], [[0,1],[0,-1],[[3,0],[0,0.01]]], [[0,1.5],[0,-1.5],[[3,0],[0,0.01]]], [[0,2],[0,-2],[[3,0],[0,0.01]]]]
    #mean = [[[0.25,0.25],[-0.25,-0.25]], [[0.5,0.5],[-0.5,-0.5]], [[1,1],[-1,-1]], [[2,2],[-2,-2]], [[4,4],[-4,-4]]]
    #cov = [[[0.10, 0],[0,0.10]], [[0.15,0],[0,0.15]], [[0.25,0],[0,0.25]], [[0.5,0],[0,0.25]], [[0.75,0],[0,0.25]], 
    #       [[1,0],[0,0.25]], [[1.25,0],[0,0.25]], [[1.5,0],[0,0.25]], [[2,0],[0,0.25]], [[3,0],[0,0.25]], [[4,0],[0,0.25]]]
    
    sample_size = 10000
    
    rng = np.random.default_rng()
    setnum = 0
    for param in params:
        catA = rng.multivariate_normal(param[0], param[2], size=sample_size) 
        catB = rng.multivariate_normal(param[1], param[2], size=sample_size)
        
        # Compute info
        w, b, sdm = compute_sdm(catA, catB)
        meanA = param[0][1]
        meanB = param[1][1]
        var_x = param[2][0][0]
        var_y = param[2][1][1]
        info = [meanA, meanB, var_x, var_y, w, b, sdm]
        
        # Save
        cat_path = os.path.join(path, 'categories', 'set_'+str(setnum))
        os.mkdir(cat_path)
        np.savez_compressed(os.path.join(cat_path, 'dataset_'+str(setnum)), a=catA, b=catB, info=info)
        setnum += 1
if __name__ == '__main__':
    main()
