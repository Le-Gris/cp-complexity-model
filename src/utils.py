import json
import os 
from os import path as osp
import argparse
import glob
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist
from scipy.stats import ks_2samp
import random
import gc

def parse_args():
    # Get utility to run
    parser = argparse.ArgumentParser(description='Utils')
    parser.add_argument('-u', help='Utility to run', required=True)
    args = parser.parse_args()
    return args.u

def categoricality(catA, catB):

    # Compute cosine similarity of neural representations
    withinA, withinB, between = compute_pairwise_dist(catA, catB, metric='cosine')
    
    # Flatten
    withinA = withinA.flatten()
    withinB = withinB.flatten()
    between = between.flatten()
    
    # Concatenate within distances
    within = np.concatenate((withinA, withinB))

    # Compute categoricality
    ksstat, _ = ks_2samp(between, within)

    return ksstat

def compute_pairwise_dist(distA, distB, metric):

    wA = cdist(distA, distA, metric=metric)
    wB = cdist(distB, distB, metric=metric)
    b = cdist(distA, distB, metric=metric)
   
    return wA, wB, b


def compute_sdm(catA, catB):
    
    # Similarities
    wA, wB, b = compute_pairwise_dist(catA, catB, metric='cosine')

    # Make sure diag is zero and verify why behaviour is not as intended
    np.fill_diagonal(wA, 0)
    np.fill_diagonal(wB, 0)

    avg_w = (np.sum(wA)/(wA.size-wA.shape[0]) + np.sum(wB)/(wB.size-wB.shape[0]))/2
    avg_b = np.mean(b)
    sdm = avg_w/avg_b
    
    return avg_w, avg_b, sdm


def random_to_struct(path_cat, save_path):
    
    # Get list category directories
    paths = glob.glob(osp.join(path_cat, '*'))
    
    # Random generator
    rng = np.random.default_rng()
    
    # Create dataframe to save results
    cols = ['catcode', 'k', 'p', 'ks_before', 'ks_after', 'ks_cp', 'sdm']
    data = []

    # Load each category, get shape, generate random, compute KS, store
    for i, p in enumerate(paths):
        
        stim = np.load(osp.join(p, osp.split(p)[1] + '.npz'))
        catA = stim['a']
        catB = stim['b']

        # This step should be commented out and replaced by SDM computation if it does not exist in info.
        sdm = stim['info'][2]
        
        randA = rng.uniform(size=catA.shape)
        randB = rng.uniform(size=catB.shape)
        
        ks_before = categoricality(randA, randB)
        ks_after = categoricality(catA, catB)
        ks_cp = ks_after - ks_before

        split_path = osp.split(p)
        catcode = split_path[1].split('-')
        k = int(catcode[1][1])
        p_ = int(catcode[2][1])
        catcode = catcode[0] + catcode[1] + catcode[2]
        data.append([catcode, k, p_, ks_before, ks_after, ks_cp, sdm])
        
        print(f'Completed category {p}')

    # Save dataframe
    df = pd.DataFrame(data=data, columns=cols)
    df.to_csv(save_path)


def bench_rename():
    
    path = osp.abspath(Path(__file__).parents[1])
    bench_paths = glob.glob(osp.join(path, 'data', 'benchmark_stimuli', '*'))
    
    for b in bench_paths:
        dim_paths = glob.glob(osp.join(b, '*'))

        for d in dim_paths:
            split = osp.split(d)
            newname = osp.join(d, split[1]+'.npz')
            oldname = osp.join(d, 'dataset.npz')
            os.rename(oldname, newname)
 
def add_sdm():

    root = input('Absolute path to category directories: ')
    print(root)
    bench_paths = glob.glob(osp.join(root, '*'))

    print(f'{len(bench_paths)} datasets to compute')
        
    for i, d in enumerate(bench_paths):

        cat_path = osp.join(d, osp.split(d)[1]+'.npz')
        stim = np.load(cat_path)
        catA = stim['a']
        catB = stim['b']

        print(f'Current dataset: catA={catA.shape}, catB={catB.shape}')
        
        if catA.shape[0] > 10000 or catB.shape[0] > 10000:
            print(f'This dataset exceeds the allowed limit: {osp.split(d)[1]}')
            continue

        w, b, sdm = compute_sdm(catA, catB)
        info = [w,b,sdm]

        os.remove(cat_path)
        np.savez_compressed(cat_path, a=catA, b=catB, info=info)

        print(f'{len(bench_paths)-(i+1)} datasets to compute')


def main(**kwargs):
    
    utility = parse_args()
    
    if utility == 'bench_rename':
        bench_rename()
    elif utility == 'add_sdm':
        add_sdm()
    elif utility == 'rand_to_struct':
        path_cat = input('Category directory path: ')
        save_path = input('Save path: ')
        random_to_struct(path_cat, save_path)

if __name__ == '__main__':
    main()
