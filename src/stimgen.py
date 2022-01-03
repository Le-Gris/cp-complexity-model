__author__ = 'Solim LeGris'

# Imports 
import numpy as np
import pandas as pd
from numpy.random import default_rng
import os, pickle
from itertools import product


def macrofeatures(i: int, k: int, l: int, m: int, s: int, path: str, filename: str, code: str,
                  s_list: list = None, save=True) -> object:
    """
     This function randomly generates binary macrofeature sets for a category A and a category B which has macrofeatures that
    are the complement of those of A or another randomly generated binary macrofeature set. The function saves a file containing
    nested dictionaries for the macrofeature sets and a pandas dataframe as a csv containing parameters and computations
    for each

    :param i: int, number of macrofeature sets to generate
    :param k: int, number of relevant macrofeatures
    :param l: int, number of macrofeatures in macrofeature space for the categories
    :param m: int, number of dimensions of the macrofeatures
    :param path: str, where to save the files
    :param filename: str, what to name the files (one .pkl and one .csv)
    :param code: str, simulation code
    :param s: int, s to use (will be used for all i if no list is provided)
    :param s_list: list, list of length i that contains s values to use (optional)
    :param save:
    :return:
    """

    indices = np.arange(2 ** m)
    macrofeature_space = generate_stimspace(m)
    col = ['name', 'k', 'l', 's', 'total_var_M', 'gen_var_M', 'total_var_MA', 'gen_var_MA', 'total_var_MB',
           'gen_var_MB', 'total_var_N',
           'gen_var_N']
    data = []
    macrofeature_dict = {}
    rng = default_rng()

    if s_list is not None:
        assert len(s_list) == i

    for j in range(i):
        if s_list is not None:
            s = s_list[j]
        set_dict = {}
        set_name = 'mf_' + code + '_' + str(j)
        params = []

        # Possible world macrofeature space
        rnd_seq_M = rng.choice(indices, l, replace=False)
        set_dict['M'] = macrofeature_space[rnd_seq_M]

        # Common macrofeatures
        if s > 0:
            rnd_seq_common = rng.choice(rnd_seq_M, s, replace=False)
            rnd_seq_M = np.setdiff1d(rnd_seq_M, rnd_seq_common)

        # Category A macrofeatures
        rnd_seq_MA = rng.choice(rnd_seq_M, k-s, replace=False)
        if s > 0:
            rnd_seq_MA = np.concatenate((rnd_seq_MA, rnd_seq_common))
        set_dict['M_A'] = macrofeature_space[rnd_seq_MA]

        # Remove category A macrofeatures from possible world
        rnd_seq_M = np.setdiff1d(rnd_seq_M, rnd_seq_MA)

        # Category B macrofeatures
        rnd_seq_MB = rng.choice(rnd_seq_M, k-s, replace=False)
        if s > 0:
            rnd_seq_MB = np.concatenate((rnd_seq_MB, rnd_seq_common))
        set_dict['M_B'] = macrofeature_space[rnd_seq_MB]

        # Remove category B macrofeatures from possible world
        rnd_seq_M = np.setdiff1d(rnd_seq_M, rnd_seq_MB)

        # Category noise
        set_dict['N'] = macrofeature_space[rnd_seq_M]

        # Compute and save data
        macrofeature_dict[set_name] = set_dict
        params.append(set_name)
        params.append(k)
        params.append(l)
        params.append(s)

        # This feature relies on ordered dictionaries by key creation (Python 3.6+)
        for key in set_dict:
            if k > 1:
                params.append(total_var(set_dict[key]))
                params.append(gen_var(set_dict[key]))
            else:
                params.append(None)
                params.append(None)
        data.append(params)

    # Save arrays
    if save:
        with open(os.path.join(path, filename + '_' + code + '.pkl'), 'wb') as f:
            pickle.dump(macrofeature_dict, f, pickle.HIGHEST_PROTOCOL)

    # Save data
    df = pd.DataFrame(data, columns=col)
    if save:
        df.to_csv(os.path.join(path, filename + '_' + code + '.csv'))

    return macrofeature_dict, df

def categories(macrofeaturesA: object, macrofeaturesB: object, noise: object, k: int, d: int, pd_i: int, pd: float,
               path: str = None, filename: str = None, N: int = 256, cat_size: int = 2048, m: int = 8,
               p_macro: object = None, dist_noise: float = 0.05, mf_name=None, save=True) -> object:
    """
    This function generates formal binary stimuli for categories A and B using macrofeatures which typically will have more
    than 1 dimension. These stimuli lie on a continuum in terms of the nature of the invariance that characterizes the
    category to which they belong, from local to distributed. Additionally, noise can be controlled through specific parameters
    that the function takes as inputs.

    Input(s):
        macrofeatureA: an array containing macrofeature arrays of 0s and 1s that will be used for category A. Must have shape
                        = (macro_size, m)
        macrofeatureB: an array containing macrofeature arrays of 0s and 1s that will be used for category B. Must have shape
                        = (macro_size, m)
        k: number of covariant features (int)
        d: distribution of covariant features (int). Must be 1<= d <= j where j is the total number of macrofeatures in a stimulus
        pd: probability of noise in a covariant location
        pd_i: distribution parameter at location d_i. The closer to k, the more uniform the distribution of relevant macrofeatures at any given location d_i.
        N: number of dimensions for the stimuli
        cat_size: size of category to be generated
        macro_size: size of the macrofeature set for each category
        m: number of dimensions of the macrofeatures
        p_macro: if None, will be uniform distribution. Else, provide an array of length macro_size with probability distribution
                    for the macrofeatures.
        dist_noise: probability of choosing mf other than the one determined by pd_i distribution (still chosen from relevant mfs)

    Output(s):
        catA, catB: numpy arrays containing the generated categories
    """
    # Verify that macrofeature size and stimulus size match
    j = N / m  # Number of macrofeatures per stimulus
    assert j % 1 == 0
    j = int(j)

    # Verify that distribution is adequate
    assert d < j

    # Verify that invariance distribution is adequate
    assert k >= pd_i >= 1

    # Verify that path and filename were provided
    if save:
        assert path is not None and filename is not None

    # Random number generator
    rng = default_rng()

    # Selection array
    select = [True, False]

    # Covariant location probability (stimulus noise) using select
    p_cov = [pd, 1 - pd]

    # Distribution noise probability using select
    p_dist = [dist_noise, 1 - dist_noise]

    # Select indices for covariant locations with uniform probability
    indices_cat = np.arange(j)
    cov_loc_A = rng.choice(indices_cat, d, replace=False)
    cov_loc_A.sort()
    cov_loc_B = rng.choice(indices_cat, d, replace=False)
    cov_loc_B.sort()

    # Generate covariant mf distribution
    indices_mf = np.arange(k)
    p_dist_A = np.zeros((d, pd_i), dtype=int)
    p_dist_B = np.zeros((d, pd_i), dtype=int)
    for pos in range(d):
        # Should replace be True or False?
        p_dist_A[pos, :] = rng.choice(indices_mf, pd_i, replace=True, p=p_macro)
        p_dist_B[pos, :] = rng.choice(indices_mf, pd_i, replace=True, p=p_macro)

    # Category matrices set to 0
    categoryA = np.zeros((cat_size, j, m))
    categoryB = np.zeros((cat_size, j, m))

    # Index of next covariant location from cov_dist
    next_cov_A = 0
    next_cov_B = 0

    # Outer loop: stimuli
    for h in range(cat_size):
        # Guarantee of at least 1 relevant macrofeature per stimulus
        g_A = False
        g_B = False
        # Inner loop: macrofeatures placement
        for i in range(j):
            # Case 0: current position i corresponds to next relevant location and coin flip gives False
            if i == cov_loc_A[next_cov_A] and not rng.choice(select, 1, p=p_cov)[0]:
                # Case 0.0: select one of the distributed macrofeatures with uniform probability
                if not rng.choice(select, 1, p=p_dist):
                    mf_choice = rng.choice(p_dist_A[next_cov_A], 1)[0]
                    categoryA[h, i, :] = macrofeaturesA[mf_choice]
                    g_A = True
                # Case 0.1: select other relevant macrofeature with uniform probability
                else:
                    categoryA[h, i, :] = rng.choice(macrofeaturesA, 1)[0]
                if next_cov_A < d-1:
                    next_cov_A += 1
            # Case 1: current position i does not correspond to next relevant location or coin flip gives True
            else:
                categoryA[h, i, :] = rng.choice(noise, 1)[0]

            # Same cases as for A (see above)
            if i == cov_loc_B[next_cov_B] and not rng.choice(select, 1, p=p_cov)[0]:
                if not rng.choice(select, 1, p=p_dist):
                    mf_choice = rng.choice(p_dist_B[next_cov_B], 1)[0]
                    categoryB[h, i, :] = macrofeaturesB[mf_choice]
                    g_B = True
                else:
                    categoryB[h, i, :] = rng.choice(macrofeaturesB, 1)[0]
                if next_cov_B < d-1:
                    next_cov_B += 1
            else:
                categoryB[h, i, :] = rng.choice(noise, 1)[0]

        # If series of random events has led to stimulus with no relevant macrofeature at relevant location,
        # force one with random element
        if not g_A:
            # Choose random index in range(d)
            r_i = rng.choice(range(d), 1)[0]
            # Use random index to determine random relevant location
            rnd_l = cov_loc_A[r_i]
            # Use random index to determine random macrofeature set for that location
            rnd_mf = rng.choice(p_dist_A[r_i], 1)[0]
            # Assign macrofeature to random relevant macrofeature
            categoryA[h, rnd_l, :] = macrofeaturesA[rnd_mf]
        if not g_B:
            # Choose random index in range(d)
            r_i = rng.choice(range(d), 1)[0]
            # Use random index to determine random relevant location
            rnd_l = cov_loc_B[r_i]
            # Use random index to determine random macrofeature set for that location
            rnd_mf = rng.choice(p_dist_B[r_i], 1)[0]
            # Assign macrofeature to random relevant macrofeature
            categoryB[h, rnd_l, :] = macrofeaturesB[rnd_mf]

    # Flatten stimuli
    catA = categoryA.reshape((cat_size, N))
    catB = categoryB.reshape((cat_size, N))
    AB = np.append(catA, catB, axis=0)

    info = []
    columns = ['set_name', 'mf_name', 'tvA', 'tvB', 'tvAB', 'gvA', 'gvB',
               'gvAB']

    #Append filename and mf set name
    info.append(filename)
    info.append(mf_name)

    # Compute total variances
    t_v_A = total_var(catA)
    info.append(t_v_A)
    t_v_B = total_var(catB)
    info.append(t_v_B)
    t_v_AB = total_var(AB)
    info.append(t_v_AB)

    # Compute general variances
    g_v_A = gen_var(catA)
    info.append(g_v_A)
    g_v_B = gen_var(catB)
    info.append(g_v_B)
    g_v_AB = gen_var(AB)
    info.append(g_v_AB)

    # Save to directory
    if save:
        np.savez_compressed(os.path.join(path, filename), a=catA, b=catB, info=info)

    return catA, catB, info


def generate_stimspace(n):
    """
    This function generates all possible binary stimuli in a given stimulus space

    Input(s)
        n: dimensions in stimulus space

    Output(s)
        stim_space: an array containing all stimuli in the n-dimensional space
    """
    stim_space = list(map(list, product([0, 1], repeat=n)))
    return np.array(stim_space, dtype='float64')

def total_var(set):
    return np.trace(np.cov(set))


def gen_var(set):
    return np.linalg.det(np.cov(set))
