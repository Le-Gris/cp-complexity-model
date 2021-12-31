from src.stimgen import macrofeatures, categories 
import pickle
import os
import json
import argparse

# Paths
path_mf = os.path.join(os.path.dirname(__file__), '..', 'data', 'macrofeatures')
macro_filenames = 'mf'
path_cat = os.path.join(os.path.dirname(__file__), '..', 'data', 'categories')

def parse_args():

    parser = argparse.ArgumentParser(description='Generate the dataset for simulations')
    parser.add_argument('-c', help='<config.json>', required=True)
    args = parser.parse_args()
    return args.config

def get_configuration(fname):
    
    with open(fname, 'r') as f:
        config = json.load(f)
    return config['dataset']

def generate_binary(i, k, l, m, s, s_list, d, pd_i, pd, mode, path_mf=path_mf, macro_filenames=macro_filenames, path_cat=path_cat):
    
    # Generate macrofeatures
    for k_i in k:
        macro_dir = os.path.join(path_mf, str(k_i))
        if not os.path.exists(macro_dir):
            os.mkdir(macro_dir)
        
        macrofeatures(i=i, k=k_i, l=l, m=m, s=s, path=macro_dir, filename=macro_filenames, code=str(k_i), s_list=s_list)
    
    # Generate categories
    for k_i in k:
        curfile = os.path.join(path_mf, str(k_i), 'mf_' + str(k_i) + '.pkl')
        with open(curfile, 'rb') as f:
            mf_sets = pickle.load(f)
            for set_num, set_name in enumerate(mf_sets):
                cur_set = mf_sets[set_name]
                for cur_d in d:
                    for cur_pdi in pd_i:
                        if cur_pdi <= k_i:
                            for cur_pd in pd:
                                cur_dir = makedir_cat(parent_dir=path_cat, k=k_i, d=cur_d, pdi=cur_pdi, pd=cur_pd)
                                categories(cur_set['M_A'], cur_set['M_B'], cur_set['N'], k=k_i, d=cur_d, pd_i=cur_pdi, 
                                           pd=cur_pd, path=cur_dir, filename='set_'+str(set_num), mf_name=set_name)

def generate_continuous():
    pass

def makedir_cat(parent_dir, k, d, pdi, pd):
    dirname = ''
    if k < 10:
        dirname += '0' + str(k)
    else:
        dirname += str(k)
    dirname += '-'
    if d < 10:
        dirname += '0' + str(d)
    else:
        dirname += str(d)
    dirname += '-'
    if pdi < 10:
        dirname += '0' + str(pdi)
    else:
        dirname += str(pdi)
    dirname += '-' + str(pd)[0] + str(pd)[2]
    path = os.path.join(parent_dir, dirname)
    try:
        os.mkdir(path)
    except:
        pass

    return path



def main():
    
    config_fname = parse_args()
    config = get_configuration(config_fname)
    mode = config['mode']
    
    if mode == 'binary':
        generate_binary(**config)
    else:
        generate_continuous(**config)


if __name__ == '__main__':
    main()
