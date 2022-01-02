import json
import argparse
from collections import OrderedDict
import os.path as osp
from pathlib import Path

def parse_args():

    parser = argparse.ArgumentParser(description='Experiment configurations')
    parser.add_argument('-save', help='<save/path/file.json>', required=True)
    args = parser.parse_args()

    return args.save


def main():
    
    fname = parse_args()

    # Experiment 1 test config
    test1 = {}
    
    # Data set
    ## Macrofeature parameters
    test1['dataset'] = {}
    test1['mode'] = 'binary'
    test1['exp_name'] = 'test1'
    test1['data_dir'] = osp.abspath(osp.join(Path(__file__).parents[1], 'test', 'test_results', 'data'))
    test1['dataset']['i'] = 1
    test1['dataset']['k'] = [1,3,5] 
    test1['dataset']['l'] = 32
    test1['dataset']['m'] = 8
    test1['dataset']['s'] = 0
    test1['dataset']['s_list'] = None
    
    ## Category parameters
    test1['dataset']['d'] = [1,5]
    test1['dataset']['pd_i'] = [1,3,5]
    test1['dataset']['pd'] = [0.0,0.1]

    # Simulations
    test1['sim'] = {}
    
    with open(fname, 'w') as f:
        json.dump(test1, f, indent=3)
        #f.write(json_string)

if __name__ == "__main__":
    main()
