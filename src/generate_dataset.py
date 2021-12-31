import stimgen as sg
import pickle
import os
import json
import argparse

def parse_args():

    parser = argparse.ArgumentParser(description='Generate the dataset for simulations')
    parser.add_argument('-c', help='<config.json>', required=True)
    #parser.add_argument('-d', help='<clean_dialog.csv file>', required=True)
    args = parser.parse_args()

    return args.config


def get_configuration(fname):
    
    with open(fname, 'r') as f:
        config = json.load(f)
    
    return config['dataset']







def main():
    
    config_fname = parse_args()
    config = get_configuration(config_fname)




if __name__ == '__main__':
    main()
