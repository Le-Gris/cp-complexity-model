import os
import numpy as np
import argparse
import glob
import pandas as pd

def parse_args():
    
    parser = argparse.ArgumentParser(description='Collect and save data as csv for analysis')
    parser.add_argument('-i', help='path/to/simulations', required=True)
    parser.add_argument('-o', help='path/to/save/filename', required=True)
    args = parser.parse_args()
    return args.i, args.o

def scrape_data(sim_dir, save_path):

    # List all directories containing simulations
    # Each dir contains the following: CP, plots and sim_run.npz
    sim_list = glob.glob((os.path.join(sim_dir, '*')))
    sim_list.sort(key=lambda p : int(os.path.split(p)[1].split('_')[1]))

    # Data to be gathered
    columns = ['sim_run', 'cat_code', 'k', 'd', 'pd', 'pdi', 'init_loss_AE', 'min_loss_AE', 'avg_loss_diff_AE', 'init_loss_cl',
               'min_loss_cl', 'max_acc', 'avg_loss_diff_cl', 'w_A_init', 'w_B_init', 'bt_init', 'w_A_bef', 'w_B_bef', 'bt_bef', 'w_A_af',
               'w_B_af', 'bt_af', 'global_CP', 'avg_dist_w_A', 'avg_dist_w_B', 'avg_dist_w', 'avg_dist_bt']

    data = []

    # Iterate through sim_dir
    for i, sim in enumerate(sim_list):

        # List to append data to
        row = []

        # Get sim filename save sim run number
        sim_name = os.path.split(sim)[1]
        row.append(int(sim_name.split('_')[1]))
        sim_filename = sim_name + '.npz'

        # Get cp path and load sim data
        cp_path = os.path.join(sim, 'cp')
        sim_data = np.load(os.path.join(sim, sim_filename))

        # Save cat_code
        cat_code = sim_data['code'][0]
        row.append(cat_code)
        cat_info = cat_code.split('-')

        # Append k
        row.append(int(cat_info[0]))

        # Append d
        row.append(int(cat_info[1]))

        # Append pd
        row.append(int(cat_info[2]))

        # Append pdi
        pdi_str = cat_info[3][0] + '.' + cat_info[3][1]
        row.append(float(pdi_str))

        # Load auto-encoder data
        ae = sim_data['ae']

        # Append initial ae test loss
        row.append(ae[1][0])

        # Append ae min test loss
        row.append(np.min(ae[1]))

        # Append ae avg test loss diff
        diff_ae = np.diff(ae[1])
        r_ae = np.abs(np.mean(diff_ae))
        row.append(r_ae)

        # Load classifier data
        cl = sim_data['classifier']

        # Append classifier initial test loss
        row.append(cl[2][0])

        # Append classifier min test loss
        row.append(np.min(cl[2]))

        # Append classifier max test accuracy
        row.append(np.max(cl[3]))

        # Append classifier avg test loss diff
        diff_cl = np.abs(np.diff(cl[2]))
        r_cl = np.mean(diff_cl)
        row.append(r_cl)

        # Load CP data
        initial = np.load(os.path.join(cp_path, 'cp_initial.npz'))
        before = np.load(os.path.join(cp_path, 'cp_before.npz'))
        after = np.load(os.path.join(cp_path, 'cp_after.npz'))

        # Append initial distances
        row.append(initial['between'])
        row.append(initial['withinA'])
        row.append(initial['withinB'])

        # Append before distances and recompute avg within
        row.append(before['between'])
        row.append(before['withinA'])
        row.append(before['withinB'])

        # Append after distances
        row.append(after['between'])
        row.append(after['withinA'])
        row.append(after['withinB'])

        # Append global CP and other CP measures
        sep = after['between'] - before['between']
        comp_af = after['withinA'] + after['withinB']
        comp_bef = before['withinA'] + before['withinB']
        avg_comp = (comp_af - comp_bef)/2

        global_cp = sep - avg_comp

        row.append(global_cp)
        row.append(after['withinA'] - before['withinA'])
        row.append(after['withinB'] - before['withinB'])
        row.append(avg_comp)
        row.append(sep)

        # Append sim list to data matrix
        data.append(row)

    df = pd.DataFrame(data=data, columns=columns)
    df.to_csv(save_path)

    return df

def main(**kwargs):

    sim_path, output_path = parse_args()

    data = scrape_data(sim_path, output_path)

    return data
    
if __name__ == '__main__':
    main()
