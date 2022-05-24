import numpy as np
import matplotlib.pyplot as plt
import random
import matplotlib.colors as colors
import matplotlib.cm as cm
import os
import os.path as osp
from os.path import join
import argparse

def gabor(sigma, theta, Lambda, psi, gamma):
    """Gabor feature extraction."""
    # sigma : standard deviation of gaussian envelope in pixel
    # theta : rotation of gabor
    # Lamdda : wavelength
    # psi : phase
    # gamma aspect ratio

    sigma_x = sigma
    sigma_y = float(sigma) / gamma

    # Grid size
    #nstds = 2.5  # Number of standard deviation sigma
    #xmax = max(abs(nstds * sigma_x * np.cos(theta)), abs(nstds * sigma_y * np.sin(theta)))
    #xmax = np.ceil(max(1, xmax))
    #ymax = max(abs(nstds * sigma_x * np.sin(theta)), abs(nstds * sigma_y * np.cos(theta)))
    #ymax = np.ceil(max(1, ymax))

    xmax=128
    ymax=128

    xmin = -xmax
    ymin = -ymax

    # make the grid square
    Xmin = min(xmin,ymin)
    Xmax = max(xmax,ymax)
    Ymin=Xmin
    Ymax=Xmax

#    (y, x) = np.meshgrid(np.arange(min, max + 1), np.arange(min, max + 1))
    (y, x) = np.meshgrid(np.arange(Ymin, Ymax + 1), np.arange(Xmin, Xmax + 1))

    # Rotation
    x_theta = x * np.cos(theta) + y * np.sin(theta)
    y_theta = -x * np.sin(theta) + y * np.cos(theta)

    # mutliply rotated gaussian by cosine wave
    gb = np.exp(-.5 * (x_theta ** 2 / sigma_x ** 2 + y_theta ** 2 / sigma_y ** 2)) * np.cos(2 * np.pi / Lambda * x_theta + psi)
    return gb

if __name__ == '__main__':
    
    # Get data dir 
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', help='path/to/data/dir', required=True)
    parser.add_argument('-l', help='<lambda1><lambda2>', required=True)
    args = parser.parse_args()
    datadir = args.o
    vals = args.l

    # Added sigma=30, lambda=5, psi=np.pi/2
    sigmas=[5, 10, 15, 20, 25, 30]
    theta=[0, 15, 30, 45, 60, 75, 90]
    Lambdas=[5, 10, 15, 20, 25, 30]
    psi=[0, np.pi/32, np.pi/16, np.pi/8, np.pi/4, np.pi/2]
    gamma=[0.2, 0.4, 0.6, 0.8, 1]

    params = [sigmas,theta,Lambdas,psi,gamma]

    count=0
    relev=2
    vals= [int(vals[0]), int(vals[1])]
    cats=['A','B']
    
    # Create set directory
    #script_path = osp.dirname(osp.realpath(__file__)) 
    gabor_dir = join(datadir, 'gabor')
    if not osp.exists(gabor_dir):
        os.mkdir(gabor_dir)
    set_dirname = 'set_lam_' + str(params[relev][vals[0]]) + '_'+ str(params[relev][vals[1]])
    if not osp.exists(join(gabor_dir, set_dirname)):
        os.mkdir(join(gabor_dir, set_dirname))

    # Generate stimuli
    for c in range(len(cats)):
        
        val=params[relev][vals[c]]

        # Create cat directory for numpy and for images
        cat_path = join(gabor_dir, set_dirname, str(cats[c]))
        if not osp.exists(cat_path):
            os.mkdir(cat_path)
        
        im_path = join(gabor_dir, set_dirname, str(cats[c] + 'images'))
        if not osp.exists(im_path):
            os.mkdir(im_path)
        
        abstr_path = join(gabor_dir, set_dirname, str(cats[c] + 'abstr'))
        if not osp.exists(abstr_path):
            os.mkdir(abstr_path)

        for s in range(len(sigmas)):
            for t in range(len(theta)):
                for p in range(len(psi)):
                    for g in range(len(gamma)):

                        gb = gabor(sigma=sigmas[s], theta=theta[t], Lambda=val, psi=psi[p], gamma=gamma[g])
                        
                        #gb = (gb-np.min(gb))/(np.max(gb)-np.min(gb)) * 2 - 1  #scale to exactly [-1, 1] for comparing when visualising
                        
                        abstract_rep = np.array([sigmas[s], theta[t], val, psi[p], gamma[g]])

                        # Filename
                        filename = 'gab_' + 'sig' + str(sigmas[s]) + '_' + 'theta' + str(theta[t]) + '_' + 'lam' + str(val) + '_' + 'psi' + str(psi[p]) + '_' + 'gam' + str(gamma[g]) 
                        # Save numpy 
                        np.save(join(cat_path, filename), gb)
                        
                        # Save abstract numpy
                        np.save(join(abstr_path, filename), abstract_rep)

                        # Save image
                        filename += '.jpeg'
                        plt.imsave(join(im_path, filename), gb, cmap='gray', vmin=-0.9, vmax=1)

                        #axs[count].imshow(gb, cmap=plt.get_cmap('gray'))
                        #axs[count].axis('off')
                        #count += 1
                        #plt.show()
