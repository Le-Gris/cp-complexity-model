import numpy as np
import os
import os.path as osp
from pathlib import Path
import glob


def main(**kwargs):
    paths = glob.glob(osp.join(osp.abspath(Path(__file__).parent), '..', 'data', 'benchmark_stimuli', 'N8', '*'))
    catpaths = []

    for p in paths:
        catpaths.append(glob.glob(osp.join(p, '*'))[0])

    count = {}

    for c in catpaths:
        stim = np.load(c)
        catA = stim['a']
        catB = stim['b']

        if catA.shape != catB.shape:
            if 'diff' in count:
                count['diff'] += 1
            else:
                count['diff'] = 1
        if catA.shape in count:
            count[catA.shape] += 1
        else:
            count[catA.shape] = 1

        if catB.shape in count:
            count[catB.shape] += 1
        else:
            count[catB.shape] = 1
    print(count)

if __name__ == '__main__':
    main()
