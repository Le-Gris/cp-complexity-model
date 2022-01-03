import unittest
from pathlib import Path
import os, sys
parentdir = Path(__file__).parents[1]
sys.path.append(parentdir)
import json
import src.gen_dataset as sgd
import src.run_simulations as srs

class CleanTest(unittest.TestCase):
    def setUp(self):
        self.entries = []
        self.entries.append(os.path.join(parentdir, 'test','fixtures', 'test1_config.json'))
   
    # Test 2
    def test_2(self):
        print('\nRunning test 1: Verify that dataset simulator is functional')
        srs.main(config_fname=self.entries[0])
    
    # Test 1 
    def test_1(self):
        print('\nRunning test 1: Verify that simulation runner is functional')
        config, data_dir, exp_name, _ = sgd.get_configuration(self.entries[0])
        path_mf, path_cat, macro_fname = sgd.create_dirstruct(data_dir, exp_name)
        sgd.generate_binary(**config, path_mf=path_mf, macro_fname=macro_fname, path_cat=path_cat)

if __name__ == '__main__':
    unittest.main()
