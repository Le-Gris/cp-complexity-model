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
        self.entries.append(os.path.join(parentdir, 'test', 'fixtures', 'test1_config.json'))
        self.entries.append(os.path.join(parentdir, 'test', 'fixtures', 'test2_config.json'))
        self.entries.append(os.path.join(parentdir, 'test', 'fixtures', 'test3_config.json'))
    
    # Test 4
    def test_4(self):
        print('\nRunning test 4: Test conv net on most difficult categories from experiment 1')
        config, data_dir, exp_name, _ = sgd.get_configuration(self.entries[2])
        path_mf, path_cat, macro_fname = sgd.create_dirstruct(data_dir, exp_name)
        sgd.generate_binary_custom(**config, path_mf=path_mf, macro_fname=macro_fname, path_cat=path_cat)
        srs.main(config_fname=self.entries[2])

    # Test 3
    def test_3(self):
        print('\nRunning test 3: Verify that conv net is working/learning')
        srs.main(config_fname=self.entries[1])

    # Test 2
    def test_2(self):
        print('\nRunning test 2: Verify that dataset simulator is functional')
        srs.main(config_fname=self.entries[0])
    
    # Test 1 
    def test_1(self):
        print('\nRunning test 1: Verify that simulation runner is functional')
        config, data_dir, exp_name, _ = sgd.get_configuration(self.entries[0])
        path_mf, path_cat, macro_fname = sgd.create_dirstruct(data_dir, exp_name)
        sgd.generate_binary(**config, path_mf=path_mf, macro_fname=macro_fname, path_cat=path_cat)

if __name__ == '__main__':
    unittest.main()
