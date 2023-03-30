# Testing the LFDD algorithms

# %%
import os
import sys
from pathlib import Path
# Set the "./../src" from the script folder
dir_name = None
try:
    dir_name = os.path.dirname(os.path.realpath(__file__))
except NameError:
    print("WARNING: __file__ not found, trying local")
    dir_name = os.path.abspath('')
lib_path = os.path.realpath(f"{Path(dir_name).parent}/src")
# Add to path
if lib_path not in sys.path:
    print(f"Adding library path: {lib_path} to PYTHONPATH")
    sys.path.append(lib_path)
else:
    print(f"Library path {lib_path} already in PYTHONPATH")


# %%
import numpy as np
from PIL import Image
import sys
import unittest
import time
from featmf import lfdd


# %%
# Test SIFT algorithm
class TestSIFT(unittest.TestCase):
    def setUp(self):
        _e = lambda x: os.path.realpath(os.path.expanduser(x))
        fn = _e(f"{dir_name}/../samples/data/butterfly.jpg")
        self.img_pil = Image.open(fn)
        self.img_np = np.array(self.img_pil)
    
    def test_sift_wrapper(self):
        start_time = time.time()
        self.sift_algo = lfdd.SIFT(root_sift=True)
        res = self.sift_algo(self.img_np)
        self.assertEqual(res.keypoints.shape, (1115, 4))
        self.assertEqual(res.descriptors.shape, (1115, 128))
        self.assertEqual(res.scores.shape, (1115,))
        end_time = time.time()
        print(f"SIFT test passed in {end_time-start_time:.2f} secs.")


# %%
if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    unittest.main()
elif __name__ == "__main__" and "ipykernel" in sys.argv[0]:
    print(f"Jupyter notebook instance detected.")
    unittest.main(argv=[''], verbosity=2, exit=False)
else:
    print(f"Module {__name__} must be run as '__main__' instead.")


# %%
# Experimental section

