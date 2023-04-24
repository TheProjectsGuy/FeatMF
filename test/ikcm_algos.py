# Testing the IKCM algorithms

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
import unittest
import numpy as np
from PIL import Image
from featmf import lfdd, ikcm
from featmf.utilities import draw_matches, get_inliers
import joblib
import time


# %%
# Test NearestNeighbor matcher
class TestNNMatcher(unittest.TestCase):
    def setUp(self):
        _e = lambda x: os.path.realpath(os.path.expanduser(x))
        fn1 = _e(f"{dir_name}/../samples/data/graf/img1.ppm")
        fn2 = _e(f"{dir_name}/../samples/data/graf/img2.ppm")
        fnH = _e(f"{dir_name}/../samples/data/graf/H1to2p")
        img1_pil, img2_pil = Image.open(fn1), Image.open(fn2)
        fn_res = _e(f"{dir_name}/../samples/data/graf/img1_to_2.png")
        self.img_res = np.array(Image.open(fn_res))
        self.img1_np = np.array(img1_pil)
        self.img2_np = np.array(img2_pil)
        self.homo_1to2 = np.loadtxt(fnH)
        self.matcher = ikcm.NN(lfdd.SIFT(root_sift=True), top_n=50)
    
    def test_sift_matcher(self):
        start_time = time.time()
        mres = self.matcher.match_images(self.img1_np, self.img2_np)
        res_c = get_inliers(mres.res[0].keypoints[mres.i1],
                mres.res[1].keypoints[mres.i2], self.homo_1to2, 1,
                ret_type="m")
        colors = [(0, 255, 0) if r else (255, 0, 0) for r in res_c]
        img_res = draw_matches(self.img1_np, self.img2_np, 
                mres.res[0].keypoints, mres.res[1].keypoints,
                np.stack((mres.i1, mres.i2), axis=1), colors, 
                hw_offset=(1.0, 0.0))
        self.assertTrue(np.allclose(img_res, self.img_res))
        end_time = time.time()
        print(f"NN test passed in {end_time-start_time:.2f} secs.")


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
