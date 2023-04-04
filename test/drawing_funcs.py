# Testing drawing functions

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
import unittest
import time
import matplotlib.pyplot as plt
from featmf import utilities as utils
from featmf.lfdd import SIFT


# %%
_e = lambda x: os.path.realpath(os.path.expanduser(x))


# %%
# Draw Keypoints function
class DrawKeypoints(unittest.TestCase):
    def setUp(self):
        fn = _e(f"{dir_name}/../samples/data/butterfly.jpg")
        self.img_pil = Image.open(fn)
        self.img_np = np.array(self.img_pil)
        self.sift_algo = SIFT(root_sift=True)
        self.res = self.sift_algo(self.img_np)
    
    def test_draw_keypoints_simple(self):
        start_time = time.time()
        res_img = utils.draw_keypoints(self.img_np, 
                self.res.keypoints, color=(255, 0, 0))
        fn = _e(f"{dir_name}/../samples/data/butterfly_sift.png")
        exp_res_img = np.array(Image.open(fn))
        self.assertTrue(np.allclose(res_img, exp_res_img), 
                "Draw keypoints not giving the right results.")
        end_time = time.time()
        print(f"Draw keypoints test passed in "\
                f"{end_time-start_time:.2f} secs.")


# %%
# Stack images
class StackImages(unittest.TestCase):
    def setUp(self) -> None:
        fn1 = _e(f"{dir_name}/../samples/data/eiffel_tower1.jpg")
        fn2 = _e(f"{dir_name}/../samples/data/eiffel_tower2.jpg")
        self.img1 = Image.open(fn1)
        self.img2 = Image.open(fn2).convert("L")
        self.img1_np = np.array(self.img1)
        self.img2_np = np.array(self.img2)
    
    def test_stack_images_simple(self):
        start_time = time.time()
        res_img = utils.stack_images(self.img1_np, self.img2_np,
                hw_offset=(1, 0.5))
        fn = _e(f"{dir_name}/../samples/data/eiffel_tower_stack.png")
        exp_res_img = np.array(Image.open(fn))
        self.assertTrue(np.allclose(res_img, exp_res_img),
                "Stack images not giving the right results.")
        end_time = time.time()
        print(f"Stack images test passed in "\
                f"{end_time-start_time:.2f} secs.")


# %%
if __name__ == "__main__" and "ipykernel" not in sys.argv[0]:
    unittest.main()
elif __name__ == "__main__" and "ipykernel" in sys.argv[0]:
    print(f"Jupyter notebook instance detected.")
    unittest.main(argv=[''], verbosity=2, exit=False)
else:
    print(f"Module {__name__} must be run as '__main__' instead.")


# %%
# Experiment section
