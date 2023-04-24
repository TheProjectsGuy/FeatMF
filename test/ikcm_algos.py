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
from featmf import lfdd
from featmf import ikcm
import joblib


# %%
_e = lambda x: os.path.realpath(os.path.expanduser(x))
fn1 = _e(f"{dir_name}/../samples/data/graf/img1.ppm")
fn2 = _e(f"{dir_name}/../samples/data/graf/img2.ppm")
img1, img2 = Image.open(fn1), Image.open(fn2)
img1_np, img2_np = np.array(img1), np.array(img2)

# %%
sift = lfdd.SIFT(root_sift=True)
matcher = ikcm.NN(sift)

# %%
res = matcher.match_images(img1_np, img2_np)

# %%
res_sort = res.copy().sort_scores(100, ascending=False)

# %%

