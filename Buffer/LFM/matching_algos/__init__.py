# All local feature matching algorithms wrapped into one directory
"""
    Contains the following implementations
    - SIFT [1]: Scale Invariant Feature Transform. From OpenCV.
    - 
    
    References:
    [1]: Lowe, David G. "Distinctive image features from 
        scale-invariant keypoints." International journal of computer 
        vision 60.2 (2004): 91-110.
"""

# %%
# Build this first
from LFM.matching_algos import utilities
# Import essential abstract classes
from LFM.matching_algos.utilities import DescMatchingResult, \
        KptDetectionDescriptionAlgo, ImageMatchingAlgo


# %%
# ----------- Feature detection algorithms -----------
from .sift import SIFTWrapper as SIFT

# %%
# ------------ Feature matching algorithms ------------
from .mnn import MNNMatcher as MNN
