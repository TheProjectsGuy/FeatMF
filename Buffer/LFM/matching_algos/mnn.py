# Mutual Nearest Neighbor matching algorithm
"""
    Class wraps the nearest neighbor algorithm along with the 
    detection and description algorithm.
"""


# %%
import numpy as np
from LFM.matching_algos import KptDetectionDescriptionAlgo, \
        ImageMatchingAlgo


# %%
# Main class
class MNNMatcher(ImageMatchingAlgo):
    """
        Mutual Nearest-Neighbor descriptor matching algorithm on a
        given local feature detection and description algorithm.
        
        Current steps:
        1. Given two images (img1, img2), extract keypoints and 
            descriptors
        2. Perform matching using faiss and matching result
        3. Return matching result
    """
    def __init__(self, kpt_algo: KptDetectionDescriptionAlgo, 
                 *args, **kwargs) -> None:
        """
            Initialize the MNN matcher with a given local feature
            matching algorithm.
            
            Parameters:
            - kpt_algo:     Keypoint detection and description 
                            algorithm (fully initialized).
            
        """
        super().__init__()
        self.algo = kpt_algo
    pass
