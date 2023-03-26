# SIFT - Scale Invariant Feature Transform
""" 
    SIFT - Scale Invariant Feature Transform
    -----------------------------------------
    
    Introduced in :ref:`Lowe2004 <lowe2004distinctive>`, SIFT is a 
    popular local feature detection and description algorithm.
    
    The following wrappers are included
    
    .. autoclass:: SIFTWrapper
        :members:
        :special-members: __init__
        
    
    .. _opencv-sift-impl: https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
"""

# %%
import cv2 as cv
import numpy as np
from featmf import KptDetDescAlgo

# %%
EPS = 1e-7      # To prevent ZeroDivisionError

# %%
class SIFTWrapper(KptDetDescAlgo):
    """
        The algorithm is directly imported from `OpenCV's SIFT \
            implementation <opencv-sift-impl_>`_.
        There is also the option for using RootSIFT descriptors 
        introduced in :ref:`Arandjelovic2012 <arandjelovic2012three>`.
        
        In detection results, the keypoints and descriptors are stored
        as `np.ndarray` objects.
        
        - keypoints: N, 2       Single scale keypoints (x, y)
        - descriptors: N, 128   Standard 128 dim SIFT descriptors
        - scores: None          SIFT implementation has no score
    """
    Result = KptDetDescAlgo.Result  #: :meta private:
    # Constructor
    def __init__(self, norm_desc: bool=False, root_sift: bool=False,
                **sift_params) -> None:
        """
            
        """
    
    pass

