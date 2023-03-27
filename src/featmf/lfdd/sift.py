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
from typing import Any
# Local imports
from featmf.utilities import kpts_cv2np


# %%
EPS = 1e-7      # To prevent ZeroDivisionError
# Types
IMG_T1 = np.ndarray


# %%
class SIFTWrapper(KptDetDescAlgo):
    """
        The algorithm is directly imported from `OpenCV's SIFT \
            implementation <opencv-sift-impl_>`_.
        There is also the option for using RootSIFT descriptors 
        introduced in :ref:`Arandjelovic2012 <arandjelovic2012three>`.
    """
    Result = KptDetDescAlgo.Result  #: :meta private:
    
    # Constructor
    def __init__(self, norm_desc: bool=False, root_sift: bool=False,
                **sift_params) -> None:
        """
            :param norm_desc:   If True, the output descriptors are 
                                normalized
            :type norm_desc:    bool
            :param root_sift:   If True, use RootSIFT descriptors from
                                :ref:`Arandjelovic2012 \
                                    <arandjelovic2012three>`
            :type root_sift:    bool
            :param **sift_params:   Parameters for ``SIFT_create`` 
                                    function in opencv
            
            Since RootSIFT normalizes descriptors, do not set 
            ``norm_desc`` and ``root_sift`` as True together.
        """
        super().__init__()
        self.sift = cv.SIFT_create(**sift_params)
        assert not (norm_desc and root_sift), "Set only one"
        self.norm_desc = norm_desc
        self.root_sift = root_sift
    
    def detect_and_describe(self, img: IMG_T1, *args: Any, 
                **kwargs: Any) -> Result:
        """
            Detect and describe keypoints in an image using SIFT.
            
            In detection results, the keypoints and descriptors are 
            stored as `np.ndarray` objects.
            
            | - keypoints:  ``[N, 4]`` array of keypoints. Each row is
                            ``[x, y, size, orientation]``. The 
                            orientation is in radians, and size is the
                            keypoint neighborhood (diameter).
            | - descriptors:    ``[N, 128]`` array of descriptors.
            | - scores:     ``[N,]`` array of scores.
            
            :param img:     Input image of shape ``[H, W, C]``, where
                            ``C`` is 1 (grayscale) or 3 (RGB). Can 
                            also be a ``[H, W]`` grayscale image.
            :type img:      np.ndarray
            :param *args:   Additional arguments (not used)
            :param **kwargs:    Additional keyword arguments 
                                (not used)
            
            :raises ValueError: If the image shape is invalid.
        """
        if len(img.shape) == 2: # Add channel for grayscale
            img = img[..., None]
        H, W, C = img.shape
        if C == 3:  # Convert to grayscale
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        elif C == 1:
            img = img[..., 0]
        else:
            raise ValueError("Invalid channels")
        # 'img' is now [H, W] grayscale image
        kp_cv, desc_cv = self.sift.detectAndCompute(img, None)
        if len(kp_cv) == 0:
            kpts = scores = descs = None
        else:
            kpts, scores = kpts_cv2np(kp_cv, parse_size=True,
                    parse_angle=True, angle_conv_rad=True, 
                    ret_response=True)
            descs = desc_cv
        res = SIFTWrapper.Result(kpts, descs, scores)
        return res

