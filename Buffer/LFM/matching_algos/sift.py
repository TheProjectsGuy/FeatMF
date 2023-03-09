# SIFT implementation
"""
    Get SIFT keypoints and descriptors from Torch image
    Contains
    - SolverSIFT for SIFT keypoints and descriptors
    - SolverASIFT for ASIFT keypoints and descriptors
"""

# %%
import cv2 as cv
import numpy as np
from LFM.matching_algos import KptDetectionDescriptionAlgo
from LFM.matching_algos.utilities import kpts_cv2np


# %%
EPS = 1e-7      # To prevent ZeroDivisionError

# %%
class SIFTWrapper(KptDetectionDescriptionAlgo):
    """
        Keypoint and descriptor solver for SIFT. Uses OpenCV's SIFT
        under the hood [1]. Currently, it can only process one image
        at a time (single size batch). There is also the option for 
        using RootSIFT [2, 3] descriptors.
        
        In detection results, the keypoints and descriptors are stored
        as `np.ndarray` objects.
        - keypoints: N, 2       Single scale keypoints (x, y)
        - descriptors: N, 128   Standard 128 dim SIFT descriptors
        - scores: None          SIFT implementation has no score
        
        > Note: Currently does not have any image rescaling 
        functionality.
        
        [1]: https://docs.opencv.org/4.x/d7/d60/classcv_1_1SIFT.html
        [2]: Arandjelović, Relja, and Andrew Zisserman. "Three things 
            everyone should know to improve object retrieval." 2012 
            IEEE conference on computer vision and pattern 
            recognition. IEEE, 2012.
        [3]: https://pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
    """
    Result = KptDetectionDescriptionAlgo.Result
    # Constructor
    def __init__(self, norm_desc: bool=False, root_sift: bool=False,
                **sift_params) -> None:
        """
            Parameters:
            - norm_desc:    If True, the descriptors are normalized
            - root_sift:    If True, use RootSIFT descriptors [2, 3]
            - **sift_params:    Parameters for `SIFT_create` function
                                in opencv
            
            > **Note**: Do not set `norm_desc` and `root_sift` as True
                        together (`root_sift` already includes a
                        descriptor normalization step)
        """
        super().__init__()
        self.sift = cv.SIFT_create(**sift_params)
        assert not (norm_desc and root_sift), "Set only one"
        self.norm_desc = norm_desc
        self.root_sift = root_sift
    
    def detect_and_describe(self, img: np.ndarray, *args, **kwargs) \
            -> Result:
        """
            Detect and describe keypoints in an image using SIFT.
            
            Parameters:
            - img:      Image of shape [H, W, 3] (uint8, RGB channels)
            
            Returns:
            - res:      Keypoints [N, 2] and descriptors [N, 128]
        """
        H, W, C = img.shape
        if C == 3:  # Grayscale only
            img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        elif C != 1:
            raise Exception("Image has invalid channels")
        # 'img' is now a H, W, 1 grayscale image
        kp_cv, desc_cv = self.sift.detectAndCompute(img, None)
        if len(kp_cv) == 0:
            kpts = descs = None
        else:
            kpts, descs = kpts_cv2np(kp_cv), desc_cv
            if self.norm_desc:
                descs /= (np.linalg.norm(descs,axis=1,keepdims=True)+\
                            EPS)
            elif self.root_sift:
                # L1 normalize descriptors
                descs /= (np.sum(descs, axis=1, keepdims=True) + EPS)
                # Take square root
                descs = np.sqrt(descs)
                # The descriptors should already be L2-normalized!
        res = SIFTWrapper.Result(kpts, descs)
        return res
