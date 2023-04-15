# Nearest Neighbor (NN) Keypoint Matching algorithms
"""
    Nearest Neighbor (NN) Matcher
    -----------------------------
    
    This algorithm does descriptor matching (descriptors obtained 
    through a :py:class:`KptDetDescAlgo \
        <featmf.templates.KptDetDescAlgo>`) and returns the keypoint
    matches.
    
    .. autoclass:: NNMatcher
        :members:
        :special-members: __init__
"""

# %%
import torch
import numpy as np
from PIL import Image
from typing import Any, Union
# Local imports
from featmf.templates import ImgKptMatchAlgo, KptDetDescAlgo


# %%
# Types
IMG_T1 = Union[np.ndarray, torch.Tensor, Image.Image]   # Images


# %%
class NNMatcher(ImgKptMatchAlgo):
    """
        A Nearest Neighbor matcher over a keypoint detection and
        description (LFDD) algorithm.
        
        Some information about the :py:class:`Result \
            <featmf.templates.ImgKptMatchAlgo.Result>` object is given
        below
        
        -   The LFDD results for both the images are stored in the
            :py:attr:`res \
                <featmf.templates.ImgKptMatchAlgo.Result.res>` object
            as a tuple ``(image1_res, image2_res)``.
        -   The :py:attr:`i1 \
            <featmf.templates.ImgKptMatchAlgo.Result.i1>` and
            :py:attr:`i2 <featmf.templates.ImgKptMatchAlgo.Result.i2>`
            attributes are of shape ``[N,]`` where ``N`` is the number
            of matches. Each element in ``i1`` and ``i2`` correspond
            to the keypoint in the first and second image ``res``
            respectively.
        -   If the ``top_n`` is not None, then the ``i1`` and ``i2``
            attributes are of shape ``[top_n,]`` and the ``res``
            results are the ``top_n`` best keypoints sorted in the
            descending order of their scores.
    """
    Result = ImgKptMatchAlgo.Result  #: :meta private:
    
    # Constructor
    def __init__(self, algo: KptDetDescAlgo, mutual: bool=True, 
                top_n: Union[int, None]=None) -> None:
        """
            :param algo:    A Local Feature Detection and Description 
                            algorithm object to use for extracting 
                            keypoints and descriptors.
            :type algo:     KptDetDescAlgo
            :param mutual:  If True, the condition of mutual matching
                            is enforced. That is, if a keypoint in the
                            first image matches with a keypoint in the
                            second image, then the keypoint in the
                            second image should also match with the
                            same keypoint in the first image.
            :type mutual:   bool
            :param top_n:   If not None, only the ``top_n`` best
                            keypoints in each image are considered for
                            matching and storing in results.
            :type top_n:    int or None
        """
        super().__init__()
        self.lfdd_algo = algo
        self.mutual = mutual
        self.top_n = top_n
    
    # Representation
    def __repr__(self) -> str:
        r = super().__repr__()
        if self.mutual:
            r += f"\n\tMutual Nearest Neighbor matcher"
        else:
            r += f"\n\tNearest Neighbor matcher"
        r += f"\n\tUsing LFDD algorithm: {self.lfdd_algo}"
    
    # Match keypoints
    def match_images(self, img1: IMG_T1, img2: IMG_T1, *args: Any, 
                **kwargs: Any) -> Result:
        """
            Detect keypoints using the LFDD ``algo`` passed in the
            constructor and match them using the Nearest Neighbor
            algorithm for the descriptors of the keypoints on the two
            images.
            
            :param img1:    The first image.
            :type img1: Union[np.ndarray, torch.Tensor, Image.Image]
            :param img2:    The second image.
            :type img2: Union[np.ndarray, torch.Tensor, Image.Image]
            
            .. note::
                Though the input image can be of any type, for the
                detection and description of the keypoints, it must be
                of type described in the LFDD algorithm's
                :py:meth:`detect_and_describe \
                    <featmf.templates.KptDetDescAlgo.detect_and_describe>`
                method. If the LFDD algorithm takes ``np.ndarray``,
                then other types cannot be given.
            
            :raises TypeError:  If the input image is not of the 
                                correct type, or if the images are
                                not of the same type.
            
            
        """
        if type(img1) != type(img2):
            raise TypeError("Both images must be of the same type")
        if type(img1) != np.ndarray and type(img1) != torch.Tensor \
                and not isinstance(img1, Image.Image):
            raise TypeError("Input image of wrong type")
        res1 = self.lfdd_algo.detect_and_describe(img1)
        res2 = self.lfdd_algo.detect_and_describe(img2)
        if self.top_n is not None:  # Top-N descending
            res1 = res1.sort_scores(self.top_n, False)
            res2 = res2.sort_scores(self.top_n, False)
        d1, d2 = res1.descriptors, res2.descriptors
        # TODO: Implement this using faiss or something
        raise NotImplementedError("Not implemented yet")

