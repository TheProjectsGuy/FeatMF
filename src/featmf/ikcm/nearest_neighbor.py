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
import einops as ein
import faiss
from PIL import Image
from typing import Any, Union, Literal
# Local imports
from featmf.templates import ImgKptMatchAlgo, KptDetDescAlgo


# %%
# Types
IMG_T1 = Union[np.ndarray, torch.Tensor, Image.Image]   # Images
MATCH_T = Literal["mutual", "union"]    # Match types
EPS = 1e-7  # Prevent ZeroDivisionError


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
            as a tuple ``(image1_res, image2_res)``. These are direct
            results. If ``top_k`` is set, then the ``res`` object will
            contain ``top_k`` keypoints (sorted). This is only for
            algorithms that give keypoint scores.
        -   The :py:attr:`i1 \
            <featmf.templates.ImgKptMatchAlgo.Result.i1>` and
            :py:attr:`i2 <featmf.templates.ImgKptMatchAlgo.Result.i2>`
            attributes are of shape ``[N,]`` where ``N`` is the number
            of matches. Each element in ``i1`` and ``i2`` correspond
            to the keypoint in the first and second image ``res``
            respectively.
    """
    Result = ImgKptMatchAlgo.Result  #: :meta private:
    
    # Constructor
    def __init__(self, algo: KptDetDescAlgo, norm_descs: bool=False,
                match_algo: MATCH_T="mutual", 
                top_k: Union[int, None]=None, top_r: int=1,
                top_n: Union[int, None]=None) -> None:
        """
            :param algo:    A Local Feature Detection and Description 
                            algorithm object to use for extracting 
                            keypoints and descriptors.
            :type algo:     KptDetDescAlgo
            :param norm_descs:  Normalize the descriptors extracted
                                from ``algo`` if True. This does not
                                change the ``res`` object of result,
                                but is for before matching stage.
            :type norm_descs:   bool
            :param match_algo: Matching algorithm to use. Must be in
            
                -   ``"mutual"``: Mutual Nearest Neighbor matching.
                    Matches from image 1 to 2, and 2 to 1 are taken
                    and the intersection of the two is used.
                -   ``"union"``: Merge the matches from image 1 to 2
                    and 2 to 1 (union of the two).
            :type match_algo:   str
            :param top_k:   The number of top keypoints to use before
                            matching. If None, all keypoints are used.
            :type top_k:    int or None
            :param top_r:   The number of top correspondences to get
                            for each keypoint in the images. Basically
                            retrievals per keypoint when matching/
                            searching.
            :type top_r:    int
            :param top_n:   The number of top matches to return. If
                            None, all matches are returned.
            :type top_n:    int or None
        """
        super().__init__()
        self.lfdd_algo = algo
        self.norm_descs = norm_descs
        self.malgo = match_algo
        self.top_k = top_k
        self.top_n = top_n
        self.top_r = top_r
    
    # Representation
    def __repr__(self) -> str:
        r = super().__repr__()
        if self.malgo == "mutual":
            r += f"\n\tMutual Nearest Neighbor matcher"
        else:
            r += f"\n\tNearest Neighbor matcher (union)"
        r += f"\n\tUsing LFDD algorithm: {self.lfdd_algo}"
        if self.norm_descs:
            r += f"\n\tDescriptors are normalized before matching"
        if self.top_k is not None:
            r += f"\n\tUsing top {self.top_k} keypoints"
        r += f"\n\tUsing {self.top_r} retrievals per keypoint"
        if self.top_n is not None:
            r += f"\n\tReturning top {self.top_n} matches"
        return r
    
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
                of the type described in the LFDD algorithm's
                :py:meth:`detect_and_describe \
                    <featmf.templates.KptDetDescAlgo.detect_and_describe>`
                method. Eg: If the LFDD algorithm takes ``np.ndarray``
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
        # Top-K keypoints
        if self.top_k is not None:
            res1 = res1.sort_scores(self.top_k)
            res2 = res2.sort_scores(self.top_k)
        res = NNMatcher.Result(res=(res1, res2))
        # Match keypoints
        d1, d2 = res1.descriptors, res2.descriptors
        d1, d2 = ein.asnumpy(d1), ein.asnumpy(d2)
        if self.norm_descs:
            d1 = d1 / (np.linalg.norm(d1, axis=1, keepdims=True)+EPS)
            d2 = d2 / (np.linalg.norm(d2, axis=1, keepdims=True)+EPS)
        D, n1, n2 = d1.shape[1], d1.shape[0], d2.shape[0]
        # Query in d1 for d2
        index_d1 = faiss.IndexFlatIP(D)
        index_d1.add(d1)
        dists2_d1, indices2_d1 = index_d1.search(d2, self.top_r)
        # Get (i, j) and scores for d2 -> d1 indices
        ij_jd2 = np.arange(n2)
        ij_jd2 = ein.repeat(ij_jd2, "i -> (i k)", k=self.top_r)
        ij_id1 = ein.rearrange(indices2_d1, "n k -> (n k)")
        l1_ij = np.stack((ij_id1, ij_jd2), axis=1)
        sc1 = ein.rearrange(dists2_d1, "n k -> (n k)")
        # Query in d2 for d1
        index_d2 = faiss.IndexFlatIP(D)
        index_d2.add(d2)
        dists1_d2, indices1_d2 = index_d2.search(d1, self.top_r)
        # Get (i, j) and scores for d1 -> d2 indices
        ij_id1 = np.arange(n1)
        ij_id1 = ein.repeat(ij_id1, "i -> (i k)", k=self.top_r)
        ij_jd2 = ein.rearrange(indices1_d2, "n k -> (n k)")
        l2_ij = np.stack((ij_id1, ij_jd2), axis=1)
        sc2 = ein.rearrange(dists1_d2, "n k -> (n k)")
        # Match result
        if self.malgo == "mutual":
            i = [t for t, (i, j) in enumerate(l1_ij) if np.any(
                np.bitwise_and(l2_ij[:, 0] == i, l2_ij[:, 1] == j))]
            mtchs = l1_ij[i]
            scrs = sc1[i]
        elif self.malgo == "union":
            i = [t for t, (i, j) in enumerate(l2_ij) if not np.any(
                np.bitwise_and(l1_ij[:, 0] == i, l1_ij[:, 1] == j))]
            mtchs = np.concatenate((l1_ij, l2_ij[i]), axis=0)
            scrs = np.concatenate((sc1, sc2[i]), axis=0)
        res.i1, res.i2, res.scores = mtchs[:, 0], mtchs[:, 1], scrs
        if self.top_n is not None:
            res.sort_scores(self.top_n)
        return res

