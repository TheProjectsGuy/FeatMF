# Generic utility functions for feature matching algorithms
"""
"""

# %%
import numpy as np
import cv2 as cv
from collections import namedtuple
from typing import Any, Union, List
from dataclasses import dataclass


# %%
# ------------------- Abstract classes -------------------
# Detect and describe keypoints
class KptDetectionDescriptionAlgo:
    """
        Abstract class for keypoint detection and description 
        algorithms.
        
        > Note: Currently, keypoints are "points" (they have x, y 
        positions only, no affine parameters). They could have a scale
        parameter as well.
        
        The child classes have to implement the function
        `detect_and_describe` which takes one image and outputs the
        `Result` (keypoint, descriptor, and probably scores). No
        default constructor is provided.
    """
    # Result container
    @dataclass
    class Result:
        """
            Contains attributes keypoints, descriptors, and scores.
            Also has length, indexing, and string representation.
        """
        keypoints: Union[np.ndarray, None] = None
        """
            Keypoints detected in the image. Shape must be in
            - [N, 3] for keypoints with scale, each row is (x, y, s)
            - [N, 2] for keypoints without scale, each row is (x, y)
            
            Where N is the number of keypoints
        """
        descriptors: Union[np.ndarray, None] = None
        """
            Descriptors for the corresponding keypoints. Shape must be
            [N, D] which means 'D' dimensional descriptors for 'N'
            keypoints
        """
        scores: Union[np.ndarray, None] = None
        """
            Keypoint detection scores (confidence on the keypoint).
            Not all algorithms give this.
        """
        # Length of the results
        def __len__(self):
            if self.keypoints is None:
                return 0
            return len(self.keypoints)
        
        def __repr__(self) -> str:
            _repr = f"Result: {len(self)} keypoints (shape: "\
                    f"{self.keypoints.shape}).\n" \
                    f"\tDescriptor is {self.descriptors.shape[1]} "\
                    f"dimensional (shape: {self.descriptors.shape})."
            if self.scores is None:
                _repr = f"{_repr} \n\tScores are absent"
            else:
                _repr = f"{_repr} \n\tScores are present"
            return _repr
        
        # Access using []
        def __getitem__(self, i):
            """
                - If type(i) == str: Return the self attribute 'i'
                - If type(i) == int: Return the particular result
                - If type(i) == slice:  Return the slice of results
                - If type(i) == list:  Selected filter (or list)
                
                In the last two cases, a `namedtuple` is returned. It 
                is a copy and modifying it won't change the original
                result. The attributes are (in order) 'keypoints', 
                'descriptors', and 'scores'.
                - If type(i) == int:
                    - 'keypoints' shape: [2,] (x, y) or [3,] (x, y, s)
                    - 'descriptors' shape: [D,] (single descriptor)
                    - 'scores': None or shape: [1,] (scalar score)
                - If type(i) == slice: A slice of the original is 
                    returned
                - If type(i) == list: Depends upon what direct 
                    indexing yields.
                
                This function makes no changes to the data.
            """
            if type(i) == str and i in self.__dict__.keys():
                return self.__getattribute__(i)
            elif type(i) in [int, slice, list]:
                _res = namedtuple("KptDetectDescribeResult", 
                        ["keypoints", "descriptors", "scores"], 
                        defaults=[None, None, None])
                kp = self.keypoints[i]
                desc = self.descriptors[i]
                sc = None
                if self.scores is not None:
                    sc = self.scores[i]
                return _res(kp, desc, sc)
            else:
                raise TypeError(f"Invalid type: {type(i)}, i={i}")
        
        # Sort result using keypoint detection scores
        def sort_scores(self, top_k: Union[int,None]=None, 
                    ascending: bool=True) -> None:
            """
                Sort the result based on the detection scores. To use
                this function, the scores should not be None.
                
                This function changes the data contained.
                
                Parameters:
                - top_k:    If not None, only the top-k detections are
                            saved (others are discarded)
                - ascending:    If True, the result is in the 
                                ascending order, else it is in the
                                descending order.
            """
            assert self.scores is not None, "Scores do not exist"
            _o = np.argsort(self.scores)    # Ascending sort
            self.scores = self.scores[_o]
            self.keypoints = self.keypoints[_o]
            self.descriptors = self.descriptors[_o]
            if top_k is not None:
                self.scores = self.scores[-top_k:]
                self.keypoints = self.keypoints[-top_k:]
                self.descriptors = self.descriptors[-top_k:]
            if not ascending:       # Descending order
                self.scores = self.scores[::-1]
                self.keypoints = self.keypoints[::-1]
                self.descriptors = self.descriptors[::-1]
    
    def __repr__(self) -> str:
        return f"Wrapper class for {self.__class__.__name__}"
    
    def detect_and_describe(self, img: np.ndarray, 
                *args, **kwargs) -> Result:
        """
            Runs a detect and describe algorithm and returns the
            keypoints, descriptors, and scores.
            Some algorithms may not implement a 'scores' metric.
            
            Parameters:
            - img:      RGB Image of type np.uint8, shape: [H, W, 3]
            - *args:    Additional arguments for the child classes
            - **kwargs: Additional keyword arguments for the child
                        classes
            
            Returns:
            - res:      Result containing the keypoints, descriptors,
                        and (optionally) the scores for the single
                        `img`.
        """
        raise NotImplementedError("Abstract function not implemented")
    
    def __call__(self, img: Union[np.ndarray, List[np.ndarray]], 
                *args: Any, **kwargs: Any) \
                -> Union[Result, List[Result]]:
        """
            Makes a call to `detect_and_describe` with the given image
            (or images).
            
            Parameters:
            - img:      A single image (shape: [H, W, 3]) or a batch 
                        of images (shape: [B, H, W, 3], or a list). 
                        Should be RGB and of type uint8.
            - *args:    Additional arguments for the child classes
            - **kwargs: Additional keyword arguments for the child
                        classes
            
            Returns:
            - res:      Result(s) for image(s). If single image, then
                        a single `Result` object. If a batch of images
                        is passed, then a list of `Result` objects
                        (length = B).
        """
        if type(img) == list or len(img.shape) == 4: # Batch of images
            return [self.detect_and_describe(im, *args, **kwargs) \
                    for im in img]
        return self.detect_and_describe(img, *args, **kwargs)


# Descriptor matching results
@dataclass
class DescMatchingResult:
    """
        Contains attributes i1, i2, and scores. Also has length, 
        indexing, and string representation.
    """
    i1: Union[None,np.ndarray] = None
    """
        Indices of keypoints (or descriptors) from image 1 (start=0).
        Shape: [N,] where N is number of matches.
    """
    i2: Union[None,np.ndarray] = None
    """
        Indices of keypoints (or descriptors) from image 2 (start=0).
        Shape: [N,] where N is number of matches.
    """
    scores: Union[None,np.ndarray] = None
    """
        Descriptor matching score. Usually a high score means a better
        match (like in cosine similarity or dot product). If choosing
        score from euclidean distance, a high value could mean a bad
        match.
        
        > Tip: If using euclidean distance, this can be -ve to show
        that high magnitude means bad match. Sorting will work then.
        
        Shape: [N,] where N is number of matches. All descriptor 
        matching algorithms should output a score (based on similarity
        search).
    """
    
    # Length of results
    def __len__(self):
        if self.scores is None:
            return 0
        return len(self.scores)
    
    # String representation
    def __repr__(self) -> str:
        return f"Descriptor matching result with {len(self)} matches"
    
    # Indexing
    def __getitem__(self, i):
        """
            - If type(i) == str: Return the self attribute 'i'
            - If type(i) == int: Return the particular result
            - If type(i) == slice:  Return the slice of results
            
            In the last two cases, a `namedtuple` is returned. It is a 
            copy and modifying it won't change the original result.
            The attributes are (in order) 'i1', 'i2', 'scores'. Direct
            slices (or index) using 'i' is taken.
            
            This function makes no changes to the data.
        """
        if type(i) == str and i in self.__dict__.keys():
            return self.__getattribute__(i)
        elif type(i) in [int, slice, list]:
            _res = namedtuple("DescMatchResult", 
                ["i1", "i2", "scores"], defaults=[None, None, None])
            return _res(self.i1[i], self.i2[i], self.scores[i])
        else:
            raise TypeError(f"Invalid type: {type(i)}, i={i}")
    
    # Sort result using matching scores
    def sort_scores(self, top_k: Union[int,None]=None, 
            ascending: bool=True) -> None:
        """
            Sort the result based on the detection scores. This 
            function changes the data (results) contained.
            
            Parameters:
            - top_k:    If not None, only the top-k detections are
                        saved (others are discarded).
            - ascending:    If True, the result is in the ascending
                            order, else it is in the descending order.
        """
        assert self.scores is not None
        _o = np.argsort(self.scores)    # Ascending sort
        self.i1 = self.i1[_o]
        self.i2 = self.i2[_o]
        self.scores = self.scores[_o]
        if top_k is not None:
            self.i1 = self.i1[-top_k:]
            self.i2 = self.i2[-top_k:]
            self.scores = self.scores[-top_k:]
        if not ascending:       # Descending order
            self.i1 = self.i1[::-1]
            self.i2 = self.i2[::-1]
            self.scores = self.scores[::-1]


# Keypoint matching result (for descriptor-less algorithms)
@dataclass
class KptMatchingResult:
    """
        Contains attributes kpts1, kpts2, scores. The 'kpts1' and 
        'kpts2' attributes have the matches (correspondences).
    """
    kpts1: Union[None, np.ndarray] = None
    """
        Keypoint (locations) for image 1. Shape is [N, 2] if keypoints
        are (x, y) values, and [N, 3] if keypoints are (x, y, s) 
        values (if they include scale information).
    """
    kpts2: Union[None, np.ndarray] = None
    """
        Keypoint (locations) for image 2. Shape is [N, 2] if keypoints
        are (x, y) values, and [N, 3] if keypoints are (x, y, s) 
        values (if they include scale information).
    """
    scores: Union[None, np.ndarray] = None
    """
        Scores for keypoint correspondences (matches). Shape is [N,].
        It could also be None (if the matching algorithm gives no 
        score). Usually a high score means a better match (like in 
        cosine similarity or dot product). If choosing score from 
        euclidean distance, a high value could mean a bad match.
        
        > Tip: If using euclidean distance, this can be -ve to show
        that high magnitude means bad match. Sorting will work then.
        
        Shape: [N,] where N is number of matches. Could also be None
        (some algorithms could give direct matches without scores).
    """
    
    # Length of results
    def __len__(self):
        if self.kpts1 is None:
            return 0
        return len(self.kpts1)
    
    # String representation
    def __repr__(self) -> str:
        _r = f"Keypoint matching result with {len(self)} matches."
        if self.scores is None:
            _r = f"{_r} \n\tScores are absent"
        else:
            _r = f"{_r} \n\tScores are present"
        return _r
    
    # Access using []
    def __getitem__(self, i):
        """
            - If type(i) == str: Return the self attribute 'i'
            - If type(i) == int: Return the particular result
            - If type(i) == slice:  Return the slice of results
            
            In the last two cases, a `namedtuple` is returned. It is a
            copy and modifying it won't change the original result.
            The attributes are (in order) "kpts1", "kpts2", "scores".
            Direct slices (or index) using 'i' is taken.
            
            This function makes no changes to the data.
        """
        if type(i) == str and i in self.__dict__.keys():
            return self.__getattribute__(i)
        elif type(i) in [int, slice, list]:
            _res = namedtuple("KptMatchResult", 
                    ["kpts1", "kpts2", "scores"], 
                    defaults=[None, None, None])
            k1 = self.kpts1[i]
            k2 = self.kpts2[i]
            sc = None
            if self.scores is not None:
                sc = self.scores[i]
            return _res(k1, k2, sc)
        else:
            raise TypeError(f"Invalid type: {type(i)}, i={i}")
    
    # Sort results using scores
    def sort_scores(self, top_k: Union[int,None]=None, 
            ascending: bool=True) -> None:
        """
            Sort the result based on the detection scores. This 
            function changes the data (results) contained.
            
            Parameters:
            - top_k:    If not None, only the top-k detections are
                        saved (others are discarded).
            - ascending:    If True, the result is in the ascending
                            order, else it is in the descending order.
        """
        assert self.scores is not None
        _o = np.argsort(self.scores)    # Ascending sort
        self.kpts1 = self.kpts1[_o]
        self.kpts2 = self.kpts2[_o]
        self.scores = self.scores[_o]
        if top_k is not None:
            self.kpts1 = self.kpts1[-top_k:]
            self.kpts2 = self.kpts2[-top_k:]
            self.scores = self.scores[-top_k:]
        if not ascending:       # Descending order
            self.kpts1 = self.kpts1[::-1]
            self.kpts2 = self.kpts2[::-1]
            self.scores = self.scores[::-1]


# Match two images
class ImageMatchingAlgo:
    """
        Abstract class for image matching algorithms (using keypoints,
        descriptors, or both). No default constructor is provided.
    """
    
    def __repr__(self) -> str:
        return f"Image matching algorithm: {self.__class__.__name__}"
    
    def match(self, img1: np.ndarray, img2: np.ndarray, 
            **kwargs) -> KptMatchingResult:
        """
            Match keypoints in two images.
            
            Parameters:
            - img1:         First image. Shape [H, W, 3] (uint8 RGB)
            - img2:         Second image. Shape [H, W, 3] (uint8 RGB)
            - **kwargs:     Other arguments to the algorithm.
            
            Returns:
            - result:   A KptMatchingResult object.
        """
        raise NotImplementedError("Abstract method called.")
    
    def __call__(self, img: Union[np.ndarray, List[np.ndarray]], 
                 *args: Any, **kwds: Any) \
                -> Union[KptMatchingResult,List[KptMatchingResult]]:
        """
            Makes a call to `match` with the given image (or images).
            
            Parameters:
            - img:      A single image (shape: [H, W, 3]) or a batch 
                        of images (shape: [B, H, W, 3] or list).
                        Should be RGB and of type uint8.
            - *args:    Other arguments to the algorithm.
            - **kwds:   Other keyword arguments to the algorithm.
            
            Returns:
            - res:      A single KptMatchingResult object or a list of
                        KptMatchingResult objects (batch of images).
        """
        if type(img) == list or len(img.shape) == 4:
            return [self.match(im, *args, **kwds) for im in img]
        else:
            return self.match(img, *args, **kwds)


# %%
# ----------------- Converter functions -----------------
# Convert List[OpenCV KeyPoint] to numpy array
def kpts_cv2np(kpts_cv: List[cv.KeyPoint]) -> np.ndarray:
    """
        Converts a list of OpenCV keypoints into a list of numpy
        keypoints. The first element in pt is x, then y, then (if 
        present) scaling factor.
        
        Parameters:
        - kpts_cv:      A list of OpenCV keypoint detections.
        
        Returns:
        - kpts:         Keypoints of shape [N, 2]. Note that OpenCV
                        KeyPoint.pt is 2D (x, y) value (no s value).
    """
    return np.array(list(map(lambda kp: list(kp.pt), kpts_cv)))


# %%


# %%
# Experimental section
