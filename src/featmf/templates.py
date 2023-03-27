# Templates for FeatMF Framework
"""
    Template classes for local image feature detection and description 
    algorithms.
"""

# %%
import numpy as np
import torch
import cv2 as cv
from PIL import Image
import copy
from dataclasses import dataclass
from typing import Union, Any


# %%
# ----------- Types -----------
# For KptDetDescAlgo
KDD_T1 = Union[np.ndarray, torch.Tensor, None]
KDD_T2 = Union[np.ndarray, torch.Tensor, Image.Image]


# %%
class KptDetDescAlgo:
    """
        A template for image local keypoint detection and description 
        algorithms.
        
        Rules for the child classes:
        
        | - The child classes must implement the function 
            :meth:`detect_and_describe` that returns a 
            :class:`KptDetDescAlgo.Result` object. They must sanitize
            the input types to suit the particular algorithm.
        | - ``__call__`` function calls :meth:`detect_and_describe`.
            The child classes need not implement this.
    """
    @dataclass
    class Result:
        """
            Result for a keypoint detection and description algorithm.
            Includes keypoints, descriptors, and scores. Each of these
            items can contain a numpy array, a torch tensor, or None.
            
            .. note:: 
            
                Check the documentation of the algorithm for types, 
                shapes, and descriptions of these items. A generic 
                description is given in this class.
        """
        keypoints: KDD_T1
        """
            Keypoints detected by the algorithm. Depending upon the
            algorithm, its shape can be one of the following
            
            | - ``[N, 2]`` for 2D keypoints ``(x, y)`` without size 
                information.
            | - ``[N, 3]`` for 2D keypoints ``(x, y, s)`` with size 
                information (diameter of the circle centered at the 
                keypoint).
            | - ``[N, 4]`` for 2D keypoints ``(x, y, s, ang)`` with
                size and angle information. The angles are in radians
                and are measured clockwise from the +ve x-axis (which
                is to the right of the image).
            
            Where ``N`` is the number of keypoints detected.
            
            .. note:: 
                Using ``[N, 3]`` as ``(x, y, ang)`` is discouraged 
                (set size as 1 and use ``[N, 4]`` instead).
            
            See the documentation of the particular algorithm for more 
            details.
        """
        descriptors: KDD_T1
        """
            Descriptors for the keypoints detected by the algorithm.
            It's shape must be ``[N, D]`` where ``N`` is the number of 
            keypoints and ``D`` is the descriptor dimension.
        """
        scores: KDD_T1
        """
            Scores for the keypoints detected by the algorithm. It's
            shape must be ``[N,]`` where ``N`` is the number of 
            keypoints. This could be None if the algorithm does not 
            provide detection scores (confidence).
        """
        
        # Length of the results
        def __len__(self):
            if self.keypoints is None:
                return 0
            return len(self.keypoints)
        
        def len(self):
            """
                Returns the number of keypoints (0 if no keypoints).
            """
            return len(self)
        
        # Descriptor dimension
        def d_dim(self) -> int:
            """
                Returns the descriptor dimension (if descriptors have
                been loaded). Otherwise, returns 0.
            """
            if self.descriptors is None:
                return 0
            return self.descriptors.shape[1]
        
        # String representation
        def __repr__(self) -> str:
            r = f"KptDetDescAlgoResult: {len(self)} keypoints "\
                f"(shape: {self.keypoints.shape}) \n" \
                f"Descriptor dimension: {self.d_dim()}"
            if self.scores is None:
                r += f"\nScores are absent"
            else:
                r += f"\nScores are present"
            return r
        
        def repr(self):
            """
                Returns a string representation of the object
                describing the keypoints, descriptors, and scores.
                
                .. tip::
                    This function is also implemented in the 
                    ``__repr__`` method for usual ``print`` calls.
            """
            return self.__repr__()
        
        # Access using []
        def get(self, i):
            """
                :param i: Item to be returned
                :type i: Union[int, slice, list[int], str]
                
                Depending upon the type of ``i``, returns one of the 
                following
                
                | - ``str``: Should be in ['keypoints', 'descriptors',
                    'scores']. Returns the corresponding item. Type
                    will be the same as the original item.
                | - ``int``: Returns a new result containing only the
                    i-th keypoint, descriptor, and score (if present). 
                    N = 1.
                | - ``slice``: Returns a new result containing only
                    the keypoints, descriptors, and scores in the
                    specified `slice <https://docs.python.org/3/glossary.html#term-slice>`_.
                | - ``list[int]``: Returns a new result containing
                    only the keypoints, descriptors, and scores at
                    the specified indices. 
                
                .. note::
                    Only in the case of ``str`` type, the returned
                    value is a direct reference (editing it will edit
                    the original result). In all other cases, a new
                    result is created.
                
                :raises IndexError: If keypoints not initialized
                :raises TypeError: If ``i`` is of an invalid type
                
                .. tip:: 
                    This function is also implemented in the 
                    ``__getitem__`` method for usual ``[i]`` indexing.
                
                :return: The requested indices (or item)
                :rtype: Union[KptDetDescAlgo.Result, np.ndarray, \
                        torch.Tensor, None]
            """
            return self[i]
        
        def __getitem__(self, i):
            # See the get function for doc
            if type(i) == str:
                assert i in ['keypoints', 'descriptors', 'scores']
                return self.__getattribute__(i)
            elif len(self) == 0:
                raise IndexError("Keypoints not initialized")
            elif type(i) == int:
                k = self.keypoints[i].reshape(1, -1)
                d = self.descriptors[i].reshape(1, -1)
                s = None if self.scores is None else self.scores[i]
                if s is not None:
                    s = type(s)([s])
                return KptDetDescAlgo.Result(k, d, s)
            elif type(i) == slice:
                s = None if self.scores is None else self.scores[i]
                return KptDetDescAlgo.Result(self.keypoints[i], 
                        self.descriptors[i], s)
            elif type(i) == list:
                assert all([type(j) == int for j in i])
                k = self.keypoints[i]
                d = self.descriptors[i]
                s = None if self.scores is None else self.scores[i]
                return KptDetDescAlgo.Result(k, d, s)
            else:
                raise TypeError(f"Invalid index type: {type(i)}")
        
        # Sort scores
        def sort_scores(self, top_k: Union[int,None]=None, 
                ascending: bool=True) -> 'KptDetDescAlgo.Result':
            """
                Sort the result based on the detection scores. To
                use this function, the scores should not be None.
                
                :param top_k: If not None, only the top-k detections
                        are retained (others are discarded). Default 
                        is ``None``.
                :type top_k: Union[int, None]
                :param ascending: If True, the scores are sorted in
                        the ascending order. If False, the scores are
                        sorted in the descending order. Default is
                        ``True``.
                :type ascending: bool
                
                :raises AssertionError: If scores are None.
                :raises TypeError: If scores are not of correct type.
                
                :returns self:  The result object (scores are sorted).
                
                .. note::
                    This function changes the data contained in the
                    called object.
                
                :rtype: KptDetDescAlgo.Result
            """
            assert self.scores is not None, "Scores do not exist"
            if type(self.scores) == np.ndarray:
                _o = np.argsort(self.scores)
            elif type(self.scores) == torch.Tensor:
                _o = torch.argsort(self.scores)
            else:
                raise TypeError(f"Scores type: {type(self.scores)}")
            self.scores = self.scores[_o]
            self.keypoints = self.keypoints[_o]
            self.descriptors = self.descriptors[_o]
            if top_k is not None:   # Keep only top-k (ascending)
                self.scores = self.scores[-top_k:]
                self.keypoints = self.keypoints[-top_k:]
                self.descriptors = self.descriptors[-top_k:]
            if not ascending:
                self.scores = self.scores[::-1]
                self.keypoints = self.keypoints[::-1]
                self.descriptors = self.descriptors[::-1]
            return self
        
        # Clone
        def copy(self) -> 'KptDetDescAlgo.Result':
            """
                Returns a deepcopy of the self item.
                
                :rtype: KptDetDescAlgo.Result
            """
            return copy.deepcopy(self)
    
    # Abstract methods
    def repr(self) -> str:
        """
            Returns a string representation of the algorithm (name)
        """
        return self.__repr__()
    
    def __repr__(self) -> str:
        return f"Wrapper class for {self.__class__.__name__}"
    
    def detect_and_describe(self, img: KDD_T2, *args: Any, 
            **kwargs: Any) -> Result:
        """
            :param img: A single image to be processed.
            :type img: Union[Image.Image, np.ndarray, torch.Tensor]
            :param args: Additional arguments
            :param kwargs: Additional keyword arguments
            
            .. note::
                The child classes should enforce type constraints. 
                This is because some algorithms may not work with
                certain types, or might require preprocessing of the
                image. See the documentation of the child classes for
                more information.
            
            :raises TypeError: If ``img`` is of an invalid type (for
                the particular algorithm).
            :raises NotImplementedError: If the child class does not
                implement this method.
            
            Abstract method that child classes should inherit to
            implement the detection and description algorithm.
            
            .. tip::
                The ``__call__`` method calls this method directly. It
                has the same inputs and outputs.
            
            :return: The result of the detection and description
            :rtype: KptDetDescAlgo.Result
        """
        raise NotImplementedError("Abstract method, not implemented")
    
    def __call__(self, img: KDD_T2, *args: Any, **kwds: Any) \
            -> Result:
        return self.detect_and_describe(img, *args, **kwds)


# %%
