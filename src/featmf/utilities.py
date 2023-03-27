# Utilities for FeatMF

# %%
import numpy as np
import cv2 as cv
from typing import List, Union, Tuple


# %%
# ----------- Converter functions -----------
# Keypoints: OpenCV to numpy
T1 = Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
def kpts_cv2np(kpts_cv: List[cv.KeyPoint], parse_size=False, 
            parse_angle=False, angle_conv_rad=True, 
            ret_response=False) -> T1:
    """
        Convert a list of OpenCV keypoints into numpy array(s). By
        default, only the keypoints ``(x, y)`` are extracted and 
        returned.
        
        :param kpts_cv:     A list of `OpenCV keypoint <https://docs.opencv.org/4.x/d2/d29/classcv_1_1KeyPoint.html>`_ objects.
        :type kpts_cv:      List[cv.KeyPoint]
        :param parse_size:  If True, the ``size`` attribute of each
                            keypoint is also parsed.
        :param parse_angle: If True, the ``angle`` attribute of each
                            keypoint is also parsed.
        :param angle_conv_rad:  If True, the angle in KeyPoint is 
                                assumed to be in degrees and is 
                                subsequently converted to radians.
        :param ret_response:    If True, the ``response`` attribute of
                                each keypoint is returned in a 
                                separate array.
        
        :return:    A numpy array containing the keypoints. If
                    ``parse_size`` is True, a tuple is returned, with
                    the second element being the array of scores.
        
        For the shape of keypoint array:
        
        | - If ``parse_size`` is False and ``parse_angle`` is False,
            the returned array is of shape ``[N, 2]`` (each row is
            ``[x, y]``).
        | - If ``parse_size`` is True, the returned array is of shape
            ``[N, 3]`` (each row is ``[x, y, size]``). 
        | - If ``parse_angle`` is also True, the returned array is of 
            shape ``[N, 4]`` (each row is ``[x, y, size, angle]``). 
        | - If ``parse_size`` is False and ``parse_angle`` is True, 
            the returned array is of shape ``[N, 3]`` (each row is 
            ``[x, y, angle]``).
        
        The shape of the scores array (returned only if 
        ``ret_response`` is True) is ``[N,]``.
    """
    kpts = np.array([k.pt for k in kpts_cv])
    if parse_size:
        sizes = np.array([k.size for k in kpts_cv])[:, None]
        kpts = np.hstack((kpts, sizes))
    if parse_angle:
        angles = np.array([k.angle for k in kpts_cv])[:, None]
        if angle_conv_rad:
            angles = np.deg2rad(angles)
        kpts = np.hstack((kpts, angles))
    if ret_response:
        scores = np.array([k.response for k in kpts_cv])
        return kpts, scores
    else:
        return kpts


# %%
# --------------------- Drawing functions ---------------------
T2 = Union[np.ndarray, Tuple[int, int, int], None]
T3 = Tuple[int, int]
def draw_keypoints(img: np.ndarray, pts: np.ndarray, 
            offset: T3=(0, 0), color: T2=None, sc=1, 
            draw_angle: bool=True) -> np.ndarray:
    """
        Draw keypoints on an image (with provision to add an offset).
        All keypoints are drawn as circles. 
        If no keypoint scale is specified, the radius of the circle 
        (keypoint neighborhood) is set to 10 pixels. 
        Note that the keypoint ``size`` is the diameter of the circle 
        centered at the keypoint. 
        If angles are not provided, no angle lines are drawn.
        
        :param img:     The image on which to draw the keypoints.
                        Shape must be ``[H, W, 3]`` (RGB image). If
                        a grayscale image is provided, it's converted
                        to RGB (same value for all channels). The 
                        passed image is not modified.
        :param pts:     The keypoints to draw. If shape is ``[N, 2]``,
                        they're assumed to be ``[x, y]``. If shape is
                        ``[N, 3]``, they're assumed to be ``[x, y,
                        size]`` (size is diameter). If shape is 
                        ``[N, 4]``, they're assumed to be 
                        ``[x, y, size, angle]`` (angle in radians, 
                        measured clockwise from +ve x-axis/right).
        :param offset:  The (x, y) offset to add to the keypoints 
                        before drawing.
        :param color:   The color to draw the keypoints. If not
                        provided, keypoints are drawn with random
                        colors. If (R, G, B) tuple is provided, all
                        keypoints are drawn with the same color. It
                        can also be a list of colors.
        :param sc:      A scalar multiple for the keypoint sizes (for
                        drawing). If ``size`` is not provided, this is
                        still used to scale the default radius of 10.
        :param draw_angle:  If False, keypoint angle lines are not
                            drawn even if keypoints contain angles.
        
        :return:    An image with the keypoints drawn on it. The
                    shape is ``[H, W, 3]`` (RGB image).
    """
    s_x, s_y = offset
    pts_xy = pts[:, :2] # [x, y] points
    npts = len(pts)
    # List of uint8 RGB colors (for each keypoint)
    colors: np.ndarray = None
    if color is None:
        colors = np.random.randint(256, size=(npts,3), dtype=np.uint8)
    elif type(color) == tuple:
        assert len(color) == 3, "Color must be (R, G, B) uint8 tuple"
        colors = np.array([color] * npts, dtype=np.uint8)
    else:
        colors = np.array(color, dtype=np.uint8)
    assert colors.shape == (npts, 3), "Colors must be of shape [N, 3]"
    # List of keypoint radii
    radii: np.ndarray = np.ones((npts,)) * sc
    if pts.shape[1] > 2:   # keypoints are [x, y, size]
        radii *= pts[:, 2]/2
    else:
        radii *= 10
    radii = np.round(radii).astype(int)
    # List of keypoint centers (with offset)
    pts_loc = np.round(pts_xy + np.array([s_x, s_y])).astype(int)
    # List of end points of circles (for angles)
    pts_end: np.ndarray = None
    if draw_angle and pts.shape[1] > 3:
        angs = np.array(pts[:, 3])
        pts_end = pts_loc + np.array([np.cos(angs), np.sin(angs)]).T \
                            * radii[:, None]
        pts_end = np.round(pts_end).astype(int)
    # Draw keypoints
    rimg: np.ndarray = img.copy()
    if len(rimg.shape) == 2:
        rimg = rimg[..., None]  # [H, W, 1]
    H, W, C = rimg.shape
    if C == 1:  # Grayscale to RGB
        rimg = np.repeat(rimg, 3, axis=2)    # [H, W, 3]
    rimg = rimg.astype(np.uint8)
    t = 1                   # Thickness
    lt = cv.LINE_AA         # Line type
    for i, pt in enumerate(pts_loc):
        c = colors[i].tolist()  # Color [R, G, B] values
        cv.circle(rimg, (pt[0], pt[1]), radii[i], c, t, lt)
        if pts_end is not None:
            pe = pts_end[i]
            cv.line(rimg, (pt[0], pt[1]), (pe[0], pe[1]), c, t, lt)
    return rimg

