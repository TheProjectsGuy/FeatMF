# Utilities for FeatMF
"""
    The ``utilities`` module contains utility functions for the
    ``featmf`` module. They are divided into the following sections.
    
    Converter functions
    -------------------
    
    -   :py:func:`featmf.utilities.kpts_cv2np`: Convert a list of
        OpenCV keypoints into numpy array(s).
    
    Drawing functions
    -----------------
    
    -   :py:func:`featmf.utilities.draw_keypoints`: Draw keypoints on
        an image.
    -   :py:func:`featmf.utilities.stack_images`: Stack two images
        side-by-side with offset and relative positioning.
    
    All functions are documented below.
"""

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
# Draw keypoints on an image
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


# Stack images
def stack_images(img1: np.ndarray, img2: np.ndarray, hw_offset=(0, 1),
            xy_pos=(0, 0), ret_alpha: bool=True) -> np.ndarray:
    """ 
        Stacks two images beside each other. There is an option to
        add an offset between the images and to move the second image
        to a different position (than adjacent). The size of the two
        images can be different. The passed images are not modified.
        
        The input images can be in range ``[0, 1]`` (``float`` or 
        ``np.float32``) or ``[0, 255]`` (uint8). The output image is
        in the same range (and of the same type) as the input images.
        
        If a greyscale image is provided, it's converted to RGB (same
        value for all channels).
        
        :param img1:    First image. Shape must be ``[H1, W1, 3]``.
        :type img1:     np.ndarray
        :param img2:    Second image. Shape must be ``[H2, W2, 3]``.
        :type img2:     np.ndarray
        :param hw_offset:   
                The ``(hf, wf)`` offset of the top-left corner of the
                frame of second image from the top-left corner of the 
                first image. ``hf`` is the vertical offset (in 
                fraction of the height of the first image) and ``wf`` 
                is the horizontal offset (in fraction of the width of 
                the first image).
        :type hw_offset:    Tuple[float, float]
        :param xy_pos:  The ``(x, y)`` position of the top-left corner
                        of the second image in the frame of the second
                        image. ``x`` is the horizontal offset (in
                        pixels) and ``y`` is the vertical offset (in
                        pixels).
        :type xy_pos:   Tuple[int, int]
        :param ret_alpha:   If True, the returned image has an alpha
                            channel. If False, the returned image is
                            RGB.
        :type ret_alpha:    bool
        
        .. tip::
            1.  The second image can placed in pure pixel locations 
                relative to the top left corner of the first image by
                setting ``hw_offset=(0, 0)`` and ``xy_pos=(x, y)``.
            2.  The second image can be placed below the first one by
                setting ``hw_offset=(1, 0)`` and ``xy_pos=(0, 0)``.
        
        :return:    The stacked image. Shape is ``[H, W, 4]`` if
                    ``ret_alpha=True`` and ``[H, W, 3]`` if
                    ``ret_alpha=False``. The ``dtype`` is the same as
                    the input images.
        :rtype:     np.ndarray
    """
    # Check input images
    assert type(img1) == np.ndarray, "img1 must be a numpy array"
    assert type(img2) == np.ndarray, "img2 must be a numpy array"
    assert len(img1.shape) == 3, "img1 must be a 3D array"
    assert len(img2.shape) == 3, "img2 must be a 3D array"
    h1, w1, c1 = img1.shape
    if c1 == 1:
        img1 = np.repeat(img1, 3, axis=2)
    h2, w2, c2 = img2.shape
    if c2 == 1:
        img2 = np.repeat(img2, 3, axis=2)
    assert img1.dtype == img2.dtype, "Images must have same dtype"
    # Both images are now RGB (H, W, 3 shape)
    hf, wf = hw_offset
    x, y = xy_pos
    ret_img = np.zeros((int(max(h1, h1 * hf + y + h2)), 
            int(max(w1, w1 * wf + x + w2)), 4), dtype=img1.dtype)
    # Insert images
    ret_img[:h1, :w1, :3] = img1
    ret_img[int(h1 * hf + y):int(h1 * hf + y + h2),
            int(w1 * wf + x):int(w1 * wf + x + w2), :3] = img2
    # Add alpha channel
    occ_val = 255 if img1.dtype == np.uint8 else 1.0
    ret_img[:h1, :w1, 3] = occ_val
    ret_img[int(h1 * hf + y):int(h1 * hf + y + h2),
            int(w1 * wf + x):int(w1 * wf + x + w2), 3] = occ_val
    # Return image
    if ret_alpha:
        return ret_img
    else:
        return ret_img[..., :3]

