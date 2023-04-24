# Utilities for FeatMF
"""
    The ``utilities`` module contains utility functions for the
    ``featmf`` module. They are divided into the following sections.
    
    Converter functions
    -------------------
    
    -   :py:func:`featmf.utilities.kpts_cv2np`: Convert a list of
        OpenCV keypoints into numpy array(s).
    
    Homography functions
    ---------------------
    
    -   :py:func:`featmf.utilities.get_inliers`: Get inliers from a
        list of keypoint matches.
    
    Drawing functions
    -----------------
    
    -   :py:func:`featmf.utilities.draw_matches`: Draw keypoint
        correspondences (matches) between two images.
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
EPS = 1e-5  # Prevent division by zero


# %% ----------- Converter functions -----------
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


# %% --------------------- Drawing functions ---------------------
# Draw keypoints on an image
T2 = Union[np.ndarray, Tuple[int, int, int], None]
T3 = Tuple[int, int]
def draw_keypoints(img: np.ndarray, pts: np.ndarray, 
            offset: T3=(0, 0), color: T2=None, sc=1, 
            draw_angle: bool=True, thickness: int=1) -> np.ndarray:
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
        :param pts:     The keypoints to draw. If shape is
                        
                        -   ``[N, 2]``, they're assumed ``[x, y]``
                        -   ``[N, 3]``, they're ``[x, y, size]``
                            (size is diameter).
                        -   ``[N, 4]``, it is ``[x, y, size, angle]``
                            (angle in radians, measured clockwise from
                            +ve x-axis/right).
                        
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
        :param thickness:   The thickness of the lines in drawing (in
                            pixels).
        
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
    t = thickness           # Thickness
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
        value for all channels). In this case, an image of shape 
        ``[H, W]`` is converted to ``[H, W, 1]``.
        
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
    if len(img1.shape) == 2:
        img1 = img1[..., None]
    if len(img2.shape) == 2:
        img2 = img2[..., None]
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


# Draw correspondences
def draw_matches(img1: np.ndarray, img2: np.ndarray, 
        kpts1: np.ndarray, kpts2: np.ndarray, 
        matches: Union[np.ndarray, None], colors = (0, 255, 0), 
        hw_offset = (0.0, 1.0), xy_pos = (0, 0), draw_kpts = False,
        kpts_size_m = 1, ln_thickness: int= 2, ret_alpha: bool=True) \
        -> np.ndarray:
    """
        Draw keypoint correspondences (matches) between two images.
        Also draws the keypoint scale and angles. Uses the functions
        :py:func:`stack_images` and :py:func:`draw_keypoints`.
        
        :param img1:    The image on which to draw the keypoints.
                        Shape must be ``[H, W, 3]`` (RGB image). The
                        type should be ``np.uint8`` (0-255).
        :type img1:     np.ndarray
        :param img2:    The second image (like the first one).
        :type img2:     np.ndarray
        :param kpts1:   The keypoints for the first image. Shape is
                        
                        -   ``[N, 2]``, they're assumed ``[x, y]``
                        -   ``[N, 3]``, they're ``[x, y, size]``
                            (size is diameter).
                        -   ``[N, 4]``, it is ``[x, y, size, angle]``
                            (angle in radians, measured clockwise from
                            +ve x-axis/right).
                        
        :type kpts1:    np.ndarray
        :param kpts2:   The keypoints for the second image (should
                        have the same number of columns as ``kpts1``).
        :type kpts2:    np.ndarray
        :param matches: The matches between the keypoints. If None,
                        then a one-to-one correspondence is assumed
                        (in which case, the number of keypoints should
                        be the same).
        :type matches:  np.ndarray or None
        :param colors:  Color of the matches. An (R, G, B) tuple or a
                        list of RGB values (each item a tuple or the
                        list can be a ``np.ndarray``)
        :type colors:   Tuple, List, np.ndarray
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
        :param draw_kpts:   Draw the keypoints for matches if True
        :type draw_kpts:    bool
        :param kpts_size_m: A multiplier for the keypoint scale.
        :type kpts_size_m:  float
        :param ln_thickness:    Thickness of the lines drawn
        :type ln_thickness:     int
        :param ret_alpha:   If True, return the alpha channel (last).
        :type ret_alpha:    bool
        
        :raises AssertionError: Keypoints are of different lengths and
                                matches is None.
        
    """
    # Matches
    if matches is None:
        assert len(kpts1) == len(kpts2), "Same length needed"
        matches = np.repeat(np.arange(len(kpts1)).reshape(-1, 1), 2, 
                            axis=1)
    if type(colors) == tuple:
        colors = np.array(colors)[np.newaxis, ...]\
                .repeat(len(kpts1), axis=0)
    elif type(colors) == list:
        colors = np.array(colors)
    # Stack images
    img_back = stack_images(img1, img2, hw_offset, xy_pos)
    img, img_alpha = img_back[..., :-1], img_back[..., [-1]]
    # Draw keypoints (for the matches)
    _img = np.array(img)
    k1 = kpts1[matches[:, 0]]
    k2 = kpts2[matches[:, 1]]
    if k1.shape[1] > 2:
        k1[:, 2] *= kpts_size_m
        k2[:, 2] *= kpts_size_m
    hf, wf = hw_offset
    x, y = xy_pos
    h1, w1, c1 = img1.shape
    h2, w2, c2 = img2.shape
    x_, y_ = w1 * wf + x, h1 * hf + y
    if draw_kpts:
        _img = draw_keypoints(_img, k1, thickness=ln_thickness)
        _img = draw_keypoints(_img, k2, (x_, y_), 
                            thickness=ln_thickness)
    # Draw lines for the matches
    offset = np.array([x_, y_])
    _al = np.repeat(img_alpha, 3, axis=-1)
    for i, (i1, i2) in enumerate(matches):
        _pt1 = tuple(kpts1[i1, :2].astype(int))
        _pt2 = tuple((kpts2[i2, :2] + offset).astype(int))
        c = tuple(map(int, colors[i]))
        _img = cv.line(_img, _pt1, _pt2, c, ln_thickness, cv.LINE_AA)
        _al = cv.line(_al, _pt1, _pt2, (255, 255, 255), ln_thickness, 
                    cv.LINE_AA)
    img_alpha = _al[:, :, [0]]
    if ret_alpha:
        img_res = np.concatenate((_img, img_alpha), axis=-1)
    else:
        img_res = _img
    return img_res


# %% -------------- Homography functions --------------
# Get inliers given keypoints and homography
def get_inliers(kpts1: np.ndarray, kpts2: np.ndarray, 
            h_1to2: np.ndarray, pix_thr=1, ret_type: str="count") \
            -> Union[int, float, bool, List[int], List[float], \
                List[bool], List[List[bool]]]:
    """
        Given two sets of keypoints (pixel coordinates) and the
        homogeneous transform from one set to another, get the inlier
        set containing the keypoints that follow the given homography.
        
        :param kpts1:   Keypoints for image 1. Shape ``[N1, d]`` with
                        ``d > 2``. The first two columns are ``x``
                        and ``y`` respectively. Other columns are not
                        used.
        :type kpts1:    np.ndarray
        :param kpts2:   Keypoints for image 2. Shape ``[N1, d]`` (same
                        as in ``kpts1``)
        :type kpts2:    np.ndarray
        :param h_1to2:  The homogeneous transform from image 1 to
                        image 2. Shape ``[3, 3]``.
        :type h_1to2:   np.ndarray
        :param pix_thr: Pixel threshold for considering inlier. If an
                        ``int`` (or ``float``), then only a single
                        threshold is used. Could also be a list of
                        thresholds (for multiple results).
        :type pix_thr:  Union[int, float, List[int, float]]
        :param ret_type:    What to return. Can be
        
                            - `"count"`: Number of inliers
                            - `"percentage"`: Inliers as %
                            - `"mask"`: For a True or False list
                            
                            It can also be ``c``, ``p``, or ``m``
                            respectively.
        
        :type ret_type: str
        :raises ValueError: If ``ret_type`` is invalid
        
        Multiple thresholds can be given in a list to ``pix_thr``. The
        return type depends on ``ret_type``.
        
        -   ``"count"``: A list of ``int``
        -   ``"percentage"``: A list of ``float`` values
        -   ``"mask"``: A list of list of ``bool``. For each 
            ``pix_thr``, the list contains a list of ``bool``. Each
            element in this list is for a keypoint: ``True`` if inlier
            and ``False`` if outlier.
        
        If ``pix_thr`` is not a list but an ``int`` (or ``float``)
        then only the first element of the list is returned (type 
        corresponds from above).
        
        :returns:   The inliers. See above for type.
    """
    _thr = pix_thr
    if type(_thr) != list:
        _thr = [_thr]
    kp1 = np.concatenate((kpts1[:, :2], 
            np.ones((kpts1.shape[0], 1))), axis=1)
    kp2 = np.concatenate((kpts2[:, :2],
            np.ones((kpts2.shape[0], 1))), axis=1)
    _kp2 = (h_1to2 @ kp1.T).T
    try:
        _kp2 = _kp2 / _kp2[:, [2]]
    except ZeroDivisionError:
        _kp2 = (EPS + _kp2) / (EPS + _kp2[:, [2]])
    dists = np.linalg.norm(_kp2 - kp2, axis=1)
    res = []
    for th in _thr:
        if ret_type in ["c", "count"]:
            res.append(np.sum(dists <= th))
        elif ret_type in ["p", "percentage"]:
            res.append(np.sum(dists <= th) / dists.shape[0])
        elif ret_type in ["m", "mask"]:
            res.append((dists <= th).tolist())
        else:
            raise ValueError(f"Invalid return type: {ret_type}")
    
    if not type(pix_thr) in [list, tuple]:
        return res[0]
    
    return res

# %%
