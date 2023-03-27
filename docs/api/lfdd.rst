LFDD Algorithms
===============

**L**\ ocal **F**\ eature **D**\ etection and **D**\ escription (LFDD) algorithms detect and describe keypoints for an image. All classes inherit :py:class:`KptDetDescAlgo <featmf.templates.KptDetDescAlgo>` and they implement a :py:meth:`detect_and_describe <featmf.templates.KptDetDescAlgo.detect_and_describe>` function to return a :py:class:`KptDetDescAlgo.Result <featmf.templates.KptDetDescAlgo.Result>` object containing detection results for a given image.

.. automodule:: featmf.lfdd
    :members:
    :special-members: __init__
