IKCM Algorithms
=================

**I**\ mage **K**\ eypoint **C**\ orrespondence and/or **M**\ atching (IKCM) algorithms match keypoints (pixel locations) in two images. All classes inherit :py:class:`ImgKptMatchAlgo <featmf.templates.ImgKptMatchAlgo>` and they implement :py:func:`match_images <featmf.templates.ImgKptMatchAlgo.match_images>` function to return a :py:class:`ImgKptMatchAlgo.Result <featmf.templates.ImgKptMatchAlgo.Result>` object containing the match results.

Most child classes take a :py:class:`KptDetDescAlgo <featmf.templates.KptDetDescAlgo>` object if they operate on keypoints. See the documentation of the particular method to know more.

.. automodule:: featmf.ikcm
    :members:
    :special-members: __init__
