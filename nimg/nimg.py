# -*- coding: utf-8 -*-

import numpy as np
import scipy
from scipy import ndimage
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

import zipfile
import sys
#import seaborn as sns
#%matplotlib inline

from skimage import data, io, filters
from skimage.morphology import disk
import skimage.feature


def im_print(im):
    print("ndim = ", im.ndim, "| shape = ", im.shape, '| max = ', im.max(),
          '| min = ', im.min(), "| size = ", im.size, "| dtype = ", im.dtype)


def myhist(im, bins=60, log=False, nf=0):
    hist, bin_edges = np.histogram(im, bins=bins)
    bin_centers = 0.5*(bin_edges[:-1] + bin_edges[1:])
    if nf:
        plt.figure()
    plt.plot(bin_centers, hist, lw=2)
    #plt.text(0.57, 0.8, 'histogram', fontsize=14, transform = plt.gca().transAxes)
    if log:
        plt.yscale('log')


def plot_im_series(im, cm=plt.cm.gray, horizontal=True, **kw):
    if horizontal:
        plt.figure(figsize=(12, 5.6))
        s = 100 + len(im) * 10 + 1
    else:
        plt.figure(figsize=(5.6, 12))
        s = len(im) * 100 + 10 + 1
    for i, img in enumerate(im):
        plt.subplot(s+i)
        plt.imshow(img, cmap=cm, **kw)
        plt.axis('off')
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0, left=0, right=1)


def plot_otsu(im, cm=plt.cm.gray):
    val = filters.threshold_otsu(im)
    mask = im > val
    plot_im_series(im * mask, cm=cm)
    return mask


def im_median(im,  radius=0, footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])):
    """Return the median filtered image *im*
    Default footprint is equivalent to skimage.morphology.disk(1). and plane by plane
    with e.g. radius=3 calculate 3D median filter.
    """
    if radius:
        return ndimage.filters.median_filter(im, size=radius)
    else:
        if im.ndim == 2:
            return ndimage.filters.median_filter(im, footprint=footprint)
        else:
            imf = np.zeros(im.shape).astype(im.dtype)
            for i, img in enumerate(im):
                imf[i] = ndimage.filters.median_filter(img, footprint=footprint)
            return  imf


def zipread(fp):
    # Read a single tif that was zipped
    # Return the image
    with zipfile.ZipFile(fp) as myzip:
        with myzip.open(myzip.filelist[0]) as myfile:
            return io.imread(myfile)


def zproject_median(im):
    """
    im: is a 3D-grayscale (pln,row,col)
    Return a single (median by default) projected image.
    TODO: %2 only for int
    """
    if not (im.ndim == 3 and len(im) == im.shape[0]):
        sys.exit('only 3D-grayscale (pln, row, col)')
    # maintain same dtype as input im
    zproj = np.zeros(im.shape[1:]).astype(im.dtype)
    if len(im)%2:
        np.median(im, axis=0, out=zproj)
    else:
        # skip first plane to project even number of pln (to correctly maintain int types)
        np.median(im[1:], axis=0, out=zproj)
    return zproj


def flat(im, dark, method='overall'):
    """Return a flat image:
        dark subtracted and normalized either at each plane 'single' or after median projection 'overall'
    """
    if not im.shape[1:] == dark.shape:
        sys.exit('flat images serie and dark image size mismatch')
    ims = im - dark
    if method == 'overall':
        flat = np.median(ims, axis=0)
        flat = im_median(flat)
        flat = flat / np.mean(flat)
    if method == 'single':
        ims = ims.astype(float)
        for i, im in enumerate(ims):
            ims[i] = im_median(im)
            ims[i] = ims[i] / np.mean(ims[i])
        flat = np.median(ims, axis=0)
    return flat
