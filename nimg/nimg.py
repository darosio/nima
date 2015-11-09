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
    Default footprint is equivalent to skimage.morphology.disk(1) (= 0.5 in
    Fiji). 
    plane by plane with e.g. radius=3 calculate 3D median filter.
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


def dark(fp, thr=95):
    """
    Read zip; median z-project; median filter(1).
    Return imf. Plot imf and its histogram.
    thr: float threshold for hot pixels calculation.
    """
    #sns.pointplot(imf.flatten(), kde=False, hist_kws={"histtype": "step", "linewidth": 3})
    #myhist(imf, log=True)sns.set_style('ticks', {'axes.grid': True})
    im = zipread(fp)
    zp = zproject_median(im)
    imf = im_median(zp)
    f = plt.figure(figsize=(6.75, 9.25))
    plt.suptitle('DARK stack')
    #
    #sns.set_style('ticks', {'axes.grid': True})
    with plt.style.context('seaborn-ticks'): 
        plt.subplot(321)
        plt.hist(imf.ravel(), bins=256, histtype='step', lw=4)
        plt.yscale('log')
        plt.title('histogram of calculated DARK')
    #
    plt.subplot(322)
    plt.imshow(imf, cmap=plt.cm.inferno_r)
    plt.colorbar()
    plt.axis('off')
    plt.title('exported DARK image')
    #
    with plt.style.context('seaborn-ticks'):
        plt.subplot(323)
        plt.hist(im.ravel(), bins=256, histtype='step', lw=4)
        plt.yscale('log')
        plt.title('histogram of the whole stack')
    #
    plt.subplot(324)
    # hot pixels; cast to float because uint screwed up to range max
    d = imf.astype(float) - zp.astype(float)
    thr = np.std(d) * thr
    hot_pixels = np.nonzero(abs(d)>thr)
    df_hp = pd.DataFrame({'row': hot_pixels[0],
                          'col': hot_pixels[1],
                          'val': zp[hot_pixels]})
    plt.imshow(zp)
    plt.plot(hot_pixels[1],hot_pixels[0],'r+',mfc='none',mec='w',ms=18)
    plt.colorbar()
    plt.axis('off')
    plt.title('projected stack')
    #
    return imf, df_hp, f

