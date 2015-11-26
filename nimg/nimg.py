# -*- coding: utf-8 -*-

import numpy as np
from scipy import ndimage
import pandas as pd
import matplotlib.pyplot as plt

import zipfile
import sys
import os

import skimage
import skimage.transform
from skimage import io, filters
from skimage.morphology import disk


def im_print(im):
    print("ndim = ", im.ndim, "| shape = ", im.shape, '| max = ', im.max(),
          '| min = ', im.min(), "| size = ", im.size, "| dtype = ", im.dtype)


def myhist(im, bins=60, log=False, nf=0):
    # sns.set_style('ticks', {'axes.grid': True})
    # sns.pointplot(imf.flatten(), kde=False,
    #               hist_kws={"histtype": "step", "linewidth": 3})
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
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0,
                        left=0, right=1)


def plot_otsu(im, cm=plt.cm.gray):
    val = filters.threshold_otsu(im)
    mask = im > val
    plot_im_series(im * mask, cm=cm)
    return mask


def im_median(im,  radius=0,
              footprint=np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]])):
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
                imf[i] = ndimage.filters.median_filter(img,
                                                       footprint=footprint)
            return imf


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
    if len(im) % 2:
        np.median(im, axis=0, out=zproj)
    else:
        # to correctly maintain uint types
        # skip first plane to project even number of pln
        np.median(im[1:], axis=0, out=zproj)
    return zproj


def flat(im, dark, method='overall'):
    """
    Return a flat image:
    dark subtracted and normalized either:
    at each plane 'single' or
    after median projection 'overall'
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
    # Pdf output
    f = plt.figure(figsize=(6.75, 9.25))
    f.suptitle('FLAT stack')
    # table
    ax = plt.subplot2grid((6, 4), (0, 0), colspan=4)
    ax.set_axis_off()
    # FLAT
    ax0 = plt.subplot2grid((6, 4), (1, 0), colspan=3, rowspan=3)
    plt.axis('off')
    img0 = ax0.imshow(flat)
    plt.title('exported FLAT image')
    ax1 = plt.subplot2grid((6, 4), (1, 3), colspan=1, rowspan=3)
    plt.axis('off')
    plt.colorbar(img0, ax=ax1, fraction=0.9, shrink=0.78, aspect=14)
    #
    # hist flat
    with plt.style.context('seaborn-ticks'):
        plt.subplot2grid((6, 4), (4, 0), colspan=2, rowspan=2)
        plt.hist(flat.ravel(), bins=256, histtype='step', lw=2,
                 color='crimson')
        plt.yscale('log')
        plt.grid()
        plt.title('FLAT image')
    # hist stack
    with plt.style.context('seaborn-ticks'):
        plt.subplot2grid((6, 4), (4, 2), colspan=2, rowspan=2)
        plt.hist(im.ravel(), bins=256, histtype='step', lw=2, color='grey')
        plt.yscale('log')
        plt.ylim(.1,)
        plt.grid()
        plt.title('original stack')
    #
    plt.tight_layout()
    return flat, f, ax


def dark(fp, thr=95):
    """
    Read zip; median z-project; median filter(1).
    Return imf. Plot imf and its histogram.
    thr: float threshold for hot pixels calculation.
    """
    im = zipread(fp)
    zp = zproject_median(im)
    imf = im_median(zp)
    f = plt.figure(figsize=(6.75, 9.25))
    plt.suptitle('DARK stack')
    #
    with plt.style.context('seaborn-ticks'):
        plt.subplot(321)
        plt.hist(imf.ravel(), bins=256, histtype='step', lw=4)
        plt.yscale('log')
        plt.title('DARK image')
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
        plt.title('original stack')
    #
    plt.subplot(324)
    # hot pixels; cast to float because uint screwed up to range max
    d = imf.astype(float) - zp.astype(float)
    thr = np.std(d) * thr
    hot_pixels = np.nonzero(abs(d) > thr)
    df_hp = pd.DataFrame({'row': hot_pixels[0],
                          'col': hot_pixels[1],
                          'val': zp[hot_pixels]})
    plt.imshow(zp)
    plt.plot(hot_pixels[1], hot_pixels[0], 'r+', mfc='none', mec='w', ms=18)
    plt.colorbar()
    plt.axis('off')
    plt.title('projected stack')
    #
    plt.tight_layout()
    return imf, df_hp, f


def common_path(path1, path2):
    """
    Input: 2 file paths

    Output: a tupla for common abs path, path 1 and path2 (relative to common)
    """
    fcommon = os.path.commonprefix([os.path.abspath(path1),
                                    os.path.abspath(path2)])
    f1 = os.path.relpath(path1, start=fcommon)
    f2 = os.path.relpath(path2, start=fcommon)
    return fcommon, f1, f2


def read_tiff(fp, channel_list):
    """
    Read multichannel tif timelapse image.
    Return a tupla with n_channel

    d_im, n_channels, n_times = read_tiff_ch(
     '/home/dati/GBM_persson/data/15.02.05_cal-GBM5-pBJclop/ph725b/1_7_14.tif')
    """
    im = io.imread(fp)
    n_channels = len(channel_list)
    if len(im) % n_channels:
        raise Exception('n_channel mismatch total lenght of tif sequence')
    else:
        d_im = {}
        for i, ch in enumerate(channel_list):
            d_im[ch] = im[i::n_channels]
        return d_im, n_channels, len(im)//n_channels


def d_show(d_im, **kws):
    """
    im: tupla (d_im, n_channels, n_times)
    """
    MAX_ROWS = 9
    n_channels = len(d_im.keys())
    f = plt.figure(figsize=(16, 16))

    for n, ch in enumerate(sorted(d_im.keys())):
        n_times = len(d_im[ch])
        if n_times <= MAX_ROWS:
            rng = range(n_times)
            n_rows = n_times
        else:
            rng = range(0, n_times, n_times // (MAX_ROWS - 1))
            n_rows = MAX_ROWS
        for i, r in enumerate(rng):
            plt.subplot(n_rows, n_channels, i * n_channels + n + 1)
            img0 = plt.imshow(d_im[ch][r], **kws)
            plt.colorbar(img0, orientation='horizontal')
            plt.axis('off')
            plt.title(ch + ' @ t = ' + str(r))
    plt.subplots_adjust(wspace=0.02, hspace=0.02, top=1, bottom=0,
                        left=0, right=1)
    return f


# median
def d_median(d_im):
    d_out = {}
    for k, im in d_im.items():
        d_out[k] = im_median(im)
    return d_out


def d_shading(d_im, dark, flat, clip=True):
    '''Works either with flat or d_flat'''
    # TODO d_flat with checking
    d_cor = {}
    for k in d_im.keys():
        d_cor[k] = d_im[k].astype(float)
        d_cor[k] -= dark.astype(float)
        # TODO dark[k]
        try:
            d_cor[k] /= flat[k]
            # print('k')
        except:
            # print('n')
            d_cor[k] /= flat
    if clip:
        for k in d_cor.keys():
            d_cor[k] = d_cor[k].clip(0)
    return d_cor


def bg(im, kind='arcsinh', perc=10, radius=10,
        adaptive_radius=None, arcsinh_perc=80):
    '''
    Return median, whole vector, figures (in a [list])
    '''
    if adaptive_radius is None:
        adaptive_radius = im.shape[1]/2
    if perc < 0 or perc > 100:
        raise Exception("perc must be in [0, 100] range")
    else:
        perc /= 100
    if kind == 'arcsinh':
        lim = np.arcsinh(im)
        lim = ndimage.percentile_filter(lim, arcsinh_perc, size=radius)
        lim_ = True
        title = radius, perc
        thr = (1 - perc) * lim.min() + perc * lim.max()
        m = lim < thr
    elif kind == 'entropy':
        if im.dtype == float:
            lim = filters.rank.entropy(im/im.max(), disk(radius))
        else:
            lim = filters.rank.entropy(im, disk(radius))
        lim_ = True
        title = radius, perc
        thr = (1 - perc) * lim.min() + perc * lim.max()
        m = lim < thr
    elif kind == 'adaptive':
        lim_ = False
        title = adaptive_radius
        f = filters.threshold_adaptive(im, adaptive_radius)
        m = ~f
    elif kind == 'li_adaptive':
        lim_ = False
        title = adaptive_radius
        li = filters.threshold_li(im.copy())
        m = im < li
        # m = skimage.morphology.binary_erosion(m, disk(3))
        imm = im * m
        f = filters.threshold_adaptive(imm, adaptive_radius)
        m = ~f*m
    elif kind == 'li_li':
        lim_ = False
        title = None
        li = filters.threshold_li(im.copy())
        m = im < li
        # m = skimage.morphology.binary_erosion(m, disk(3))
        imm = im * m
        # To avoid zeros generated after first thesholding, clipping to the
        # min value of original image is needed before second thesholding.
        thr2 = filters.threshold_li(imm.clip(np.min(im)))
        m = im < thr2
        # ###mm = skimage.morphology.binary_closing(mm)
    # result = np.percentile(im[m], [25, 50, 75]), im[m].mean(), masked
    result = im[m]
    iqr = np.percentile(result, [25, 50, 75])
    #
    f1 = plt.figure(figsize=(9, 5))
    plt.subplot(121)
    masked = im * m
    img0 = plt.imshow(masked, cmap=plt.cm.inferno)
    plt.colorbar(img0, orientation='horizontal')
    plt.title(kind + ' ' + str(title) + '\n' + str(iqr))
    #
    plt.subplot(122)
    myhist(im[m], log=True)
    plt.tight_layout()

    if lim_:
        f2 = plt.figure(figsize=(14, 5))
        plt.subplot(131)
        img0 = plt.imshow(lim)
        plt.colorbar(img0, orientation='horizontal')
        #
        plt.subplot(132)
        myhist(lim)
        #
        host = plt.subplot(133)
        # plot bg vs. perc
        ave, sd, median = ([], [], [])
        delta = lim.max() - lim.min()
        delta /= 2
        rng = np.linspace(lim.min()+delta/20, lim.min()+delta, 20)
        par = host.twiny()
        # plt.make_patch_spines_invisible(par)
        # Second, show the right spine.
        par.spines["bottom"].set_visible(True)
        par.set_xlabel('perc')
        par.set_xlim(0, 0.5)
        par.grid()
        host.set_xlim(lim.min(), lim.min()+delta)
        p = np.linspace(.025, .5, 20)
        for t in rng:
            m = lim < t
            ave.append(im[m].mean())
            sd.append(im[m].std()/10)
            median.append(np.median(im[m]))
        host.plot(rng, median, "o")
        par.errorbar(p, ave, sd)
        return iqr[1], result, [f1, f2]
    else:
        return iqr[1], result, [f1]


def d_bg(d_im, downscale=None, kind='li_adaptive', clip=True, **kw):
    '''
    input: d_im
    out: d_im - bg, bg_df (median), [figs], d_bg_values
    '''
    d_bg = {}
    d_bg_values = {}
    d_cor = {}
    d_fig = {}
    for k in d_im.keys():
        d_bg[k] = []
        d_bg_values[k] = []
        d_cor[k] = []
        d_fig[k] = []
        for t, im in enumerate(d_im[k]):
            if downscale:
                im = skimage.transform.downscale_local_mean(im, downscale)
            med, v, ff = bg(im, kind, **kw)
            d_bg[k].append(med)
            d_bg_values[k].append(v)
            d_cor[k].append(d_im[k][t] - med)
            d_fig[k].append(ff)
        d_cor[k] = np.array(d_cor[k])
    if clip:
        for k in d_cor.keys():
            d_cor[k] = d_cor[k].clip(0)
    return d_cor, d_bg, d_fig, d_bg_values
