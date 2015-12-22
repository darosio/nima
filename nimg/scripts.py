from docopt import docopt
from nimg import nimg
import os
import sys
from skimage import io
import pandas as pd
import matplotlib as mpl
mpl.rcParams['figure.max_open_warning'] = 99

import matplotlib.pyplot as plt
from scipy import ndimage
import tifffile
# import numpy as np

__version__ = "0.0.3"
__author__ = "Daniele Arosio"
__license__ = "MIT"


def main():
    '''
    Analyze a multichannel tiff stack. ([G, R, C, G2] in timelapse)
    create bg folder and save:
    - bg plots and dataframe.
    in the current dir.

    Usage:
      nimg [options] bg -m=METHOD TIFFSTK [(-d DARK... -f FLAT...)] CH_NAMES...
      nimg -h | --help
      nimg --version

    Options:
      -h --help     Show this screen.
      --version     Show version.
      --silent      Do not print; verbose=0.
      --downscale X,Y           X and Y comma separeted (no spaces) for optional downscaling.
      -m MTH, --method MTH      Method for bg estimation - must be indicated.
                                Available method: entropy, arcsinh, adaptive, li_adaptive, li_li.
      --hotpixel, --hotpixels   Execute median filter with radius=0.5 to remove hotpixel.
      -d DARK, --dark DARK      Dark for shading correction.
      -f FLAT, --flat FLAT      FLAT for shading correction.

      --radius R                Radius for entropy or arcsinh methods.
      --adaptive-radius R       Radius for adaptive methods.
      --percentile P            Percentile for entropy or arcsinh methods.
      --percentile-filter P     Percentile filter for arcsinh method.

      --threshold_method MTH    Method for fg estimation - must be indicated.
                                Available method: li, 
      --min_size PIXELS         
      --wiener
      --watershed
      --randomwalk


    '''
    methods_bg = ('entropy', 'arcsinh', 'adaptive', 'li_adaptive', 'li_li')
    args = docopt(main.__doc__, version=__version__)
    # print(args)

    if args['bg']:
        # method
        method = args['--method']
        if method not in methods_bg:
            print(main.__doc__)
            print("METHOD must be one of: ", methods_bg)
            sys.exit()
        downscale = args['--downscale']
        if downscale:
            downscale = downscale.split(',')
            downscale = int(downscale[0]), int(downscale[1])
        # INPUT
        channel_list = args['CH_NAMES']
        d_im, _, t = nimg.read_tiff(args['TIFFSTK'], channel_list)
        if not args['--silent']:
            print("  File: ", args['TIFFSTK'])
            print("  Channels: ", channel_list)
            print("  Times: ", t)
        if args['--hotpixels']:
            d_im = nimg.d_median(d_im)
        if args['--flat']:
            dark = io.imread(args['--dark'][0])
            flat = io.imread(args['--flat'][0])  # FIXME works only single flat
            d_im = nimg.d_shading(d_im, dark, flat, clip=True)

        # method and their kwargs
        kwargs = {}
        if args['--radius']:
            kwargs['radius'] = float(args['--radius'])
        if args['--adaptive-radius']:
            kwargs['adaptive_radius'] = float(args['--adaptive-radius'])
        if args['--percentile']:
            kwargs['perc'] = float(args['--percentile'])
        if args['--percentile-filter']:
            kwargs['arcsinh_perc'] = float(args['--percentile-filter'])
        d_im_bg, bgs, ff, _bgv = nimg.d_bg(d_im, downscale=downscale,
                                           kind=method, **kwargs)
        # output plots
        bname = os.path.basename(args['TIFFSTK'])
        bname = os.path.splitext(bname)[0]
        #bname = os.path.join('bg', bname)
        bname = os.path.join('nimg', bname)
        # if not os.path.exists(bname):
            # os.makedirs(bname)
        bname_bg = os.path.join(bname, "bg")
        if not os.path.exists(bname_bg):
            os.makedirs(bname_bg)
        for k in ff.keys():
            for t, f in enumerate(ff[k]):
                fname = method + '-' + k + '-t' + str(t) + '.png'
                f[0].savefig(os.path.join(bname_bg, fname))
                if len(f) == 2:
                    fname1 = method + '1-' + k + '-t' + str(t) + '.png'
                    f[1].savefig(os.path.join(bname_bg, fname1))
        bgs.to_csv(bname_bg + '.csv')
        # TODO: plt.close('all') or control mpl warning
        if not args['--silent']:
            # print(bgs)
            pass
        ## dim
        # i got a problem with 'li' and unique label for 19 1.10_15 af16 ds
        nimg.d_mask_label(d_im_bg, channels=channel_list, threshold_method='yen',
                  min_size=12000, watershed=False, randomwalk=False)

        # show all channels and labels only.
        d = {ch: d_im_bg[ch] for ch in channel_list}
        d['labels'] = d_im_bg['labels']
        f = nimg.d_show(d, cmap=plt.cm.inferno_r)
        f.savefig(bname + '_dim.png')
        ## meas
        # TODO channels_cl and channels_pH in input
        meas, pr = nimg.d_meas_props(d_im_bg, channels=channel_list,
                channels_cl=[channel_list[2], channel_list[1]],
                channels_pH=[channel_list[0], channel_list[2]])
        # TODO channel_list --> channels and add optparse for channels_pH
        # channels_cl
        #meas3, pr3 = nimg.d_meas_props(dim3, channels=['C', 'G', 'G2', 'R'], channels_pH=['G2', 'C'] )
        ##
        # f, f_ch = nimg.d_plot_meas(meas, channels=channel_list)
        # f.savefig(bname + '_meas.png')
        # f_ch.savefig(bname + '_meas_ch.png')
        f = nimg.d_plot_meas(bgs, meas, channels=channel_list)
        f.savefig(bname + '_meas.png')

        # fig = nimg.d_plot_bg_ratio_ch(meas, channels=channel_list)
        # fig.savefig(bname + '_meas.png')
        ## meas csv
        for k, df in meas.items():
            df.to_csv(os.path.join(bname, "label" + str(k) + ".csv"))
        ## labelX_{rcl,rpH}.tif ### require r_cl and r_pH present in d_im
        objs = ndimage.find_objects(d_im_bg['labels'])
        for k, o in enumerate(objs):
            name = os.path.join(bname, "label" + str(k + 1) + "_rcl.tif")
            tifffile.imsave(name, d_im_bg['r_cl'][o], compress=9)
            name = os.path.join(bname, "label" + str(k + 1) + "_rpH.tif")
            tifffile.imsave(name, d_im_bg['r_pH'][o], compress=9)


def dark():
    '''
    Read a stack of dark images (tiff-zip) and save:
    - plot (histograms, median, projection, hot pixels)
    - a single DARK image i.e. median filter of median projection
    - txt file containing coordinates of hotpixels
    in the current dir.

    Usage:
      dark <zipfile>
      dark -h | --help
      dark --version

    Options:
      -h --help     Show this screen.
      --version     Show version.
    '''
    args = docopt(dark.__doc__, version=__version__)
    if args['<zipfile>']:
        dark_im, dark_hotpixels, f = nimg.dark(args['<zipfile>'])
        bname = 'dark-' + \
                os.path.splitext(os.path.basename(args['<zipfile>']))[0]
        f.savefig(bname + '.pdf')
        io.imsave(bname + '.tif', dark_im, plugin='tifffile')
        dark_hotpixels.to_csv(bname + '.csv')


def flat():
    '''
    Read a stack of flat images (tiff-zip) and a dark reference image.
    Save:
    - plot (stack histograms, flat image and its histogram - name of stack and
      reference dark image)
    - a single FLAT image i.e. median filter of median projection
    in the current dir.

    Usage:
      flat <zipfile> <darkfile>
      flat -h | --help
      flat --version

    Options:
      -h --help     Show this screen.
      --version     Show version.
    '''
    args = docopt(flat.__doc__, version=__version__)
    if args['<zipfile>']:
        # read files
        im = nimg.zipread(args['<zipfile>'])
        dark = io.imread(args['<darkfile>'], plugin='tifffile')
        # computation
        flat_im, f, ax = nimg.flat(im, dark)
        # output
        bname = 'flat-' + \
                os.path.splitext(os.path.basename(args['<zipfile>']))[0] + \
                '-' + \
                os.path.splitext(os.path.basename(args['<darkfile>']))[0]
        fdark = args['<darkfile>']
        fflat = args['<zipfile>']
        fcommon, fdark, fflat = nimg.common_path(fdark, fflat)
        data = pd.Series([fcommon, fdark, fflat], name='Files',
                         index=['root', 'dark', 'flat'])
        # http://nipunbatra.github.io/2014/08/latexify/
        params = {
                  'axes.labelsize': 8,  # fontsize for x and y labels (was 10)
                  'axes.titlesize': 8, 'text.fontsize': 8,  # was 10
                  'legend.fontsize': 8,  # was 10
                  'xtick.labelsize': 8,
                  'ytick.labelsize': 8,
                  'font.family': 'serif'}
        # 'backend': 'ps',
        # 'text.latex.preamble': ['\usepackage{gensymb}'],
        # 'text.usetex': True,
        # 'figure.figsize': [fig_width,fig_height],
        mpl.rcParams.update(params)
        pd.tools.plotting.table(ax=ax, data=data, loc=3)
        f.savefig(bname + '.pdf')
        mpl.rcdefaults()
        io.imsave(bname + '.tif', flat_im, plugin='tifffile')


if __name__ == '__main__':
    pass
