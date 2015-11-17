from docopt import docopt
from nimg import nimg
import os
from skimage import io
import pandas as pd
import matplotlib as mpl

__version__ = "0.0.2"
__author__ = "Daniele Arosio"
__license__ = "MIT"


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
