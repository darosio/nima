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

from docopt import docopt
from nimg import nimg
import os
from skimage import io

__version__ = "0.0.1"
__author__ = "Daniele Arosio"
__license__ = "MIT"


def main():
    '''Main entry point for the {{ cookiecutter.script_name }} CLI.'''
    args = docopt(__doc__, version=__version__)
    if args['<zipfile>']:
        dark_im, dark_hotpixels, f = nimg.dark(args['<zipfile>'])
        bname = 'dark-' + \
                os.path.splitext(os.path.basename(args['<zipfile>']))[0]
        f.savefig(bname + '.pdf')
        io.imsave(bname + '.png', dark_im)
        dark_hotpixels.to_csv(bname + '.csv')


if __name__ == '__main__':
    main()
