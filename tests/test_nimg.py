"""
Tests for nimg module.
"""
# import pytest
# from nimg import nimg
import numpy as np
from numpy.testing import assert_array_equal
from docopt import docopt
# from numpy.testing import assert_array_equal, assert_allclose
import nimg.scripts
import nimg



class Test_script(object):
    def test_nimg(self):
        TestFile = './data/1b_c16_15.tif'
        # args = docopt(nimg.scripts.__doc__, argv=[TestFile, 'G R C', '-m', 'li_adaptive'])
        args = docopt(nimg.scripts.__doc__, argv=['nimg', 'flat', '--version'])
        # assert args == '0.1.1'
        print(args)

class Test_d_shading(object):

    @classmethod
    def setup_class(cls):
        cls.d_im = {'C': np.ones((5, 5, 5)) * 2, 'C2': np.ones((5, 5, 5)) * 4}
        cls.dark = np.ones((5, 5))
        cls.flat = np.ones((5, 5)) * 2
        cls.d_flat = {'C': cls.flat, 'C2': np.ones((5, 5)) * 3}

    def test_single_dark_and_single_flat(self):
        d_cor = nimg.d_shading(self.d_im, self.dark, self.flat, clip=True)
        # assert_allclose(d_cor, np.ones((5,5,5)) / 2)
        assert_array_equal(d_cor['C'], np.ones((5, 5, 5)) / 2)

    def test_single_dark_and_d_flat(self):
        d_cor = nimg.d_shading(self.d_im, self.dark, self.d_flat, clip=True)
        # assert_allclose(d_cor, np.ones((5,5,5)) / 2)
        assert_array_equal(d_cor['C'], np.ones((5, 5, 5)) / 2)
        assert_array_equal(d_cor['C2'], np.ones((5, 5, 5)))
