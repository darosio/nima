#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
test_nimg
----------------------------------

Tests for `nimg` module.
"""

import unittest
import numpy as np

from nimg import nimg


class TestNimg(unittest.TestCase):

    def setUp(self):
        pass

    def tearDown(self):
        pass

    def test_000_something(self):
        pass

class TestScript(unittest.TestCase):

    def setUp(self):
        im = np.random.rand(20, 50)
        im[10, 25] = 10

        a = np.random.poisson(2, 1000)
        a = np.floor( np.random.normal(2.5,.6, 1000) + np.random.normal(1.3,.3, 1000) )
        a.reshape(20,50).astype('uint')

        pass

    def tearDown(self):
        pass

    def test_000_something(self):
        nimg.dark()
        pass

if __name__ == '__main__':
    import sys
    sys.exit(unittest.main())
