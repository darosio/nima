"""
Tests for nimg script.
"""
import os
# from nimg import script
# script.main()

# test data
TestFile = './data/1b_c16_15.tif'


def test():
    cmd_line = 'nimg ' + TestFile + ' G R C'
    os.system(cmd_line)


if __name__ == '__main__':
    test()
