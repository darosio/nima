'''dark

Usage:
  dark ship new <name>...
  dark ship <name> move <x> <y> [--speed=<kn>]
  dark ship shoot <x> <y>
  dark mine (set|remove) <x> <y> [--moored|--drifting]
  dark -h | --help
  dark --version

Options:
  -h --help     Show this screen.
  --version     Show version.
  --speed=<kn>  Speed in knots [default: 10].
  --moored      Moored (anchored) mine.
  --drifting    Drifting mine.
'''

#from __future__ import unicode_literals, print_function
from docopt import docopt

__version__ = "{{ cookiecutter.version }}"
__author__ = "{{ cookiecutter.full_name }}"
__license__ = "MIT"


def main():
    '''Main entry point for the {{ cookiecutter.script_name }} CLI.'''
    args = docopt(__doc__, version=__version__)
    print(args)

if __name__ == '__main__':
    main()
