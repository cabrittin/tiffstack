"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction


import tiffstack.loader as loader
from tiffstack.loader import Loader
from tiffstack.sequence import Sequence
import tiffstack.preprocess as pp

CONFIG = 'config/config.ini'

def set_roi(args):
    L = Loader(args.config,args.dir)
    path = loader.format_path(L)
    stacks = loader.stacks_from_path(path) 
    S = Sequence(stacks)
    pp.set_roi(S) 



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('dir',
                        action = 'store',
                        help = 'Directory path')


    parser.add_argument('-c','--config',
                dest = 'config',
                action = 'store',
                default = CONFIG,
                required = False,
                help = 'Config file')

    args = parser.parse_args()
    eval(args.mode + '(args)')

