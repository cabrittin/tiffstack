#!/opt/stack/bin/python
"""
@name: viewer.py                         
@description:                  
    Functions for viewing tiffstacks

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
import os
import argparse
from inspect import getmembers,isfunction
from tqdm import tqdm

from tiffstack.viewer import image_looper

def timelapse(args):
    from tiffstack.datasets import TimeLapseFast
    T = TimeLapseFast(args.fin)
    T.preprocess() 
    T.init_window() 
    image_looper(T)

def timelapsemax(args):
    from tiffstack.datasets import TimeLapseMax
    T = TimeLapseMax(args.fin)
    T.preprocess()
    T.init_window() 
    image_looper(T)

def ndtiff(args):
    from tiffstack.datasets import NDTiff
    T = NDTiff(args.fin[0])
    T.preprocess() 
    T.init_window()
    image_looper(T)

def ndtiff_max(args):
    from tiffstack.datasets import NDTiffMax 
    T = NDTiffMax(args.fin[0])
    T.preprocess() 
    T.init_window()
    image_looper(T)

def ndtiff_track(args):
    from tiffstack.datasets import NDTiffTrack
    T = NDTiffTrack(args.fin[0])
    T.preprocess() 
    T.init_window()
    T.init_objects() 
    image_looper(T)

def ls_sequence(args):
    from tiffstack.datasets import TimeLapse
    T = TimeLapse(args.fin)
    for s in T.S.iter_stacks(): print(s)

def ndtiff_to_tiff(args):
    from tiffstack.datasets import NDTiff
    from tifffile import imwrite
    import numpy as np
    T = NDTiff(args.fin[0])
    
    print(f'Saving {args.fin[0]} to {args.fout}')
    data = np.zeros((T.sequence_size,T.stack_size,2,T.shape[0],T.shape[1]),'uint16')
    #for i in tqdm(range(T.sequence_size),desc='Sequence processed'):
    for i in tqdm(range(2),desc='Sequence processed'):
        for k in range(2): 
            for j in range(T.stack_size):
                data[i,j,k,:,:]  = np.array(T.A[i,k,j,:,:])
            
    imwrite(args.fout,data)

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('fin',
                action = 'store',
                nargs = '*',
                help = 'Path to input file(s). Use * for multiple files or consult your OS')
    
    parser.add_argument('-o','--output',
                dest = 'fout',
                action = 'store',
                default = None,
                required = False,
                help = 'Path to input file(s). Use * for multiple files or consult your OS')
    
    args = parser.parse_args()
    eval(args.mode + '(args)')


