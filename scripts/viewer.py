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
    from tiffstack.datasets import NDTiffZ as NDTiff
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

def ndtiff_to_tiff_split(args):
    from tiffstack.datasets import NDTiffZ as NDTiff
    from tifffile import imwrite
    import numpy as np
    T = NDTiff(args.fin[0])
    z_shift = 4 
    split_col = T.shape[1]
    print(f'Saving {args.fin[0]} to {args.fout}')
    data = np.zeros((T.stack_size,T.shape[0],2*T.shape[1]),'uint16')
    for i in tqdm(range(T.sequence_size),desc='Sequence processed'):
        for j in range(T.stack_size):
            _jdx = (j + z_shift) % T.stack_size
            for k in range(2): 
                if k == 1: 
                    data[j,:,split_col:]  = np.fliplr(np.array(T.A[i,k,_jdx,:,:]))
                else: 
                    data[j,:,:split_col]  = np.array(T.A[i,k,_jdx,:,:])
        
        _fout = args.fout.split('.')
        fout = ''.join([_fout[0]] + [f'_t{i}.'] + [_fout[1]])
        imwrite(fout,data)


def ndtiff_to_tiff(args):
    from tiffstack.datasets import NDTiff
    from tifffile import imwrite
    import numpy as np
    T = NDTiff(args.fin[0])
    
    print(f'Saving {args.fin[0]} to {args.fout}')
    data = np.zeros((T.sequence_size,T.stack_size,2,T.shape[0],T.shape[1]),'uint16')
    for i in tqdm(range(T.sequence_size),desc='Sequence processed'):
    #for i in tqdm(range(2),desc='Sequence processed'):
        for k in range(2): 
            for j in range(T.stack_size):
                if k == 1: 
                    data[i,j,k,:,:]  = np.fliplr(np.array(T.A[i,k,j,:,:]))
                else: 
                    data[i,j,k,:,:]  = np.array(T.A[i,k,j,:,:])
            
    imwrite(args.fout,data)

def ndtiffmax_to_tiff(args):
    from tiffstack.datasets import NDTiff
    from tifffile import imwrite
    import numpy as np
    T = NDTiff(args.fin[0])
    
    print(f'Saving {args.fin[0]} to {args.fout}')
    data = np.zeros((T.sequence_size,1,2,T.shape[0],T.shape[1]),'uint16')
    for i in tqdm(range(T.sequence_size),desc='Sequence processed'):
    #for i in tqdm(range(2),desc='Sequence processed'):
        for k in range(2): 
            if k == 1: 
                data[i,0,k,:,:]  = np.fliplr(np.array(T.A[i,k,:,:,:].max(axis=0)))
            else:
                data[i,0,k,:,:]  = np.array(T.A[i,k,:,:,:].max(axis=0))
            
    imwrite(args.fout,data)

def ndtiffmax_to_gif(args):
    from tiffstack.datasets import NDTiff,compute_lut
    import imageio
    import numpy as np
    import cv2

    T = NDTiff(args.fin[0])
    T.preprocess()
    T.pxlut = compute_lut(567,3289)
    k = 0 
    print(f'Saving {args.fin[0]} to {args.fout}')
    data = [] 
    for i in tqdm(range(T.sequence_size),desc='Sequence processed'):
    #for i in tqdm(range(2),desc='Sequence processed'):
        frame = T.map_uint16_to_uint8(np.array(T.A[i,k,:,:,:].max(axis=0)))
        data.append(cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB))


    imageio.mimsave(args.fout,data,fps=1)

   

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


