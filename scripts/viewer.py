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
import cv2
import glob
import re
from tqdm import tqdm
import numpy as np
from concurrent.futures.process import ProcessPoolExecutor
import concurrent.futures
import multiprocess as mp
from random import sample

from tiffstack.viewer import Stack,image_looper
from tiffstack.loader import Session
from tiffstack import loader

MAX_PIXEL = 2**16

class TimeLapse(Stack):
    def __init__(self,*args,**kwargs):
        super(Stack,self).__init__()
        self.path = args[0] 
        if isinstance(self.path,list): self.path = self.path[0]

        self.S = Session(args[0])
        self.sequence_size = len(self.S)
        self.stack_size = self.S.depth
        
        self.jdx = 0
        self.idx = 0
   
        self.pxmin = 0
        self.pxmax = 0

        self.lut_checks = [0]*len(self.S)
    
    def init_window(self):
        self.wtitle = 'Time point %d/%d ::: Z %d/%d'
        self.win ='Volume'
        cv2.namedWindow(self.win)
        cv2.moveWindow(self.win,800,500)
        self.update_title()

    def update_title(self):
        wtitle = self.wtitle%(self.jdx,self.sequence_size,self.idx,self.stack_size)
        cv2.setWindowTitle(self.win,wtitle)
 
    def load_stack(self,jdx):
        self.jdx = jdx 
        stack = self.S.get_stack(jdx)
        #Check if LUT needs to be updated
        if self.lut_checks[jdx] == 0:
            px = get_min_max(stack)
            update_px_range = False
            if px[0] < self.pxmin:
                self.pxmin = px[0]
                update_px_range = True
            if px[1] > self.pxmax:
                self.pxmax = px[1]
                update_px_range = True
            if update_px_range: 
                self.pxlut = compute_lut(self.pxmin,self.pxmax)
            self.lut_checks[jdx] = 1

        self.stack = loader.tif_from_stack(stack)
        update_display(self)

    def display(self,idx):
        self.idx = idx 
        image = self.stack.pages[idx].asarray()
        #image = loader.array_16bit_to_8bit(image) 
        image = self.map_uint16_to_uint8(image) 
        self.update_title() 
        return image

    def preprocess(self):
        """
        Makes lookup table to convert 16-bit image to 8-bit image 
        """
        cpu_count = mp.cpu_count() 
        rstacks = sample(self.S.stacks,min(len(self.S),cpu_count))
        
        futures = []
        with ProcessPoolExecutor(max_workers=cpu_count) as executor: 
            for (idx,stack) in enumerate(rstacks):
                futures.append(executor.submit(get_min_max,stack))
        
        futures, _ = concurrent.futures.wait(futures)
        
        self.pxmin = min([f.result()[0] for f in futures])
        self.pxmax = max([f.result()[1] for f in futures])
        
        self.pxlut = compute_lut(self.pxmin,self.pxmax)

    def map_uint16_to_uint8(self,img):
        """
        Maps image from uint16 to uint8
        """
        return self.pxlut[img].astype(np.uint8)
    

    def user_update(self,key,sequence_jdx,stack_idx):
        jdx = sequence_jdx
        idx = stack_idx

        if key == ord('b'):
            self.pxmax = max(self.pxmin,self.pxmax-100)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('t'):
            self.pxmax = min(MAX_PIXEL,self.pxmax+100)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('v'):
            self.pxmax = max(self.pxmin,self.pxmax-1)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('r'):
            self.pxmax = min(MAX_PIXEL,self.pxmax+1)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
 
        elif key == ord('w'):
            self.pxmin = min(self.pxmax,self.pxmin+100)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('x'):
            self.pxmin = max(0,self.pxmin-100)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('e'):
            self.pxmin = min(self.pxmax,self.pxmin+1)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)
        
        elif key == ord('c'):
            self.pxmin = max(0,self.pxmin-1)
            self.pxlut = compute_lut(self.pxmin,self.pxmax)
            update_display(self)

class TimeLapseMax(TimeLapse):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.stack_size = 1
        self.Z = np.zeros((self.S.depth,self.S.height,self.S.width))

    def display(self,idx):
        self.idx = idx 
        
        for (zdx,page) in enumerate(self.stack.pages):
            image = page.asarray()
            #image = self.map_uint16_to_uint8(image) 
            self.Z[zdx,:,:] = image 

        image = self.Z.max(axis=0).astype(np.uint)
        image = self.map_uint16_to_uint8(image) 
        self.update_title() 
        return image

def get_min_max(stack):
    pxmin = MAX_PIXEL
    pxmax = 0
    tif = loader.tif_from_stack(stack)
    for page in tif.pages:
        frame = page.asarray()
        pxmin = min(pxmin,frame.min())
        pxmax = max(pxmax,frame.max())
    return (pxmin,pxmax)

def compute_lut(pxmin,pxmax):
    pxlut = np.concatenate([
        np.zeros(pxmin, dtype=np.uint16),
        np.linspace(0,255,pxmax - pxmin).astype(np.uint16),
        np.ones(MAX_PIXEL - pxmax, dtype=np.uint16) * 255
        ])
    
    return pxlut

def update_display(T):
    cout = f'''
        Viewing {T.path} 
        Sequence length: {T.sequence_size}

        Keys:
        -----
        Quit: q 

        Next (prev) timepoint: l(h)
        Next (prev) z-slice: k(j)
        
        Raise max pixel +100 (+1): t(r)
        Lower max pixel -100 (-1): b(v)
        Raise min pixel +100 (+1): w(e)
        Lower min pixel -100 (-1): x(c)

        Timelapse:
        ----------
        Max pixel = {T.pxmax} 
        Min pixel = {T.pxmin} 
        
        '''
 
    LINE_UP = '\033[1A'
    LINE_CLEAR = '\x1b[2K'
    
    loop = range(1, len(cout) + 1)
    for idx in reversed(loop): print(LINE_UP, end=LINE_CLEAR)

    print(cout,end='\r')


def timelapse(args):
    T = TimeLapse(args.fin)
    T.preprocess() 
    T.init_window() 
    image_looper(T)

def timelapsemax(args):
    T = TimeLapseMax(args.fin)
    T.preprocess()
    T.init_window() 
    image_looper(T)

def ls_sequence(args):
    T = TimeLapse(args.fin)
    for s in T.S.iter_stacks(): print(s)

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
    
    
    args = parser.parse_args()
    eval(args.mode + '(args)')


