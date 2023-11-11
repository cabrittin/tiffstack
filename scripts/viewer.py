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

from tiffstack.viewer import Stack,image_looper
from tiffstack.loader import Session
from tiffstack import loader

class TimeLapse(Stack):
    def __init__(self,*args,**kwargs):
        super(Stack,self).__init__()
        self.path = args[0] 
        self.S = Session(args[0])
        self.sequence_size = len(self.S)
        self.stack_size = self.S.depth
        
        self.jdx = 0
        self.idx = 0
        
        self.wtitle = 'Time point %d/%d ::: Z %d/%d'
        self.win ='Volume'
        cv2.namedWindow(self.win)
        cv2.moveWindow(self.win,800,500)
        self.update_title()
   
        self.pxmin = 0
        self.pxmax = 0
        self.MAX_PX = 2**16

        self.preprocess()

    def update_title(self):
        wtitle = self.wtitle%(self.jdx,self.sequence_size,self.idx,self.stack_size)
        cv2.setWindowTitle(self.win,wtitle)
 
    def load_stack(self,jdx):
        self.jdx = jdx 
        self.stack = loader.tif_from_stack(self.S.get_stack(jdx))
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
        for stack in tqdm(self.S.iter_stacks(),total=self.sequence_size,desc="Scaling pixels"):
            tif = loader.tif_from_stack(stack)

            for page in tif.pages:
                frame = page.asarray()
                self.pxmin = min(self.pxmin,frame.min())
                self.pxmax = max(self.pxmax,frame.max())
        
        self.compute_lut()

    def compute_lut(self):
        self.pxlut = np.concatenate([
            np.zeros(self.pxmin, dtype=np.uint16),
            np.linspace(0,255,self.pxmax - self.pxmin).astype(np.uint16),
            np.ones(2**16 - self.pxmax, dtype=np.uint16) * 255
            ])

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
            self.compute_lut()
            update_display(self)
        
        elif key == ord('t'):
            self.pxmax = min(self.MAX_PX,self.pxmax+100)
            self.compute_lut()
            update_display(self)
        
        elif key == ord('v'):
            self.pxmax = max(self.pxmin,self.pxmax-1)
            self.compute_lut()
            update_display(self)
        
        elif key == ord('r'):
            self.pxmax = min(self.MAX_PX,self.pxmax+1)
            self.compute_lut()
            update_display(self)
 
        elif key == ord('w'):
            self.pxmin = min(self.pxmax,self.pxmin+100)
            self.compute_lut()
            update_display(self)
        
        elif key == ord('x'):
            self.pxmin = max(0,self.pxmin-100)
            self.compute_lut()
            update_display(self)
        
        elif key == ord('e'):
            self.pxmin = min(self.pxmax,self.pxmin+1)
            self.compute_lut()
            update_display(self)
        
        elif key == ord('c'):
            self.pxmin = max(0,self.pxmin-1)
            self.compute_lut()
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



def update_display(T):
    cout = f'''
        Viewing {T.path} 

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
    assert args.fin is not None, "You must provide an input file (-i)"
    T = TimeLapse(args.fin)
    image_looper(T)

def timelapsemax(args):
    assert args.fin is not None, "You must provide an input file (-i)"
    T = TimeLapseMax(args.fin)
    image_looper(T)



if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('-d','--input_dir',
                action = 'store',
                dest = 'din',
                required = False,
                default = None,
                help = 'Path to Input directory')
    
    parser.add_argument('-i','--input',
                action = 'store',
                dest = 'fin',
                required = False,
                default = None,
                type=str,
                help = 'Path to input file(s). Use * for multiple files.')
    
    args = parser.parse_args()
    eval(args.mode + '(args)')


