"""
@name: tiffstack.datasets.py
@description:                  
    Classes for formatting datasets for viewing.

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 
"""

import cv2
import re
from tqdm import tqdm
import numpy as np
import multiprocess as mp
from multiprocessing import RawArray
from concurrent.futures.process import ProcessPoolExecutor
import concurrent.futures
from random import sample
import time
from ndstorage.ndtiff_index import NDTiffIndexEntry
from ndstorage import Dataset 

from tiffstack.viewer import Stack,image_looper
from tiffstack.loader import Session
from tiffstack import loader

MAX_PIXEL = 2**16 - 1

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

def stack_to_array(stacks,dims,label=None):
    k = len(stacks) 
    V = np.zeros((dims[0],dims[1],dims[2],k),dtype=np.uint16)
    for i in tqdm(range(k),desc='Num stacks loaded'):
       stack = loader.tif_from_stack(stacks[i])
       for j in range(dims[2]):
           V[:,:,j,i] = stack.pages[j].asarray()
    
    if label is not None: V = (label,V)
    return V

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

class TimeLapseFast(Stack):
    def __init__(self,*args,**kwargs):
        super(Stack,self).__init__()
        self.path = args[0] 
        if isinstance(self.path,list): self.path = self.path[0]

        self.S = Session(args[0])
        self.sequence_size = len(self.S)
        self.stack_size = self.S.depth
        self.shape = (self.S.height,self.S.width)

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
        update_display(self)

    def display(self,idx):
        self.idx = idx 
        image = self.V[:,:,self.idx,self.jdx] 
        #image = self.stack.pages[idx].asarray()
        #image = loader.array_16bit_to_8bit(image) 
        image = self.map_uint16_to_uint8(image) 
        self.update_title() 
        return image

    def _preprocess(self):
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
    
    def preprocess(self):
        cpu_count = mp.cpu_count()//4
        stacks = [self.S.get_stack(i) for i in range(self.sequence_size)]
        dims = (self.S.height,self.S.width,self.S.depth)
        futures = []
        with ProcessPoolExecutor(max_workers=cpu_count) as executor: 
            for (idx,_stacks) in enumerate(loader.split_n(stacks,cpu_count)):
                futures.append(executor.submit(stack_to_array,_stacks,dims,idx))
    
        futures, _ = concurrent.futures.wait(futures)
        futures = sorted(futures, key=lambda f: f.result()[0])
        
        
        print('Building 4D array....')
        start = time.time()
        V = np.concatenate(tuple([f.result()[1] for f in futures]),axis=3)
        self.pxmin = V.min()
        self.pxmax = V.max()
        self.pxlut = compute_lut(self.pxmin,self.pxmax)
        self.V = self.map_uint16_to_uint8(V)
        print(f'Time to build: {time.time() - start: .2f}') 

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


class NDTiff():
    def __init__(self,*args,**kwargs):
        self.path = args[0] 
        self.A = Dataset(args[0]).as_array()
        self.sequence_size = self.A.shape[0]
        self.stack_size = self.A.shape[2]
        self.shape = (self.A.shape[3],self.A.shape[4])
        self.n_channels = self.A.shape[1]
         
        self.z_shift = 6
        self.jdx = 0
        self.idx = 0
        self.cdx = 0

        self.flip_channel = True
    
    def get_start_indicies(self):
        return self.jdx,self.idx

    def init_window(self):
        self.wtitle = 'Time point %d/%d ::: Z %d/%d ::: Channel %d/%d'
        self.win ='Volume'
        cv2.namedWindow(self.win)
        cv2.moveWindow(self.win,800,500)
        self.update_title()

    def update_title(self):
        wtitle = self.wtitle%(self.jdx,self.sequence_size,self.idx,self.stack_size,self.cdx,self.n_channels-1)
        cv2.setWindowTitle(self.win,wtitle)
    
    def load_stack(self,jdx):
        self.jdx = jdx

    def display(self,idx):
        self.idx = idx 
        self.update_title()
        _idx = (self.idx + self.z_shift) % self.stack_size
        #img  = np.array(self.A[self.jdx,self.cdx,self.idx,:,:])
        img  = np.array(self.A[self.jdx,self.cdx,_idx,:,:])
        if self.flip_channel and self.cdx == 1: img = np.fliplr(img) 

        return self.map_uint16_to_uint8(img)
    
    def preprocess(self):
        self.pxmin = int(self.A[0,0,:,:,:].min())
        self.pxmax= int(self.A[0,0,:,:,:].max())
        self.pxlut = compute_lut(self.pxmin,self.pxmax)

    def get_sequence_size(self):
        return self.sequence_size

    def get_stack_size(self):
        return self.stack_size
    
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

        elif key == ord('i'):
            self.cdx = (self.cdx+1) % self.n_channels
            update_display(self)
        
        elif key == ord('r'):
            update_display(self)
        
        self._update_display()

    def _update_display(self):
        pass

class NDTiffMax(NDTiff):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
    
    """
    def preprocess(self):
        self.pxmin = int(self.A[0,0,:,:,:].min())
        self.pxmax= int(self.A[0,0,:,:,:].max())
        self.pxlut = compute_lut(self.pxmin,self.pxmax)


        self.Amax = np.zeros((self.sequence_size,self.n_channels,self.shape[0],self.shape[1]),dtype=np.uint8)
        for i in tqdm(range(self.sequence_size),desc='Sequence processed'):
            for j in range(self.n_channels):
                img = np.array(self.A[i,j,:,:,:]).max(axis=0)
                self.Amax[i,j,:,:] = self.map_uint16_to_uint8(img)

    """
    def display(self,idx):
        self.idx = idx 
        self.update_title() 
        img  = np.array(self.A[self.jdx,self.cdx,:,:,:]).max(axis=0)
        if self.flip_channel and self.cdx == 1: img = np.fliplr(img) 
        return self.map_uint16_to_uint8(img)


class NDTiffTrack(NDTiff):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        
        self.jnum = 1
        self.inum = 2
        self.z_shift = 6
        
        self.idx = self.inum
        self.jdx = self.jnum

        self.D = np.zeros(((2*self.jnum+1)*self.shape[0],(2*self.inum+1)*self.shape[1]),dtype=int) 
        #self.Ddisplay = loader.image_to_rgb(self.map_uint16_to_uint8(self.D))

        self.max_objects = 100
        self.obj = (0,0) 


    def init_window(self):
        self.wtitle = 'Time point %d/%d ::: Z %d/%d ::: Channel %d/%d'
        self.win ='Volume'
        cv2.namedWindow(self.win)
        cv2.moveWindow(self.win,800,500)
        cv2.setMouseCallback(self.win,draw_roi,self)
        self.update_title()


    def load_stack(self,jdx):
        if (jdx >= self.jnum) and (jdx < self.sequence_size - self.jnum):
            self.jdx = jdx

    def display(self,idx):
        if (idx >= self.inum) and (idx < self.stack_size - self.inum):
            self.idx = idx 
        
        self.update_title() 
        
        for (jdx,j) in enumerate(range(self.jdx-self.jnum,self.jdx+self.jnum+1)):
            for (idx,i) in enumerate(range(self.idx-self.inum,self.idx+self.inum+1)):
                _i = (i + self.z_shift) % self.stack_size
                img  = np.array(self.A[j,self.cdx,_i,:,:])
                if self.flip_channel and self.cdx == 1: img = np.fliplr(img) 
                self.D[self.shape[0]*jdx:self.shape[0]*(jdx+1):,self.shape[1]*idx:self.shape[1]*(idx+1)] = img
        
        self.Ddisplay = loader.image_to_rgb(self.map_uint16_to_uint8(self.D))
        self._update_display()

        return self.Ddisplay
        
    def add_object(self,yt,xt):
        self.obj = (yt,xt)
   
    
    def _update_display(self):
        (y,x) = self.obj
        cv2.circle(self.Ddisplay,(x,y),5,(0,0,255),2)
        print(y,x)


def draw_roi(event,x,y,flags,V):
    if event == cv2.EVENT_LBUTTONDOWN:
        V.add_object(y,x)
        V._update_display()





