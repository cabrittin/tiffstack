"""
@name: loader.py                         
@description:                  

Class object for loading tiff filess

Maintains a buffer cache

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2023-07         
"""

from configparser import ConfigParser,ExtendedInterpolation
import os
import glob
import re

from tifffile import TiffFile

from pycsvparser.read import parse_file

class Session(object):
    def __init__(self,dname):
        self.dname  = dname
        ini = os.sep.join([dname,'config.ini'])
        self.cfg = ConfigParser(interpolation=ExtendedInterpolation())
        self.cfg.read(ini)  
        self.img_dir = os.sep.join([dname,self.cfg['images']['dir']]) 
        self.ext_dir = os.sep.join([dname,self.cfg['images']['rois']])
        self.roi_out = os.sep.join([self.ext_dir,'roi.npy'])
        glob_path = format_glob_path(self.img_dir,self.cfg['images']['extension']) 
        self.stacks = stacks_from_path(glob_path) 
    
    def get_stack(self,idx):
        return self.stacks[idx]
    
    def iter_stacks(self):
        for s in self.stacks:
            yield s
    
    def chunk_stacks(self,nchunks):
        return split_n(self.stacks,nchunks)

    def get_roi_file(self):
        return os.sep.join([self.dname,self.cfg['roi']['file']])
    
    def roi_dims(self):
        dx = self.cfg.getint('roi','dx')
        dy = self.cfg.getint('roi','dy')
        return dx,dy
    
    def num_stacks(self):
        return len(self.stacks)

class Buffer(object):
    def __init__(self,stacks,buffer_size=10):
        self.bsize = buffer_size
        self.num_stacks = len(stacks)
        self.stacks_ptr = stacks
        self.cur = 0
        self.stack_loaded = [0]*self.num_stacks
        self.stack_loaded[:self.bsize+1] = [1]*(self.bsize+1)

    def next(self):
        self.cur = min(self.cur+1,self.num_stacks-1)
        bmax = self.cur + self.bsize
        bmin = self.cur - self.bsize - 1 
        if bmax < self.num_stacks: self.stack_loaded[bmax] = 1
        if bmin >= 0: self.stack_loaded[bmin] = 0

    def prev(self):
        self.cur = max(self.cur-1,0)
        bmax = self.cur + self.bsize
        bmin = self.cur - self.bsize - 1 
        if bmax < self.num_stacks: self.stack_loaded[bmax] = 0
        if bmin >= 0: self.stack_loaded[bmin] = 1


def split_n(sequence, num_chunks):
    chunk_size, remaining = divmod(len(sequence), num_chunks)
    for i in range(num_chunks):
        begin = i * chunk_size + min(i, remaining)
        end = (i + 1) * chunk_size + min(i + 1, remaining)
        yield sequence[begin:end]

def rois_from_file(fname):
    @parse_file(fname,multi_dim=True)
    def row_into_container(container,row=None,**kwargs):
        roi = [(int(row[0]),int(row[1])),(int(row[2]),int(row[3]))]
        container.append(roi)
    
    container = []
    row_into_container(container)
    return container

def tif_from_stack(fname):
    return TiffFile(fname)

def array_from_stack(fname,zdx=0):
    return tif_from_stack(fname).pages[zdx].asarray()

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]

def format_path(L):
    ext = L.cfg['extension']
    path = os.sep.join([L.dname,'*'+ext])
    return path

def format_glob_path(directory,extension):
    return os.sep.join([directory,'*'+extension])
    
def stacks_from_path(path,order_stacks=True):
    stacks = glob.glob(path)
    if order_stacks: stacks.sort(key=natural_keys)
    return stacks
