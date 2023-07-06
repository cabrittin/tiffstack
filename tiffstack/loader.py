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

class Loader(object):
    def __init__(self,_cfg,dname):
        cfg = ConfigParser(interpolation=ExtendedInterpolation())
        cfg.read(_cfg)  
        self.cfg = cfg['loader']
        self.dname = dname

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
    
def stacks_from_path(path,order_stacks=True):
    stacks = glob.glob(path)
    if order_stacks: stacks.sort(key=natural_keys)
    return stacks
