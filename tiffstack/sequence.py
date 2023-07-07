"""
@name: sequence.py                       
@description:                  
Class object for dealing with a sequence of stacks

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

from tifffile import TiffFile




class Sequence(object):
    def __init__(self,stacks):
        self.stacks = stacks

    def load(self,idx):
        f = self.stacks[idx]
        return TiffFile(f)
