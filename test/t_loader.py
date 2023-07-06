"""
@name:                         
@description:                  


@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import tiffstack.loader as loader
from tiffstack.loader import Loader,Buffer

CONFIG = 'config/loader.cfg'
L = Loader(CONFIG,'data')

def test_format_path():
    path = loader.format_path(L)
    assert(path == 'data/*.tif')

def test_stacks_from_path():
    L = Loader(CONFIG,'test/test_data_1')
    path = loader.format_path(L)
    stacks = loader.stacks_from_path(path) 
    assert(len(stacks) == 10)
    assert(stacks[0] == 'test/test_data_1/f_1_stuff.tif')
    assert(stacks[-1] == 'test/test_data_1/f_10_stuff.tif')

def test_stacks_buffer():
    L = Loader(CONFIG,'test/test_data_1')
    path = loader.format_path(L)
    stacks = loader.stacks_from_path(path) 
    B = Buffer(stacks,buffer_size=int(L.cfg['buffer_size']))
    assert(len(B.stacks_ptr) == 10)
    print(B.stack_loaded)
    for i in range(15):
        B.next()
        print(B.cur,B.stack_loaded)
    
