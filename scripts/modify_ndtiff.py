"""
@name: modify_ndtiff.py 
@description:                  
    Scripts for modifying NDTiff.index files for pycromanager reads 
@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 
"""

import sys
from configparser import ConfigParser,ExtendedInterpolation
import argparse
from inspect import getmembers,isfunction

import struct 


def get_axes(data,position=0):
    (axes_length,) = struct.unpack("I", data[position: position + 4])
    axes_str = data[position + 4: position + 4 + axes_length].decode("utf-8")
    return axes_str

def axes_update(axes,label):
    return '{' + axes.replace('}',',"'+label+'":{0:1d}}}')

def parse_entry_breaks(data,position):
    ebr = [[position,position+4],[0,0]] 
    (axes_length,) = struct.unpack("I", data[position: position + 4])
    ebr[1][0] = position + axes_length + 4 
    position += axes_length + 4
    (filename_length,) = struct.unpack("I", data[position: position + 4])
    position += 4 + filename_length
    position += 32
    ebr[1][1] = position
    return ebr


def add_channel(args):
    """
    Adds channel axes to NDTiff.index file 
    """
    bkup = args.fin + '.bkup'
    print(f'Adding channel axis to: {args.fin}') 

    position = 0
    dnew = bytearray() 
    ecounter = 0 
    n_channels = 2
    with open(args.fin, "rb") as index_file:
        dold = bytearray(index_file.read())

        ##Backup file
        with open(bkup, "wb") as binary_file:
            binary_file.write(dold)
            print(f"Backed up file to: {bkup}")

        axes_str = get_axes(dold,position)
        axes_new = axes_update(axes_str,'channel')
    
        while position < len(dold):
            print("\rReading index... {:.1f}%       ".format(100 * (1 - (len(dold) - position) / len(dold))), end="")
            tmp = bytearray() 
            cdx = ecounter % n_channels 
            axes_str = get_axes(dold,position)
            axes_new = axes_update(axes_str,'channel').format(cdx)
            axes_len = len(axes_new)
            [ebr0,ebr1] = parse_entry_breaks(dold,position) 
            tmp.extend(struct.pack("I",axes_len))
            tmp.extend(bytearray(axes_new.encode('utf-8')))
            tmp.extend(dold[ebr1[0]:ebr1[1]])
            dnew.extend(tmp)
            position = ebr1[1]
            ecounter += 1
    print("\r") 
    
    with open(args.fin, "wb") as binary_file:
        binary_file.write(dnew)
        print(f"Modified file written to: {args.fin}")

if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')
    
    parser.add_argument('fin',
                action = 'store',
                help = 'Path to input file NDTiff.index.')
 
    args = parser.parse_args()
    eval(args.mode + '(args)')


