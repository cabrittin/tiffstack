"""
@name: clean_data.py                        
@description:                  
   Functions for cleaning data  

@author: Christopher Brittin   
@email: "cabrittin"+ <at>+ "gmail"+ "."+ "com"
@date: 2019-12-05              
"""

import sys
import argparse
from inspect import getmembers,isfunction
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def pixels_above_background(args):
    P = np.load(args.fin) 
    wsize = 60
    Pm = np.zeros((P.shape[0],P.shape[1] - (wsize - 1))) 
    
    for i in range(P.shape[0]):
        Pm[i,:] = np.convolve(P[i,:],np.ones(wsize),"valid") / float(wsize)



    fig,ax = plt.subplots(1,1,figsize=(15,5))
    sns.heatmap(ax=ax,data=Pm,cmap='viridis') 
    ax.set_xticklabels([])
    ax.set_xlabel('time index')
    ax.set_ylabel('Embryos')
    if args.fout is not None:
        fout = args.fout.replace('.png','_heatmap.png')
        plt.savefig(fout,dpi=300)
    
    t = np.arange(Pm.shape[1])
    t = t / 60. + 430
    mu = Pm.mean(axis=0)
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    for i in range(Pm.shape[0]):
        ax.plot(t,Pm[i,:],linestyle='-',c='#cdcdcd')
    
    ax.plot(t,mu,'-k')
    ax.set_ylim([0.25,0.65])
    ax.set_xlim(xmax=700)
    ax.set_ylabel('% pixel change',fontsize=12)
    ax.set_xlabel('time (s)',fontsize=12)
    if args.fout is not None:
        plt.savefig(args.fout,dpi=300)

    plt.show()
 


if __name__=="__main__":
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('mode',
                        action = 'store',
                        choices = [t for (t,o) in getmembers(sys.modules[__name__]) if isfunction(o)],
                        help = 'Function call')

    parser.add_argument('fin',
                        action = 'store',
                        help = 'Path to file')
    
    parser.add_argument('--fout',
            dest = 'fout',
            action = 'store',
            default = None,
            required = False,
            help = 'Output file')
 
    params = parser.parse_args()
    eval(params.mode + '(params)')

