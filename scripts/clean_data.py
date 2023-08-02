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
from scipy import fftpack

def pixels_above_background(args):
    P = np.load(args.fin) 
    f = args.sample_freq 
    wsize = args.ksize
    Pm = np.zeros((P.shape[0],P.shape[1] - (wsize - 1))) 
    
    for i in range(P.shape[0]):
        Pm[i,:] = np.convolve(P[i,:],np.ones(wsize)/float(wsize),"valid")
    
    fig,ax = plt.subplots(1,1,figsize=(15,5))
    sns.heatmap(ax=ax,data=Pm,cmap='viridis') 
    ax.set_xticklabels([])
    ax.set_xlabel('time index')
    ax.set_ylabel('Embryos')
    if args.fout is not None:
        fout = args.fout.replace('.png','_heatmap.png')
        plt.savefig(fout,dpi=300)
    
    t = np.arange(Pm.shape[1])*f
    #t = t / 60. + 430
    t = t / 60. + 400
    mu = Pm.mean(axis=0)
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    for i in range(Pm.shape[0]):
        ax.plot(t,Pm[i,:],linestyle='-',c='#cdcdcd')
    
    ax.plot(t,mu,'-k')
    ax.set_ylim([0.25,0.75])
    #ax.set_xlim(xmax=700)
    ax.set_xlim([400,800])
    ax.set_ylabel('% pixel change',fontsize=12)
    ax.set_xlabel('time (s)',fontsize=12)
    if args.fout is not None:
        plt.savefig(args.fout,dpi=300)
    
    plt.show()
    

def check_autocorrelation(args):
    P = np.load(args.fin)
    idx = 2
    
    y0 = P[idx,:-1]
    y1 = P[idx,1:]
    
    print(np.corrcoef(y0,y1))
    fig,ax = plt.subplots(1,1,figsize=(5,5))
    ax.scatter(y1,y0,s=2)
    plt.show()


def moving_fano(args):
    P = np.load(args.fin) 
    #P = np.delete(P,(6),axis=0)
    f = args.sample_freq 
    t = np.arange(P.shape[1])*f
    t = t / 60. + 430
    #t = t / 60. + 400
    fig,ax = plt.subplots(1,1,figsize=(10,5))
    for i in range(P.shape[0]):
        ax.plot(t,P[i,:],linestyle='-',label=f'{i}')
    
    #ax.legend()
    #ax.set_xlim(xmax=700)
    ax.set_xlim([500,800])
    ax.set_ylim([0,0.4]) 
    #ax.set_ylabel('Fano factor (Var/Mean)',fontsize=12)
    ax.set_ylabel("Motor 'jerkiness' (Var/Mean)",fontsize=12)
    ax.set_xlabel('time (min)',fontsize=12)
    
    if args.fout is not None:
        plt.savefig(args.fout,dpi=300)

   
    plt.show()


def fft_prop_pixels(args):
    P = np.load(args.fin)
    idx = 0
    Fs = 1.
    
    t = np.arange(P.shape[1])
    fft = fftpack.fft(P[idx,:])
    n = np.size(t)
    fr = Fs/2 * np.linspace(0,1,n//2)
    y_m = 2/n * abs(fft[0:np.size(fr)])
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(15,5))
    ax[0].plot(t, P[idx,:],color='#cdcdcd')    # plot time series
    ax[1].stem(fr, y_m) # plot freq domain
    ax[1].set_xlim([0,0.5]) 

    plt.show()

def viz_matrix_heatmap(args):
    P = np.load(args.fin)
    fig,ax = plt.subplots(1,1,figsize=(15,5))
    sns.heatmap(ax=ax,data=P,cmap='viridis') 
    ax.set_xticklabels([])
    ax.set_xlabel('time index')
    ax.set_ylabel('ROIs')

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
    
    parser.add_argument('--dir',
            dest = 'dir',
            action = 'store',
            default = None,
            required = False,
            help = 'Input directory')
    
    parser.add_argument('--sample_freq',
            dest = 'sample_freq',
            action = 'store',
            default = 1,
            type=int,
            required = False,
            help = 'Will sample at every nth timepoint')
    
    parser.add_argument('--kernel_size',
            dest = 'ksize',
            action = 'store',
            default = 60,
            type=int,
            required = False,
            help = 'Will sample at every nth timepoint')



    params = parser.parse_args()
    eval(params.mode + '(params)')

