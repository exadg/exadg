#! /usr/bin/env python

import argparse
import os
from math import pi
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Create a random array and write it to a binary file. "
                        "Binary format analogous to given text format: "
                        "first two entries are int32 giving the matrix size, rest of data are float64 values.")
    parser.add_argument("dim", type=int, help="Array size in first dimension.")
    parser.add_argument("cells", type=int, help="Array size in first dimension.")
    parser.add_argument("degree", type=int, help="Array size in second dimension.")
    parser.add_argument("out", type=str, help="Path to output file.")

    args = parser.parse_args()
    #assert args.size1 > 0
    #assert args.size2 > 0
    #assert args.size2 > 0
    assert not os.path.isdir(args.out)
    


    # row-wise output
    type = 1
    N = int(args.cells*(args.degree+1))
    Nx = (N/2+1)*2
    
    print N
    print Nx
    
    # x-axis
    x = np.linspace(0, 2*pi, N+1); x = x[:-1]
    
    if args.dim == 2:
        # values
        xx, yy = np.meshgrid(x, x)
        xval = xx#np.cos(1*xx)*np.cos(2*yy);
        yval = 0*np.cos(4*xx)*np.cos(3*yy);
        
        # pad values
        xxval = np.zeros((N,Nx)); xxval[:,:N-Nx] = xval;
        yyval = np.zeros((N,Nx)); yyval[:,:N-Nx] = yval;
    elif args.dim ==3:
        # values
        xx, yy, zz = np.meshgrid(x, x, x)
        xval = np.cos(1*xx);
        yval = np.cos(2*yy);
        zval = np.cos(4*zz)

        # pad values
        xxval = np.zeros((N,N,Nx)); xxval[:,:,:N-Nx] = xval;
        yyval = np.zeros((N,N,Nx)); yyval[:,:,:N-Nx] = yval;
        zzval = np.zeros((N,N,Nx)); zzval[:,:,:N-Nx] = zval;
    else:
        raise NameError("This program only works for dim=2,3")

    # write file: collect header...
    size = np.array([type, args.dim, args.cells, args.degree, 0, 0, 0, 0], dtype=np.int32)
    with open(args.out, "wb") as f:
        # ... write header to file
        size.tofile(f)
        # ... write values to file
        xxval.tofile(f)
        yyval.tofile(f)
        if args.dim==3:
            zzval.tofile(f)