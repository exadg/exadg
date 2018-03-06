#! /usr/bin/env python

import argparse
import morton
import numpy as np


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
            description="Create a random array and write it to a binary file. "
                        "Binary format analogous to given text format: "
                        "first two entries are int32 giving the matrix size, rest of data are float64 values.")
    parser.add_argument("dim", type=int, help="Array size in first dimension.")
    parser.add_argument("cells", type=int, help="Array size in first dimension.")
    parser.add_argument("degree", type=int, help="Array size in second dimension.")
    parser.add_argument("outA", type=str, help="Path to output file.")
    parser.add_argument("outB", type=str, help="Path to output file.")

    args = parser.parse_args()
    dim = args.dim; cells = args.cells; points = args.degree+1;

    dofs_per_field = int(pow(cells*points,dim))

    dofs = np.zeros((int(pow(cells*points,dim))*dim,1));

    counter = 0

    m = morton.Morton(dimensions=dim, bits=32)
    
    for C in range(0,int(pow(cells,dim))):
        
        values = m.unpack(C); 
        
        if dim==2:
            I = values[0]; J = values[1];

            # for u
            for j in range(0, points):
                for i in range(0, points):
                    dofs[counter]=(J*points+j)*cells*points+(I*points+i)+dofs_per_field*0
                    counter=counter+1

            # for v
            for j in range(0, points):
                for i in range(0, points):
                    dofs[counter]=(J*points+j)*cells*points+(I*points+i)+dofs_per_field*1
                    counter=counter+1
        elif dim==3:
            I = values[0]; J = values[1]; K = values[2];

            # for u
            for k in range(0, points):
                for j in range(0, points):
                    for i in range(0, points):
                        dofs[counter]=(K*points+k)*cells**2*points**2+(J*points+j)*cells*points+(I*points+i)+dofs_per_field*0
                        counter=counter+1

            # for v
            for k in range(0, points):
                for j in range(0, points):
                    for i in range(0, points):
                        dofs[counter]=(K*points+k)*cells**2*points**2+(J*points+j)*cells*points+(I*points+i)+dofs_per_field*1
                        counter=counter+1

            # for w
            for k in range(0, points):
                for j in range(0, points):
                    for i in range(0, points):
                        dofs[counter]=(K*points+k)*cells**2*points**2+(J*points+j)*cells*points+(I*points+i)+dofs_per_field*2
                        counter=counter+1

        else:
            print "Error: only dimensions 2 and 3 supported!"

    #print(dofs.reshape(-1,int(pow(points,dim))))
    
    size = np.array([0, args.dim, args.cells, args.degree, 0, 0, 0, 0], dtype=np.int32)
    with open(args.outA, "wb") as f:
        size.tofile(f)
        dofs.tofile(f)
    
    # expected result
    N  = points*cells
    Nx = (N/2+1)*2
        
    dofs2 = np.arange(0,dofs_per_field*dim)
    dofs2 = dofs2.reshape((-1,points*cells))
    if dim==2:
        dofs3 = np.zeros((N*dim,Nx)); dofs3[:,:N-Nx] = dofs2;
    elif dim==3:
        dofs3 = np.zeros((N*dim,N,Nx)); dofs3[:,:,:N-Nx] = dofs2;
    
    #print dofs3
    
    size = np.array([1, dim, cells*points,0, 0, 0, 0, 0], dtype=np.int32)
    with open(args.outB, "wb") as f:
        size.tofile(f)
        dofs3.tofile(f)