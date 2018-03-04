#! /usr/bin/env python

import argparse
import morton
import numpy as np
from math import pi
from math import sqrt 

def func2D(xx,yy):
    # ... some spectral function
    #return [np.cos(1*2*pi*xx)*np.cos(1*2*pi*yy),
    #    np.cos(2*2*pi*xx)*np.cos(2*2*pi*yy)]
    return [np.cos(1*2*pi*xx),
        np.cos(2*2*pi*xx)*0]

    # ... superduper borring function
    #return [xx, yy*0];

def func3D(xx,yy,zz):
    # ... some spectral function
    return [
        np.cos(1*2*pi*xx)*np.cos(1*2*pi*yy)*np.cos(1*2*pi*zz),
        np.cos(1*2*pi*xx)*np.cos(1*2*pi*yy)*np.cos(1*2*pi*zz),
        np.cos(1*2*pi*xx)*np.cos(1*2*pi*yy)*np.cos(1*2*pi*zz)]

    # ... superduper borring function
    return [xx, yy, zz];

def toLex(dim,cells,points,out, f2D, f3D):
    degree = points - 1
    N = int(cells*(degree+1))
    Nx = (N/2+1)*2
    
    # x-axis
    x = np.linspace(0.5, N+0.5, N+1); x = x[:-1] ; x = x / N;
    
    if dim == 2:
        # values
        xx, yy = np.meshgrid(x, x)
        temp = f2D(xx,yy); xval = temp[0]; yval = temp[1]
        
        # pad values
        xxval = np.zeros((N,Nx)); xxval[:,:N-Nx] = xval;
        yyval = np.zeros((N,Nx)); yyval[:,:N-Nx] = yval;
    elif dim ==3:
        # values
        xx, yy, zz = np.meshgrid(x, x, x)
        temp = f3D(xx,yy,zz); xval = temp[0]; yval = temp[1]; zval = temp[2]

        # pad values
        xxval = np.zeros((N,N,Nx)); xxval[:,:,:N-Nx] = xval;
        yyval = np.zeros((N,N,Nx)); yyval[:,:,:N-Nx] = yval;
        zzval = np.zeros((N,N,Nx)); zzval[:,:,:N-Nx] = zval;
    else:
        raise NameError("This program only works for dim=2,3")

    # write file: collect header...
    size = np.array([1, dim, cells, degree, 0, 0, 0, 0], dtype=np.int32)
    with open(out, "wb") as f:
        # ... write header to file
        size.tofile(f)
        # ... write values to file
        xxval.tofile(f)
        yyval.tofile(f)
        if dim==3:
            zzval.tofile(f)

def toMorton(dim,cells,points, out, f2D, f3D):
    degree = points - 1
    
    dofs_per_field = int(pow(cells*points,dim))

    dofs = np.zeros((int(pow(cells*points,dim)),1));

    counter = 0

    m = morton.Morton(dimensions=dim, bits=32)
    
    for C in range(0,int(pow(cells,dim))):
        
        values = m.unpack(C); 
        
        if dim==2:
            I = values[0]; J = values[1];

            for j in range(0, points):
                for i in range(0, points):
                    dofs[counter]=(J*points+j)*cells*points+(I*points+i)
                    counter=counter+1

        elif dim==3:
            I = values[0]; J = values[1]; K = values[2];

            for k in range(0, points):
                for j in range(0, points):
                    for i in range(0, points):
                        dofs[counter]=(K*points+k)*cells**2*points**2+(J*points+j)*cells*points+(I*points+i)
                        counter=counter+1

        else:
            print "Error: only dimensions 2 and 3 supported!"

    if   degree==1:
        glp = np.array([-1.0, +1.0])
    elif   degree==2:
        glp = np.array([-1.0, 0.0, +1.0])
    elif degree==3:
        glp = np.array([-1.0, -sqrt(1.0/5.0), +sqrt(1.0/5.0), +1.0])
    elif degree==4:
        glp = np.array([-1.0, -sqrt(3.0/7.0), 0.0, +sqrt(3.0/7.0), +1.0])
    elif degree==5:
        v1 = sqrt(1.0/3.0+2*sqrt(7)/21)
        v2 = sqrt(1.0/3.0-2*sqrt(7)/21)
        glp = np.array([-1.0, -v1, -v2, +v1, +v2 +1.0])
    elif degree==6:
        v1 = sqrt(5.0/11.0+2.0/11.0*sqrt(5.0/3.0))
        v2 = sqrt(5.0/11.0-2.0/11.0*sqrt(5.0/3.0))
        glp = np.array([-1.0, -v1, -v2, 0.0, +v1, +v2 +1.0])
    else:
        print "Error: only degree up to 6 supported!"

    glp = (glp + 1.0)/2.0

    x = np.zeros(cells * points)
    for i in range(0, cells):
        for j in range(0, points):
            x[i*points+j] = i + glp[j];
    x = x/cells
    if dim==2:
        xx, yy = np.meshgrid(x, x)
        xx = xx.reshape(-1,1);
        yy = yy.reshape(-1,1);

        # permute morton
        xx = xx[dofs.astype(int)]; xx =  xx.flatten()
        yy = yy[dofs.astype(int)]; yy =  yy.flatten()
        
        # init u and v with values
        temp = f2D(xx,yy); xval = temp[0]; yval = temp[1]
    elif dim==3:
        xx, yy, zz = np.meshgrid(x, x, x)
        xx = xx.reshape(-1,1);
        yy = yy.reshape(-1,1);
        zz = zz.reshape(-1,1);

        # init u,v, and w with values
        xx = xx[dofs.astype(int)]; xx =  xx.flatten()
        yy = yy[dofs.astype(int)]; yy =  yy.flatten()
        zz = zz[dofs.astype(int)]; zz =  zz.flatten()
        
        # ... superduper borring function
        temp = f3D(xx,yy,zz); xval = temp[0]; yval = temp[1]; zval = temp[2]
        
    # indices for cell interleaving
    ind = np.zeros(dofs_per_field)
    for i in range(0,cells**dim):
        for j in range(0,points**dim):
            ind[i*points**dim+j] = dim*i*points**dim+j

    # ... permute data such that cell interleaved
    res = np.zeros(dofs_per_field*dim)
    res[(ind+0*points**dim).astype(int)] = xval
    res[(ind+1*points**dim).astype(int)] = yval
    if dim==3:
        res[(ind+2*points**dim).astype(int)] = zval
            
    
    # write file: collect header...
    size = np.array([0, dim, cells, degree, 0, 0, 0, 0], dtype=np.int32)
    with open(out, "wb") as f:
        # ... write header to file
        size.tofile(f)
        # ... write values to file
        res.tofile(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="...")
    parser.add_argument("type",   type=int, help="Array size in first dimension.")
    parser.add_argument("dim",    type=int, help="Array size in first dimension.")
    parser.add_argument("cells",  type=int, help="Array size in first dimension.")
    parser.add_argument("points", type=int, help="Array size in second dimension.")
    parser.add_argument("out",    type=str, help="Path to output file.")

    args = parser.parse_args()
    dim = args.dim; cells = args.cells; points = args.points; out = args.out;

    if args.type == 0:
        toMorton(dim, cells, points, out, func2D, func3D)
    else:
        toLex(   dim, cells, points, out, func2D, func3D)
