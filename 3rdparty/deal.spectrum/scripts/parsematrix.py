#! /usr/bin/env python

import argparse
import os
import numpy as np


def read_matrix(filename, print_all):
    with open(filename, "rb") as f:
        settings = np.fromfile(f, count=8, dtype=np.int32)
        
        # get matrix settings
        type=settings[0];dim=settings[1];cells=settings[2];degree=settings[3]
        
        # ... and print it
        print
        print("   file = {}".format(os.path.abspath(filename)))
        print("   type = {}   dim = {}   cells = {}   degree = {}"
                    .format(settings[0],settings[1],settings[2],settings[3]))
        print
        
        np.set_printoptions(threshold=np.inf)
        # print content:
        if print_all:
            # ... distinguishe file formats...
            if type==1:
                # ...row wise 
                N  = int(cells*(degree+1))
                Nx = (N/2+1)*2

                # ... print u
                values = np.fromfile(f, count=int(pow(cells*(degree+1),dim-1)*Nx), 
                    dtype=np.float64)
                print("u: ")
                print(values.reshape((-1,Nx)))

                # ... print v
                values = np.fromfile(f, count=int(pow(cells*(degree+1),dim-1)*Nx), 
                    dtype=np.float64)
                print("v: ")
                print(values.reshape((-1,Nx)))

                if dim==3:
                    # ... print w
                    values = np.fromfile(f, count=int(pow(cells*(degree+1),dim-1)*Nx), 
                        dtype=np.float64)
                    print("w: ")
                    print(values.reshape((-1,Nx)))
            else:
                # ...cell wise along sfc
                values = np.fromfile(f, count=int(dim*pow(cells*(degree+1),dim)), 
                    dtype=np.float64)
                print(values.reshape(-1,pow(degree+1,dim)))
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A * B = C. Test if cannon result matrix C is correct.")
    parser.add_argument("input", type=str, help="Input filename of A")
    parser.add_argument('-a', action='store_true')
    args = parser.parse_args()
    assert os.path.isfile(args.input)

    read_matrix(args.input, args.a)
