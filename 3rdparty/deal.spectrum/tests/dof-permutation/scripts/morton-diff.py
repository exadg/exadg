#! /usr/bin/env python

import argparse
import os
import numpy as np
import sys

def read_matrix(filename1, filename2):
    with open(filename1, "rb") as f1:
        with open(filename2, "rb") as f2:
            settings1 = np.fromfile(f1, count=8, dtype=np.int32)
            settings2 = np.fromfile(f2, count=8, dtype=np.int32)

            # get matrix settings
            type=settings1[0];dim=settings1[1];cells=settings1[2];degree=settings1[3]


            # ... distinguishe file formats...
            if type==1:
                # ...row wise 
                N  = int(cells*(degree+1))
                Nx = (N/2+1)*2

                # ... print u
                values1 = np.fromfile(f1, count=int(pow(cells*(degree+1),dim-1)*Nx), 
                    dtype=np.float64)
                values2 = np.fromfile(f2, count=int(pow(cells*(degree+1),dim-1)*Nx), 
                    dtype=np.float64)

                d = np.amax(np.absolute(values1-values2))
                
                if d > 1e-10:
                    print "ERROR: In field u of magnituede {}".format(d)
                    sys.exit(1)

                # ... print v
                values1 = np.fromfile(f1, count=int(pow(cells*(degree+1),dim-1)*Nx), 
                    dtype=np.float64)
                values2 = np.fromfile(f2, count=int(pow(cells*(degree+1),dim-1)*Nx), 
                    dtype=np.float64)

                d = np.amax(np.absolute(values1-values2))
                
                if d > 1e-10:
                    print "ERROR: In field u of magnituede {}".format(d)
                    sys.exit(1)

                if dim==3:
                    # ... print w
                    values1 = np.fromfile(f1, count=int(pow(cells*(degree+1),dim-1)*Nx), 
                        dtype=np.float64)
                    values2 = np.fromfile(f2, count=int(pow(cells*(degree+1),dim-1)*Nx), 
                        dtype=np.float64)

                    d = np.amax(np.absolute(values1-values2))

                    if d > 1e-10:
                        print "ERROR: In field u of magnituede {}".format(d)
                        sys.exit(1)
            else:
                print("ERROR: Type not supported")
                sys.exit(1)
            


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="A * B = C. Test if cannon result matrix C is correct.")
    parser.add_argument("input1", type=str, help="Input filename of A")
    parser.add_argument("input2", type=str, help="Input filename of A")
    args = parser.parse_args()
    assert os.path.isfile(args.input1)
    assert os.path.isfile(args.input2)

    read_matrix(args.input1,args.input2)
    print "PASSED"