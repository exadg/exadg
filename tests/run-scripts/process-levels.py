from argparse import ArgumentParser
import os
import csv
import sys
from itertools import izip
import re
import numpy
import numpy as np
from numpy import genfromtxt

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")
    
    # python process-levels.py p-transfer.csv deg1 dofs1 restrict prolongate
    # python process-levels.py dg-to-cg-transfer.csv deg dofs_dg toCG toDG
    
    parser.add_argument(
        "file",
        help="Name of file to be processed (.csv).")
    
    parser.add_argument(
        "x",
        help="Name of file to be processed (.csv).")
    
    parser.add_argument(
        "th_x",
        help="Name of file to be processed (.csv).")
    
    parser.add_argument(
        "th_y1",
        help="Name of file to be processed (.csv).")
    
    parser.add_argument(
        "th_y2",
        help="Name of file to be processed (.csv).")
    
    arguments = parser.parse_args()

    return arguments

def main():
    options = parseArguments()
    
    file_name = options.file
    
    if not file_name.endswith(".csv"):
        sys.exit("The file has to end with .csv.")

    with open(file_name) as f:
        header_str = f.readline()
        header = ' '.join(header_str.split())
        header = header.split()

    x     = header.index(options.x)
    th_x  = header.index(options.th_x)
    th_y1 = header.index(options.th_y1)
    th_y2 = header.index(options.th_y2)
    
    header.append(header[th_y1] + "_th")
    header.append(header[th_y2] + "_th")
    header_str = ' '.join(header) +"\n"
    
    my_data = genfromtxt(options.file, skip_header=1)
    my_data = numpy.column_stack((my_data,my_data[:,th_x]/my_data[:,th_y1]/1e9))
    my_data = numpy.column_stack((my_data,my_data[:,th_x]/my_data[:,th_y2]/1e9))
   
    print my_data
    
    print my_data.shape
    ys = []
    zs = []
    nr_rows = my_data.shape[0]
    for i in range(0, nr_rows-1):
        if my_data[i, x] != my_data[i+1, x]:
            ys.append(i)
            zs.append(my_data[i,x])
    ys.append(nr_rows-1)
    zs.append(my_data[nr_rows-1,x])

    with open(file_name.replace(".csv","-last.csv"), "w") as file:
        file.write(header_str)
        np.savetxt(file, my_data[ys,:])

    for i in zs:
        with open(file_name.replace(".csv","-" + str(int(i)) + ".csv"), "w") as file:
            file.write(header_str)
            np.savetxt(file, my_data[my_data[:,x]==i])
        
    
    
    #print my_data
    print ys
    print zs

if __name__ == "__main__":
    main()
