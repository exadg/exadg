from argparse import ArgumentParser
import os
import csv
from itertools import izip
import re
import numpy
import numpy as np
from numpy import genfromtxt

def parseArguments():
    parser = ArgumentParser(description="Submit a simulation as a batch job")
    
    parser.add_argument(
        "folder",
        help="Directory to store job configuration and results")
    
    parser.add_argument(
        "labelformat",
        help="Directory to store job configuration and results")
    
    parser.add_argument(
        "parameter",
        help="Directory to store job configuration and results")

    arguments = parser.parse_args()

    return arguments

def main():
    options = parseArguments()
    
    #folder      = "results/job-likwid-2017-11-10-13-25-06/mass_vec_mpi/likwid/"
    #labelformat = "cell_laplacian_dim_(\d+)-metric.csv"
    #parameter   = "Dimensions"
    folder      = options.folder + "/likwid/"
    labelformat = options.labelformat
    parameter   = options.parameter
    
    print folder
    print labelformat
    print parameter

    folderform         = folder + "/{}"
    folderregionsform  = folder + "/{}/regions"
    folderregionstform = folder + "/{}/regions-transpose"
    folderranksform    = folder + "/{}/ranks"
    folderranksstaform = folder + "/{}/ranks-statistics"

    def mkdir(lab):
        if not os.path.exists(lab):
            os.mkdir(lab)

        for f in os.listdir(lab):
            os.remove(os.path.join(lab, f))

    for metric in os.listdir(folder):
        if (".csv" not in metric) and  (".ods" not in metric) and os.path.isfile(folder + "/" +metric):

            # name of the actual file to be processes
            filename = os.path.join(folder, metric);
            print("Prosecss: " + filename)

            Metric = metric.upper()

            folder_        = folderform.format(Metric)
            folderregions  = folderregionsform.format(Metric)
            folderregionst = folderregionstform.format(Metric)
            folderranks    = folderranksform.format(Metric)
            folderrankssta = folderranksstaform.format(Metric)
            if not os.path.exists(folder_):
                os.mkdir(folder_)
            mkdir(folderregions)
            mkdir(folderregionst)
            mkdir(folderranks)
            mkdir(folderrankssta)

            def processaditional(i):
                with open(filename, "r") as f:
                    with open(folder_ + "/" + str(i) + ".csv", "w") as ff:

                        line = f.readline()

                        while line:
                            if ">>> (" + str(i) + ")" in line:
                                ff.write(re.sub(' ',',', re.sub(' +',' ',line[7:])[1:]))
                            line = f.readline()

            for i in range(0,1):
                processaditional(i)


            with open(filename, "r") as f:
                print filename
                line = f.readline()

                # process regions
                while line:
                    if ("Region,") in line:
                        region = line.replace("Region,","").replace(",","").replace("\n","")

                        line = f.readline()
                        line = f.readline()


                        # Events
                        with open(folderregions + "/" + region + "-events.csv", "w") as fw:
                            while True:
                                fw.write(line)
                                line = f.readline()

                                if "Event,Counter," in line or  "Metric," in line:
                                    break
                        os.remove(folderregions + "/" + region + "-events.csv")

                        # Events (statistics)
                        if "Event,Counter," in line:
                            with open(folderregions + "/" + region + "-events-stat.csv", "w") as fw:
                                while True:
                                    fw.write(line)
                                    line = f.readline()

                                    if "Metric," in line:
                                        break
                            os.remove(folderregions + "/" + region + "-events-stat.csv")

                        # Metric
                        with open(folderregions + "/" + region + "-metric.csv", "w") as fw:
                            while line and True:
                                fw.write(line)
                                line = f.readline()

                                if "Metric," in line or "Region," in line:
                                    break

                        # Matric (statistics)
                        if "Metric," in line:
                            with open(folderregions + "/" + region + "-metric-stat.csv", "w") as fw:
                                while line and True:
                                    fw.write(line)
                                    line = f.readline()

                                    if "Region," in line:
                                        break
                            os.remove(folderregions + "/" + region + "-metric-stat.csv")

                    else:
                        line = f.readline()

            ranks = 0

            for file in os.listdir(folderregions):
                if (".csv" in file) and os.path.isfile(folderregions + "/" + file):
                    a = izip(*csv.reader(open(os.path.join(folderregions, file), "rb")))
                    csv.writer(open(os.path.join(folderregionst, file), "wb")).writerows(a)
                    a = izip(*csv.reader(open(os.path.join(folderregions, file), "rb")))
                    ranksname = open(os.path.join(folderregions, file), "rb").readline().split(',')[1:]
                    ranksname = [ r for r in ranksname if r != "" and r != "\n" ]
                    ranks = len(ranksname) 

            print ranksname

            list1 = os.listdir(folderregionst)

            print list1

            if labelformat == "":
                list2 = list1
            else:
                listtemp = list1
                list1 = []
                list2 = []
                for l in listtemp:
                    print l
                    t = re.findall(labelformat, l)
                    print t
                    if len(t)>0:
                        list1.append(l)
                        list2.append(int(t[0]))
                #list2 = [int(re.findall(labelformat, l)[0]) for l in list1 ]
                list2, list1 = zip(*sorted(zip(list2, list1)))
                list2 = [str(l) for l in list2]

            for i in range(0,len(list1)):
                file = folderregionst + "/" + list1[i]
                if (".csv" in file) and os.path.isfile(file):
                    with open(file, 'r') as f:
                        header = f.readline().replace("Metric",parameter)
                        line = f.readline()

                        for j in range(0, ranks):
                            line = list2[i] +  line[line.find(',') :]

                            fff = folderranks + "/" + ranksname[j] + ".csv"

                            if i == 0:
                                with open(fff,'w') as ff:
                                    ff.write(header)
                                    ff.write(line)
                            else :
                                with open(fff,'a') as ff:
                                    ff.write(line)

                            line = f.readline()

            # get arguments
            files = [os.path.join(folderranks,s) for s in os.listdir(folderranks) if s.endswith(".csv")]
            print files

            def load(fname):
                data = genfromtxt(fname, delimiter=',',skip_header=1)
                return data[:,1:]

            # first file -> fill arrays
            ff = files[0]
            data = load(ff)
            data_min = data
            data_max = data
            data_sum = data

            # extract first row and column of file
            groups = []
            first_line = ""

            with open(ff, 'r') as f:
                first_line = f.readline()
                line = f.readline()
                while line:
                    groups.append(line.split(",")[0])
                    line = f.readline()

            # process remaining files of other ranks
            counter = 1

            for f in files[1:]:
                
                
                data = load(f)
                print f
                print data
                print data_min
                data_min = numpy.minimum(data_min, data)
                data_max = numpy.maximum(data_max, data)
                data_sum = data_sum + data
                counter = counter + 1

            # determine average
            data_ave = data_sum / counter

            # dermine variance
            if counter == 1:
                data_var = load(ff) * 0
            else:
                data = load(ff)
                data_var = (data - data_ave)**2
                for f in files[1:]:
                    data = load(f)
                    data_var = data_var + (data - data_ave)**2

                data_var = data_var / (counter - 1)
                data_var = data_var**0.5

            def outputtocsv(data, file):
                with open(file, "w") as f:
                    f.write(first_line)
                    for i in range(0,len(groups)):
                        f.write(str(groups[i]) + "," +",".join(np.char.mod('%f',data[i,:]))+"\n")

            # output results to files
            outputtocsv(data_min, os.path.join(folderrankssta, "min.csv"))
            outputtocsv(data_max, os.path.join(folderrankssta, "max.csv"))
            outputtocsv(data_sum, os.path.join(folderrankssta, "sum.csv"))
            outputtocsv(data_ave, os.path.join(folderrankssta, "ave.csv"))
            outputtocsv(data_var, os.path.join(folderrankssta, "var.csv"))
            
if __name__ == "__main__":
    main()