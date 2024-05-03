
# Plot the stability test results in a single graph for an overview.

import numpy as np
import matplotlib.pyplot as plt
import glob, os   
   
if __name__ == "__main__":

    # Read the all the txt files and plot the stability test results
    plt.figure()
    os.chdir("/home/richardschussnig/dealii-candi/exadg/build/applications/structure/stability_test/")
    for file in glob.glob("stability_forward_test_*"):
    
        print("Parsing file:")
        print(file)
        print("")
        
        # Parse each file.
        f = open(file, 'r')

        rows = np.zeros((1000000,3))
        idx_line = 0
        n_header_lines = 4
        for line in f:
        
            # Skip parsing header lines.
            idx_line = idx_line + 1
            if idx_line <= n_header_lines:
                continue
                
            # Split on any whitespace (including tab characters)
            row = line.split()            
            print(row)
            for i in range(len(row)):
                rows[idx_line-n_header_lines-1, i] = float(row[i])

	# shrink array to actually used size
        rows = rows[0:idx_line-n_header_lines, :]
        print(len(rows))

        plt.loglog(rows[:,0], rows[:,1], label=file + ', |stress|') #, color='blue', marker='*')
        plt.loglog(rows[:,0], rows[:,2], label=file + ', |Jacobian|') #, color='blue', marker='*')

    plt.title("Relativer Fehler $\epsilon_\mathrm{rel} = \mathrm{max}_i || (.)_\mathrm{f64} - (.)_\mathrm{f32} ||_\infty / ||(.)_\mathrm{f64}||_\infty$")
    plt.legend()
    plt.xlabel('Strain scale')
    plt.ylabel('$Relativer Fehler \epsilon_\mathrm{rel}$')
    plt.xlim(xmin=1e-9, xmax=1e3)
    plt.ylim(ymin=1e-10, ymax=1e2)
    plt.show()
    
