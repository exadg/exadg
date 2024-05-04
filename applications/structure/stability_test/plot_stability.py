
# Plot the stability test results in a single graph for an overview.

import numpy as np
import matplotlib.pyplot as plt
import glob, os   
import re
   
if __name__ == "__main__":

    skip_jacobian = not False
    skip_stress = False

    skip_STVK = True
    skip_cNH  = not True
    skip_iNH  = True
    skip_iHGO = True
    skip_spatial_integration  = False
    skip_material_integration = False
    skip_stable_formulation   = False
    skip_unstable_formulation = False

    # Read the all the txt files and plot the stability test results
    plt.figure()

    # plot horizontal lines at specific error margins
    plt.loglog([0.0, 1e20], [1e-5, 1e-5], label=None, color='black', \
    linestyle='dotted', linewidth=1.0)
    plt.loglog([0.0, 1e20], [1e-4, 1e-4], label=None, color='black', \
    linestyle='dotted', linewidth=1.0)
    plt.loglog([0.0, 1e20], [1e-3, 1e-3], label=None, color='black', \
    linestyle='dotted', linewidth=1.0)
    plt.loglog([0.0, 1e20], [1e-2, 1e-2], label=None, color='black', \
    linestyle='dotted', linewidth=1.0)
    plt.loglog([0.0, 1e20], [1e-1, 1e-1], label=None, color='black', \
    linestyle='dotted', linewidth=1.0)
    plt.loglog([0.0, 1e20], [1e-0, 1e-0], label=None, color='black', \
    linestyle='dotted', linewidth=1.0)

    # os.chdir("/home/richardschussnig/dealii-candi/exadg/build/applications/structure/stability_test/")
    os.chdir("/home/richard/dealii-candi/exadg/build/applications/structure/stability_test/")
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
            # print(row)
            for i in range(len(row)):
                rows[idx_line-n_header_lines-1, i] = float(row[i])

	# shrink array to actually used size
        rows = rows[0:idx_line-n_header_lines, :]

        # get the legend entry from the file name
        underscore_indices = np.zeros(9)
        idx = 0
        for match in re.finditer('_', file):
            underscore_indices[idx]= match.start()
            idx = idx + 1
            
        for match in re.finditer('.', file):
            point_idx = match.start()
            
        spatial_integration = file[int(underscore_indices[4]+1):int(underscore_indices[4]+2)]
        stable_formulation = file[int(underscore_indices[7]+1):int(underscore_indices[7]+2)]
        material_model = file[int(underscore_indices[8]+1):int(point_idx-3)]     

        if stable_formulation == '1':
            if skip_stable_formulation:
                continue
        else:
            if skip_unstable_formulation:
                continue
            
        if spatial_integration == '1':
            if skip_spatial_integration:
                continue
        else:
            if skip_material_integration:
                continue

        line_style_stress = 'solid'
        line_style_jacobian = 'dotted'
        
        line_color = 'black'
        if spatial_integration == '1':
            Omega_0_or_t = '$\Omega_t$';
            if stable_formulation == '1':
                line_color = 'tab:blue'
            else:
                line_color = 'tab:green'
        else:
            Omega_0_or_t = '$\Omega_0$';
            if skip_material_integration:
                continue
            if stable_formulation == '1':
                line_color = 'tab:red'
            else:
                line_color = 'tab:orange'
 
        line_width = 1.0
        if material_model == "StVenantKirchhoff":
            if skip_STVK:
                continue
            line_width = 0.5
        elif material_model == "CompressibleNeoHookean":
            if skip_cNH:
                continue
            line_width = 1.0
        elif material_model == "IncompressibleNeoHookean":
            if skip_iNH:
                continue
            line_width = 1.5
        elif material_model == "IncompressibleFibrousTissue":
            if skip_iHGO:
                continue
            line_width = 2.0
            
        if not skip_stress:
            plt.loglog(rows[:,0], rows[:,1], label=material_model + ', ' + Omega_0_or_t + \
            ', stable: ' + stable_formulation + ', |stress|', color=line_color, \
            linestyle=line_style_stress, linewidth=line_width)
        
        if not skip_jacobian:
            plt.loglog(rows[:,0], rows[:,2], label=material_model + ', ' + Omega_0_or_t + \
            ', stable: ' + stable_formulation + ', |Jacobian|', color=line_color, \
            linestyle=line_style_jacobian, linewidth=line_width)
 
    plt.title("Relativer Fehler $\epsilon_\mathrm{rel} = \mathrm{max}_i || (.)_\mathrm{f64} - (.)_\mathrm{f32} ||_\infty / ||(.)_\mathrm{f64}||_\infty$")
    plt.legend()

    plt.xlabel('Grad $\mathbf{u}$ scale')
    plt.ylabel('Relativer Fehler $\epsilon_\mathrm{rel}$')

    x_min = 1e-8
    x_max = 1e3
    y_min = 1e-10
    y_max = 1e3
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)
    plt.show()
    
