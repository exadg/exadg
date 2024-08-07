
# Plot the stability test results in a single graph for an overview.

import numpy as np
import math
import matplotlib.pyplot as plt
import glob, os
import re
import scipy.interpolate

if __name__ == "__main__":

    skip_jacobian = False
    skip_stress = False

    skip_STVK = True
    skip_cNH  = True
    skip_iNH  = True
    skip_iHGO = not True
    skip_spatial_integration  = False
    skip_material_integration = False
    skip_stable_formulation   = False
    skip_unstable_formulation = False

    sampling = 5

    # Read the all the txt files and plot the stability test results
    plt.figure()

    # plot horizontal lines at specific error margins
    #plt.loglog([0.0, 1e20], [1e-6, 1e-6], label=None, color='black', linestyle='dotted', linewidth=1.0)
    plt.loglog([0.0, 1e20], [1e-5, 1e-5], label=None, color='black', linestyle='dotted', linewidth=1.0)
    #plt.loglog([0.0, 1e20], [1e-4, 1e-4], label=None, color='black', linestyle='dotted', linewidth=1.0)
    plt.loglog([0.0, 1e20], [1e-3, 1e-3], label=None, color='black', linestyle='dotted', linewidth=1.0)
    #plt.loglog([0.0, 1e20], [1e-2, 1e-2], label=None, color='black', linestyle='dotted', linewidth=1.0)
    plt.loglog([0.0, 1e20], [1e-1, 1e-1], label=None, color='black', linestyle='dotted', linewidth=1.0)
    #plt.loglog([0.0, 1e20], [1e-0, 1e-0], label=None, color='black', linestyle='dotted', linewidth=1.0)
    plt.loglog([0.0, 1e20], [1e+1, 1e+1], label=None, color='black', linestyle='dotted', linewidth=1.0)
    #plt.loglog([0.0, 1e20], [1e+2, 1e+2], label=None, color='black', linestyle='dotted', linewidth=1.0)
    plt.loglog([0.0, 1e20], [1e+3, 1e+3], label=None, color='black', linestyle='dotted', linewidth=1.0)
    #plt.loglog([0.0, 1e20], [1e+4, 1e+4], label=None, color='black', linestyle='dotted', linewidth=1.0)

    #os.chdir("/home/richardschussnig/dealii-candi/exadg/build/applications/structure/stability_test/")
    #os.chdir("/home/richard/dealii-candi/exadg/build/applications/structure/stability_test/")

    #os.chdir("/home/richard/dealii-candi/exadg/build/applications/structure/stability_test/stability_results/")
    #os.chdir("/home/richard/dealii-candi/exadg/build/applications/structure/stability_test/stability_results_FastExp/")
    #os.chdir("/home/richard/dealii-candi/exadg/build/applications/structure/stability_test/stability_results_FastExp_JpowNewton/")
    os.chdir("/home/richard/dealii-candi/exadg/build/applications/structure/stability_test/stability_results_JpowNewton/")

    files = sorted(glob.glob("stability_forward_test_*"))
    for file in files:

        print("Parsing file:")
        print(file)
        print("")

        # Parse each file.
        f = open(file, 'r')

        contained_infinite_values = False

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
                val = float(row[i])
                if math.isinf(val) or math.isnan(val):
                    contained_infinite_values = True
                if val > 1e20:
                    val = 1e20

                rows[idx_line-n_header_lines-1, i] = val

        if contained_infinite_values:
            print("INF/NAN VALUES IN: " + file + "\n")

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
            stable_formulation_lgd_entry = "stable"
            if skip_stable_formulation:
                continue
        else:
            stable_formulation_lgd_entry = "standard"
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
        abbreviation_material_model = ""
        if material_model == "StVenantKirchhoff":
            if skip_STVK:
                continue
            #line_width = 0.5
            abbreviation_material_model = "St.Venant-Kirchhoff"
        elif material_model == "CompressibleNeoHookean":
            if skip_cNH:
                continue
            #line_width = 0.5
            abbreviation_material_model = "cNH"
        elif material_model == "IncompressibleNeoHookean":
            if skip_iNH:
                continue
            #line_width = 0.5
            abbreviation_material_model = "iNH"
        elif material_model == "IncompressibleFibrousTissue":
            if skip_iHGO:
                continue
            #line_width = 0.5
            abbreviation_material_model = "fiber"

        if not skip_stress:
            plt.loglog(rows[1:-1:sampling,0], rows[1:-1:sampling,1], label=abbreviation_material_model + ', ' + Omega_0_or_t + \
            ', ' + stable_formulation_lgd_entry + ', stress', color=line_color, \
            linestyle=line_style_stress, linewidth=line_width)

        if not skip_jacobian:
            plt.loglog(rows[1:-1:sampling,0], rows[1:-1:sampling,2], label=abbreviation_material_model + ', ' + Omega_0_or_t + \
            ', ' + stable_formulation_lgd_entry + ', D/Du stress', color=line_color, \
            linestyle=line_style_jacobian, linewidth=line_width)

    #plt.title("Relative error $\epsilon_\mathrm{rel} = \mathrm{max}_i || (.)_\mathrm{f64} - (.)_\mathrm{f32} ||_\infty / ||(.)_\mathrm{f64}||_\infty$")
    plt.legend()

    plt.xlabel('Grad $\mathbf{u}$ scale')
    plt.ylabel('Relative error $\epsilon_\mathrm{rel}$')

    x_min = 1e-8
    x_max = 1e2  # 1e0
    y_min = 1e-7 # 1e-21
    y_max = 1e4  # 1e21
    plt.xlim(xmin=x_min, xmax=x_max)
    plt.ylim(ymin=y_min, ymax=y_max)

    # plt.autoscale(enable=True, axis='x', tight=True)
    # plt.autoscale(enable=True, axis='y', tight=True)

    plt.show()

