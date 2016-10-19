/*
 * PrintInputParameters.h
 *
 *  Created on: Aug 8, 2016
 *      Author: krank
 */

#ifndef INCLUDE_PRINTINPUTPARAMETERS_H_
#define INCLUDE_PRINTINPUTPARAMETERS_H_

namespace PrintInputParams
{
  void Header(ConditionalOStream & pcout)
  {
    pcout << std::endl << std::endl << std::endl
    << "_________________________________________________________________________________" << std::endl
    << "                                                                                 " << std::endl
    << "                High-order discontinuous Galerkin solver for the                 " << std::endl
    << "                     incompressible Navier-Stokes equations                      " << std::endl
    << "                based on a semi-explicit dual-splitting approach                 " << std::endl
    << "_________________________________________________________________________________" << std::endl
    << std::endl;
  }

  template<int dim>
  void print_solver_parameters(ConditionalOStream & pcout, const InputParametersNavierStokes<dim> & param)
  {
    pcout << std::endl << "general solver parameters:" << std::endl;
    pcout << " - viscosity:                           " << param.viscosity << std::endl;
    pcout << " - IP_factor_pressure:                  " << param.IP_factor_pressure << std::endl;
    pcout << " - IP_factor_viscous:                   " << param.IP_factor_viscous << std::endl;
    pcout << " - penalty factor divergence:           " << param.penalty_factor_divergence << std::endl;
    pcout << " - penalty factor continuity:           " << param.penalty_factor_continuity << std::endl;
    pcout << " - restart interval time:               " << param.restart_interval_time << std::endl;
    pcout << " - restart interval wall time:          " << param.restart_interval_wall_time << std::endl;
    pcout << " - max number of time steps:            " << param.max_number_of_time_steps << std::endl;
    pcout << " - prefix:                              " << param.output_data.output_prefix << std::endl;
  }

  template<int dim>
  void print_xwall_parameters(ConditionalOStream & pcout, const InputParametersNavierStokes<dim> & param, const int N_Q_POINTS_1D_XWALL)
  {
    pcout << std::endl << "xwall parameters:" << std::endl;
    pcout << " - number of quad points for xwall:     " << N_Q_POINTS_1D_XWALL << std::endl;
    pcout << " - fix tauw to 1.0:                     " << not param.variabletauw << std::endl;
    pcout << " - max wall distance of xwall:          " << param.max_wdist_xwall << std::endl;
    pcout << " - increment of tauw:                   " << param.dtauw << std::endl;
  }

  template<int dim>
  void print_turbulence_parameters(ConditionalOStream & pcout, const InputParametersNavierStokes<dim> & param,const double grid_stretch_fac)
  {
    pcout << std::endl << "turbulence parameters:" << std::endl;
    pcout << " - Smagorinsky constant                 " << param.cs << std::endl;
    pcout << " - grid stretching:                     " << grid_stretch_fac << std::endl;
    pcout << " - statistics start time:               " << param.turb_stat_data.statistics_start_time << std::endl;
    pcout << " - statistics every:                    " << param.turb_stat_data.statistics_every << std::endl;
    pcout << " - statistics end time:                 " << param.turb_stat_data.statistics_end_time << std::endl;
  }

  template<int dim>
  void print_linear_solver_tolerances_dual_splitting(ConditionalOStream & pcout, const InputParametersNavierStokes<dim> & param)
  {
    pcout << std::endl << "solver tolerances:" << std::endl;
    pcout << " - Poisson problem (abs)                " << param.abs_tol_pressure << std::endl;
    pcout << " - Poisson problem (rel)                " << param.rel_tol_pressure << std::endl;
    pcout << " - Projection (abs)                     " << param.abs_tol_projection << std::endl;
    pcout << " - Projection (rel)                     " << param.rel_tol_projection << std::endl;
    pcout << " - Helmholtz problem (abs)              " << param.abs_tol_viscous << std::endl;
    pcout << " - Helmholtz problem (rel)              " << param.rel_tol_viscous << std::endl;
  }

  void print_spatial_discretization(ConditionalOStream & pcout, const int dim, const int refinements, const int cells, const int faces, const int verteces)
  {
    pcout << std::endl
          << "Generating grid for "     << dim << "-dimensional problem" << std::endl << std::endl
          << "  number of refinements:" << std::fixed << std::setw(10) << std::right << refinements << std::endl
          << "  number of cells:      " << std::fixed << std::setw(10) << std::right << cells << std::endl
          << "  number of faces:      " << std::fixed << std::setw(10) << std::right << faces << std::endl
          << "  number of vertices:   " << std::fixed << std::setw(10) << std::right << verteces << std::endl;
  }
}

#endif /* INCLUDE_PRINTINPUTPARAMETERS_H_ */
