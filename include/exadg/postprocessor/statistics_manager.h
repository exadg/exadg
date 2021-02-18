/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_POSTPROCESSOR_STATISTICS_MANAGER_H_
#define INCLUDE_EXADG_POSTPROCESSOR_STATISTICS_MANAGER_H_

// deal.II
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
using namespace dealii;

// turbulent channel data

struct TurbulentChannelData
{
  TurbulentChannelData()
    : calculate_statistics(false),
      cells_are_stretched(false),
      sample_start_time(0.0),
      sample_end_time(1.0),
      sample_every_timesteps(1),
      viscosity(1.0),
      density(1.0),
      filename_prefix("indexa")
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate_statistics == true)
    {
      pcout << "  Turbulent channel statistics:" << std::endl;
      print_parameter(pcout, "Calculate statistics", calculate_statistics);
      print_parameter(pcout, "Cells are stretched", cells_are_stretched);
      print_parameter(pcout, "Sample start time", sample_start_time);
      print_parameter(pcout, "Sample end time", sample_end_time);
      print_parameter(pcout, "Sample every timesteps", sample_every_timesteps);
      print_parameter(pcout, "Dynamic viscosity", viscosity);
      print_parameter(pcout, "Density", density);
      print_parameter(pcout, "Filename prefix", filename_prefix);
    }
  }

  // calculate statistics?
  bool calculate_statistics;

  // are cells stretched, i.e., is a volume manifold applied?
  bool cells_are_stretched;

  // start time for sampling
  double sample_start_time;

  // end time for sampling
  double sample_end_time;

  // perform sampling every ... timesteps
  unsigned int sample_every_timesteps;

  // dynamic viscosity
  double viscosity;

  // density
  double density;

  std::string filename_prefix;
};

template<int dim, typename Number>
class StatisticsManager
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  StatisticsManager(DoFHandler<dim> const & dof_handler_velocity, Mapping<dim> const & mapping);

  // The argument grid_transform indicates how the y-direction that is initially distributed from
  // [0,1] is mapped to the actual grid. This must match the transformation applied to the
  // triangulation, otherwise the identification of data will fail
  void
  setup(std::function<double(double const &)> const & grid_tranform,
        TurbulentChannelData const &                  turb_channel_data);

  void
  evaluate(VectorType const & velocity, double const & time, unsigned int const & time_step_number);

  void
  evaluate(VectorType const & velocity);

  void
  evaluate(const std::vector<VectorType> & velocity);

  void
  write_output(const std::string output_prefix,
               double const      dynamic_viscosity,
               double const      density);

  void
  reset();

private:
  static unsigned int const n_points_y_per_cell_linear = 11;
  unsigned int              n_points_y_per_cell;

  void
  do_evaluate(const std::vector<VectorType const *> & velocity);

  DoFHandler<dim> const & dof_handler;
  Mapping<dim> const &    mapping;
  MPI_Comm                communicator;

  // vector of y-coordinates at which statistical quantities are computed
  std::vector<double> y_glob;

  // mean velocity <u_i>, i=1,...,d (for all y-coordinates)
  std::vector<std::vector<double>> vel_glob;

  // square velocity <u_i²>, i=1,...,d (for all y-coordinates)
  std::vector<std::vector<double>> velsq_glob;

  // <u_1*u_2> = <u*v> (for all y-coordinates)
  std::vector<double> veluv_glob;

  // number of samples
  int number_of_samples;

  bool write_final_output;

  TurbulentChannelData turb_channel_data;
};

} // namespace ExaDG

#endif
