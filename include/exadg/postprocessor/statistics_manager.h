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
#include <exadg/postprocessor/time_control_statistics.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
// turbulent channel data

struct TurbulentChannelData
{
  TurbulentChannelData()
    : cells_are_stretched(false),
      viscosity(1.0),
      density(1.0),
      directory("output/"),
      filename("channel")
  {
  }

  void
  print(dealii::ConditionalOStream & pcout)
  {
    if(time_control_data_statistics.time_control_data.is_active)
    {
      pcout << "  Turbulent channel statistics:" << std::endl;

      // only implemented for unsteady problem
      pcout << "    Time control:" << std::endl;
      time_control_data_statistics.print(pcout, true /*unsteady*/);

      print_parameter(pcout, "Cells are stretched", cells_are_stretched);
      print_parameter(pcout, "Dynamic viscosity", viscosity);
      print_parameter(pcout, "Density", density);
      print_parameter(pcout, "Directory of output files", directory);
      print_parameter(pcout, "Filename", filename);
    }
  }

  TimeControlDataStatistics time_control_data_statistics;

  // are cells stretched, i.e., is a volume manifold applied?
  bool cells_are_stretched;

  // dynamic viscosity
  double viscosity;

  // density
  double density;

  // directory and filename
  std::string directory;
  std::string filename;
};

template<int dim, typename Number>
class StatisticsManager
{
public:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  StatisticsManager(dealii::DoFHandler<dim> const & dof_handler_velocity,
                    dealii::Mapping<dim> const &    mapping);

  // The argument grid_transform indicates how the y-direction that is initially distributed from
  // [0,1] is mapped to the actual grid. This must match the transformation applied to the
  // triangulation, otherwise the identification of data will fail
  void
  setup(std::function<double(double const &)> const & grid_tranform,
        TurbulentChannelData const &                  data);

  void
  evaluate(VectorType const & velocity, bool const unsteady);

  void
  write_output();

  void
  reset();

  TimeControlStatistics time_control_statistics;

private:
  static unsigned int const n_points_y_per_cell_linear = 11;
  unsigned int              n_points_y_per_cell;

  void
  evaluate_statistics(VectorType const & velocity);

  void
  evaluate_statistics(const std::vector<VectorType> & velocity);

  void
  do_evaluate(const std::vector<VectorType const *> & velocity);

  void
  do_write_output(std::string const filename, double const dynamic_viscosity, double const density);

  dealii::DoFHandler<dim> const & dof_handler;
  dealii::Mapping<dim> const &    mapping;
  MPI_Comm                        mpi_comm;

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

  TurbulentChannelData data;
};

} // namespace ExaDG

#endif
