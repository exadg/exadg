/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2022 by the ExaDG authors
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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_H_

#include <deal.II/lac/la_parallel_vector.h>

// application
#include <exadg/fluid_structure_interaction/precice/precice_adapter.h>
#include <exadg/fluid_structure_interaction/precice/precice_parameters.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/fluid.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/structure.h>
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>
// grid
#include <exadg/grid/get_dynamic_mapping.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace FSI
{
namespace preCICE
{
template<int dim, typename Number>
class Driver
{
private:
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

public:
  Driver(std::string const &                           input_file,
         MPI_Comm const &                              comm,
         std::shared_ptr<ApplicationBase<dim, Number>> app,
         bool const                                    is_test)
    : mpi_comm(comm),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(comm) == 0),
      application(app),
      precice_parameters(ExaDG::preCICE::ConfigurationParameters(input_file)),
      is_test(is_test)
  {
    print_general_info<Number>(pcout, mpi_comm, is_test);
  }

  virtual void
  setup() = 0;

  virtual void
  solve() const = 0;

  virtual void
  print_performance_results(double const total_time) const = 0;

  virtual ~Driver() = default;

protected:
  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>>                  application;
  std::shared_ptr<ExaDG::preCICE::Adapter<dim, dim, VectorType>> precice;
  ExaDG::preCICE::ConfigurationParameters                        precice_parameters;

  // Nomenclature
  // time window: denotes time interval after which FSi coupling takes place
  // subcycling: a single field solver performs several time steps until the next FSI coupling time
  // window is reached
  // time step size: refers to single field solvers. In case of subcycling, one
  // time window consists of several time step sizes.
  // serial coupling scheme: the single-field solvers are called sequentially/consecutively with
  // data transfer from one field to another after in single-field solver.
  // parallel coupling scheme: the single-field solvers are called simultaneously with data transfer
  // in both directions afterwards.

  // The time-window size is determined through the preCICE configuration by the user:
  // The <coupling-scheme.. /> tag has options for
  // <time-window-size method="fixed" ... /> which refers to a constant time window size
  // <time-window-size method="first-participant" ... /> which refers to an adaptive time window
  // size prescribed by the first (which is usually the Fluid) participant. The latter is only
  // applicable in serial coupling schemes.
  // Both solvers can exploit subcycling, but they need to synchronize at the end of a certain
  // time-window size. Therefore, the allowed time-step size until we reach the next synchronization
  // point needs to be determined.

  // maximum allowed time-step size until we reach a new coupling time window
  mutable double time_until_next_coupling_window = 0;

  // do not print wall times if is_test
  bool const is_test;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;
};

} // namespace preCICE
} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_H_ */
