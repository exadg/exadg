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
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>
#include <exadg/fluid_structure_interaction_precice/precice_adapter.h>
#include <exadg/fluid_structure_interaction_precice/precice_parameter.h>

namespace ExaDG
{
namespace FSI
{
using namespace dealii;

template<int dim, typename Number>
class Driver
{
private:
  using VectorType = LinearAlgebra::distributed::Vector<Number>;

public:
  Driver(std::string const & input_file, MPI_Comm const & comm, bool const is_test)
    : mpi_comm(comm),
      pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0),
      is_test(is_test)
  {
    dealii::ParameterHandler prm;

    add_parameters(prm);

    precice_parameters.add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  static void
  add_parameters(dealii::ParameterHandler & prm)
  {
    (void)prm;
  }

  virtual void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const                            degree_fluid,
        unsigned int const                            degree_structure,
        unsigned int const                            refine_space_fluid,
        unsigned int const                            refine_space_structure) = 0;

  virtual void
  solve() const = 0;

  virtual void
  print_performance_results(double const total_time) const = 0;

protected:
  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>>           application;
  std::shared_ptr<Adapter::Adapter<dim, dim, VectorType>> precice;
  Parameters::PreciceAdapterConfiguration                 precice_parameters;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;
};
} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_PRECICE_DRIVER_H_ */
