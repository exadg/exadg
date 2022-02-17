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

#ifndef INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_DRIVER_H_
#define INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_DRIVER_H_

// application
#include <exadg/fluid_structure_interaction/user_interface/application_base.h>

// FSI
#include <exadg/fluid_structure_interaction/acceleration_schemes/parameters.h>
#include <exadg/fluid_structure_interaction/acceleration_schemes/partitioned_solver.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/fluid.h>
#include <exadg/fluid_structure_interaction/single_field_solvers/structure.h>

// utilities
#include <exadg/functions_and_boundary_conditions/interface_coupling.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace FSI
{
template<int dim, typename Number>
class Driver
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  Driver(std::string const &                           input_file,
         MPI_Comm const &                              comm,
         std::shared_ptr<ApplicationBase<dim, Number>> application,
         bool const                                    is_test);

  void
  setup();

  void
  solve() const;

  void
  print_performance_results(double const total_time) const;

private:
  void
  setup_interface_coupling();

  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  void
  coupling_structure_to_ale(VectorType const & displacement_structure) const;

  void
  coupling_structure_to_fluid(bool const extrapolate) const;

  void
  coupling_fluid_to_structure(bool const end_of_time_step) const;

  void
  apply_dirichlet_neumann_scheme(VectorType &       d_tilde,
                                 VectorType const & d,
                                 unsigned int       iteration) const;

  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  std::shared_ptr<SolverStructure<dim, Number>> structure;

  std::shared_ptr<SolverFluid<dim, Number>> fluid;

  // interface coupling
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> structure_to_fluid;
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> structure_to_ale;
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> fluid_to_structure;

  // Parameters for partitioned FSI schemes
  Parameters parameters;

  // Computation time
  mutable TimerTree timer_tree;

  // Partitioned FSI solver
  std::shared_ptr<PartitionedSolver<dim, Number>> partitioned_solver;
};

} // namespace FSI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_FLUID_STRUCTURE_INTERACTION_DRIVER_H_ */
