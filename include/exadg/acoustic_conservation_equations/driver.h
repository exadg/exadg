/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_DRIVER_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_DRIVER_H_

#include <exadg/acoustic_conservation_equations/postprocessor/postprocessor_base.h>
#include <exadg/acoustic_conservation_equations/spatial_discretization/spatial_operator.h>
#include <exadg/acoustic_conservation_equations/time_integration/time_int_abm.h>
#include <exadg/acoustic_conservation_equations/user_interface/application_base.h>
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/get_dynamic_mapping.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/finite_element.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace Acoustics
{
enum class OperatorType
{
  AcousticOperator,   // gradient operator for scalar pressure and divergence operator for
                      // vectorial velocity
  InverseMassOperator // inverse mass operator: vectorial quantity (velocity) and scalar quantity
                      // (pressure)
};

inline unsigned int
get_dofs_per_element(unsigned int const       dim,
                     unsigned int const       degree,
                     ExaDG::ElementType const element_type)
{
  unsigned int const pressure_dofs_per_element =
    ExaDG::get_dofs_per_element(element_type, true /* is_dg */, 1 /* n_components */, degree, dim);

  unsigned int const velocity_dofs_per_element = ExaDG::get_dofs_per_element(
    element_type, true /* is_dg */, dim /* n_components */, degree, dim);

  return velocity_dofs_per_element + pressure_dofs_per_element;
}

template<int dim, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const &                              comm,
         std::shared_ptr<ApplicationBase<dim, Number>> application,
         bool const                                    is_test,
         bool const                                    is_throughput_study);

  void
  setup();

  void
  solve() const;

  void
  print_performance_results(double const total_time) const;

  /*
   * Throughput study
   */
  std::tuple<unsigned int, dealii::types::global_dof_index, double>
  apply_operator(OperatorType const & operator_type,
                 unsigned int const   n_repetitions_inner,
                 unsigned int const   n_repetitions_outer) const;

private:
  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // do not set up certain data structures (solver, postprocessor) in case of throughput study
  bool const is_throughput_study;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  std::shared_ptr<Grid<dim>> grid;

  std::shared_ptr<dealii::Mapping<dim>> mapping;

  /*
   * Spatial discretization
   */
  std::shared_ptr<SpatialOperator<dim, Number>> pde_operator;

  /*
   * Postprocessor
   */
  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;

  /*
   * Temporal discretization
   */

  // unsteady solver
  std::shared_ptr<TimeIntAdamsBashforthMoulton<Number>> time_integrator;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;
};

} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_DRIVER_H_ */
