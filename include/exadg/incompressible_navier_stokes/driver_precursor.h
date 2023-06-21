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

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_PRECURSOR_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_PRECURSOR_H_

#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor_base.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/user_interface/application_base_precursor.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
class Solver
{
public:
  void
  setup(std::shared_ptr<Grid<dim> const>               grid,
        std::shared_ptr<dealii::Mapping<dim> const>    mapping,
        std::shared_ptr<BoundaryDescriptor<dim> const> boundary_descriptor,
        std::shared_ptr<FieldFunctions<dim> const>     field_functions,
        Parameters const &                             parameters,
        std::string const &                            field,
        MPI_Comm const &                               mpi_comm,
        bool const                                     is_test)
  {
    // ALE is not used for this solver
    std::shared_ptr<HelpersALE<Number>> helpers_ale_dummy;

    // initialize pde_operator
    pde_operator = create_operator<dim, Number>(
      grid, mapping, boundary_descriptor, field_functions, parameters, field, mpi_comm);

    // initialize matrix_free
    matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
    matrix_free_data->append(pde_operator);

    matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
    if(parameters.use_cell_based_face_loops)
      Categorization::do_cell_based_loops(*grid->triangulation, matrix_free_data->data);
    matrix_free->reinit(*mapping,
                        matrix_free_data->get_dof_handler_vector(),
                        matrix_free_data->get_constraint_vector(),
                        matrix_free_data->get_quadrature_vector(),
                        matrix_free_data->data);

    // setup Navier-Stokes operator
    pde_operator->setup(matrix_free, matrix_free_data);

    // setup postprocessor
    postprocessor->setup(*pde_operator);

    // Setup time integrator
    time_integrator = create_time_integrator<dim, Number>(
      pde_operator, helpers_ale_dummy, postprocessor, parameters, mpi_comm, is_test);

    // setup time integrator before calling setup_solvers (this is necessary since the setup of the
    // solvers depends on quantities such as the time_step_size or gamma0!)
    time_integrator->setup(parameters.restarted_simulation);

    // setup solvers
    pde_operator->setup_solvers(time_integrator->get_scaling_factor_time_derivative_term(),
                                time_integrator->get_velocity());
  }

  /*
   * MatrixFree
   */
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;

  /*
   * Spatial discretization
   */
  std::shared_ptr<SpatialOperatorBase<dim, Number>> pde_operator;

  /*
   * Postprocessor
   */
  typedef PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor;

  /*
   * Temporal discretization
   */
  std::shared_ptr<TimeIntBDF<dim, Number>> time_integrator;
};

template<int dim, typename Number>
class DriverPrecursor
{
public:
  DriverPrecursor(MPI_Comm const &                                       mpi_comm,
                  std::shared_ptr<ApplicationBasePrecursor<dim, Number>> application,
                  bool const                                             is_test);

  void
  setup();

  void
  solve() const;

  void
  print_performance_results(double const total_time) const;

private:
  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // application
  std::shared_ptr<ApplicationBasePrecursor<dim, Number>> application;

  Solver<dim, Number> main, precursor;

  bool use_adaptive_time_stepping;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_PRECURSOR_H_ */
