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
#include <exadg/incompressible_navier_stokes/precursor/user_interface/application_base.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/create_operator.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/create_time_integrator.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace IncNS
{
namespace Precursor
{
template<int dim, typename Number>
class Solver
{
public:
  void
  setup(std::shared_ptr<Domain<dim, Number>> & domain,
        std::vector<std::string> const &       subsection_names_parameters,
        std::string const &                    field,
        MPI_Comm const &                       mpi_comm,
        bool const                             is_test)
  {
    // setup application
    domain->setup(grid, mapping, subsection_names_parameters);

    // TODO: needs to be shifted to application in order to allow mappings realized as
    // MappingDoFVector
    multigrid_mappings = std::make_shared<MultigridMappings<dim, Number>>(mapping);

    // ALE is not used for this solver
    std::shared_ptr<HelpersALE<Number>> helpers_ale_dummy;

    // initialize pde_operator
    pde_operator = create_operator<dim, Number>(grid,
                                                mapping,
                                                multigrid_mappings,
                                                domain->get_boundary_descriptor(),
                                                domain->get_field_functions(),
                                                domain->get_parameters(),
                                                field,
                                                mpi_comm);

    // initialize matrix_free
    matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
    matrix_free_data->append(pde_operator);

    matrix_free = std::make_shared<dealii::MatrixFree<dim, Number>>();
    if(domain->get_parameters().use_cell_based_face_loops)
      Categorization::do_cell_based_loops(*grid->triangulation, matrix_free_data->data);
    matrix_free->reinit(*mapping,
                        matrix_free_data->get_dof_handler_vector(),
                        matrix_free_data->get_constraint_vector(),
                        matrix_free_data->get_quadrature_vector(),
                        matrix_free_data->data);

    // setup Navier-Stokes operator
    pde_operator->setup(matrix_free, matrix_free_data);

    // setup postprocessor
    postprocessor = domain->create_postprocessor();
    postprocessor->setup(*pde_operator);

    // Setup time integrator
    time_integrator = create_time_integrator<dim, Number>(
      pde_operator, helpers_ale_dummy, postprocessor, domain->get_parameters(), mpi_comm, is_test);

    time_integrator->setup(domain->get_parameters().restarted_simulation);
  }

  /*
   * Grid and mapping
   */
  std::shared_ptr<Grid<dim>>            grid;
  std::shared_ptr<dealii::Mapping<dim>> mapping;

  std::shared_ptr<MultigridMappings<dim, Number>> multigrid_mappings;

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

private:
  /*
   * MatrixFree
   */
  std::shared_ptr<MatrixFreeData<dim, Number>>     matrix_free_data;
  std::shared_ptr<dealii::MatrixFree<dim, Number>> matrix_free;
};

template<int dim, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const &                              mpi_comm,
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
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  void
  consistency_checks() const;

  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  dealii::ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  Solver<dim, Number> solver_main, solver_precursor;

  bool use_adaptive_time_stepping;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;
};

} // namespace Precursor
} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_PRECURSOR_H_ */
