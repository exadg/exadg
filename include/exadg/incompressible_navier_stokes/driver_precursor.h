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
#include <exadg/grid/mapping_degree.h>
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor_base.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_coupled.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/user_interface/application_base.h>
#include <exadg/matrix_free/matrix_free_wrapper.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
class DriverPrecursor
{
public:
  DriverPrecursor(MPI_Comm const & mpi_comm, bool const is_test);

  void
  setup(std::shared_ptr<ApplicationBasePrecursor<dim, Number>> application,
        unsigned int const                                     degree,
        unsigned int const                                     refine_space);

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
  ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // application
  std::shared_ptr<ApplicationBasePrecursor<dim, Number>> application;

  /*
   * Mesh
   */

  // triangulation
  std::shared_ptr<Triangulation<dim>> triangulation_pre, triangulation;

  // mapping
  std::shared_ptr<Mapping<dim>> mapping_pre, mapping;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces_pre, periodic_faces;

  std::shared_ptr<FieldFunctions<dim>>      field_functions_pre, field_functions;
  std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity_pre,
    boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure_pre,
    boundary_descriptor_pressure;

  /*
   * Parameters
   */
  InputParameters param_pre, param;

  /*
   * MatrixFree
   */
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data_pre, matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free_pre, matrix_free;

  /*
   * Spatial discretization
   */
  std::shared_ptr<SpatialOperatorBase<dim, Number>> pde_operator_pre;
  std::shared_ptr<SpatialOperatorBase<dim, Number>> pde_operator;

  /*
   * Postprocessor
   */
  typedef PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor_pre, postprocessor;

  /*
   * Temporal discretization
   */
  std::shared_ptr<TimeIntBDF<dim, Number>> time_integrator_pre, time_integrator;

  bool use_adaptive_time_stepping;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_PRECURSOR_H_ */
