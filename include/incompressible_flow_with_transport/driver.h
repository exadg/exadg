/*
 * driver.h
 *
 *  Created on: 31.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_DRIVER_H_
#define INCLUDE_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_DRIVER_H_

// application
#include "user_interface/application_base.h"


// GENERAL

#include "../functions_and_boundary_conditions/verify_boundary_conditions.h"
#include "../grid/mapping_degree.h"
#include "../grid/moving_mesh.h"
#include "../matrix_free/matrix_free_wrapper.h"
#include "../utilities/print_functions.h"
#include "../utilities/timings_hierarchical.h"

// CONVECTION-DIFFUSION

// spatial discretization
#include "../convection_diffusion/spatial_discretization/dg_operator.h"

// time integration
#include "../convection_diffusion/time_integration/time_int_bdf.h"
#include "../convection_diffusion/time_integration/time_int_explicit_runge_kutta.h"


// NAVIER-STOKES

// spatial discretization
#include "../incompressible_navier_stokes/spatial_discretization/dg_coupled_solver.h"
#include "../incompressible_navier_stokes/spatial_discretization/dg_dual_splitting.h"
#include "../incompressible_navier_stokes/spatial_discretization/dg_pressure_correction.h"

// temporal discretization
#include "../incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "../incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h"
#include "../utilities/print_general_infos.h"

namespace FTI
{
template<int dim, typename Number = double>
class Driver
{
public:
  Driver(MPI_Comm const & mpi_comm, unsigned int const n_scalars);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const &                          degree,
        unsigned int const &                          refine_space);

  void
  solve() const;

  void
  print_statistics(double const total_time) const;

private:
  void
  print_header() const;

  void
  communicate_scalar_to_fluid() const;

  void
  communicate_fluid_to_all_scalars() const;

  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

  // MPI communicator
  MPI_Comm const & mpi_comm;

  // number of scalar quantities
  unsigned int const n_scalars;

  // output to std::cout
  ConditionalOStream pcout;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  /*
   * Mesh
   */

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  // mapping (static and moving meshes)
  std::shared_ptr<Mesh<dim>>                         mesh;
  std::shared_ptr<MovingMeshAnalytical<dim, Number>> moving_mesh;

  bool use_adaptive_time_stepping;

  // INCOMPRESSIBLE NAVIER-STOKES
  std::shared_ptr<IncNS::FieldFunctions<dim>>      fluid_field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> fluid_boundary_descriptor_velocity;
  std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> fluid_boundary_descriptor_pressure;

  IncNS::InputParameters fluid_param;

  //  MatrixFree
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;

  typedef IncNS::DGNavierStokesBase<dim, Number>               DGBase;
  typedef IncNS::DGNavierStokesCoupled<dim, Number>            DGCoupled;
  typedef IncNS::DGNavierStokesDualSplitting<dim, Number>      DGDualSplitting;
  typedef IncNS::DGNavierStokesPressureCorrection<dim, Number> DGPressureCorrection;

  std::shared_ptr<DGBase>               navier_stokes_operator;
  std::shared_ptr<DGCoupled>            navier_stokes_operator_coupled;
  std::shared_ptr<DGDualSplitting>      navier_stokes_operator_dual_splitting;
  std::shared_ptr<DGPressureCorrection> navier_stokes_operator_pressure_correction;

  typedef IncNS::PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> fluid_postprocessor;

  typedef IncNS::TimeIntBDF<dim, Number>                   TimeInt;
  typedef IncNS::TimeIntBDFCoupled<dim, Number>            TimeIntCoupled;
  typedef IncNS::TimeIntBDFDualSplitting<dim, Number>      TimeIntDualSplitting;
  typedef IncNS::TimeIntBDFPressureCorrection<dim, Number> TimeIntPressureCorrection;

  std::shared_ptr<TimeInt> fluid_time_integrator;

  // SCALAR TRANSPORT

  std::vector<ConvDiff::InputParameters> scalar_param;

  std::vector<std::shared_ptr<ConvDiff::FieldFunctions<dim>>>     scalar_field_functions;
  std::vector<std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>>> scalar_boundary_descriptor;

  std::vector<std::shared_ptr<ConvDiff::DGOperator<dim, Number>>> conv_diff_operator;

  std::vector<std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>> scalar_postprocessor;

  std::vector<std::shared_ptr<TimeIntBase>> scalar_time_integrator;

  mutable LinearAlgebra::distributed::Vector<Number> temperature;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;
};

} // namespace FTI



#endif /* INCLUDE_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_DRIVER_H_ */
