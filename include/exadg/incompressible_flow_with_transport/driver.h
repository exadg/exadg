/*
 * driver.h
 *
 *  Created on: 31.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_DRIVER_H_
#define INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_DRIVER_H_

// application
#include <exadg/incompressible_flow_with_transport/user_interface/application_base.h>

// utilities
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/mapping_degree.h>
#include <exadg/grid/moving_mesh_function.h>
#include <exadg/matrix_free/matrix_free_wrapper.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/timer_tree.h>

// ConvDiff
#include <exadg/convection_diffusion/spatial_discretization/dg_operator.h>
#include <exadg/convection_diffusion/time_integration/time_int_bdf.h>
#include <exadg/convection_diffusion/time_integration/time_int_explicit_runge_kutta.h>

// IncNS
#include <exadg/incompressible_navier_stokes/spatial_discretization/dg_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/dg_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/dg_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/driver_steady_problems.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>

namespace ExaDG
{
namespace FTI
{
using namespace dealii;

template<int dim, typename Number = double>
class Driver
{
public:
  Driver(MPI_Comm const & mpi_comm);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const                            degree,
        unsigned int const                            refine_space,
        bool const                                    is_test);

  void
  solve() const;

  void
  print_performance_results(double const total_time, bool const is_test) const;

private:
  void
  ale_update() const;

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
  unsigned int n_scalars;

  // output to std::cout
  ConditionalOStream pcout;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  /*
   * Mesh
   */

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;

  // mapping
  std::shared_ptr<Mapping<dim>> mapping;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  // mesh (static or moving)
  std::shared_ptr<Mesh<dim>> mesh;

  // moving mesh (ALE)
  std::shared_ptr<MovingMeshFunction<dim, Number>> moving_mesh;

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

  // steady solver
  typedef IncNS::DriverSteadyProblems<dim, Number> DriverSteady;

  std::shared_ptr<DriverSteady> fluid_driver_steady;

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

  mutable unsigned int N_time_steps;
};

} // namespace FTI
} // namespace ExaDG


#endif /* INCLUDE_EXADG_INCOMPRESSIBLE_FLOW_WITH_TRANSPORT_DRIVER_H_ */
