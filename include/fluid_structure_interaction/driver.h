/*
 * driver.h
 *
 *  Created on: 01.04.2020
 *      Author: fehn
 */

#ifndef INCLUDE_FLUID_STRUCTURE_INTERACTION_DRIVER_H_
#define INCLUDE_FLUID_STRUCTURE_INTERACTION_DRIVER_H_

// application
#include "user_interface/application_base.h"

// IncNS: postprocessor
#include "../incompressible_navier_stokes/postprocessor/postprocessor.h"

// IncNS: spatial discretization
#include "../incompressible_navier_stokes/spatial_discretization/dg_coupled_solver.h"
#include "../incompressible_navier_stokes/spatial_discretization/dg_dual_splitting.h"
#include "../incompressible_navier_stokes/spatial_discretization/dg_pressure_correction.h"

// IncNS: temporal discretization
#include "../incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "../incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h"

// Poisson: spatial discretization
#include "../poisson/spatial_discretization/operator.h"

// grid
#include "../grid/mapping_degree.h"
#include "../grid/moving_mesh.h"

// matrix-free
#include "../matrix_free/matrix_free_wrapper.h"

// functionalities
#include "../functions_and_boundary_conditions/interface_coupling.h"
#include "../functions_and_boundary_conditions/verify_boundary_conditions.h"
#include "../utilities/print_general_infos.h"

namespace FSI
{
template<int dim, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const & comm);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const &                          degree_fluid,
        unsigned int const &                          degree_poisson,
        unsigned int const &                          refine_space);


  void
  solve() const;

  void
  analyze_computing_times() const;

private:
  void
  print_header() const;

  // MPI communicator
  MPI_Comm const & mpi_comm;

  // output to std::cout
  ConditionalOStream pcout;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  /****************************************** FLUID *******************************************/

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> fluid_triangulation;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    fluid_periodic_faces;

  // solve mesh deformation by a Poisson problem
  Poisson::InputParameters poisson_param;

  std::shared_ptr<Poisson::FieldFunctions<dim>>        poisson_field_functions;
  std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> poisson_boundary_descriptor;

  // static mesh for Poisson problem
  std::shared_ptr<Mesh<dim>> poisson_mesh;

  std::shared_ptr<MatrixFreeWrapper<dim, Number>>      poisson_matrix_free_wrapper;
  std::shared_ptr<Poisson::Operator<dim, Number, dim>> poisson_operator;

  IncNS::InputParameters fluid_param;

  std::shared_ptr<IncNS::FieldFunctions<dim>>      fluid_field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> fluid_boundary_descriptor_velocity;
  std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> fluid_boundary_descriptor_pressure;

  // moving mesh for fluid problem
  std::shared_ptr<Mesh<dim>>                   fluid_mesh;
  std::shared_ptr<MovingMeshBase<dim, Number>> fluid_moving_mesh;

  std::shared_ptr<MatrixFreeWrapper<dim, Number>> fluid_matrix_free_wrapper;

  // Spatial discretization
  typedef IncNS::DGNavierStokesBase<dim, Number>               DGBase;
  typedef IncNS::DGNavierStokesCoupled<dim, Number>            DGCoupled;
  typedef IncNS::DGNavierStokesDualSplitting<dim, Number>      DGDualSplitting;
  typedef IncNS::DGNavierStokesPressureCorrection<dim, Number> DGPressureCorrection;

  std::shared_ptr<DGBase>               fluid_operator;
  std::shared_ptr<DGCoupled>            fluid_operator_coupled;
  std::shared_ptr<DGDualSplitting>      fluid_operator_dual_splitting;
  std::shared_ptr<DGPressureCorrection> fluid_operator_pressure_correction;

  // Temporal discretization
  typedef IncNS::TimeIntBDF<dim, Number>                   TimeInt;
  typedef IncNS::TimeIntBDFCoupled<dim, Number>            TimeIntCoupled;
  typedef IncNS::TimeIntBDFDualSplitting<dim, Number>      TimeIntDualSplitting;
  typedef IncNS::TimeIntBDFPressureCorrection<dim, Number> TimeIntPressureCorrection;

  std::shared_ptr<TimeInt> fluid_time_integrator;

  // Postprocessor
  typedef IncNS::PostProcessorBase<dim, Number> Postprocessor;
  std::shared_ptr<Postprocessor>                fluid_postprocessor;

  /****************************************** FLUID *******************************************/


  /**************************************** STRUCTURE *****************************************/

  // TODO

  /**************************************** STRUCTURE *****************************************/


  /******************************* FLUID - STRUCTURE - INTERFACE ******************************/

  // TODO
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> structure_to_fluid;
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> structure_to_moving_mesh;
  std::shared_ptr<InterfaceCoupling<dim, dim, Number>> fluid_to_structure;

  /******************************* FLUID - STRUCTURE - INTERFACE ******************************/

  /*
   * Computation time (wall clock time).
   */
  Timer          timer;
  mutable double overall_time;
  double         setup_time;
  mutable double ale_update_time;
};

} // namespace FSI


#endif /* INCLUDE_FLUID_STRUCTURE_INTERACTION_DRIVER_H_ */
