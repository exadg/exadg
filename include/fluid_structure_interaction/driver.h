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

// Structure: spatial discretization
#include "../structure/spatial_discretization/operator.h"

// Structure: time integration
#include "../structure/time_integration/time_int_gen_alpha.h"

// grid
#include "../grid/mapping_degree.h"
#include "../grid/moving_mesh.h"

// matrix-free
#include "../matrix_free/matrix_free_wrapper.h"

// functionalities
#include "../functions_and_boundary_conditions/interface_coupling.h"
#include "../functions_and_boundary_conditions/verify_boundary_conditions.h"
#include "../utilities/print_general_infos.h"
#include "../utilities/timings_hierarchical.h"

namespace FSI
{
template<int dim, typename Number>
class Driver
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  Driver(MPI_Comm const & comm);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const &                          degree_fluid,
        unsigned int const &                          degree_ale,
        unsigned int const &                          degree_structure,
        unsigned int const &                          refine_space_fluid,
        unsigned int const &                          refine_space_structure);

  void
  solve() const;

  void
  print_statistics(double const total_time) const;

private:
  void
  print_header() const;

  void
  set_start_time() const;

  void
  synchronize_time_step_size() const;

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

  std::shared_ptr<MatrixFreeData<dim, Number>>         poisson_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>             poisson_matrix_free;
  std::shared_ptr<Poisson::Operator<dim, Number, dim>> poisson_operator;

  IncNS::InputParameters fluid_param;

  std::shared_ptr<IncNS::FieldFunctions<dim>>      fluid_field_functions;
  std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> fluid_boundary_descriptor_velocity;
  std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> fluid_boundary_descriptor_pressure;

  // moving mesh for fluid problem
  std::shared_ptr<Mesh<dim>>                   fluid_mesh;
  std::shared_ptr<MovingMeshBase<dim, Number>> fluid_moving_mesh;

  std::shared_ptr<MatrixFreeData<dim, Number>> fluid_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     fluid_matrix_free;

  // spatial discretization
  std::shared_ptr<IncNS::DGNavierStokesBase<dim, Number>>          fluid_operator;
  std::shared_ptr<IncNS::DGNavierStokesCoupled<dim, Number>>       fluid_operator_coupled;
  std::shared_ptr<IncNS::DGNavierStokesDualSplitting<dim, Number>> fluid_operator_dual_splitting;
  std::shared_ptr<IncNS::DGNavierStokesPressureCorrection<dim, Number>>
    fluid_operator_pressure_correction;

  // temporal discretization
  std::shared_ptr<IncNS::TimeIntBDF<dim, Number>> fluid_time_integrator;

  // Postprocessor
  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> fluid_postprocessor;

  /****************************************** FLUID *******************************************/


  /**************************************** STRUCTURE *****************************************/

  // input parameters
  Structure::InputParameters structure_param;

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> structure_triangulation;

  // mapping
  std::shared_ptr<Mesh<dim>> structure_mesh;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    structure_periodic_faces;

  // material descriptor
  std::shared_ptr<Structure::MaterialDescriptor> structure_material_descriptor;

  // boundary conditions
  std::shared_ptr<Structure::BoundaryDescriptor<dim>> structure_boundary_descriptor;

  // field functions
  std::shared_ptr<Structure::FieldFunctions<dim>> structure_field_functions;

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>> structure_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     structure_matrix_free;

  // spatial discretization
  std::shared_ptr<Structure::Operator<dim, Number>> structure_operator;

  // temporal discretization
  std::shared_ptr<Structure::TimeIntGenAlpha<dim, Number>> structure_time_integrator;

  // postprocessor
  std::shared_ptr<Structure::PostProcessor<dim, Number>> structure_postprocessor;

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
  mutable TimerTree timer_tree;
};

} // namespace FSI


#endif /* INCLUDE_FLUID_STRUCTURE_INTERACTION_DRIVER_H_ */
