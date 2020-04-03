/*
 * driver_precursor.h
 *
 *  Created on: 29.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_PRECURSOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_PRECURSOR_H_

// postprocessor
#include "postprocessor/postprocessor_base.h"

// spatial discretization
#include "spatial_discretization/dg_coupled_solver.h"
#include "spatial_discretization/dg_dual_splitting.h"
#include "spatial_discretization/dg_pressure_correction.h"

// temporal discretization
#include "time_integration/time_int_bdf_coupled_solver.h"
#include "time_integration/time_int_bdf_dual_splitting.h"
#include "time_integration/time_int_bdf_pressure_correction.h"

// application
#include "user_interface/application_base.h"

// general functionalities
#include "../functionalities/mapping_degree.h"
#include "../functionalities/matrix_free_wrapper.h"
#include "../functionalities/print_general_infos.h"
#include "../functionalities/verify_boundary_conditions.h"

namespace IncNS
{
template<int dim, typename Number>
class DriverPrecursor
{
public:
  DriverPrecursor(MPI_Comm const & mpi_comm);

  void
  setup(std::shared_ptr<ApplicationBasePrecursor<dim, Number>> application,
        unsigned int const &                                   degree,
        unsigned int const &                                   refine_space);

  void
  solve() const;

  void
  analyze_computing_times() const;

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
  std::shared_ptr<ApplicationBasePrecursor<dim, Number>> application;

  /*
   * Mesh
   */

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation_pre, triangulation;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces_pre, periodic_faces;

  // mapping (static and moving meshes)
  std::shared_ptr<Mesh<dim>> mesh_pre, mesh;

  bool use_adaptive_time_stepping;

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
  std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper_pre, matrix_free_wrapper;

  /*
   * Spatial discretization
   */
  typedef DGNavierStokesBase<dim, Number>               DGBase;
  typedef DGNavierStokesCoupled<dim, Number>            DGCoupled;
  typedef DGNavierStokesDualSplitting<dim, Number>      DGDualSplitting;
  typedef DGNavierStokesPressureCorrection<dim, Number> DGPressureCorrection;

  std::shared_ptr<DGBase>               navier_stokes_operator_pre;
  std::shared_ptr<DGCoupled>            navier_stokes_operator_coupled_pre;
  std::shared_ptr<DGDualSplitting>      navier_stokes_operator_dual_splitting_pre;
  std::shared_ptr<DGPressureCorrection> navier_stokes_operator_pressure_correction_pre;

  std::shared_ptr<DGBase>               navier_stokes_operator;
  std::shared_ptr<DGCoupled>            navier_stokes_operator_coupled;
  std::shared_ptr<DGDualSplitting>      navier_stokes_operator_dual_splitting;
  std::shared_ptr<DGPressureCorrection> navier_stokes_operator_pressure_correction;

  /*
   * Postprocessor
   */
  typedef PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor_pre, postprocessor;

  /*
   * Temporal discretization
   */
  typedef TimeIntBDF<dim, Number>                   TimeInt;
  typedef TimeIntBDFCoupled<dim, Number>            TimeIntCoupled;
  typedef TimeIntBDFDualSplitting<dim, Number>      TimeIntDualSplitting;
  typedef TimeIntBDFPressureCorrection<dim, Number> TimeIntPressureCorrection;

  std::shared_ptr<TimeInt> time_integrator_pre, time_integrator;

  /*
   * Computation time (wall clock time).
   */
  Timer          timer;
  mutable double overall_time;
  double         setup_time;

  unsigned int const length = 15;

  void
  analyze_iterations(InputParameters const &        param,
                     std::shared_ptr<TimeInt> const time_integrator) const;

  double
  analyze_computing_times(InputParameters const &        param,
                          std::shared_ptr<TimeInt> const time_integrator) const;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_PRECURSOR_H_ */
