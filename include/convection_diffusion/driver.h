/*
 * driver.h
 *
 *  Created on: 22.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_DRIVER_H_
#define INCLUDE_CONVECTION_DIFFUSION_DRIVER_H_

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// general functionalities
#include "../include/functionalities/mapping_degree.h"
#include "../include/functionalities/matrix_free_wrapper.h"
#include "../include/functionalities/moving_mesh.h"
#include "../include/functionalities/print_functions.h"
#include "../include/functionalities/print_general_infos.h"
#include "../include/functionalities/verify_boundary_conditions.h"

// spatial discretization
#include "../include/convection_diffusion/spatial_discretization/dg_operator.h"
#include "../include/convection_diffusion/spatial_discretization/interface.h"

// temporal discretization
#include "../include/convection_diffusion/time_integration/driver_steady_problems.h"
#include "../include/convection_diffusion/time_integration/time_int_bdf.h"
#include "../include/convection_diffusion/time_integration/time_int_explicit_runge_kutta.h"

// postprocessor
#include "../include/convection_diffusion/postprocessor/postprocessor_base.h"

// user interface, etc.
#include "../include/convection_diffusion/user_interface/analytical_solution.h"
#include "../include/convection_diffusion/user_interface/application_base.h"
#include "../include/convection_diffusion/user_interface/boundary_descriptor.h"
#include "../include/convection_diffusion/user_interface/field_functions.h"
#include "../include/convection_diffusion/user_interface/input_parameters.h"

using namespace dealii;

namespace ConvDiff
{
template<int dim, typename Number = double>
class Driver
{
public:
  Driver(MPI_Comm const & mpi_comm);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const &                          degree,
        unsigned int const &                          refine_space,
        unsigned int const &                          refine_time);

  void
  solve();

  void
  analyze_computing_times() const;

private:
  void
  print_header();

  // MPI communicator
  MPI_Comm const & mpi_comm;

  // output to std::cout
  ConditionalOStream pcout;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;

  // mapping (static and moving meshes)
  std::shared_ptr<Mesh<dim>>                         mesh;
  std::shared_ptr<MovingMeshAnalytical<dim, Number>> moving_mesh;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  InputParameters param;

  std::shared_ptr<FieldFunctions<dim>>        field_functions;
  std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor;

  std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper;

  std::shared_ptr<DGOperator<dim, Number>> conv_diff_operator;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;

  std::shared_ptr<TimeIntBase> time_integrator;

  std::shared_ptr<DriverSteadyProblems<Number>> driver_steady;

  /*
   * Computation time (wall clock time).
   */
  Timer          timer;
  mutable double overall_time;
  double         setup_time;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_DRIVER_H_ */
