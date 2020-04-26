/*
 * driver.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_DRIVER_H_
#define INCLUDE_STRUCTURE_DRIVER_H_

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

// application
#include "user_interface/application_base.h"

// functionalities
#include "../include/grid/mapping_degree.h"
#include "../include/grid/mesh.h"
#include "../utilities/print_general_infos.h"
#include "../utilities/timings_hierarchical.h"

// spatial discretization
#include "../structure/spatial_discretization/operator.h"

// time integration
#include "../structure/time_integration/driver_quasi_static_problems.h"
#include "../structure/time_integration/driver_steady_problems.h"
#include "../structure/time_integration/time_int_gen_alpha.h"

using namespace dealii;

namespace Structure
{
template<int dim, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const & comm);

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

  // MPI communicator
  MPI_Comm mpi_comm;

  // output to std::cout
  ConditionalOStream pcout;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  // user input parameters
  InputParameters param;

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;

  // mapping
  std::shared_ptr<Mesh<dim>> mesh;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  // material descriptor
  std::shared_ptr<MaterialDescriptor> material_descriptor;

  // boundary conditions
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

  // field functions
  std::shared_ptr<FieldFunctions<dim>> field_functions;

  // operator
  typedef Operator<dim, Number> PDEOperator;
  std::shared_ptr<PDEOperator>  pde_operator;

  // postprocessor
  typedef PostProcessor<dim, Number> Postprocessor;
  std::shared_ptr<Postprocessor>     postprocessor;

  // driver steady-state
  std::shared_ptr<DriverSteady<dim, Number>> driver_steady;

  // driver quasi-static
  std::shared_ptr<DriverQuasiStatic<dim, Number>> driver_quasi_static;

  // time integration scheme
  std::shared_ptr<TimeIntGenAlpha<dim, Number>> time_integrator;

  // computation time
  mutable TimerTree timer_tree;
};

} // namespace Structure

#endif /* INCLUDE_STRUCTURE_DRIVER_H_ */
