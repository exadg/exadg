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

// functionalities
#include "../include/functionalities/mapping_degree.h"
#include "../include/functionalities/mesh.h"
#include "../include/functionalities/print_general_infos.h"

// application
#include "../structure/user_interface/application_base.h"

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
  analyze_computing_times() const;

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
  Timer          timer;
  mutable double overall_time;
  double         setup_time;
};

} // namespace Structure

#endif /* INCLUDE_STRUCTURE_DRIVER_H_ */
