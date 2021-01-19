/*
 * driver.h
 *
 *  Created on: 24.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_POISSON_DRIVER_H_
#define INCLUDE_EXADG_POISSON_DRIVER_H_

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/base/timer.h>

// ExaDG

#include <exadg/convection_diffusion/postprocessor/postprocessor_base.h>
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/calculate_maximum_aspect_ratio.h>
#include <exadg/grid/mapping_degree.h>
#include <exadg/grid/mesh.h>
#include <exadg/matrix_free/matrix_free_wrapper.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/poisson/user_interface/analytical_solution.h>
#include <exadg/poisson/user_interface/application_base.h>
#include <exadg/poisson/user_interface/field_functions.h>
#include <exadg/poisson/user_interface/input_parameters.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/timer_tree.h>
#include <exadg/utilities/timings.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

inline unsigned int
get_dofs_per_element(std::string const & operator_type_string,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  (void)operator_type_string;

  unsigned int dofs_per_element = 1;

  if(operator_type_string == "Scalar-CG")
    dofs_per_element = std::pow(degree, dim);
  else if(operator_type_string == "Scalar-DG")
    dofs_per_element = std::pow(degree + 1, dim);
  else if(operator_type_string == "Vectorial-CG")
    dofs_per_element = dim * std::pow(degree, dim);
  else if(operator_type_string == "Vectorial-DG")
    dofs_per_element = dim * std::pow(degree + 1, dim);

  return dofs_per_element;
}

template<int dim, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const & mpi_comm);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const &                          degree,
        unsigned int const &                          refine_space,
        bool const &                                  is_throughput_study = false);

  void
  solve();

  Timings
  print_statistics(double const total_time) const;

  std::tuple<unsigned int, types::global_dof_index, double>
  apply_operator(unsigned int const  degree,
                 std::string const & operator_type_string,
                 unsigned int const  n_repetitions_inner,
                 unsigned int const  n_repetitions_outer) const;

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

  // mapping
  std::shared_ptr<Mesh<dim>> mesh;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  InputParameters param;

  std::shared_ptr<FieldFunctions<dim>>        field_functions;
  std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor;

  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data;

  std::shared_ptr<Operator<dim, Number>> poisson_operator;

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> postprocessor;

  // number of iterations
  mutable unsigned int iterations;

  // Computation time (wall clock time)
  mutable TimerTree timer_tree;
  mutable double    solve_time;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_DRIVER_H_ */
