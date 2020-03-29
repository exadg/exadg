/*
 * driver.h
 *
 *  Created on: 24.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_DRIVER_H_
#define INCLUDE_POISSON_DRIVER_H_

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/base/timer.h>

// spatial discretization
#include "../include/poisson/spatial_discretization/operator.h"

// postprocessor
#include "../include/convection_diffusion/postprocessor/postprocessor_base.h"

// user interface, etc.
#include "../include/convection_diffusion/user_interface/boundary_descriptor.h"
#include "../include/poisson/user_interface/analytical_solution.h"
#include "../include/poisson/user_interface/application_base.h"
#include "../include/poisson/user_interface/field_functions.h"
#include "../include/poisson/user_interface/input_parameters.h"

// functionalities
#include "../include/functionalities/calculate_maximum_aspect_ratio.h"
#include "../include/functionalities/mapping_degree.h"
#include "../include/functionalities/matrix_free_wrapper.h"
#include "../include/functionalities/mesh.h"
#include "../include/functionalities/print_functions.h"
#include "../include/functionalities/print_general_infos.h"
#include "../include/functionalities/timings.h"
#include "../include/functionalities/verify_boundary_conditions.h"

namespace Poisson
{
enum class OperatorType
{
  MatrixFree,
  MatrixBased
};

inline std::string
enum_to_string(OperatorType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case OperatorType::MatrixFree:  string_type = "MatrixFree";  break;
    case OperatorType::MatrixBased: string_type = "MatrixBased"; break;
    default: AssertThrow(false, ExcMessage("Not implemented.")); break;
      // clang-format on
  }

  return string_type;
}

inline void
string_to_enum(OperatorType & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "MatrixFree")  enum_type = OperatorType::MatrixFree;
  else if(string_type == "MatrixBased") enum_type = OperatorType::MatrixBased;
  else AssertThrow(false, ExcMessage("Unknown operator type. Not implemented."));
  // clang-format on
}

inline unsigned int
get_dofs_per_element(std::string const & operator_type_string,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  (void)operator_type_string;

  unsigned int const dofs_per_element = std::pow(degree + 1, dim);

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
        unsigned int const &                          refine_space);

  void
  solve();

  Timings
  analyze_computing_times() const;

  std::tuple<unsigned int, types::global_dof_index, double>
  apply_operator(std::string const & operator_type_string,
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

  std::shared_ptr<FieldFunctions<dim>>                  field_functions;
  std::shared_ptr<ConvDiff::BoundaryDescriptor<0, dim>> boundary_descriptor;

  std::shared_ptr<MatrixFreeWrapper<dim, Number>> matrix_free_wrapper;

  std::shared_ptr<Operator<dim, Number>> poisson_operator;

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> postprocessor;

  // Computation time (wall clock time)
  Timer          timer;
  mutable double overall_time;
  double         setup_time;

  // number of iterations
  mutable unsigned int iterations;
  mutable double wall_time_vector_init, wall_time_rhs, wall_time_solver, wall_time_postprocessing;
};

} // namespace Poisson



#endif /* INCLUDE_POISSON_DRIVER_H_ */
