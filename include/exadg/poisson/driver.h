/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_EXADG_POISSON_DRIVER_H_
#define INCLUDE_EXADG_POISSON_DRIVER_H_

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/base/timer.h>

// ExaDG
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/calculate_maximum_aspect_ratio.h>
#include <exadg/grid/mapping_degree.h>
#include <exadg/grid/mapping_dof_vector.h>
#include <exadg/matrix_free/matrix_free_wrapper.h>
#include <exadg/poisson/spatial_discretization/operator.h>
#include <exadg/poisson/user_interface/application_base.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/solver_result.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

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
get_dofs_per_element(std::string const & input_file,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  std::string spatial_discretization = "DG";

  ParameterHandler prm;
  prm.enter_subsection("Discretization");
  prm.add_parameter("SpatialDiscretization",
                    spatial_discretization,
                    "Spatial discretization (CG vs. DG).",
                    Patterns::Selection("CG|DG"),
                    true);
  prm.leave_subsection();

  prm.parse_input(input_file, "", true, true);

  unsigned int dofs_per_element = 1;

  if(spatial_discretization == "CG")
    dofs_per_element = Utilities::pow(degree, dim);
  else if(spatial_discretization == "DG")
    dofs_per_element = Utilities::pow(degree + 1, dim);
  else
    AssertThrow(false, ExcMessage("Not implemented."));

  return dofs_per_element;
}

template<int dim, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const & mpi_comm, bool const is_test);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const                            degree,
        unsigned int const                            refine_space,
        bool const                                    is_throughput_study);

  void
  solve();

  SolverResult
  print_performance_results(double const total_time) const;

  std::tuple<unsigned int, types::global_dof_index, double>
  apply_operator(unsigned int const  degree,
                 std::string const & operator_type_string,
                 unsigned int const  n_repetitions_inner,
                 unsigned int const  n_repetitions_outer) const;

private:
  // MPI communicator
  MPI_Comm const mpi_comm;

  // output to std::cout
  ConditionalOStream pcout;

  // do not print wall times if is_test
  bool const is_test;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  // triangulation
  std::shared_ptr<Triangulation<dim>> triangulation;

  // mapping
  std::shared_ptr<Mapping<dim>> mapping;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  InputParameters param;

  std::shared_ptr<FieldFunctions<dim>>        field_functions;
  std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor;

  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data;

  std::shared_ptr<Operator<dim, Number>> poisson_operator;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;

  // number of iterations
  mutable unsigned int iterations;

  // Computation time (wall clock time)
  mutable TimerTree timer_tree;
  mutable double    solve_time;
};

} // namespace Poisson
} // namespace ExaDG


#endif /* INCLUDE_EXADG_POISSON_DRIVER_H_ */
