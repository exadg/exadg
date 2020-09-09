/*
 * driver.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_STRUCTURE_DRIVER_H_
#define INCLUDE_EXADG_STRUCTURE_DRIVER_H_

// deal.II
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>

// ExaDG
#include <exadg/grid/mapping_degree.h>
#include <exadg/grid/mesh.h>
#include <exadg/matrix_free/matrix_free_wrapper.h>
#include <exadg/structure/spatial_discretization/operator.h>
#include <exadg/structure/time_integration/driver_quasi_static_problems.h>
#include <exadg/structure/time_integration/driver_steady_problems.h>
#include <exadg/structure/time_integration/time_int_gen_alpha.h>
#include <exadg/structure/user_interface/application_base.h>
#include <exadg/utilities/print_general_infos.h>
#include <exadg/utilities/timer_tree.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

enum class OperatorType
{
  Nonlinear,
  Linearized
};

inline std::string
enum_to_string(OperatorType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case OperatorType::Nonlinear:  string_type = "Nonlinear";  break;
    case OperatorType::Linearized: string_type = "Linearized"; break;
    default: AssertThrow(false, ExcMessage("Not implemented.")); break;
      // clang-format on
  }

  return string_type;
}

inline void
string_to_enum(OperatorType & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "Nonlinear")  enum_type = OperatorType::Nonlinear;
  else if(string_type == "Linearized") enum_type = OperatorType::Linearized;
  else AssertThrow(false, ExcMessage("Unknown operator type. Not implemented."));
  // clang-format on
}

inline unsigned int
get_dofs_per_element(std::string const & operator_type_string,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  (void)operator_type_string;

  unsigned int const dofs_per_element = std::pow(degree, dim) * dim;

  return dofs_per_element;
}

template<int dim, typename Number>
class Driver
{
public:
  Driver(MPI_Comm const & comm);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const &                          degree,
        unsigned int const &                          refine_space,
        unsigned int const &                          refine_time,
        bool const &                                  is_throughput_study = false);

  void
  solve() const;

  void
  print_statistics(double const total_time) const;

  std::tuple<unsigned int, types::global_dof_index, double>
  apply_operator(std::string const & operator_type_string,
                 unsigned int const  n_repetitions_inner,
                 unsigned int const  n_repetitions_outer) const;

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

  // matrix-free
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;

  // operator
  std::shared_ptr<Operator<dim, Number>> pde_operator;

  // postprocessor
  std::shared_ptr<PostProcessor<dim, Number>> postprocessor;

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
} // namespace ExaDG

#endif /* INCLUDE_EXADG_STRUCTURE_DRIVER_H_ */
