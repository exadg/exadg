/*
 * driver.h
 *
 *  Created on: 22.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_EXADG_CONVECTION_DIFFUSION_DRIVER_H_
#define INCLUDE_EXADG_CONVECTION_DIFFUSION_DRIVER_H_

// deal.II
#include <deal.II/base/revision.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>

// ExaDG
#include <exadg/convection_diffusion/postprocessor/postprocessor_base.h>
#include <exadg/convection_diffusion/spatial_discretization/dg_operator.h>
#include <exadg/convection_diffusion/spatial_discretization/interface.h>
#include <exadg/convection_diffusion/time_integration/driver_steady_problems.h>
#include <exadg/convection_diffusion/time_integration/time_int_bdf.h>
#include <exadg/convection_diffusion/time_integration/time_int_explicit_runge_kutta.h>
#include <exadg/convection_diffusion/user_interface/analytical_solution.h>
#include <exadg/convection_diffusion/user_interface/application_base.h>
#include <exadg/convection_diffusion/user_interface/boundary_descriptor.h>
#include <exadg/convection_diffusion/user_interface/field_functions.h>
#include <exadg/convection_diffusion/user_interface/input_parameters.h>
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/mapping_degree.h>
#include <exadg/grid/moving_mesh_function.h>
#include <exadg/matrix_free/matrix_free_wrapper.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

enum class Operatortype
{
  MassOperator,
  ConvectiveOperator,
  DiffusiveOperator,
  MassConvectionDiffusionOperator
};

inline std::string
enum_to_string(Operatortype const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case Operatortype::MassOperator:                    string_type = "MassOperator";                    break;
    case Operatortype::ConvectiveOperator:              string_type = "ConvectiveOperator";              break;
    case Operatortype::DiffusiveOperator:               string_type = "DiffusiveOperator";               break;
    case Operatortype::MassConvectionDiffusionOperator: string_type = "MassConvectionDiffusionOperator"; break;
    default: AssertThrow(false, ExcMessage("Not implemented.")); break;
      // clang-format on
  }

  return string_type;
}

inline void
string_to_enum(Operatortype & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "MassOperator")                    enum_type = Operatortype::MassOperator;
  else if(string_type == "ConvectiveOperator")              enum_type = Operatortype::ConvectiveOperator;
  else if(string_type == "DiffusiveOperator")               enum_type = Operatortype::DiffusiveOperator;
  else if(string_type == "MassConvectionDiffusionOperator") enum_type = Operatortype::MassConvectionDiffusionOperator;
  else AssertThrow(false, ExcMessage("Unknown operator type. Not implemented."));
  // clang-format on
}

inline unsigned int
get_dofs_per_element(std::string const & input_file,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  (void)input_file;

  unsigned int const dofs_per_element = std::pow(degree + 1, dim);

  return dofs_per_element;
}

template<int dim, typename Number = double>
class Driver
{
public:
  Driver(MPI_Comm const & mpi_comm);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const                            degree,
        unsigned int const                            refine_space,
        unsigned int const                            refine_time,
        bool const                                    is_test,
        bool const                                    is_throughput_study);

  void
  solve();

  std::tuple<unsigned int, types::global_dof_index, double>
  apply_operator(unsigned int const  degree,
                 std::string const & operator_type,
                 unsigned int const  n_repetitions_inner,
                 unsigned int const  n_repetitions_outer,
                 bool const          is_test) const;

  void
  print_performance_results(double const total_time, bool const is_test) const;

private:
  void
  ale_update() const;

  // MPI communicator
  MPI_Comm const & mpi_comm;

  // output to std::cout
  ConditionalOStream pcout;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;

  // mapping (static and moving meshes)
  std::shared_ptr<Mesh<dim>>                       mesh;
  std::shared_ptr<MovingMeshFunction<dim, Number>> moving_mesh;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  InputParameters param;

  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;

  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;

  std::shared_ptr<DGOperator<dim, Number>> conv_diff_operator;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;

  std::shared_ptr<TimeIntBase> time_integrator;

  std::shared_ptr<DriverSteadyProblems<Number>> driver_steady;

  // Computation time (wall clock time)
  mutable TimerTree timer_tree;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_EXADG_CONVECTION_DIFFUSION_DRIVER_H_ */
