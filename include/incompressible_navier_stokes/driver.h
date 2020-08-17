/*
 * driver.h
 *
 *  Created on: 26.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_H_

// matrix-free
#include "../matrix_free/matrix_free_wrapper.h"

// postprocessor
#include "postprocessor/postprocessor_base.h"

// spatial discretization
#include "spatial_discretization/dg_coupled_solver.h"
#include "spatial_discretization/dg_dual_splitting.h"
#include "spatial_discretization/dg_pressure_correction.h"

// temporal discretization
#include "time_integration/driver_steady_problems.h"
#include "time_integration/time_int_bdf_coupled_solver.h"
#include "time_integration/time_int_bdf_dual_splitting.h"
#include "time_integration/time_int_bdf_pressure_correction.h"

// application
#include "user_interface/application_base.h"

// general functionalities
#include "../functions_and_boundary_conditions/verify_boundary_conditions.h"
#include "../grid/mapping_degree.h"
#include "../grid/moving_mesh_function.h"
#include "../grid/moving_mesh_poisson.h"
#include "../utilities/print_general_infos.h"

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

// Note: Make sure that the correct time integration scheme is selected in the input file that is
//       compatible with the Operator type specified here. This also includes the treatment of the
//       convective term (explicit/implicit), e.g., specifying VelocityConvDiffOperator together
//       with an explicit treatment of the convective term will only apply the Helmholtz-like
//       operator.

// clang-format off
enum class Operator{
  CoupledNonlinearResidual, // nonlinear residual of coupled system of equations
  CoupledLinearized,        // linearized system of equations for coupled solution approach
  PressurePoissonOperator,  // negative Laplace operator (scalar quantity, pressure)
  ConvectiveOperator,       // convective term (vectorial quantity, velocity)
  HelmholtzOperator,        // mass + viscous (vectorial quantity, velocity)
  ProjectionOperator,       // mass + divergence penalty + continuity penalty (vectorial quantity, velocity)
  VelocityConvDiffOperator, // mass + convective + viscous (vectorial quantity, velocity)
  InverseMassMatrix         // inverse mass matrix operator (vectorial quantity, velocity)
};
// clang-format on

inline std::string
enum_to_string(Operator const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case Operator::CoupledNonlinearResidual: string_type = "CoupledNonlinearResidual"; break;
    case Operator::CoupledLinearized:        string_type = "CoupledLinearized";        break;
    case Operator::PressurePoissonOperator:  string_type = "PressurePoissonOperator";  break;
    case Operator::ConvectiveOperator:       string_type = "ConvectiveOperator";       break;
    case Operator::HelmholtzOperator:        string_type = "HelmholtzOperator";        break;
    case Operator::ProjectionOperator:       string_type = "ProjectionOperator";       break;
    case Operator::VelocityConvDiffOperator: string_type = "VelocityConvDiffOperator"; break;
    case Operator::InverseMassMatrix:        string_type = "InverseMassMatrix";        break;

    default:AssertThrow(false, ExcMessage("Not implemented.")); break;
      // clang-format on
  }

  return string_type;
}

inline void
string_to_enum(Operator & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "CoupledNonlinearResidual")  enum_type = Operator::CoupledNonlinearResidual;
  else if(string_type == "CoupledLinearized")         enum_type = Operator::CoupledLinearized;
  else if(string_type == "PressurePoissonOperator")   enum_type = Operator::PressurePoissonOperator;
  else if(string_type == "ConvectiveOperator")        enum_type = Operator::ConvectiveOperator;
  else if(string_type == "HelmholtzOperator")         enum_type = Operator::HelmholtzOperator;
  else if(string_type == "ProjectionOperator")        enum_type = Operator::ProjectionOperator;
  else if(string_type == "VelocityConvDiffOperator")  enum_type = Operator::VelocityConvDiffOperator;
  else if(string_type == "InverseMassMatrix")         enum_type = Operator::InverseMassMatrix;
  else AssertThrow(false, ExcMessage("Unknown operator type. Not implemented."));
  // clang-format on
}

inline unsigned int
get_dofs_per_element(std::string const & operator_type_string,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  Operator operator_type;
  string_to_enum(operator_type, operator_type_string);

  unsigned int const velocity_dofs_per_element = dim * std::pow(degree + 1, dim);
  // assume mixed-order polynomials
  unsigned int const pressure_dofs_per_element = std::pow(degree, dim);

  if(operator_type == Operator::CoupledNonlinearResidual ||
     operator_type == Operator::CoupledLinearized)
  {
    return velocity_dofs_per_element + pressure_dofs_per_element;
  }
  // velocity
  else if(operator_type == Operator::ConvectiveOperator ||
          operator_type == Operator::VelocityConvDiffOperator ||
          operator_type == Operator::HelmholtzOperator ||
          operator_type == Operator::ProjectionOperator ||
          operator_type == Operator::InverseMassMatrix)
  {
    return velocity_dofs_per_element;
  }
  // pressure
  else if(operator_type == Operator::PressurePoissonOperator)
  {
    return pressure_dofs_per_element;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  return 0;
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
  apply_operator(std::string const & operator_type,
                 unsigned int const  n_repetitions_inner,
                 unsigned int const  n_repetitions_outer) const;

private:
  void
  print_header() const;

  void
  ale_update() const;

  // MPI communicator
  MPI_Comm const & mpi_comm;

  // output to std::cout
  ConditionalOStream pcout;

  // application
  std::shared_ptr<ApplicationBase<dim, Number>> application;

  /*
   * Mesh
   */

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  // mapping for static mesh
  std::shared_ptr<Mesh<dim>> mesh;

  // mapping for moving mesh
  std::shared_ptr<MovingMeshBase<dim, Number>> moving_mesh;

  // solve mesh deformation by a Poisson problem
  Poisson::InputParameters poisson_param;

  std::shared_ptr<Poisson::FieldFunctions<dim>>        poisson_field_functions;
  std::shared_ptr<Poisson::BoundaryDescriptor<1, dim>> poisson_boundary_descriptor;

  // static mesh for Poisson problem
  std::shared_ptr<Mesh<dim>> poisson_mesh;

  std::shared_ptr<MatrixFreeData<dim, Number>>         poisson_matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>             poisson_matrix_free;
  std::shared_ptr<Poisson::Operator<dim, Number, dim>> poisson_operator;

  /*
   * Functions and boundary conditions
   */
  std::shared_ptr<FieldFunctions<dim>>      field_functions;
  std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure;

  /*
   * Parameters
   */
  InputParameters param;

  /*
   * MatrixFree
   */
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;

  /*
   * Spatial discretization
   */
  typedef DGNavierStokesBase<dim, Number>               DGBase;
  typedef DGNavierStokesCoupled<dim, Number>            DGCoupled;
  typedef DGNavierStokesDualSplitting<dim, Number>      DGDualSplitting;
  typedef DGNavierStokesPressureCorrection<dim, Number> DGPressureCorrection;

  std::shared_ptr<DGBase>               navier_stokes_operator;
  std::shared_ptr<DGCoupled>            navier_stokes_operator_coupled;
  std::shared_ptr<DGDualSplitting>      navier_stokes_operator_dual_splitting;
  std::shared_ptr<DGPressureCorrection> navier_stokes_operator_pressure_correction;

  /*
   * Postprocessor
   */
  typedef PostProcessorBase<dim, Number> Postprocessor;

  std::shared_ptr<Postprocessor> postprocessor;

  /*
   * Temporal discretization
   */

  // unsteady solvers
  typedef TimeIntBDF<dim, Number>                   TimeInt;
  typedef TimeIntBDFCoupled<dim, Number>            TimeIntCoupled;
  typedef TimeIntBDFDualSplitting<dim, Number>      TimeIntDualSplitting;
  typedef TimeIntBDFPressureCorrection<dim, Number> TimeIntPressureCorrection;

  std::shared_ptr<TimeInt> time_integrator;

  // steady solver
  typedef DriverSteadyProblems<dim, Number> DriverSteady;

  std::shared_ptr<DriverSteady> driver_steady;

  /*
   * Computation time (wall clock time).
   */
  mutable TimerTree timer_tree;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_DRIVER_H_ */
