/*
 * driver.h
 *
 *  Created on: 25.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_DRIVER_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_DRIVER_H_

// matrix-free
#include "../include/matrix_free/matrix_free_wrapper.h"

// spatial discretization
#include "spatial_discretization/dg_operator.h"

// temporal discretization
#include "time_integration/time_int_explicit_runge_kutta.h"

// user interface
#include "user_interface/analytical_solution.h"
#include "user_interface/application_base.h"
#include "user_interface/boundary_descriptor.h"
#include "user_interface/field_functions.h"
#include "user_interface/input_parameters.h"

// grid
#include "../include/grid/mapping_degree.h"
#include "../include/grid/mesh.h"

// general functionalities
#include "../include/functions_and_boundary_conditions/verify_boundary_conditions.h"
#include "../utilities/print_general_infos.h"

namespace CompNS
{
// Select the operator to be applied
enum class Operator
{
  ConvectiveTerm,
  ViscousTerm,
  ViscousAndConvectiveTerms,
  InverseMassMatrix,
  InverseMassMatrixDstDst,
  VectorUpdate,
  EvaluateOperatorExplicit
};

inline std::string
enum_to_string(Operator const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case Operator::ConvectiveTerm:            string_type = "ConvectiveTerm";           break;
    case Operator::ViscousTerm:               string_type = "ViscousTerm";              break;
    case Operator::ViscousAndConvectiveTerms: string_type = "ViscousAndConvectiveTerms";break;
    case Operator::InverseMassMatrix:         string_type = "InverseMassMatrix";        break;
    case Operator::InverseMassMatrixDstDst:   string_type = "InverseMassMatrixDstDst";  break;
    case Operator::VectorUpdate:              string_type = "VectorUpdate";             break;
    case Operator::EvaluateOperatorExplicit:  string_type = "EvaluateOperatorExplicit"; break;

    default:AssertThrow(false, ExcMessage("Not implemented.")); break;
      // clang-format on
  }

  return string_type;
}

inline void
string_to_enum(Operator & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "ConvectiveTerm")            enum_type = Operator::ConvectiveTerm;
  else if(string_type == "ViscousTerm")               enum_type = Operator::ViscousTerm;
  else if(string_type == "ViscousAndConvectiveTerms") enum_type = Operator::ViscousAndConvectiveTerms;
  else if(string_type == "InverseMassMatrix")         enum_type = Operator::InverseMassMatrix;
  else if(string_type == "InverseMassMatrixDstDst")   enum_type = Operator::InverseMassMatrixDstDst;
  else if(string_type == "VectorUpdate")              enum_type = Operator::VectorUpdate;
  else if(string_type == "EvaluateOperatorExplicit")  enum_type = Operator::EvaluateOperatorExplicit;
  else AssertThrow(false, ExcMessage("Unknown operator type. Not implemented."));
  // clang-format on
}

inline unsigned int
get_dofs_per_element(std::string const & operator_type_string,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  (void)operator_type_string;

  unsigned int const dofs_per_element = (dim + 2) * std::pow(degree + 1, dim);

  return dofs_per_element;
}

template<int dim, typename Number = double>
class Driver
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  Driver(MPI_Comm const & comm);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const &                          degree,
        unsigned int const &                          refine_space,
        unsigned int const &                          refine_time,
        bool const &                                  is_throughput_study = false);

  void
  solve();

  void
  print_statistics(double const total_time) const;

  std::tuple<unsigned int, types::global_dof_index, double>
  apply_operator(std::string const & operator_type,
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

  InputParameters param;

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;

  // mapping
  std::shared_ptr<Mesh<dim>> mesh;

  // periodic boundaries
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  std::shared_ptr<FieldFunctions<dim>>           field_functions;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_density;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_velocity;
  std::shared_ptr<BoundaryDescriptor<dim>>       boundary_descriptor_pressure;
  std::shared_ptr<BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy;

  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data;
  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free;

  std::shared_ptr<DGOperator<dim, Number>> comp_navier_stokes_operator;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;

  std::shared_ptr<TimeIntExplRK<Number>> time_integrator;

  // Computation time (wall clock time)
  mutable TimerTree timer_tree;
};

} // namespace CompNS

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_DRIVER_H_ */
