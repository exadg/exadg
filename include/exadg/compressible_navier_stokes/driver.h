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

#ifndef INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_DRIVER_H_
#define INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_DRIVER_H_

#include <exadg/compressible_navier_stokes/spatial_discretization/operator.h>
#include <exadg/compressible_navier_stokes/time_integration/time_int_explicit_runge_kutta.h>
#include <exadg/compressible_navier_stokes/user_interface/analytical_solution.h>
#include <exadg/compressible_navier_stokes/user_interface/application_base.h>
#include <exadg/compressible_navier_stokes/user_interface/boundary_descriptor.h>
#include <exadg/compressible_navier_stokes/user_interface/field_functions.h>
#include <exadg/compressible_navier_stokes/user_interface/input_parameters.h>
#include <exadg/functions_and_boundary_conditions/verify_boundary_conditions.h>
#include <exadg/grid/mapping_degree.h>
#include <exadg/grid/mapping_finite_element.h>
#include <exadg/matrix_free/matrix_free_wrapper.h>
#include <exadg/utilities/print_general_infos.h>

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

// Select the operator to be applied
enum class OperatorType
{
  ConvectiveTerm,
  ViscousTerm,
  ViscousAndConvectiveTerms,
  InverseMassOperator,
  InverseMassOperatorDstDst,
  VectorUpdate,
  EvaluateOperatorExplicit
};

inline std::string
enum_to_string(OperatorType const enum_type)
{
  std::string string_type;

  switch(enum_type)
  {
    // clang-format off
    case OperatorType::ConvectiveTerm:            string_type = "ConvectiveTerm";           break;
    case OperatorType::ViscousTerm:               string_type = "ViscousTerm";              break;
    case OperatorType::ViscousAndConvectiveTerms: string_type = "ViscousAndConvectiveTerms";break;
    case OperatorType::InverseMassOperator:       string_type = "InverseMassOperator";      break;
    case OperatorType::InverseMassOperatorDstDst: string_type = "InverseMassOperatorDstDst";break;
    case OperatorType::VectorUpdate:              string_type = "VectorUpdate";             break;
    case OperatorType::EvaluateOperatorExplicit:  string_type = "EvaluateOperatorExplicit"; break;

    default:AssertThrow(false, ExcMessage("Not implemented.")); break;
      // clang-format on
  }

  return string_type;
}

inline void
string_to_enum(OperatorType & enum_type, std::string const string_type)
{
  // clang-format off
  if     (string_type == "ConvectiveTerm")            enum_type = OperatorType::ConvectiveTerm;
  else if(string_type == "ViscousTerm")               enum_type = OperatorType::ViscousTerm;
  else if(string_type == "ViscousAndConvectiveTerms") enum_type = OperatorType::ViscousAndConvectiveTerms;
  else if(string_type == "InverseMassOperator")       enum_type = OperatorType::InverseMassOperator;
  else if(string_type == "InverseMassOperatorDstDst") enum_type = OperatorType::InverseMassOperatorDstDst;
  else if(string_type == "VectorUpdate")              enum_type = OperatorType::VectorUpdate;
  else if(string_type == "EvaluateOperatorExplicit")  enum_type = OperatorType::EvaluateOperatorExplicit;
  else AssertThrow(false, ExcMessage("Unknown operator type. Not implemented."));
  // clang-format on
}

inline unsigned int
get_dofs_per_element(std::string const & input_file,
                     unsigned int const  dim,
                     unsigned int const  degree)
{
  (void)input_file;

  unsigned int const dofs_per_element = (dim + 2) * std::pow(degree + 1, dim);

  return dofs_per_element;
}

template<int dim, typename Number = double>
class Driver
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  Driver(MPI_Comm const & comm, bool const is_test);

  void
  setup(std::shared_ptr<ApplicationBase<dim, Number>> application,
        unsigned int const                            degree,
        unsigned int const                            refine_space,
        unsigned int const                            refine_time,
        bool const                                    is_throughput_study);

  void
  solve();

  void
  print_performance_results(double const total_time) const;

  std::tuple<unsigned int, types::global_dof_index, double>
  apply_operator(unsigned int const  degree,
                 std::string const & operator_type,
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

  InputParameters param;

  // triangulation
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;

  // mapping
  std::shared_ptr<Mapping<dim>> mapping;

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

  std::shared_ptr<Operator<dim, Number>> comp_navier_stokes_operator;

  std::shared_ptr<PostProcessorBase<dim, Number>> postprocessor;

  std::shared_ptr<TimeIntExplRK<Number>> time_integrator;

  // Computation time (wall clock time)
  mutable TimerTree timer_tree;
};

} // namespace CompNS
} // namespace ExaDG

#endif /* INCLUDE_EXADG_COMPRESSIBLE_NAVIER_STOKES_DRIVER_H_ */
