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

// likwid
#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// ExaDG
#include <exadg/compressible_navier_stokes/driver.h>
#include <exadg/utilities/print_solver_results.h>
#include <exadg/utilities/throughput_parameters.h>

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm, bool const is_test)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    is_test(is_test)
{
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const                            degree,
                           unsigned int const                            refine_space,
                           unsigned int const                            refine_time,
                           bool const                                    is_throughput_study)
{
  Timer timer;
  timer.restart();

  print_exadg_header(pcout);
  pcout << "Setting up compressible Navier-Stokes solver:" << std::endl;

  if(not(is_test))
  {
    print_dealii_info(pcout);
    print_matrixfree_info<Number>(pcout);
  }
  print_MPI_info(pcout, mpi_comm);

  application = app;

  application->set_input_parameters(param);
  param.check_input_parameters();
  param.print(pcout, "List of input parameters:");

  // grid
  GridData grid_data;
  grid_data.triangulation_type = param.triangulation_type;
  grid_data.n_refine_global    = refine_space;
  grid_data.mapping_degree     = get_mapping_degree(param.mapping, degree);

  grid = application->create_grid(grid_data, mpi_comm);
  print_grid_info(pcout, *grid);

  boundary_descriptor = std::make_shared<BoundaryDescriptor<dim>>();
  application->set_boundary_conditions(boundary_descriptor);
  CompNS::verify_boundary_conditions<dim>(boundary_descriptor, *grid);

  field_functions = std::make_shared<FieldFunctions<dim>>();
  application->set_field_functions(field_functions);

  // initialize compressible Navier-Stokes operator
  pde_operator = std::make_shared<Operator<dim, Number>>(
    grid, degree, boundary_descriptor, field_functions, param, "fluid", mpi_comm);

  // initialize matrix_free
  matrix_free_data = std::make_shared<MatrixFreeData<dim, Number>>();
  matrix_free_data->append(pde_operator);

  matrix_free = std::make_shared<MatrixFree<dim, Number>>();
  matrix_free->reinit(*grid->mapping,
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  // setup compressible Navier-Stokes operator
  pde_operator->setup(matrix_free, matrix_free_data);

  // initialize postprocessor
  if(!is_throughput_study)
  {
    postprocessor = application->create_postprocessor(degree, mpi_comm);
    postprocessor->setup(*pde_operator);

    // initialize time integrator
    time_integrator = std::make_shared<TimeIntExplRK<Number>>(
      pde_operator, param, refine_time, mpi_comm, is_test, postprocessor);
    time_integrator->setup(param.restarted_simulation);
  }

  timer_tree.insert({"Compressible flow", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve()
{
  time_integrator->timeloop();
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_performance_results(double const total_time) const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << "Performance results for compressible Navier-Stokes solver:" << std::endl;

  // Wall times
  timer_tree.insert({"Compressible flow"}, total_time);

  timer_tree.insert({"Compressible flow"}, time_integrator->get_timings());

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  types::global_dof_index const DoFs            = pde_operator->get_number_of_dofs();
  unsigned int const            N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);
  unsigned int const            N_time_steps    = time_integrator->get_number_of_time_steps();

  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  print_throughput_unsteady(pcout, DoFs, overall_time_avg, N_time_steps, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, overall_time_avg, N_mpi_processes);

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

template<int dim, typename Number>
std::tuple<unsigned int, types::global_dof_index, double>
Driver<dim, Number>::apply_operator(unsigned int const  degree,
                                    std::string const & operator_type_string,
                                    unsigned int const  n_repetitions_inner,
                                    unsigned int const  n_repetitions_outer) const
{
  (void)degree;

  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  OperatorType operator_type;
  string_to_enum(operator_type, operator_type_string);

  // Vectors
  VectorType dst, src;

  // initialize vectors
  pde_operator->initialize_dof_vector(src);
  pde_operator->initialize_dof_vector(dst);
  src = 1.0;
  dst = 1.0;

  const std::function<void(void)> operator_evaluation = [&](void) {
    if(operator_type == OperatorType::ConvectiveTerm)
      pde_operator->evaluate_convective(dst, src, 0.0);
    else if(operator_type == OperatorType::ViscousTerm)
      pde_operator->evaluate_viscous(dst, src, 0.0);
    else if(operator_type == OperatorType::ViscousAndConvectiveTerms)
      pde_operator->evaluate_convective_and_viscous(dst, src, 0.0);
    else if(operator_type == OperatorType::InverseMassOperator)
      pde_operator->apply_inverse_mass(dst, src);
    else if(operator_type == OperatorType::InverseMassOperatorDstDst)
      pde_operator->apply_inverse_mass(dst, dst);
    else if(operator_type == OperatorType::VectorUpdate)
      dst.sadd(2.0, 1.0, src);
    else if(operator_type == OperatorType::EvaluateOperatorExplicit)
      pde_operator->evaluate(dst, src, 0.0);
    else
      AssertThrow(false, ExcMessage("Specified operator type not implemented"));
  };

  // do the measurements
  double const wall_time = measure_operator_evaluation_time(
    operator_evaluation, degree, n_repetitions_inner, n_repetitions_outer, mpi_comm);

  // calculate throughput
  types::global_dof_index const dofs = pde_operator->get_number_of_dofs();

  double const throughput = (double)dofs / wall_time;

  unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  if(not(is_test))
  {
    // clang-format off
    pcout << std::endl
          << std::scientific << std::setprecision(4)
          << "DoFs/sec:        " << throughput << std::endl
          << "DoFs/(sec*core): " << throughput/(double)N_mpi_processes << std::endl;
    // clang-format on
  }

  pcout << std::endl << " ... done." << std::endl << std::endl;

  return std::tuple<unsigned int, types::global_dof_index, double>(
    pde_operator->get_polynomial_degree(), dofs, throughput);
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace CompNS
} // namespace ExaDG
