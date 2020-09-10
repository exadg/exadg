/*
 * driver.cpp
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

// likwid
#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// ExaDG
#include <exadg/structure/driver.h>
#include <exadg/utilities/print_throughput.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm)
  : mpi_comm(comm), pcout(std::cout, Utilities::MPI::this_mpi_process(comm) == 0)
{
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_header() const
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                    High-order matrix-free elasticity solver                     " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const &                          degree,
                           unsigned int const &                          refine_space,
                           unsigned int const &                          refine_time,
                           bool const &                                  is_throughput_study)
{
  Timer timer;
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  application = app;

  application->set_input_parameters(param);

  param.check_input_parameters();
  param.print(pcout, "List of input parameters:");

  // triangulation
  if(param.triangulation_type == TriangulationType::Distributed)
  {
    triangulation.reset(new parallel::distributed::Triangulation<dim>(
      mpi_comm,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
  }
  else if(param.triangulation_type == TriangulationType::FullyDistributed)
  {
    triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(mpi_comm));
  }
  else
  {
    AssertThrow(false, ExcMessage("Invalid parameter triangulation_type."));
  }

  application->create_grid(triangulation, refine_space, periodic_faces);

  print_grid_data(pcout, refine_space, *triangulation);

  // boundary conditions
  boundary_descriptor.reset(new BoundaryDescriptor<dim>());
  application->set_boundary_conditions(boundary_descriptor);

  // material_descriptor
  material_descriptor.reset(new MaterialDescriptor);
  application->set_material(*material_descriptor);

  // field functions and boundary conditions
  field_functions.reset(new FieldFunctions<dim>());
  application->set_field_functions(field_functions);

  // mapping
  unsigned int const mapping_degree = get_mapping_degree(param.mapping, degree);
  mesh.reset(new Mesh<dim>(mapping_degree));

  // setup spatial operator
  pde_operator.reset(new Operator<dim, Number>(*triangulation,
                                               mesh->get_mapping(),
                                               degree,
                                               periodic_faces,
                                               boundary_descriptor,
                                               field_functions,
                                               material_descriptor,
                                               param,
                                               "elasticity",
                                               mpi_comm));

  // initialize matrix_free
  matrix_free_data.reset(new MatrixFreeData<dim, Number>());
  matrix_free_data->data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;
  pde_operator->fill_matrix_free_data(*matrix_free_data);

  matrix_free.reset(new MatrixFree<dim, Number>());
  matrix_free->reinit(mesh->get_mapping(),
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  pde_operator->setup(matrix_free, matrix_free_data);

  if(!is_throughput_study)
  {
    // initialize postprocessor
    postprocessor = application->construct_postprocessor(degree, mpi_comm);
    postprocessor->setup(pde_operator->get_dof_handler(), pde_operator->get_mapping());

    // initialize time integrator/driver
    if(param.problem_type == ProblemType::Unsteady)
    {
      time_integrator.reset(new TimeIntGenAlpha<dim, Number>(
        pde_operator, postprocessor, refine_time, param, mpi_comm));
      time_integrator->setup(param.restarted_simulation);
    }
    else if(param.problem_type == ProblemType::Steady)
    {
      driver_steady.reset(
        new DriverSteady<dim, Number>(pde_operator, postprocessor, param, mpi_comm));
      driver_steady->setup();
    }
    else if(param.problem_type == ProblemType::QuasiStatic)
    {
      driver_quasi_static.reset(
        new DriverQuasiStatic<dim, Number>(pde_operator, postprocessor, param, mpi_comm));
      driver_quasi_static->setup();
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    pde_operator->setup_solver();
  }

  timer_tree.insert({"Elasticity", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve() const
{
  if(param.problem_type == ProblemType::Unsteady)
  {
    time_integrator->timeloop();
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    driver_steady->solve_problem();
  }
  else if(param.problem_type == ProblemType::QuasiStatic)
  {
    driver_quasi_static->solve_problem();
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_statistics(double const total_time) const
{
  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;

  pcout << "Performance results for elasticity solver:" << std::endl;

  // Iterations
  if(param.problem_type == ProblemType::QuasiStatic)
  {
    pcout << std::endl << "Average number of iterations:" << std::endl;
    driver_quasi_static->print_iterations();
  }
  else if(param.problem_type == ProblemType::Unsteady)
  {
    pcout << std::endl << "Average number of iterations:" << std::endl;
    time_integrator->print_iterations();
  }

  timer_tree.insert({"Elasticity"}, total_time);

  if(param.problem_type == ProblemType::Unsteady)
  {
    timer_tree.insert({"Elasticity"}, time_integrator->get_timings());
  }
  else if(param.problem_type == ProblemType::Steady)
  {
    timer_tree.insert({"Elasticity"}, driver_steady->get_timings());
  }
  else if(param.problem_type == ProblemType::QuasiStatic)
  {
    timer_tree.insert({"Elasticity"}, driver_quasi_static->get_timings());
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  pcout << std::endl << "Timings for level 2:" << std::endl;
  timer_tree.print_level(pcout, 2);

  // Throughput in DoFs/s per time step per core
  types::global_dof_index const DoFs            = pde_operator->get_number_of_dofs();
  unsigned int const            N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  Utilities::MPI::MinMaxAvg total_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const              total_time_avg  = total_time_data.avg;

  if(param.problem_type == ProblemType::Unsteady)
  {
    unsigned int const N_time_steps = time_integrator->get_number_of_time_steps();
    print_throughput_unsteady(pcout, DoFs, total_time_avg, N_time_steps, N_mpi_processes);
  }
  else
  {
    print_throughput_steady(pcout, DoFs, total_time_avg, N_mpi_processes);
  }

  // computational costs in CPUh
  print_costs(pcout, total_time_avg, N_mpi_processes);

  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

template<int dim, typename Number>
std::tuple<unsigned int, types::global_dof_index, double>
Driver<dim, Number>::apply_operator(unsigned int const degree,
				    std::string const & operator_type_string,
                                    unsigned int const  n_repetitions_inner,
                                    unsigned int const  n_repetitions_outer) const
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  OperatorType operator_type;
  string_to_enum(operator_type, operator_type_string);

  LinearAlgebra::distributed::Vector<Number> dst, src, linearization;
  pde_operator->initialize_dof_vector(src);
  pde_operator->initialize_dof_vector(dst);
  src = 1.0;

  if(param.large_deformation && operator_type == OperatorType::Linearized)
  {
    pde_operator->initialize_dof_vector(linearization);
    linearization = 1.0;
  }

  // Timer and wall times
  Timer  timer;
  double wall_time = std::numeric_limits<double>::max();

  for(unsigned int i_outer = 0; i_outer < n_repetitions_outer; ++i_outer)
  {
    double current_wall_time = 0.0;

    // apply matrix-vector product several times
    for(unsigned int i = 0; i < n_repetitions_inner; ++i)
    {
      timer.restart();

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_START(("degree_" + std::to_string(degree)).c_str());
#endif

      if(param.large_deformation)
      {
        if(operator_type == OperatorType::Nonlinear)
        {
          pde_operator->apply_nonlinear_operator(dst, src, 1.0, 0.0);
        }
        else if(operator_type == OperatorType::Linearized)
        {
          pde_operator->set_solution_linearization(linearization);
          pde_operator->apply_linearized_operator(dst, src, 1.0, 0.0);
        }
      }
      else
      {
        pde_operator->apply_linear_operator(dst, src, 1.0, 0.0);
      }

#ifdef LIKWID_PERFMON
      LIKWID_MARKER_STOP(("degree_" + std::to_string(degree)).c_str());
#endif

      Utilities::MPI::MinMaxAvg wall_time_local =
        Utilities::MPI::min_max_avg(timer.wall_time(), mpi_comm);

      current_wall_time += wall_time_local.avg;
    }

    // compute average wall time
    current_wall_time /= (double)n_repetitions_inner;

    wall_time = std::min(wall_time, current_wall_time);
  }

  if(wall_time * n_repetitions_inner * n_repetitions_outer < 1.0 /*wall time in seconds*/)
  {
    this->pcout
      << std::endl
      << "WARNING: One should use a larger number of matrix-vector products to obtain reproducible results."
      << std::endl;
  }

  types::global_dof_index dofs = pde_operator->get_number_of_dofs();

  double dofs_per_walltime = (double)dofs / wall_time;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4)
        << "DoFs/sec:        " << dofs_per_walltime << std::endl
        << "DoFs/(sec*core): " << dofs_per_walltime/(double)N_mpi_processes << std::endl;
  // clang-format on

  pcout << std::endl << " ... done." << std::endl << std::endl;

  return std::tuple<unsigned int, types::global_dof_index, double>(pde_operator->get_degree(),
                                                                   dofs,
                                                                   dofs_per_walltime);
}

template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace Structure
} // namespace ExaDG
