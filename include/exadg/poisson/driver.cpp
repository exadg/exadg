/*
 * driver.cpp
 *
 *  Created on: 24.03.2020
 *      Author: fehn
 */

// likwid
#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

// ExaDG
#include <exadg/poisson/driver.h>
#include <exadg/utilities/print_throughput.h>
#include <exadg/utilities/throughput_study.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    iterations(0),
    solve_time(0.0)
{
}

template<int dim, typename Number>
void
Driver<dim, Number>::print_header()
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                            scalar Poisson equation                              " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
  // clang-format on
}

template<int dim, typename Number>
void
Driver<dim, Number>::setup(std::shared_ptr<ApplicationBase<dim, Number>> app,
                           unsigned int const &                          degree,
                           unsigned int const &                          refine_space,
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

  boundary_descriptor.reset(new BoundaryDescriptor<0, dim>());
  application->set_boundary_conditions(boundary_descriptor);
  verify_boundary_conditions(*boundary_descriptor, *triangulation, periodic_faces);

  field_functions.reset(new FieldFunctions<dim>());
  application->set_field_functions(field_functions);

  // mapping
  unsigned int const mapping_degree = get_mapping_degree(param.mapping, degree);
  mesh.reset(new Mesh<dim>(mapping_degree));

  // compute aspect ratio
  if(false)
  {
    // this variant is only for comparison
    double AR = calculate_aspect_ratio_vertex_distance(*triangulation, mpi_comm);
    pcout << std::endl << "Maximum aspect ratio vertex distance = " << AR << std::endl;

    QGauss<dim> quadrature(degree + 1);
    AR = GridTools::compute_maximum_aspect_ratio(mesh->get_mapping(), *triangulation, quadrature);
    pcout << std::endl << "Maximum aspect ratio Jacobian = " << AR << std::endl;
  }

  // initialize Poisson operator
  poisson_operator.reset(new Operator<dim, Number>(*triangulation,
                                                   mesh->get_mapping(),
                                                   degree,
                                                   periodic_faces,
                                                   boundary_descriptor,
                                                   field_functions,
                                                   param,
                                                   "Poisson",
                                                   mpi_comm));

  // initialize matrix_free
  matrix_free_data.reset(new MatrixFreeData<dim, Number>());
  matrix_free_data->data.tasks_parallel_scheme =
    MatrixFree<dim, Number>::AdditionalData::partition_partition;
  if(param.enable_cell_based_face_loops)
  {
    auto tria =
      std::dynamic_pointer_cast<parallel::distributed::Triangulation<dim> const>(triangulation);
    Categorization::do_cell_based_loops(*tria, matrix_free_data->data);
  }
  poisson_operator->fill_matrix_free_data(*matrix_free_data);
  matrix_free.reset(new MatrixFree<dim, Number>());
  matrix_free->reinit(mesh->get_mapping(),
                      matrix_free_data->get_dof_handler_vector(),
                      matrix_free_data->get_constraint_vector(),
                      matrix_free_data->get_quadrature_vector(),
                      matrix_free_data->data);

  poisson_operator->setup(matrix_free, matrix_free_data);
  poisson_operator->setup_solver();

  if(!is_throughput_study)
  {
    // initialize postprocessor
    postprocessor = application->construct_postprocessor(degree, mpi_comm);
    postprocessor->setup(poisson_operator->get_dof_handler(), mesh->get_mapping());
  }

  timer_tree.insert({"Poisson", "Setup"}, timer.wall_time());
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve()
{
  // initialization of vectors
  Timer timer;
  timer.restart();
  LinearAlgebra::distributed::Vector<Number> rhs;
  LinearAlgebra::distributed::Vector<Number> sol;
  poisson_operator->initialize_dof_vector(rhs);
  poisson_operator->initialize_dof_vector(sol);
  poisson_operator->prescribe_initial_conditions(sol);
  timer_tree.insert({"Poisson", "Vector init"}, timer.wall_time());

  // postprocessing of results
  timer.restart();
  postprocessor->do_postprocessing(sol);
  timer_tree.insert({"Poisson", "Postprocessing"}, timer.wall_time());

  // calculate right-hand side
  timer.restart();
  poisson_operator->rhs(rhs);
  timer_tree.insert({"Poisson", "Right-hand side"}, timer.wall_time());

  // solve linear system of equations
  timer.restart();
  iterations = poisson_operator->solve(sol, rhs, 0.0 /* time */);
  solve_time += timer.wall_time();
  timer_tree.insert({"Poisson", "Solve"}, solve_time);

  // postprocessing of results
  timer.restart();
  postprocessor->do_postprocessing(sol);
  timer_tree.insert({"Poisson", "Postprocessing"}, timer.wall_time());
}

template<int dim, typename Number>
Timings
Driver<dim, Number>::print_statistics(double const total_time) const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << "Performance results for Poisson solver:" << std::endl;

  // Iterations
  double const n_10 = poisson_operator->get_n10();

  this->pcout << std::endl << "Number of iterations:" << std::endl;

  this->pcout << "  Iterations n         = " << std::fixed << iterations << std::endl
              << "  Iterations n_10      = " << std::fixed << std::setprecision(1) << n_10
              << std::endl
              << "  Convergence rate rho = " << std::fixed << std::setprecision(4)
              << poisson_operator->get_average_convergence_rate() << std::endl;

  // wall times
  timer_tree.insert({"Poisson"}, total_time);

  pcout << std::endl << "Timings for level 1:" << std::endl;
  timer_tree.print_level(pcout, 1);

  // throughput
  types::global_dof_index const DoFs = poisson_operator->get_number_of_dofs();

  unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  // Throughput of linear solver in DoFs/s per core
  double const t_10 = solve_time * n_10 / iterations;
  print_throughput_10(pcout, DoFs, t_10, N_mpi_processes);

  // Throughput in DoFs/s per core (overall costs)
  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(total_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;
  print_throughput_steady(pcout, DoFs, overall_time_avg, N_mpi_processes);

  // computational costs in CPUh
  print_costs(pcout, overall_time_avg, N_mpi_processes);

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  double const tau_10 = t_10 * (double)N_mpi_processes / DoFs;
  return Timings(poisson_operator->get_degree(), DoFs, n_10, tau_10);
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

  bool const matrix_free = true;

  LinearAlgebra::distributed::Vector<Number> dst, src;
  poisson_operator->initialize_dof_vector(src);
  poisson_operator->initialize_dof_vector(dst);
  src = 1.0;

#ifdef DEAL_II_WITH_TRILINOS
  typedef double                                     TrilinosNumber;
  LinearAlgebra::distributed::Vector<TrilinosNumber> dst_trilinos, src_trilinos;
  src_trilinos = src;
  dst_trilinos = dst;

  TrilinosWrappers::SparseMatrix system_matrix;
#endif

  if(!matrix_free)
  {
#ifdef DEAL_II_WITH_TRILINOS
    poisson_operator->init_system_matrix(system_matrix);
    poisson_operator->calculate_system_matrix(system_matrix);
#else
    AssertThrow(false, ExcMessage("Activate DEAL_II_WITH_TRILINOS for matrix-based computations."));
#endif
  }

  const std::function<void(void)> operator_evaluation = [&](void) {
    if(matrix_free)
    {
      poisson_operator->vmult(dst, src);
    }
    else
    {
#ifdef DEAL_II_WITH_TRILINOS
      poisson_operator->vmult_matrix_based(dst_trilinos, system_matrix, src_trilinos);
#else
      AssertThrow(false, ExcMessage("Activate DEAL_II_WITH_TRILINOS."));
#endif
    }
  };

  // do the measurements
  double const wall_time = measure_operator_evaluation_time(
    operator_evaluation, degree, n_repetitions_inner, n_repetitions_outer, mpi_comm);

  // calculate throughput
  types::global_dof_index const dofs = poisson_operator->get_number_of_dofs();

  double const throughput = (double)dofs / wall_time;

  unsigned int const N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4)
        << "DoFs/sec:        " << throughput << std::endl
        << "DoFs/(sec*core): " << throughput/(double)N_mpi_processes << std::endl;
  // clang-format on

  pcout << std::endl << " ... done." << std::endl << std::endl;

  return std::tuple<unsigned int, types::global_dof_index, double>(poisson_operator->get_degree(),
                                                                   dofs,
                                                                   throughput);
}


template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace Poisson
} // namespace ExaDG
