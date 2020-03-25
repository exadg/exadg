/*
 * driver.cpp
 *
 *  Created on: 24.03.2020
 *      Author: fehn
 */

#include "driver.h"

namespace Poisson
{
template<int dim, typename Number>
Driver<dim, Number>::Driver(MPI_Comm const & comm)
  : mpi_comm(comm),
    pcout(std::cout, Utilities::MPI::this_mpi_process(mpi_comm) == 0),
    overall_time(0.0),
    setup_time(0.0),
    iterations(0),
    wall_time_vector_init(0.0),
    wall_time_rhs(0.0),
    wall_time_solver(0.0),
    wall_time_postprocessing(0.0)
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
                           unsigned int const &                          refine_space)
{
  timer.restart();

  print_header();
  print_dealii_info<Number>(pcout);
  print_MPI_info(pcout, mpi_comm);

  application = app;

  application->set_input_parameters(param);
  // some parameters have to be overwritten
  param.degree        = degree;
  param.h_refinements = refine_space;

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

  application->create_grid(triangulation, param.h_refinements, periodic_faces);
  print_grid_data(pcout, param.h_refinements, *triangulation);

  boundary_descriptor.reset(new ConvDiff::BoundaryDescriptor<0, dim>());
  application->set_boundary_conditions(boundary_descriptor);
  verify_boundary_conditions(*boundary_descriptor, *triangulation, periodic_faces);

  field_functions.reset(new FieldFunctions<dim>());
  application->set_field_functions(field_functions);

  // mapping
  unsigned int const mapping_degree = get_mapping_degree(param.mapping, param.degree);
  mesh.reset(new Mesh<dim>(mapping_degree));

  // compute aspect ratio
  if(false)
  {
    // this variant is only for comparison
    double AR = calculate_aspect_ratio_vertex_distance(*triangulation, mpi_comm);
    pcout << std::endl << "Maximum aspect ratio vertex distance = " << AR << std::endl;

    QGauss<dim> quadrature(param.degree + 1);
    AR = GridTools::compute_maximum_aspect_ratio(*triangulation, mesh->get_mapping(), quadrature);
    pcout << std::endl << "Maximum aspect ratio Jacobian = " << AR << std::endl;
  }

  // initialize Poisson operator
  poisson_operator.reset(new Operator<dim, Number>(*triangulation,
                                                   mesh->get_mapping(),
                                                   periodic_faces,
                                                   boundary_descriptor,
                                                   field_functions,
                                                   param,
                                                   mpi_comm));

  // initialize matrix_free
  matrix_free_wrapper.reset(new MatrixFreeWrapper<dim, Number>(mesh->get_mapping()));
  matrix_free_wrapper->append_data_structures(*poisson_operator);
  matrix_free_wrapper->reinit(param.enable_cell_based_face_loops, triangulation);

  poisson_operator->setup(matrix_free_wrapper);
  poisson_operator->setup_solver();

  // initialize postprocessor
  postprocessor = application->construct_postprocessor(param, mpi_comm);
  postprocessor->setup(poisson_operator->get_dof_handler(), mesh->get_mapping());

  setup_time = timer.wall_time();
}

template<int dim, typename Number>
void
Driver<dim, Number>::solve()
{
  Timer timer_local;

  // initialization of vectors
  timer_local.restart();
  LinearAlgebra::distributed::Vector<Number> rhs;
  LinearAlgebra::distributed::Vector<Number> sol;
  poisson_operator->initialize_dof_vector(rhs);
  poisson_operator->initialize_dof_vector(sol);
  poisson_operator->prescribe_initial_conditions(sol);
  wall_time_vector_init = timer_local.wall_time();

  // postprocessing of results
  timer_local.restart();
  postprocessor->do_postprocessing(sol);
  wall_time_postprocessing = timer_local.wall_time();

  // calculate right-hand side
  timer_local.restart();
  poisson_operator->rhs(rhs);
  wall_time_rhs = timer_local.wall_time();

  // solve linear system of equations
  timer_local.restart();
  iterations       = poisson_operator->solve(sol, rhs);
  wall_time_solver = timer_local.wall_time();

  // postprocessing of results
  timer_local.restart();
  postprocessor->do_postprocessing(sol);
  wall_time_postprocessing += timer_local.wall_time();

  overall_time += this->timer.wall_time();
}

template<int dim, typename Number>
Timings
Driver<dim, Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  this->pcout << "Performance results for Poisson solver:" << std::endl;

  double const n_10 = poisson_operator->get_n10();
  // Iterations
  {
    this->pcout << std::endl << "Number of iterations:" << std::endl;

    this->pcout << "  Iterations n         = " << std::fixed << iterations << std::endl;

    this->pcout << "  Iterations n_10      = " << std::fixed << std::setprecision(1) << n_10
                << std::endl;

    this->pcout << "  Convergence rate rho = " << std::fixed << std::setprecision(4)
                << poisson_operator->get_average_convergence_rate() << std::endl;
  }

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg overall_time_data = Utilities::MPI::min_max_avg(overall_time, mpi_comm);
  double const              overall_time_avg  = overall_time_data.avg;

  // wall times
  this->pcout << std::endl << "Wall times:" << std::endl;

  std::vector<std::string> names = {"Initialization of vectors",
                                    "Right-hand side",
                                    "Linear solver",
                                    "Postprocessing"};

  std::vector<double> computing_times;
  computing_times.resize(4);
  computing_times[0] = wall_time_vector_init;
  computing_times[1] = wall_time_rhs;
  computing_times[2] = wall_time_solver;
  computing_times[3] = wall_time_postprocessing;

  unsigned int length = 1;
  for(unsigned int i = 0; i < names.size(); ++i)
  {
    length = length > names[i].length() ? length : names[i].length();
  }

  double sum_of_substeps = 0.0;
  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(computing_times[i], mpi_comm);
    this->pcout << "  " << std::setw(length + 2) << std::left << names[i] << std::setprecision(2)
                << std::scientific << std::setw(10) << std::right << data.avg << " s  "
                << std::setprecision(2) << std::fixed << std::setw(6) << std::right
                << data.avg / overall_time_avg * 100 << " %" << std::endl;

    sum_of_substeps += data.avg;
  }

  Utilities::MPI::MinMaxAvg setup_time_data = Utilities::MPI::min_max_avg(setup_time, mpi_comm);
  double const              setup_time_avg  = setup_time_data.avg;
  this->pcout << "  " << std::setw(length + 2) << std::left << "Setup" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << setup_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << setup_time_avg / overall_time_avg * 100 << " %" << std::endl;

  double const other = overall_time_avg - sum_of_substeps - setup_time_avg;
  this->pcout << "  " << std::setw(length + 2) << std::left << "Other" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << other << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << other / overall_time_avg * 100 << " %" << std::endl;

  this->pcout << "  " << std::setw(length + 2) << std::left << "Overall" << std::setprecision(2)
              << std::scientific << std::setw(10) << std::right << overall_time_avg << " s  "
              << std::setprecision(2) << std::fixed << std::setw(6) << std::right
              << overall_time_avg / overall_time_avg * 100 << " %" << std::endl;

  // computational costs in CPUh
  // Throughput in DoF/s per time step per core
  unsigned int                  N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);
  types::global_dof_index const DoFs            = poisson_operator->get_number_of_dofs();

  this->pcout << std::endl
              << "Computational costs and throughput:" << std::endl
              << "  Number of MPI processes = " << N_mpi_processes << std::endl
              << "  Degrees of freedom      = " << DoFs << std::endl
              << std::endl;

  this->pcout << "Overall costs (including setup + postprocessing):" << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << overall_time_avg << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << overall_time_avg * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl
              << "  Throughput              = " << std::scientific << std::setprecision(2)
              << DoFs / (overall_time_avg * N_mpi_processes) << " DoF/s/core" << std::endl
              << std::endl;

  this->pcout << "Linear solver:" << std::endl
              << "  Wall time               = " << std::scientific << std::setprecision(2)
              << wall_time_solver << " s" << std::endl
              << "  Computational costs     = " << std::scientific << std::setprecision(2)
              << wall_time_solver * (double)N_mpi_processes / 3600.0 << " CPUh" << std::endl
              << "  Throughput              = " << std::scientific << std::setprecision(2)
              << DoFs / (wall_time_solver * N_mpi_processes) << " DoF/s/core" << std::endl
              << std::endl;

  double const t_10   = wall_time_solver * n_10 / iterations;
  double const tau_10 = t_10 * (double)N_mpi_processes / DoFs;
  this->pcout << "Linear solver (numbers based on n_10):" << std::endl
              << "  Wall time t_10          = " << std::scientific << std::setprecision(2) << t_10
              << " s" << std::endl
              << "  tau_10                  = " << std::scientific << std::setprecision(2) << tau_10
              << " s*core/DoF" << std::endl
              << "  Throughput E_10         = " << std::scientific << std::setprecision(2)
              << 1.0 / tau_10 << " DoF/s/core" << std::endl;

  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  return Timings(param.degree, DoFs, n_10, tau_10);
}

template<int dim, typename Number>
std::tuple<unsigned int, types::global_dof_index, double>
Driver<dim, Number>::apply_operator(std::string const & operator_type_string,
                                    unsigned int const  n_repetitions_inner,
                                    unsigned int const  n_repetitions_outer) const
{
  pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

  OperatorType operator_type;
  string_to_enum(operator_type, operator_type_string);

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

  if(operator_type == OperatorType::MatrixBased)
  {
#ifdef DEAL_II_WITH_TRILINOS
    poisson_operator->init_system_matrix(system_matrix);
    poisson_operator->calculate_system_matrix(system_matrix);

    // TODO
//  pcout << "Number of nonzero elements = " << system_matrix.n_nonzero_elements() << std::endl;
//  pcout << "Number of nonzero elements block diagonal = " <<
//  triangulation->n_global_active_cells()*std::pow(param.degree+1, 2*param.dim) << std::endl;
#else
    AssertThrow(false, ExcMessage("Activate DEAL_II_WITH_TRILINOS for matrix-based computations."));
#endif
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

      if(operator_type == OperatorType::MatrixFree)
      {
        poisson_operator->vmult(dst, src);
      }
      else if(operator_type == OperatorType::MatrixBased)
      {
#ifdef DEAL_II_WITH_TRILINOS
        poisson_operator->vmult_matrix_based(dst_trilinos, system_matrix, src_trilinos);
#else
        AssertThrow(false,
                    ExcMessage("Activate DEAL_II_WITH_TRILINOS for matrix-based computations."));
#endif
      }

      current_wall_time += timer.wall_time();
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

  types::global_dof_index dofs = poisson_operator->get_number_of_dofs();

  double dofs_per_walltime = (double)dofs / wall_time;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(mpi_comm);

  // clang-format off
  pcout << std::endl
        << std::scientific << std::setprecision(4)
        << "DoFs/sec:        " << dofs_per_walltime << std::endl
        << "DoFs/(sec*core): " << dofs_per_walltime/(double)N_mpi_processes << std::endl;
  // clang-format on

  pcout << std::endl << " ... done." << std::endl << std::endl;

  return std::tuple<unsigned int, types::global_dof_index, double>(param.degree,
                                                                   dofs,
                                                                   dofs_per_walltime);
}


template class Driver<2, float>;
template class Driver<3, float>;

template class Driver<2, double>;
template class Driver<3, double>;

} // namespace Poisson
