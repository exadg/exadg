/*
 * DriverSteadyProblems.h
 *
 *  Created on: Jul 4, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DRIVERSTEADYPROBLEMS_H_
#define INCLUDE_DRIVERSTEADYPROBLEMS_H_


template<int dim> class PostProcessor;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class DriverSteadyProblems
{
public:
  DriverSteadyProblems(std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree,
                                fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > ns_operation_in,
                                std_cxx11::shared_ptr<PostProcessor<dim> >           postprocessor_in,
                                InputParameters const                                &param_in)
    :
    ns_operation(std::dynamic_pointer_cast<DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > (ns_operation_in)),
    postprocessor(postprocessor_in),
    param(param_in),
    total_time(0.0)
  {}

  void setup();

  void solve_steady_problem();

  void analyze_computing_times() const;

private:
  void initialize_vectors();
  void initialize_solution();

  void solve();
  void postprocessing() const;

  std_cxx11::shared_ptr<DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > ns_operation;

  std_cxx11::shared_ptr<PostProcessor<dim> > postprocessor;
  InputParameters const &param;

  Timer global_timer;
  value_type total_time;

  parallel::distributed::BlockVector<value_type> solution;
  parallel::distributed::BlockVector<value_type> rhs_vector;
  parallel::distributed::Vector<value_type> vorticity;
  parallel::distributed::Vector<value_type> divergence;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by using the analytical solution or a guess of the velocity and pressure field
//  initialize_solution();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_vectors()
{
  // solution
  ns_operation->initialize_block_vector_velocity_pressure(solution);

  // rhs_vector
  if(this->param.equation_type == EquationType::Stokes)
    ns_operation->initialize_block_vector_velocity_pressure(rhs_vector);

  // vorticity
  ns_operation->initialize_vector_vorticity(vorticity);

  // divergence
  if(this->param.compute_divergence == true)
  {
    ns_operation->initialize_vector_velocity(divergence);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_solution()
{
  ns_operation->prescribe_initial_conditions(solution.block(0),solution.block(1),0);
}


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
solve()
{
  Timer timer;
  timer.restart();

  // Steady Stokes equations
  if(this->param.equation_type == EquationType::Stokes)
  {
    // calculate rhs vector
    ns_operation->rhs_stokes_problem(rhs_vector);

    // solve coupled system of equations
    unsigned int iterations = ns_operation->solve_linear_stokes_problem(solution,rhs_vector);
    // write output
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "Solve linear Stokes problem:" << std::endl
                << "  Iterations: " << std::setw(6) << std::right << iterations
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }
  else // Steady Navier-Stokes equations
  {
    // Newton solver
    unsigned int iterations = ns_operation->solve_nonlinear_problem(solution);

    // write output
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "Solve nonlinear Navier-Stokes problem:" << std::endl
                << "  Newton iterations: " << std::setw(6) << std::right << iterations
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }

  // adjust pressure level in case of pure Dirichlet BC
  if(this->param.pure_dirichlet_bc)
  {
    ns_operation->shift_pressure(solution.block(1));
  }
}



template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
solve_steady_problem()
{
  global_timer.restart();

  postprocessing();

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "Solving steady state problem ..." << std::endl;

  solve();

  postprocessing();

  total_time += global_timer.wall_time();

  analyze_computing_times();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
postprocessing() const
{
  this->postprocessor->do_postprocessing(solution.block(0),solution.block(1),vorticity,divergence);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "_________________________________________________________________________________" << std::endl
        << std::endl << "Computing times:          min        avg        max        rel      p_min  p_max" << std::endl;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (this->total_time, MPI_COMM_WORLD);
  pcout  << "  Global time:         " << std::scientific
         << std::setprecision(4) << std::setw(10) << data.min << " "
         << std::setprecision(4) << std::setw(10) << data.avg << " "
         << std::setprecision(4) << std::setw(10) << data.max << " "
         << "          " << "  "
         << std::setw(6) << std::left << data.min_index << " "
         << std::setw(6) << std::left << data.max_index << std::endl
         << "_________________________________________________________________________________"
         << std::endl << std::endl;
}

#endif /* INCLUDE_DRIVERSTEADYPROBLEMS_H_ */
