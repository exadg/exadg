/*
 * DriverSteadyProblems.h
 *
 *  Created on: Jul 4, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DRIVERSTEADYPROBLEMS_H_
#define INCLUDE_DRIVERSTEADYPROBLEMS_H_


template<int dim> class PostProcessorBase;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class DriverSteadyProblems
{
public:
  DriverSteadyProblems(std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree,
                         fe_degree_p, fe_degree_xwall, xwall_quad_rule> >   ns_operation_in,
                       std_cxx11::shared_ptr<PostProcessorBase<dim> >           postprocessor_in,
                       InputParametersNavierStokes<dim> const                   &param_in)
    :
    ns_operation(std::dynamic_pointer_cast<DGNavierStokesCoupled<dim, fe_degree,
                    fe_degree_p, fe_degree_xwall, xwall_quad_rule> > (ns_operation_in)),
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
  void postprocessing();

  std_cxx11::shared_ptr<DGNavierStokesCoupled<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule> > ns_operation;

  std_cxx11::shared_ptr<PostProcessorBase<dim> > postprocessor;
  InputParametersNavierStokes<dim> const &param;

  Timer global_timer;
  value_type total_time;

  parallel::distributed::BlockVector<value_type> solution;
  parallel::distributed::BlockVector<value_type> rhs_vector;
  parallel::distributed::Vector<value_type> vorticity;
  parallel::distributed::Vector<value_type> divergence;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
setup()
{
  // initialize global solution vectors (allocation)
  initialize_vectors();

  // initialize solution by using the analytical solution or a guess of the velocity and pressure field
  initialize_solution();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
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
  if(this->param.output_data.compute_divergence == true)
  {
    ns_operation->initialize_vector_velocity(divergence);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
initialize_solution()
{
  double time = 0.0;
  ns_operation->prescribe_initial_conditions(solution.block(0),solution.block(1),time);
}


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
solve()
{
  Timer timer;
  timer.restart();

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "Solving steady state problem ..." << std::endl;

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
      std::cout << std:: endl
                << "Solve linear Stokes problem:" << std::endl
                << "  Iterations: " << std::setw(6) << std::right << iterations
                << "\t Wall time [s]: " << std::scientific << std::setprecision(4) << timer.wall_time() << std::endl;
    }
  }
  else // Steady Navier-Stokes equations
  {
    // Newton solver
    unsigned int newton_iterations;
    double average_linear_iterations;
    ns_operation->solve_nonlinear_steady_problem(solution,newton_iterations,average_linear_iterations);

    // write output
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << std:: endl
                << "Solve nonlinear Navier-Stokes problem:" << std::endl
                << "  Linear iterations (avg):" << std::setw(12) << std::scientific << std::setprecision(4) << std::right << average_linear_iterations << std::endl
                << "  Newton iterations:      " << std::setw(12) << std::right << newton_iterations
                << "\t Wall time [s]: " << std::scientific << std::setprecision(4) << timer.wall_time() << std::endl;
    }
  }

  // special case: pure Dirichlet BC's
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.error_data.analytical_solution_available == true)
      ns_operation->shift_pressure(solution.block(1));
    else // analytical_solution_available == false
      ns_operation->apply_zero_mean(solution.block(1));
  }

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl << "... done!" << std::endl;
}



template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
solve_steady_problem()
{
  global_timer.restart();

  postprocessing();

  solve();

  postprocessing();

  total_time += global_timer.wall_time();

  analyze_computing_times();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
postprocessing()
{
  // calculate divergence
  if(this->param.output_data.compute_divergence == true)
  {
    ns_operation->compute_divergence(divergence, solution.block(0));
  }

  // calculate vorticity
  ns_operation->compute_vorticity(vorticity,solution.block(0));

  this->postprocessor->do_postprocessing(solution.block(0),
                                         solution.block(0), // intermediate_velocity = velocity (inteface!)
                                         solution.block(1),
                                         vorticity,
                                         divergence);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
void DriverSteadyProblems<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>::
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
