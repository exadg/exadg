/*
 * TimeIntBDFCoupled.h
 *
 *  Created on: Jun 13, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTBDFCOUPLED_H_
#define INCLUDE_TIMEINTBDFCOUPLED_H_

#include "TimeIntBDFNavierStokes.h"
#include "PushBackVectors.h"

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
class TimeIntBDFCoupled : public TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>
{
public:
  TimeIntBDFCoupled(std_cxx11::shared_ptr<NavierStokesOperation>   navier_stokes_operation_in,
                    std_cxx11::shared_ptr<PostProcessorBase<dim> > postprocessor_in,
                    InputParametersNavierStokes<dim> const         &param_in,
                    unsigned int const                             n_refine_time_in,
                    bool const                                     use_adaptive_time_stepping)
    :
    TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>
            (navier_stokes_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
    solution(this->order),
    vec_convective_term(this->order),
    navier_stokes_operation(navier_stokes_operation_in),
    N_iter_linear_average(0.0),
    N_iter_newton_average(0.0),
    solver_time_average(0.0)
  {}

  virtual ~TimeIntBDFCoupled(){}

  virtual void analyze_computing_times() const;

private:
  virtual void setup_derived();

  virtual void initialize_vectors();

  virtual void initialize_current_solution();
  virtual void initialize_former_solution();

  void calculate_vorticity() const;
  void calculate_divergence() const;
  void initialize_vec_convective_term();

  virtual void solve_timestep();
  virtual void postprocessing() const;

  virtual void prepare_vectors_for_next_timestep();

  virtual parallel::distributed::Vector<value_type> const & get_velocity();

  virtual void read_restart_vectors(boost::archive::binary_iarchive & ia);
  virtual void write_restart_vectors(boost::archive::binary_oarchive & oa) const;

  parallel::distributed::BlockVector<value_type> solution_np;
  std::vector<parallel::distributed::BlockVector<value_type> > solution;

  parallel::distributed::Vector<value_type> sum_alphai_ui;
  parallel::distributed::BlockVector<value_type> rhs_vector;

  std::vector<parallel::distributed::Vector<value_type> > vec_convective_term;

  mutable parallel::distributed::Vector<value_type> vorticity;

  mutable parallel::distributed::Vector<value_type> divergence;

  std_cxx11::shared_ptr<NavierStokesOperation> navier_stokes_operation;

  // performance analysis: average number of iterations and solver time
  double N_iter_linear_average, N_iter_newton_average;
  double solver_time_average;
};

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  if(this->param.equation_type == EquationType::Stokes ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    pcout << std::endl << "Number of time steps = " << (this->time_step_number-1) << std::endl
                       << "Average number of iterations = " << std::scientific << std::setprecision(3) << N_iter_linear_average/(this->time_step_number-1) << std::endl
                       << "Average wall time per time step = " << std::scientific << std::setprecision(3) << solver_time_average/(this->time_step_number-1) << std::endl;
  }
  else
  {
    pcout << std::endl << "Number of time steps = " << (this->time_step_number-1) << std::endl
                           << "Average number of linear iterations = " << std::fixed << std::setprecision(3) << N_iter_linear_average/(this->time_step_number-1) << std::endl
                           << "Average number of Newton iterations = " << std::fixed << std::setprecision(3) << N_iter_newton_average/(this->time_step_number-1) << std::endl
                           << "Average wall time per time step = " << std::scientific << std::setprecision(3) << solver_time_average/(this->time_step_number-1) << std::endl;
  }

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

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_vectors()
{
  // solution
  for(unsigned int i=0;i<solution.size();++i)
    navier_stokes_operation->initialize_block_vector_velocity_pressure(solution[i]);
  navier_stokes_operation->initialize_block_vector_velocity_pressure(solution_np);

  // convective term
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    for(unsigned int i=0;i<vec_convective_term.size();++i)
      navier_stokes_operation->initialize_vector_velocity(vec_convective_term[i]);
  }

  // temporal derivative term: sum_i (alpha_i * u_i)
  navier_stokes_operation->initialize_vector_velocity(sum_alphai_ui);

  // rhs_vector
  if(this->param.equation_type == EquationType::Stokes ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    navier_stokes_operation->initialize_block_vector_velocity_pressure(rhs_vector);

  // vorticity
  navier_stokes_operation->initialize_vector_vorticity(vorticity);

  // divergence
  if(this->param.output_data.compute_divergence == true)
    navier_stokes_operation->initialize_vector_velocity(divergence);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_current_solution()
{
  navier_stokes_operation->prescribe_initial_conditions(solution[0].block(0),solution[0].block(1),this->time);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_former_solution()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<solution.size();++i)
    navier_stokes_operation->prescribe_initial_conditions(solution[i].block(0),solution[i].block(1),this->time - double(i)*this->time_steps[0]);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
setup_derived()
{
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit &&
     this->param.start_with_low_order == false)
    initialize_vec_convective_term();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
calculate_vorticity() const
{
  navier_stokes_operation->compute_vorticity(vorticity, solution[0].block(0));
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
calculate_divergence() const
{
  if(this->param.output_data.compute_divergence == true)
  {
    navier_stokes_operation->compute_divergence(divergence, solution[0].block(0));
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<vec_convective_term.size();++i)
  {
    navier_stokes_operation->evaluate_convective_term(vec_convective_term[i],
                                                   solution[i].block(0),
                                                   this->time - double(i)*this->time_steps[0]);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
parallel::distributed::Vector<value_type> const &  TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
get_velocity()
{
  return solution[0].block(0);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  Vector<double> tmp;
  for (unsigned int i=0; i<solution.size(); i++)
  {
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(),
              solution[i].block(0).begin());
  }
  for (unsigned int i=0; i<solution.size(); i++)
  {
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(),
              solution[i].block(1).begin());
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
write_restart_vectors(boost::archive::binary_oarchive & oa) const
{
  VectorView<double> tmp(solution[0].block(0).local_size(),
                         solution[0].block(0).begin());
  oa << tmp;
  for (unsigned int i=1; i<solution.size(); i++)
  {
    tmp.reinit(solution[i].block(0).local_size(),
        solution[i].block(0).begin());
    oa << tmp;
  }
  for (unsigned int i=0; i<solution.size(); i++)
  {
    tmp.reinit(solution[i].block(1).local_size(),
        solution[i].block(1).begin());
    oa << tmp;
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
solve_timestep()
{
  Timer timer;
  timer.restart();

  // write output
  if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl
          << "______________________________________________________________________"
          << std::endl
          << std::endl
          << " Number of TIME STEPS: " << std::left << std::setw(8) << this->time_step_number
          << "t_n = " << std::scientific << std::setprecision(4) << this->time
          << " -> t_n+1 = " << this->time + this->time_steps[0] << std::endl
          << "______________________________________________________________________"
          << std::endl << std::endl;
  }

  // extrapolate old solution to obtain a good initial guess for the solver
  solution_np.equ(this->beta[0],solution[0]);
  for(unsigned int i=1;i<solution.size();++i)
    solution_np.add(this->beta[i],solution[i]);

  // calculate sum (alpha_i/dt * u_i)
  sum_alphai_ui.equ(this->alpha[0]/this->time_steps[0],solution[0].block(0));
  for (unsigned int i=1;i<solution.size();++i)
  {
    sum_alphai_ui.add(this->alpha[i]/this->time_steps[0],solution[i].block(0));
  }

  // if the problem to be solved is linear
  if(this->param.equation_type == EquationType::Stokes ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    // calculate rhs vector for the Stokes problem, i.e., the convective term is neglected in this step
    navier_stokes_operation->rhs_stokes_problem(rhs_vector, &sum_alphai_ui, this->time + this->time_steps[0]);

    // evaluate convective term and add extrapolation of convective term to the rhs (-> minus sign!)
    if(this->param.equation_type == EquationType::NavierStokes)
    {
      navier_stokes_operation->evaluate_convective_term(vec_convective_term[0],solution[0].block(0),this->time);

      for(unsigned int i=0;i<vec_convective_term.size();++i)
        rhs_vector.block(0).add(-this->beta[i],vec_convective_term[i]);
    }

    // solve coupled system of equations
    unsigned int iterations = navier_stokes_operation->solve_linear_stokes_problem(solution_np,
                                                                                   rhs_vector,
                                                                                   this->get_scaling_factor_time_derivative_term());

    N_iter_linear_average += iterations;
    solver_time_average += timer.wall_time();

    // write output
    if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
    {
      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      pcout << "Solve linear Stokes problem:" << std::endl
            << "  Iterations: " << std::setw(6) << std::right << iterations
            << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }
  else // a nonlinear system of equations has to be solved
  {
    // Newton solver
    unsigned int newton_iterations = 0;
    double average_linear_iterations = 0.0;
    navier_stokes_operation->solve_nonlinear_problem(solution_np,
                                                     sum_alphai_ui,
                                                     this->time + this->time_steps[0],
                                                     this->get_scaling_factor_time_derivative_term(),
                                                     newton_iterations,
                                                     average_linear_iterations);

    N_iter_newton_average += newton_iterations;
    N_iter_linear_average += average_linear_iterations;
    solver_time_average += timer.wall_time();

    // write output
    if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
    {
      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
      pcout << "Solve nonlinear Navier-Stokes problem:" << std::endl
            << "  Linear iterations (avg): " << std::setw(6) << std::right << average_linear_iterations << std::endl
            << "  Newton iterations:       " << std::setw(6) << std::right << newton_iterations
            << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }

  // special case: pure Dirichlet BC's
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.error_data.analytical_solution_available == true)
      navier_stokes_operation->shift_pressure(solution_np.block(1),this->time + this->time_steps[0]);
    else // analytical_solution_available == false
      navier_stokes_operation->apply_zero_mean(solution_np.block(1));
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
postprocessing() const
{
  calculate_vorticity();
  calculate_divergence();

  this->postprocessor->do_postprocessing(solution[0].block(0),
                                         solution[0].block(0), // intermediate_velocity = velocity
                                         solution[0].block(1),
                                         vorticity,
                                         divergence,
                                         this->time,
                                         this->time_step_number);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
prepare_vectors_for_next_timestep()
{
  push_back(solution);
  solution[0].swap(solution_np);

  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    push_back(vec_convective_term);
  }
}

#endif /* INCLUDE_TIMEINTBDFCOUPLED_H_ */
