/*
 * TimeIntBDFCoupled.h
 *
 *  Created on: Jun 13, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/parallel_block_vector.h>

#include "../../incompressible_navier_stokes/time_integration/time_int_bdf_navier_stokes.h"
#include "time_integration/push_back_vectors.h"

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
class TimeIntBDFCoupled : public TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>
{
public:
  TimeIntBDFCoupled(std::shared_ptr<NavierStokesOperation>              navier_stokes_operation_in,
                    std::shared_ptr<PostProcessorBase<dim,value_type> > postprocessor_in,
                    InputParametersNavierStokes<dim> const              &param_in,
                    unsigned int const                                  n_refine_time_in,
                    bool const                                          use_adaptive_time_stepping)
    :
    TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>
            (navier_stokes_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
    solution(this->order),
    vec_convective_term(this->order),
    navier_stokes_operation(navier_stokes_operation_in),
    N_iter_linear_average(0.0),
    N_iter_newton_average(0.0),
    solver_time_average(0.0),
    scaling_factor_continuity(1.0),
    characteristic_element_length(1.0)
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
  void postprocess_velocity();

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

  std::shared_ptr<NavierStokesOperation> navier_stokes_operation;

  // performance analysis: average number of iterations and solver time
  double N_iter_linear_average, N_iter_newton_average;
  double solver_time_average;

  // scaling factor continuity equation
  double scaling_factor_continuity;
  double characteristic_element_length;
};

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
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    navier_stokes_operation->initialize_block_vector_velocity_pressure(rhs_vector);
  }

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
  // scaling factor continuity equation:
  // Calculate characteristic element length h
  characteristic_element_length = calculate_minimum_vertex_distance(
        navier_stokes_operation->get_dof_handler_u().get_triangulation());

  characteristic_element_length = calculate_characteristic_element_length(characteristic_element_length,fe_degree_u);

  // convective term treated explicitly (additive decomposition)
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit &&
     this->param.start_with_low_order == false)
  {
    initialize_vec_convective_term();
  }
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
  VectorView<value_type> tmp(solution[0].block(0).local_size(),
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

  // update scaling factor of continuity equation
  if(this->param.use_scaling_continuity == true)
  {
    scaling_factor_continuity = this->param.scaling_factor_continuity*characteristic_element_length/this->time_steps[0];
    navier_stokes_operation->set_scaling_factor_continuity(scaling_factor_continuity);
  }
  else // use_scaling_continuity == false
  {
    scaling_factor_continuity = 1.0;
  }

  // extrapolate old solution to obtain a good initial guess for the solver
  solution_np.equ(this->extra.get_beta(0),solution[0]);
  for(unsigned int i=1;i<solution.size();++i)
    solution_np.add(this->extra.get_beta(i),solution[i]);

  // update of turbulence model
  if(this->param.use_turbulence_model == true)
  {
    Timer timer_turbulence;
    timer_turbulence.restart();

    navier_stokes_operation->update_turbulence_model(solution_np.block(0));

    if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
    {
      this->pcout << std::endl
                  << "Update of turbulent viscosity:   Wall time [s]: "
                  << std::scientific << timer_turbulence.wall_time() << std::endl;
    }
  }

  // calculate auxiliary variable p^{*} = 1/scaling_factor * p
  solution_np.block(1) *= 1.0/scaling_factor_continuity;

  // TODO
  // Update divegence and continuity penalty operator
//  parallel::distributed::Vector<value_type> const * velocity_ptr = nullptr;
//
//  // extrapolate velocity to time t_n+1 and use this velocity field to
//  // caculate the penalty parameter for the divergence and continuity penalty term
//  if(this->param.use_divergence_penalty == true ||
//     this->param.use_continuity_penalty == true)
//  {
//    parallel::distributed::Vector<value_type> velocity_extrapolated(solution[0].block(0));
//    velocity_extrapolated = 0;
//    for (unsigned int i=0; i<solution.size(); ++i)
//      velocity_extrapolated.add(this->extra.get_beta(i),solution[i].block(0));
//
//    velocity_ptr = &velocity_extrapolated;
//  }
//
//  if(this->param.use_divergence_penalty == true)
//  {
//    //navier_stokes_operation->update_divergence_penalty_operator(solution[0].block(0));
//    navier_stokes_operation->update_divergence_penalty_operator(velocity_ptr);
//  }
//  if(this->param.use_continuity_penalty == true)
//  {
//    //navier_stokes_operation->update_continuity_penalty_operator(solution[0].block(0));
//    navier_stokes_operation->update_continuity_penalty_operator(velocity_ptr);
//  }

  // if the problem to be solved is linear
  if(this->param.equation_type == EquationType::Stokes ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    // calculate rhs vector for the Stokes problem, i.e., the convective term is neglected in this step
    navier_stokes_operation->rhs_stokes_problem(rhs_vector, this->time + this->time_steps[0]);

    // Add the convective term to the right-hand side of the equations
    // if the convective term is treated explicitly (additive decomposition):
    // evaluate convective term and add extrapolation of convective term to the rhs (-> minus sign!)
    if(this->param.equation_type == EquationType::NavierStokes &&
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    {
      navier_stokes_operation->evaluate_convective_term(vec_convective_term[0],solution[0].block(0),this->time);

      for(unsigned int i=0;i<vec_convective_term.size();++i)
        rhs_vector.block(0).add(-this->extra.get_beta(i),vec_convective_term[i]);
    }

    //TODO OIF splitting
    // calculate sum (alpha_i/dt * u_tilde_i) in case of explicit treatment of convective term
    // and operator-integration-factor splitting
    if(this->param.equation_type == EquationType::NavierStokes &&
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
    {
      // fill vectors with old velocity solutions and old time instants for
      // interpolation of velocity field
      std::vector<parallel::distributed::Vector<value_type> *> solutions;
      std::vector<double> times;
      for(unsigned int i=0;i<solution.size();++i)
      {
        solutions.push_back(&solution[i].block(0));
        times.push_back(this->time - (double)(i) * this->time_steps[0]);
      }

      // Loop over all previous time instants required by the BDF scheme
      // and calculate u_tilde by substepping algorithm, i.e.,
      // integrate over time interval t_{n-i} <= t <= t_{n+1}
      // using explicit Runge-Kutta methods.
      for(unsigned int i=0;i<solution.size();++i)
      {
        // initialize solution: u_tilde(s=0) = u(t_{n-i})
        this->solution_tilde_m = solution[i].block(0);

        // calculate start time t_{n-i} (assume equidistant time step sizes!!!)
        double const time_n_i = this->time - (double)(i) * this->time_steps[i];

        // time loop substepping: t_{n-i} <= t <= t_{n+1}
        for(unsigned int m=0; m<this->M*(i+1);++m)
        {
          // solve time step
          this->rk_time_integrator_OIF->solve_timestep(this->solution_tilde_mp,
                                                       this->solution_tilde_m,
                                                       time_n_i + this->delta_s*m,
                                                       this->delta_s,
                                                       solutions,
                                                       times);

          this->solution_tilde_mp.swap(this->solution_tilde_m);
        }

        // calculate sum (alpha_i/dt * u_tilde_i)
        if(i==0)
          sum_alphai_ui.equ(this->bdf.get_alpha(i)/this->time_steps[0],this->solution_tilde_m);
        else // i>0
          sum_alphai_ui.add(this->bdf.get_alpha(i)/this->time_steps[0],this->solution_tilde_m);
      }
    }
    // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
    else
    {
      sum_alphai_ui.equ(this->bdf.get_alpha(0)/this->time_steps[0],solution[0].block(0));
      for (unsigned int i=1;i<solution.size();++i)
      {
        sum_alphai_ui.add(this->bdf.get_alpha(i)/this->time_steps[0],solution[i].block(0));
      }
    }
    //TODO OIF splitting

    // apply mass matrix to sum_alphai_ui and add to rhs vector
    navier_stokes_operation->apply_mass_matrix_add(rhs_vector.block(0),sum_alphai_ui);

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
    // calculate sum (alpha_i/dt * u_i)
    sum_alphai_ui.equ(this->bdf.get_alpha(0)/this->time_steps[0],solution[0].block(0));
    for (unsigned int i=1;i<solution.size();++i)
    {
      sum_alphai_ui.add(this->bdf.get_alpha(i)/this->time_steps[0],solution[i].block(0));
    }

    // Newton solver
    unsigned int newton_iterations = 0;
    unsigned int linear_iterations = 0;
    navier_stokes_operation->solve_nonlinear_problem(solution_np,
                                                     sum_alphai_ui,
                                                     this->time + this->time_steps[0],
                                                     this->get_scaling_factor_time_derivative_term(),
                                                     newton_iterations,
                                                     linear_iterations);

    N_iter_newton_average += newton_iterations;
    N_iter_linear_average += linear_iterations;
    solver_time_average += timer.wall_time();

    // write output
    if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
    {
      ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

      pcout << "Solve nonlinear Navier-Stokes problem:" << std::endl
            << "  Newton iterations: " << std::setw(6) << std::right << newton_iterations
            << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl
            << "  Linear iterations: " << std::setw(6) << std::fixed << std::setprecision(2) << std::right << double(linear_iterations)/(double)newton_iterations << " (avg)" << std::endl
            << "  Linear iterations: " << std::setw(6) << std::fixed << std::setprecision(2) << std::right << linear_iterations << " (tot)" << std::endl;
    }
  }

  // reconstruct pressure solution p from auxiliary variable p^{*}: p = scaling_factor * p^{*}
  solution_np.block(1) *= scaling_factor_continuity;

  // special case: pure Dirichlet BC's
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
      navier_stokes_operation->shift_pressure(solution_np.block(1),this->time + this->time_steps[0]);
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyZeroMeanValue)
      navier_stokes_operation->apply_zero_mean(solution_np.block(1));
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalMeanValue)
      navier_stokes_operation->shift_pressure_mean_value(solution_np.block(1),this->time + this->time_steps[0]);
    else
      AssertThrow(false,ExcMessage("Specified method to adjust pressure level is not implemented."));
  }

  // postprocess velocity field using divergence and/or continuity penalty terms
  if(this->param.use_divergence_penalty == true ||
     this->param.use_continuity_penalty == true)
  {
    postprocess_velocity();
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
postprocess_velocity()
{
  Timer timer;
  timer.restart();

  parallel::distributed::Vector<value_type> temp(solution_np.block(0));
  navier_stokes_operation->apply_mass_matrix(temp,solution_np.block(0));

  // extrapolate velocity to time t_n+1 and use this velocity field to
  // caculate the penalty parameter for the divergence and continuity penalty term
  parallel::distributed::Vector<value_type> velocity_extrapolated(solution[0].block(0));
  velocity_extrapolated = 0;
  for (unsigned int i=0; i<solution.size(); ++i)
    velocity_extrapolated.add(this->extra.get_beta(i),solution[i].block(0));

  // update projection operator
  navier_stokes_operation->update_projection_operator(velocity_extrapolated,this->time_steps[0]);

  // calculate inhomongeneous boundary faces integrals and add to rhs
  navier_stokes_operation->rhs_projection_add(temp,this->time + this->time_steps[0]);

  // solve projection (the preconditioner is updated here)
  unsigned int iterations = navier_stokes_operation->solve_projection(solution_np.block(0),temp);

  // write output
  if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    this->pcout << std::endl
                << "Postprocessing of velocity field:" << std::endl
                << "  Iterations: " << std::setw(6) << std::right << iterations
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
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

  // check pressure error and formation of numerical boundary layers for standard vs. rotational formulation
//  parallel::distributed::Vector<value_type> velocity_exact;
//  navier_stokes_operation->initialize_vector_velocity(velocity_exact);
//
//  parallel::distributed::Vector<value_type> pressure_exact;
//  navier_stokes_operation->initialize_vector_pressure(pressure_exact);
//
//  navier_stokes_operation->prescribe_initial_conditions(velocity_exact,pressure_exact,this->time);
//
//  velocity_exact.add(-1.0,solution[0].block(0));
//  pressure_exact.add(-1.0,solution[0].block(1));
//
//  this->postprocessor->do_postprocessing(velocity_exact,
//                                         solution[0].block(0),
//                                         pressure_exact,
//                                         vorticity,
//                                         divergence,
//                                         this->time,
//                                         this->time_step_number);
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

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFCoupled<dim, fe_degree_u, value_type, NavierStokesOperation>::
analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  pcout << std::endl << "Number of MPI processes = " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;

  if(this->param.equation_type == EquationType::Stokes ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    pcout << std::endl << "Number of time steps = " << (this->time_step_number-1) << std::endl
                       << "Average number of iterations = " << std::scientific << std::setprecision(3) << N_iter_linear_average/(this->time_step_number-1) << std::endl
                       << "Average wall time per time step = " << std::scientific << std::setprecision(3) << solver_time_average/(this->time_step_number-1) << std::endl;
  }
  else
  {
    double n_iter_nonlinear_average = N_iter_newton_average/(this->time_step_number-1);
    double n_iter_linear_average_accumulated = N_iter_linear_average/(this->time_step_number-1);

    pcout << std::endl << "Number of time steps = " << (this->time_step_number-1) << std::endl
                       << "Average number of Newton iterations = " << std::fixed << std::setprecision(3) << n_iter_nonlinear_average << std::endl
                       << "Average number of linear iterations = " << std::fixed << std::setprecision(3)
                       << n_iter_linear_average_accumulated/n_iter_nonlinear_average << " (per nonlinear iteration)" << std::endl
                       << "Average number of linear iterations = " << std::fixed << std::setprecision(3)
                       << n_iter_linear_average_accumulated << " (accumulated)" << std::endl
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

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_COUPLED_SOLVER_H_ */
