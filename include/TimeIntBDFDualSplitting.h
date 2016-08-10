/*
 * TimeIntBDFDualSplitting.h
 *
 *  Created on: May 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTBDFDUALSPLITTING_H_
#define INCLUDE_TIMEINTBDFDUALSPLITTING_H_

#include "TimeIntBDF.h"

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class TimeIntBDFDualSplitting : public TimeIntBDF<dim,fe_degree,fe_degree_p,fe_degree_xwall,n_q_points_1d_xwall,value_type>
{
public:
  TimeIntBDFDualSplitting(std_cxx11::shared_ptr<DGNavierStokesBase<dim, fe_degree,
                            fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> >  ns_operation_in,
                          std_cxx11::shared_ptr<PostProcessor<dim> >              postprocessor_in,
                          InputParameters const                                   &param_in,
                          unsigned int const                                      n_refine_time_in)
    :
    TimeIntBDF<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>
            (ns_operation_in,postprocessor_in,param_in,n_refine_time_in),
    velocity(this->order),
    vorticity(this->order),
    vec_convective_term(this->order),
    computing_times(5),
    pressure(this->order),
    ns_operation_splitting (std::dynamic_pointer_cast<DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > (ns_operation_in)),
    N_iter_pressure_average(0.0),
    N_iter_viscous_average(0.0)
  {}

  virtual ~TimeIntBDFDualSplitting(){}

  virtual void analyze_computing_times() const;

protected:
  virtual void setup_derived();

  virtual void solve_timestep();

  std::vector<parallel::distributed::Vector<value_type> > velocity;


  parallel::distributed::Vector<value_type> velocity_np;

  std::vector<parallel::distributed::Vector<value_type> > vorticity;

  std::vector<parallel::distributed::Vector<value_type> > vec_convective_term;

private:
  virtual void postprocessing() const;

  virtual void initialize_vectors();
  virtual void initialize_current_solution();
  virtual void initialize_former_solution();
  
  void initialize_vorticity();
  void initialize_vec_convective_term();

  void convective_step();
  void pressure_step();
  void projection_step();
  void viscous_step();
  
  void rhs_pressure (const parallel::distributed::Vector<value_type>  &src,
                     parallel::distributed::Vector<value_type>        &dst);

  virtual void prepare_vectors_for_next_timestep();
  void push_back_solution();
  void push_back_vorticity();
  void push_back_vec_convective_term();

  virtual parallel::distributed::Vector<value_type> const & get_velocity();

  virtual void read_restart_vectors(boost::archive::binary_iarchive & ia);
  virtual void write_restart_vectors(boost::archive::binary_oarchive & oa) const;

  std::vector<value_type> computing_times;

  parallel::distributed::Vector<value_type> pressure_np;
  std::vector<parallel::distributed::Vector<value_type> > pressure;

  parallel::distributed::Vector<value_type> vorticity_extrapolated;

  // solve convective step implicitly
  parallel::distributed::Vector<value_type> sum_alphai_ui;

  parallel::distributed::Vector<value_type> rhs_vec_pressure;
  parallel::distributed::Vector<value_type> rhs_vec_pressure_temp;
  parallel::distributed::Vector<value_type> dummy;

  parallel::distributed::Vector<value_type> rhs_vec_projection;
  parallel::distributed::Vector<value_type> rhs_vec_viscous;

  // postprocessing: divergence of intermediate velocity u_hathat
  parallel::distributed::Vector<value_type> divergence;

  std_cxx11::shared_ptr<DGNavierStokesDualSplitting<dim, fe_degree,
    fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall> > ns_operation_splitting;

  double N_iter_pressure_average, N_iter_viscous_average;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
setup_derived()
{
  initialize_vorticity();

  if(this->param.equation_type == EquationType::NavierStokes && this->param.start_with_low_order == false)
    initialize_vec_convective_term();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_vectors()
{
  // velocity
  for(unsigned int i=0;i<velocity.size();++i)
    ns_operation_splitting->initialize_vector_velocity(velocity[i]);
  ns_operation_splitting->initialize_vector_velocity(velocity_np);

  // pressure
  for(unsigned int i=0;i<pressure.size();++i)
    ns_operation_splitting->initialize_vector_pressure(pressure[i]);
  ns_operation_splitting->initialize_vector_pressure(pressure_np);

  // vorticity
  for(unsigned int i=0;i<vorticity.size();++i)
    ns_operation_splitting->initialize_vector_vorticity(vorticity[i]);
  ns_operation_splitting->initialize_vector_vorticity(vorticity_extrapolated);

  // vec_convective_term
  if(this->param.equation_type == EquationType::NavierStokes)
  {
    for(unsigned int i=0;i<vec_convective_term.size();++i)
      ns_operation_splitting->initialize_vector_velocity(vec_convective_term[i]);
  }

  // Sum_i (alpha_i/dt * u_i)
  ns_operation_splitting->initialize_vector_velocity(sum_alphai_ui);

  // rhs vector pressure
  ns_operation_splitting->initialize_vector_pressure(rhs_vec_pressure);
  ns_operation_splitting->initialize_vector_pressure(rhs_vec_pressure_temp);

  // rhs vector projection, viscous
  ns_operation_splitting->initialize_vector_velocity(rhs_vec_projection);
  ns_operation_splitting->initialize_vector_velocity(rhs_vec_viscous);

  // divergence
  if(this->param.compute_divergence == true)
  {
    ns_operation_splitting->initialize_vector_velocity(divergence);
  }
}


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_current_solution()
{
  ns_operation_splitting->prescribe_initial_conditions(velocity[0],pressure[0],this->time);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_former_solution()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<velocity.size();++i)
    ns_operation_splitting->prescribe_initial_conditions(velocity[i],pressure[i],this->time - value_type(i)*this->time_steps[0]);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_vorticity()
{
  ns_operation_splitting->compute_vorticity(vorticity[0], velocity[0]);

  if(this->param.start_with_low_order == false)
  {
    for(unsigned int i=1;i<vorticity.size();++i)
      ns_operation_splitting->compute_vorticity(vorticity[i], velocity[i]);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<vec_convective_term.size();++i)
  {
    ns_operation_splitting->evaluate_convective_term(vec_convective_term[i],velocity[i],this->time - value_type(i)*this->time_steps[0]);
    ns_operation_splitting->apply_inverse_mass_matrix(vec_convective_term[i],vec_convective_term[i]);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
parallel::distributed::Vector<value_type> const & TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
get_velocity()
{
  return velocity[0];
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  Vector<double> tmp;
  for (unsigned int i=0; i<velocity.size(); i++)
  {
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(),
              velocity[i].begin());
  }
  for (unsigned int i=0; i<pressure.size(); i++)
  {
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(),
              pressure[i].begin());
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
write_restart_vectors(boost::archive::binary_oarchive & oa) const
{
  VectorView<double> tmp(velocity[0].local_size(),
                         velocity[0].begin());
  oa << tmp;
  for (unsigned int i=1; i<velocity.size(); i++)
  {
    tmp.reinit(velocity[i].local_size(),
               velocity[i].begin());
    oa << tmp;
  }
  for (unsigned int i=0; i<pressure.size(); i++)
  {
    tmp.reinit(pressure[i].local_size(),
               pressure[i].begin());
    oa << tmp;
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
postprocessing() const
{
  this->postprocessor->do_postprocessing(velocity[0],pressure[0],vorticity[0],divergence,this->time,this->time_step_number);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
solve_timestep()
{
  // set the parameters that NavierStokesOperation depends on
  ns_operation_splitting->set_time(this->time);
  ns_operation_splitting->set_time_step(this->time_steps[0]);
  ns_operation_splitting->set_scaling_factor_time_derivative_term(this->gamma0/this->time_steps[0]);

  // perform the four substeps of the dual-splitting method
  convective_step();

  pressure_step();

  projection_step();

  viscous_step();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
convective_step()
{
  Timer timer;
  timer.restart();

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    std::cout << std::endl << "______________________________________________________________________" << std::endl
              << std::endl << " Number of TIME STEPS: " << std::left << std::setw(8) << this->time_step_number
                           << "t_n = " << std::scientific << std::setprecision(4) << this->time << " -> t_n+1 = " << this->time + this->time_steps[0] << std::endl
                           << "______________________________________________________________________" << std::endl;
  }

  // compute body force vector
  ns_operation_splitting->calculate_body_force(velocity_np,this->time+this->time_steps[0]);

  // compute convective term and extrapolate convective term (if not Stokes equations)
  if(this->param.equation_type == EquationType::NavierStokes)
  {
    ns_operation_splitting->evaluate_convective_term(vec_convective_term[0],velocity[0],this->time);
    ns_operation_splitting->apply_inverse_mass_matrix(vec_convective_term[0],vec_convective_term[0]);
    for(unsigned int i=0;i<vec_convective_term.size();++i)
      velocity_np.add(-this->beta[i],vec_convective_term[i]);
  }

  // calculate sum (alpha_i/dt * u_i)
  sum_alphai_ui.equ(this->alpha[0]/this->time_steps[0],velocity[0]);
  for (unsigned int i=1;i<velocity.size();++i)
    sum_alphai_ui.add(this->alpha[i]/this->time_steps[0],velocity[i]);

  // solve discrete temporal derivative term for intermediate velocity u_hat (if not STS approach)
  if(this->param.small_time_steps_stability == false)
  {
    velocity_np.add(1.0,sum_alphai_ui);
    velocity_np *= this->time_steps[0]/this->gamma0;
  }

  if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    // write output explicit case
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
    {
      std::cout << std::endl << "Solve nonlinear convective problem explicitly:" << std::endl
                << "  iterations:        " << std::setw(4) << std::right << "-" << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }
  else // param.formulation_of_convective_term == Implicit
  {
    AssertThrow(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit &&
                !(this->param.equation_type == EquationType::Stokes || this->param.small_time_steps_stability),
        ExcMessage("Use TREATMENT_OF_CONVECTIVE_TERM = Explicit when solving the Stokes equations or when using the STS approach."));

    unsigned int newton_iterations;
    double average_linear_iterations;
    ns_operation_splitting->solve_nonlinear_convective_problem(velocity_np,newton_iterations,average_linear_iterations,sum_alphai_ui);

    // write output implicit case
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
    {
      std::cout << std::endl << "Solve nonlinear convective problem for intermediate velocity:" << std::endl
                << "  Linear iterations (avg): " << std::setw(6) << std::right << average_linear_iterations << std::endl
                << "  Newton iterations: " << std::setw(4) << std::right << newton_iterations
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }

  computing_times[0] += timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
pressure_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand-side vector
  rhs_pressure(velocity_np,rhs_vec_pressure);

  // extrapolate old solution to get a good initial estimate for the solver
  pressure_np = 0;
  for(unsigned int i=0;i<pressure.size();++i)
  {
    pressure_np.add(this->beta[i],pressure[i]);
  }

  // solve linear system of equations
  unsigned int pres_niter = ns_operation_splitting->solve_pressure(pressure_np, rhs_vec_pressure);

  if(this->param.pure_dirichlet_bc)
  {
    ns_operation_splitting->shift_pressure(pressure_np);
  }

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    std::cout << std::endl << "Solve Poisson equation for pressure p:" << std::endl
              << "  PCG iterations:    " << std::setw(4) << std::right << pres_niter << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }
  /*
  if(time_step_number%param.output_solver_info_every_timesteps == 0)
  {
    Utilities::System::MemoryStats stats;
    Utilities::System::get_memory_stats(stats);
    Utilities::MPI::MinMaxAvg memory = Utilities::MPI::min_max_avg (stats.VmRSS/1024., MPI_COMM_WORLD);
    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "  Memory [MB]: " << memory.min << " (proc" << memory.min_index << ") <= "
                << memory.avg << " (avg)" << " <= " << memory.max << " (proc" << memory.max_index << ")" << std::endl;
    }
  }
  */
  computing_times[1] += timer.wall_time();

  N_iter_pressure_average += pres_niter;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
rhs_pressure (const parallel::distributed::Vector<value_type>  &src,
              parallel::distributed::Vector<value_type>        &dst)
{
  /******************************** I. calculate divergence term ********************************/
  ns_operation_splitting->rhs_pressure_divergence_term(dst, src, this->time+this->time_steps[0]);
  /**********************************************************************************************/

  /***** II. calculate terms originating from inhomogeneous parts of boundary face integrals ****/

  // II.1. BC terms depending on prescribed boundary data,
  //       i.e. pressure Dirichlet boundary conditions on Gamma_N and
  //       body force vector, temporal derivative of velocity on Gamma_D
  ns_operation_splitting->rhs_pressure_BC_term(dst, dummy);

  // II.2. viscous term of pressure Neumann boundary condition on Gamma_D
  //       extrapolate vorticity and subsequently evaluate boundary face integral
  //       (this is possible since pressure Neumann BC is linear in vorticity)
  vorticity_extrapolated = 0;
  for(unsigned int i=0;i<vorticity.size();++i)
    vorticity_extrapolated.add(this->beta[i], vorticity[i]);

  ns_operation_splitting->rhs_pressure_viscous_term(dst, vorticity_extrapolated);

  // II.3. convective term of pressure Neumann boundary condition on Gamma_D
  //       (only if we do not solve the Stokes equations)
  //       evaluate convective term and subsequently extrapolate rhs vectors
  //       (the convective term is nonlinear!)
  if(this->param.equation_type == EquationType::NavierStokes)
  {
    for(unsigned int i=0;i<velocity.size();++i)
    {
      rhs_vec_pressure_temp = 0;
      ns_operation_splitting->rhs_pressure_convective_term(rhs_vec_pressure_temp, velocity[i]);
      dst.add(this->beta[i], rhs_vec_pressure_temp);
    }
  }
  /**********************************************************************************************/

  if(this->param.pure_dirichlet_bc)
    ns_operation_splitting->apply_nullspace_projection(dst);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
projection_step()
{
  Timer timer;
  timer.restart();

  // when using the STS stability approach vector updates have to be performed to obtain the
  // intermediate velocity u_hat which is used to calculate the rhs of the projection step
  if(this->param.small_time_steps_stability == true)
  {
    velocity_np.add(1.0,sum_alphai_ui);
    velocity_np *= this->time_steps[0]/this->gamma0;
  }

  // compute right-hand-side vector
  ns_operation_splitting->rhs_projection(rhs_vec_projection, velocity_np, pressure_np);

  // solve linear system of equations
  unsigned int iterations_projection = ns_operation_splitting->solve_projection(velocity_np,rhs_vec_projection,velocity[0],this->cfl);

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    std::cout << std::endl << "Solve projection step for intermediate velocity:" << std::endl
              << "  PCG iterations:    " << std::setw(4) << std::right << iterations_projection << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  // postprocessing related to the analysis of different projection algorithms
  if(this->param.compute_divergence == true)
  {
    this->postprocessor->analyze_divergence_error(velocity_np,this->time+this->time_steps[0],this->time_step_number);
    ns_operation_splitting->compute_divergence(divergence, velocity_np);
  }

  computing_times[2] += timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
viscous_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand-side vector
  ns_operation_splitting->rhs_viscous(rhs_vec_viscous, velocity_np);

  // extrapolate old solution to get a good initial estimate for the solver
  velocity_np = 0;
  for (unsigned int i=0; i<velocity.size(); ++i)
    velocity_np.add(this->beta[i],velocity[i]);

  // solve linear system of equations
  unsigned int iterations_viscous = ns_operation_splitting->solve_viscous(velocity_np, rhs_vec_viscous);

  // write output
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0 && this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    std::cout << std::endl << "Solve viscous step for velocity u:" << std::endl
              << "  PCG iterations:    " << std::setw(4) << std::right << iterations_viscous << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  computing_times[3] += timer.wall_time();

  N_iter_viscous_average += iterations_viscous;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
prepare_vectors_for_next_timestep()
{
  Timer timer;
  timer.restart();

  push_back_solution();

  push_back_vorticity();

  if(this->param.equation_type == EquationType::NavierStokes)
    push_back_vec_convective_term();

  computing_times[4] += timer.wall_time();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
push_back_solution()
{
  /*
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}    t_{n+1}
   *  _______________|_________|________|_________|___________\
   *                 |         |        |         |           /
   *
   *  sol-vec:    sol[2]    sol[1]    sol[0]    sol_np
   *
   * <- sol[2] <- sol[1] <- sol[0] <- sol_np <- sol[2] <--
   * |___________________________________________________|
   *
   */

  // solution at t_{n-i} <-- solution at t_{n-i+1}
  for(unsigned int i=velocity.size()-1; i>0; --i)
  {
    velocity[i].swap(velocity[i-1]);
    pressure[i].swap(pressure[i-1]);
  }
  velocity[0].swap(velocity_np);
  pressure[0].swap(pressure_np);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
push_back_vorticity()
{
  // solution at t_{n-i} <-- solution at t_{n-i+1}
  for(unsigned int i=vorticity.size()-1; i>0; --i)
  {
    vorticity[i].swap(vorticity[i-1]);
  }
  ns_operation_splitting->compute_vorticity(vorticity[0], velocity[0]);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
push_back_vec_convective_term()
{
  // solution at t_{n-i} <-- solution at t_{n-i+1}
  for(unsigned int i=vec_convective_term.size()-1; i>0; --i)
  {
    vec_convective_term[i].swap(vec_convective_term[i-1]);
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
void TimeIntBDFDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::
analyze_computing_times() const
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  if(true)
  {
    pcout << std::endl << "Number of time steps = " << (this->time_step_number-1) << std::endl
                       << "Average number of iterations pressure Poisson = " << std::scientific << std::setprecision(3) << N_iter_pressure_average/(this->time_step_number-1) << std::endl
                       << "Average number of iterations viscous step = " << std::scientific << std::setprecision(3) << N_iter_viscous_average/(this->time_step_number-1) << std::endl
                       << "Average wall time per time step = " << std::scientific << std::setprecision(3) << this->total_time/(this->time_step_number-1) << std::endl;
  }

  std::string names[5] = {"Convection   ","Pressure     ","Projection   ","Viscous      ","Other        "};
  pcout << std::endl << "_________________________________________________________________________________" << std::endl
        << std::endl << "Computing times:          min        avg        max        rel      p_min  p_max" << std::endl;
  value_type total_avg_time = 0.0;
  for (unsigned int i=0; i<computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (computing_times[i], MPI_COMM_WORLD);
        total_avg_time += data.avg;
  }
  for (unsigned int i=0; i<computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (computing_times[i], MPI_COMM_WORLD);
    pcout << "  Step " << i+1 <<  ": " << names[i]  << std::scientific
          << std::setprecision(4) << std::setw(10) << data.min << " "
          << std::setprecision(4) << std::setw(10) << data.avg << " "
          << std::setprecision(4) << std::setw(10) << data.max << " "
          << std::setprecision(4) << std::setw(10) << data.avg/total_avg_time << "  "
          << std::setw(6) << std::left << data.min_index << " "
          << std::setw(6) << std::left << data.max_index << std::endl;
  }
  pcout  << "  Time in steps 1-" << computing_times.size() << ":              "
         << std::setprecision(4) << std::setw(10) << total_avg_time
         << "            "
         << std::setprecision(4) << std::setw(10) << total_avg_time/total_avg_time << std::endl;
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


#endif /* INCLUDE_TIMEINTBDFDUALSPLITTING_H_ */
