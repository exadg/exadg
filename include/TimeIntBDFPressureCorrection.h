/*
 * TimeIntBDFPressureCorrection.h
 *
 *  Created on: Oct 26, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_TIMEINTBDFPRESSURECORRECTION_H_
#define INCLUDE_TIMEINTBDFPRESSURECORRECTION_H_


#include "TimeIntBDFNavierStokes.h"
#include "PushBackVectors.h"


template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
class TimeIntBDFPressureCorrection : public TimeIntBDFNavierStokes<dim,fe_degree_u,value_type,NavierStokesOperation>
{
public:
  TimeIntBDFPressureCorrection(std_cxx11::shared_ptr<NavierStokesOperation >  navier_stokes_operation_in,
                               std_cxx11::shared_ptr<PostProcessorBase<dim> > postprocessor_in,
                               InputParametersNavierStokes<dim> const         &param_in,
                               unsigned int const                             n_refine_time_in,
                               bool const                                     use_adaptive_time_stepping)
    :
    TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>
            (navier_stokes_operation_in,postprocessor_in,param_in,n_refine_time_in,use_adaptive_time_stepping),
    velocity(this->order),
    pressure(this->order),
    vec_convective_term(this->order),
    vorticity(this->order),
    navier_stokes_operation(navier_stokes_operation_in),
    order_pressure_extrapolation(this->param.order_pressure_extrapolation),
    extra_pressure_gradient(this->param.order_pressure_extrapolation,this->param.start_with_low_order),
    vec_pressure_gradient_term(this->param.order_pressure_extrapolation),
    computing_times(3),
    N_iter_pressure_average(0.0),
    N_iter_linear_momentum_average(0.0),
    N_iter_nonlinear_momentum_average(0.0)
  {}

  virtual ~TimeIntBDFPressureCorrection(){}

  virtual void analyze_computing_times() const;

private:
  virtual void initialize_time_integrator_constants();
  virtual void update_time_integrator_constants();

  virtual void initialize_vectors();

  virtual void initialize_current_solution();
  virtual void initialize_former_solution();

  virtual void setup_derived();

  void initialize_vec_convective_term();
  void initialize_vec_pressure_gradient_term();

  virtual void solve_timestep();

  void momentum_step();
  void rhs_momentum();
  void pressure_step();
  void projection_step();
  void rhs_projection();
  void pressure_update();

  void calculate_chi(double &chi) const;

  void rhs_pressure();

  virtual void prepare_vectors_for_next_timestep();

  virtual void postprocessing() const;

  void calculate_vorticity() const;
  void calculate_divergence() const;

  virtual void read_restart_vectors(boost::archive::binary_iarchive & ia);
  virtual void write_restart_vectors(boost::archive::binary_oarchive & oa) const;

  virtual parallel::distributed::Vector<value_type> const & get_velocity();

  std::vector<parallel::distributed::Vector<value_type> > velocity;
  parallel::distributed::Vector<value_type> velocity_np;

  std::vector<parallel::distributed::Vector<value_type> > pressure;
  parallel::distributed::Vector<value_type> pressure_np;
  parallel::distributed::Vector<value_type> pressure_increment;

  std::vector<parallel::distributed::Vector<value_type> > vec_convective_term;

  mutable parallel::distributed::Vector<value_type> vorticity;

  // solve convective step implicitly
  parallel::distributed::Vector<value_type> sum_alphai_ui;

  // rhs vector momentum step
  parallel::distributed::Vector<value_type> rhs_vec_momentum;

  // rhs vector pressur step
  parallel::distributed::Vector<value_type> rhs_vec_pressure;
  parallel::distributed::Vector<value_type> rhs_vec_pressure_temp;

  // rhs vector projection step
  parallel::distributed::Vector<value_type> rhs_vec_projection;
  parallel::distributed::Vector<value_type> rhs_vec_projection_temp;

  // postprocessing: divergence of intermediate velocity
  mutable parallel::distributed::Vector<value_type> divergence;

  std_cxx11::shared_ptr<NavierStokesOperation> navier_stokes_operation;

  // incremental formulation of pressure-correction scheme
  unsigned int order_pressure_extrapolation;

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_gradient;

  std::vector<parallel::distributed::Vector<value_type> > vec_pressure_gradient_term;

  std::vector<value_type> computing_times;

  double N_iter_pressure_average, N_iter_linear_momentum_average,
      N_iter_nonlinear_momentum_average;
};

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_time_integrator_constants()
{
  // call function of base class to initialize the standard time integrator constants
  TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::initialize_time_integrator_constants();

  // set time integrator constants for extrapolation scheme of pressure gradient term
  // in case of incremental formulation of pressure-correction scheme
  if(extra_pressure_gradient.get_order()>0)
  {
    extra_pressure_gradient.initialize();
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::update_time_integrator_constants();

  // update time integrator constants for extrapolation scheme of pressure gradient term

  // if start_with_low_order == true (no analytical solution available) the pressure is unknown at t = t_0:
  // -> use no extrapolation (order=0, non-incremental) in first time step (the pressure solution is calculated in the second sub step)
  // -> use first order extrapolation in second time steps, second order extrapolation in third time step, etc.
  if(this->adaptive_time_stepping == false)
  {
    extra_pressure_gradient.update(this->time_step_number-1);
  }
  else // adaptive time stepping
  {
    extra_pressure_gradient.update(this->time_step_number-1, this->time_steps);
  }

  // use this function to check the correctness of the time integrator constants
//  std::cout << "Coefficients extrapolation scheme pressure: Time step = " << this->time_step_number << std::endl;
//  extra_pressure_gradient.print();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
setup_derived()
{
  if(this->param.equation_type == EquationType::NavierStokes && this->param.start_with_low_order == false
      && this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
    initialize_vec_convective_term();

  if(extra_pressure_gradient.get_order()>0)
    initialize_vec_pressure_gradient_term();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_vectors()
{
  // velocity
  for(unsigned int i=0;i<velocity.size();++i)
    navier_stokes_operation->initialize_vector_velocity(velocity[i]);
  navier_stokes_operation->initialize_vector_velocity(velocity_np);

  // pressure
  for(unsigned int i=0;i<pressure.size();++i)
    navier_stokes_operation->initialize_vector_pressure(pressure[i]);
  navier_stokes_operation->initialize_vector_pressure(pressure_np);
  navier_stokes_operation->initialize_vector_pressure(pressure_increment);

  // vorticity
  navier_stokes_operation->initialize_vector_vorticity(vorticity);

  // vec_convective_term
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    for(unsigned int i=0;i<vec_convective_term.size();++i)
      navier_stokes_operation->initialize_vector_velocity(vec_convective_term[i]);
  }

  // vec_pressure_gradient_term
  for(unsigned int i=0;i<vec_pressure_gradient_term.size();++i)
    navier_stokes_operation->initialize_vector_velocity(vec_pressure_gradient_term[i]);

  // Sum_i (alpha_i/dt * u_i)
  navier_stokes_operation->initialize_vector_velocity(sum_alphai_ui);

  // rhs vector momentum
  navier_stokes_operation->initialize_vector_velocity(rhs_vec_momentum);

  // rhs vector pressure
  navier_stokes_operation->initialize_vector_pressure(rhs_vec_pressure);
  navier_stokes_operation->initialize_vector_pressure(rhs_vec_pressure_temp);

  // rhs vector projection
  navier_stokes_operation->initialize_vector_velocity(rhs_vec_projection);
  navier_stokes_operation->initialize_vector_velocity(rhs_vec_projection_temp);

  // divergence
  if(this->param.output_data.compute_divergence == true)
  {
    navier_stokes_operation->initialize_vector_velocity(divergence);
  }
}


template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_current_solution()
{
  navier_stokes_operation->prescribe_initial_conditions(velocity[0],pressure[0],this->time);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_former_solution()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<velocity.size();++i)
    navier_stokes_operation->prescribe_initial_conditions(velocity[i],pressure[i],this->time - double(i)*this->time_steps[0]);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<vec_convective_term.size();++i)
  {
    navier_stokes_operation->evaluate_convective_term(vec_convective_term[i],velocity[i],this->time - double(i)*this->time_steps[0]);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
initialize_vec_pressure_gradient_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i=1;i<vec_pressure_gradient_term.size();++i)
  {
    navier_stokes_operation->evaluate_pressure_gradient_term(vec_pressure_gradient_term[i],pressure[i],this->time - double(i)*this->time_steps[0]);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
parallel::distributed::Vector<value_type> const & TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
get_velocity()
{
  return velocity[0];
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
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

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
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

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
postprocessing() const
{
  calculate_vorticity();
  calculate_divergence();

  // check pressure error and formation of numerical boundary layers for standard vs. rotational formulation
//  parallel::distributed::Vector<value_type> velocity_exact;
//  navier_stokes_operation->initialize_vector_velocity(velocity_exact);
//
//  parallel::distributed::Vector<value_type> pressure_exact;
//  navier_stokes_operation->initialize_vector_pressure(pressure_exact);
//
//  navier_stokes_operation->prescribe_initial_conditions(velocity_exact,pressure_exact,this->time);
//
//  velocity_exact.add(-1.0,velocity[0]);
//  pressure_exact.add(-1.0,pressure[0]);
//
//  this->postprocessor->do_postprocessing(velocity_exact,
//                                         velocity[0],
//                                         pressure_exact,
//                                         vorticity,
//                                         divergence,
//                                         this->time,
//                                         this->time_step_number);

  this->postprocessor->do_postprocessing(velocity[0],
                                         velocity[0],
                                         pressure[0],
                                         vorticity,
                                         divergence,
                                         this->time,
                                         this->time_step_number);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
solve_timestep()
{
  // write output
  if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    this->pcout << std::endl
                << "______________________________________________________________________" << std::endl << std::endl
                << " Number of TIME STEPS: " << std::left << std::setw(8) << this->time_step_number
                << "t_n = " << std::scientific << std::setprecision(4) << this->time
                << " -> t_n+1 = " << this->time + this->time_steps[0] << std::endl
                << "______________________________________________________________________" << std::endl;
  }

  // perform the substeps of the pressure-correction scheme
  momentum_step();

  pressure_step();

  projection_step();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
momentum_step()
{
  Timer timer;
  timer.restart();

  /*  Calculate the right-hand side of the linear system of equations
   *  (in case of an explicit formulation of the convective term or Stokes equations)
   *  or the vector that is constant when solving the nonlinear momentum equation
   *  (where constant means that the vector does not change from one Newton iteration
   *  to the next, i.e., it does not depend on the current solution of the nonlinear solver)
   */
  rhs_momentum();

  /*
   *  Extrapolate old solution to get a good initial estimate for the solver.
   */
  velocity_np = 0.0;
  for(unsigned int i=0;i<velocity.size();++i)
  {
    velocity_np.add(this->extra.get_beta(i),velocity[i]);
  }

  /*
   *  Solve the linear or nonlinear problem.
   */
  if(this->param.equation_type == EquationType::Stokes ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    // solve linear system of equations
    unsigned int linear_iterations_momentum;
    navier_stokes_operation->solve_linear_momentum_equation(velocity_np,
                                                            rhs_vec_momentum,
                                                            this->get_scaling_factor_time_derivative_term(),
                                                            linear_iterations_momentum);

    // write output explicit case
    if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
    {
      this->pcout << std::endl
                  << "Solve linear momentum equation for intermediate velocity:" << std::endl
                  << "  Iterations:        " << std::setw(6) << std::right << linear_iterations_momentum
                  << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }

    N_iter_linear_momentum_average += linear_iterations_momentum;
  }
  else // treatment of convective term == Implicit
  {
    AssertThrow(this->param.equation_type == EquationType::NavierStokes &&
                this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit,
        ExcMessage("There is a logical error. Probably, the specified combination of input parameters is not implemented."));

    double linear_iterations_momentum;
    unsigned int nonlinear_iterations_momentum;
    navier_stokes_operation->solve_nonlinear_momentum_equation(velocity_np,
                                                              rhs_vec_momentum,
                                                              this->time + this->time_steps[0],
                                                              this->get_scaling_factor_time_derivative_term(),
                                                              nonlinear_iterations_momentum,
                                                              linear_iterations_momentum);

    // write output implicit case
    if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
    {
      this->pcout << std::endl
                  << "Solve nonlinear momentum equation for intermediate velocity:" << std::endl
                  << "  Linear iterations: " << std::setw(6) << std::right << std::fixed << std::setprecision(2) << linear_iterations_momentum << std::endl
                  << "  Newton iterations: " << std::setw(6) << std::right << nonlinear_iterations_momentum
                  << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }

    N_iter_linear_momentum_average += linear_iterations_momentum;
    N_iter_nonlinear_momentum_average += nonlinear_iterations_momentum;
  }

  computing_times[0] += timer.wall_time();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
rhs_momentum()
{
  /*
   *  calculate sum (alpha_i/dt * u_i): This term is relevant for both the explicit
   *  and the implicit formulation of the convective term
   */
  sum_alphai_ui.equ(this->bdf.get_alpha(0)/this->time_steps[0],velocity[0]);
  for (unsigned int i=1;i<velocity.size();++i)
  {
    sum_alphai_ui.add(this->bdf.get_alpha(i)/this->time_steps[0],velocity[i]);
  }
  navier_stokes_operation->apply_mass_matrix(rhs_vec_momentum,sum_alphai_ui);

  /*
   *  Add extrapolation of pressure gradient term to the rhs in case of incremental formulation
   */
  if(extra_pressure_gradient.get_order()>0)
  {
    navier_stokes_operation->evaluate_pressure_gradient_term(vec_pressure_gradient_term[0],
                                                             pressure[0],
                                                             this->time);

    for(unsigned int i=0;i<extra_pressure_gradient.get_order();++i)
      rhs_vec_momentum.add(-extra_pressure_gradient.get_beta(i),vec_pressure_gradient_term[i]);
  }

  /*
   *  Body force term
   */
  if(this->param.right_hand_side == true)
  {
    navier_stokes_operation->evaluate_add_body_force_term(rhs_vec_momentum,this->time+this->time_steps[0]);
  }

  /*
   *  Convective term formulated explicitly:
   *  Evaluate convective term and add extrapolation of convective term to the rhs
   */
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    navier_stokes_operation->evaluate_convective_term(vec_convective_term[0],velocity[0],this->time);

    for(unsigned int i=0;i<vec_convective_term.size();++i)
      rhs_vec_momentum.add(-this->extra.get_beta(i),vec_convective_term[i]);
  }

  /*
   *  Right-hand side viscous term:
   *  If a linear system of equations has to be solved,
   *  inhomogeneous parts of boundary face integrals of the viscous operator
   *  have to be shifted to the right-hand side of the equation.
   */
  if(this->param.equation_type == EquationType::Stokes ||
      this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    navier_stokes_operation->rhs_add_viscous_term(rhs_vec_momentum,this->time+this->time_steps[0]);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
pressure_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand side vector
  rhs_pressure();

  // extrapolate old solution to get a good initial estimate for the solver

  // calculate initial guess for pressure solve
  pressure_increment = 0.0;

  // extrapolate old solution to get a good initial estimate for the
  // pressure solution p_{n+1} at time t^{n+1}
  for(unsigned int i=0;i<pressure.size();++i)
  {
    pressure_increment.add(this->extra.get_beta(i),pressure[i]);
  }

  // incremental formulation
  if(extra_pressure_gradient.get_order()>0)
  {
    // Subtract extrapolation of pressure since the PPE is solved for the
    // pressure increment phi = p_{n+1} - sum_i (beta_pressure_extra_i * pressure_i)
    // where p_{n+1} is approximated by extrapolation of order J (=order of BDF scheme).
    // Note that divergence correction term in case of rotational formulation is not
    // considered when calculating a good initial guess for the solution of the PPE,
    // which will slightly increase the number of iterations compared to the standard
    // formulation of the pressure-correction scheme.
    for(unsigned int i=0;i<extra_pressure_gradient.get_order();++i)
    {
      pressure_increment.add(-this->extra_pressure_gradient.get_beta(i),pressure[i]);
    }
  }

  // solve linear system of equations
  unsigned int iterations_pressure = navier_stokes_operation->solve_pressure(pressure_increment, rhs_vec_pressure);

  // calculate pressure p^{n+1} from pressure increment
  pressure_update();

  // Special case: pure Dirichlet BC's
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  // For some test cases it was found that ApplyZeroMeanValue works better than ApplyAnalyticalSolutionInPoint
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
      navier_stokes_operation->shift_pressure(pressure_np,this->time + this->time_steps[0]);
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyZeroMeanValue)
      navier_stokes_operation->apply_zero_mean(pressure_np);
  }

  // write output
  if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    this->pcout << std::endl
                << "Solve Poisson equation for pressure p:" << std::endl
                << "  Iterations:        " << std::setw(6) << std::right << iterations_pressure
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  computing_times[1] += timer.wall_time();

  N_iter_pressure_average += iterations_pressure;
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
calculate_chi(double &chi) const
{
  if(this->param.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    chi = 1.0;
  else if(this->param.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
    chi = 2.0;
  else
    AssertThrow(this->param.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation &&
                this->param.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation,
                ExcMessage("Not implemented!"));
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
rhs_pressure()
{
  /*
   *  I. calculate divergence term
   */
  navier_stokes_operation->evaluate_velocity_divergence_term(rhs_vec_pressure_temp, velocity_np, this->time+this->time_steps[0]);
  rhs_vec_pressure.equ(-this->bdf.get_gamma0()/this->time_steps[0],rhs_vec_pressure_temp);


  /*
   *  II. calculate terms originating from inhomogeneous parts of boundary face integrals,
   *  i.e., pressure Dirichlet boundary conditions on Gamma_N and
   *  pressure Neumann boundary conditions on Gamma_D (always h=0 for pressure-correction scheme!)
   */
  navier_stokes_operation->rhs_ppe_laplace_add(rhs_vec_pressure, this->time+this->time_steps[0]);

  // incremental formulation of pressure-correction scheme
  for(unsigned int i=0;i<extra_pressure_gradient.get_order();++i)
  {
    double time_offset = 0.0;
    for(unsigned int k=0; k<=i;++k)
      time_offset += this->time_steps[k];

    rhs_vec_pressure_temp = 0.0; // set rhs_vec_pressure_temp to zero since rhs_ppe_laplace_add() adds into dst-vector
    navier_stokes_operation->rhs_ppe_laplace_add(rhs_vec_pressure_temp, this->time + this->time_steps[0] - time_offset);
    rhs_vec_pressure.add(-extra_pressure_gradient.get_beta(i),rhs_vec_pressure_temp);
  }

  // TODO remove this
  // rotational formulation of pressure-correction scheme
//  if(this->param.rotational_formulation == true)
//  {
//    rhs_vec_pressure_temp = 0.0; // set rhs_vec_pressure_temp to zero since rhs_ppe_nbc_divergence_term_add() adds into dst-vector
//    navier_stokes_operation->rhs_ppe_divergence_term_add(rhs_vec_pressure_temp,velocity_np);
//
//    double chi = 0.0;
//    calculate_chi(chi);
//
//    rhs_vec_pressure.add(chi * this->param.viscosity,rhs_vec_pressure_temp);
//  }

  // special case: pure Dirichlet BC's
  // TODO:
  // check if this is really necessary, because from a theoretical
  // point of view one would expect that the mean value of the rhs of the
  // presssure Poisson equation is zero if consistent Dirichlet boundary
  // conditions are prescribed.
  // In principle, it works (since the linear system of equations is consistent)
  // but we detected no convergence for some test cases and specific parameters.
  // Hence, for reasons of robustness we also solve a transformed linear system of equations
  // in case of the pressure-correction scheme.

  if(this->param.pure_dirichlet_bc)
    navier_stokes_operation->apply_zero_mean(rhs_vec_pressure);

}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
pressure_update()
{
  // First set pressure solution to zero.
  pressure_np = 0.0;

  // Rotational formulation only (this step is performed first in order
  // to avoid the storage of another temporary variable).
  if(this->param.rotational_formulation == true)
  {
    // Automatically sets pressure_np to zero before operator evaluation.
    navier_stokes_operation->evaluate_velocity_divergence_term(pressure_np,velocity_np,this->time + this->time_steps[0]);
    navier_stokes_operation->apply_inverse_pressure_mass_matrix(pressure_np,pressure_np);

    double chi = 0.0;
    calculate_chi(chi);

    pressure_np *= - chi * this->param.viscosity;
  }

  // This is done for both the incremental and the non-incremental formulation,
  // the standard and the rotational formulation.
  pressure_np.add(1.0,pressure_increment);

  // Incremental formulation only.

  // add extrapolation of pressure to the pressure-increment solution in order to obtain
  // the pressure solution at the end of the time step, i.e.,
  // p^{n+1} = (pressure_increment)^{n+1} + sum_i (beta_pressure_extrapolation_i * p^{n-i});
  for(unsigned int i=0;i<extra_pressure_gradient.get_order();++i)
  {
    pressure_np.add(this->extra_pressure_gradient.get_beta(i),pressure[i]);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
rhs_projection()
{
  /*
   *  I. calculate mass matrix term
   */
  navier_stokes_operation->apply_mass_matrix(rhs_vec_projection,velocity_np);

  /*
   *  II. calculate pressure gradient term including boundary condition g_p(t_{n+1})
   */
  navier_stokes_operation->evaluate_pressure_gradient_term(rhs_vec_projection_temp,pressure_increment,this->time + this->time_steps[0]);
  rhs_vec_projection.add(-this->time_steps[0]/this->bdf.get_gamma0(),rhs_vec_projection_temp);

  /*
   *  III. pressure gradient term: boundary conditions g_p(t_{n-i})
   *       in case of incremental formulation of pressure-correction scheme
   */
  for(unsigned int i=0;i<extra_pressure_gradient.get_order();++i)
  {
    double time_offset = 0.0;
    for(unsigned int k=0; k<=i;++k)
      time_offset += this->time_steps[k];

    // evaluate inhomogeneous parts of boundary face integrals
    navier_stokes_operation->rhs_pressure_gradient_term(rhs_vec_projection_temp, this->time + this->time_steps[0] - time_offset);
    rhs_vec_projection.add(-extra_pressure_gradient.get_beta(i)*this->time_steps[0]/this->bdf.get_gamma0(),rhs_vec_projection_temp);
  }

  // TODO remove this
  /*
   *  IV. pressure gradient term: boundary condition chi*nu*div(u_hat)
   *      in case of rotational formulation of pressure-correction scheme
   */
//  if(this->param.rotational_formulation == true)
//  {
//    rhs_vec_projection_temp = 0.0;
//    navier_stokes_operation->pressure_gradient_bc_term_div_term_add(rhs_vec_projection_temp,velocity_np);
//
//    double chi = 0.0;
//    calculate_chi(chi);
//    rhs_vec_projection.add(-chi*this->param.viscosity*this->time_steps[0]/this->bdf.get_gamma0(),rhs_vec_projection_temp);
//  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
projection_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand-side vector
  rhs_projection();

  // solve linear system of equations
  unsigned int iterations_projection = navier_stokes_operation->solve_projection(velocity_np,rhs_vec_projection,velocity[0],this->cfl,this->time_steps[0]);

  // write output
  if(this->time_step_number%this->param.output_solver_info_every_timesteps == 0)
  {
    this->pcout << std::endl
                << "Solve projection step for intermediate velocity:" << std::endl
                << "  Iterations:        " << std::setw(6) << std::right << iterations_projection
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  computing_times[2] += timer.wall_time();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
prepare_vectors_for_next_timestep()
{
  push_back(velocity);
  velocity[0].swap(velocity_np);

  push_back(pressure);
  pressure[0].swap(pressure_np);

  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    push_back(vec_convective_term);
  }

  if(extra_pressure_gradient.get_order()>0)
  {
    push_back(vec_pressure_gradient_term);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
calculate_vorticity() const
{
  navier_stokes_operation->compute_vorticity(vorticity, velocity[0]);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
calculate_divergence() const
{
  if(this->param.output_data.compute_divergence == true)
  {
    navier_stokes_operation->compute_divergence(divergence, velocity[0]);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void TimeIntBDFPressureCorrection<dim, fe_degree_u, value_type, NavierStokesOperation>::
analyze_computing_times() const
{
  this->pcout << std::endl
              << "Number of MPI processes = " << Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) << std::endl;

  if(this->param.equation_type == EquationType::Stokes ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    this->pcout << std::endl
                << "Number of time steps = " << (this->time_step_number-1) << std::endl
                << "Average number of linear iterations momentum step = " << std::scientific << std::setprecision(3)
                << N_iter_linear_momentum_average/(this->time_step_number-1) << std::endl;
  }
  else
  {
    this->pcout << std::endl
                << "Number of time steps = " << (this->time_step_number-1) << std::endl
                << "Average number of linear iterations momentum step = " << std::scientific << std::setprecision(3)
                << N_iter_linear_momentum_average/(this->time_step_number-1) << std::endl
                << "Average number of nonlinear iterations momentum step = " << std::scientific << std::setprecision(3)
                << N_iter_nonlinear_momentum_average/(this->time_step_number-1) << std::endl;
  }

  this->pcout << "Average number of iterations pressure Poisson = " << std::scientific << std::setprecision(3)
              << N_iter_pressure_average/(this->time_step_number-1) << std::endl
              << "Average wall time per time step = " << std::scientific << std::setprecision(3)
              << this->total_time/(this->time_step_number-1) << std::endl;

  std::string names[5] = {"Momentum     ","Pressure     ","Projection   "};
  this->pcout << std::endl
              << "_________________________________________________________________________________" << std::endl << std::endl
              << "Computing times:          min        avg        max        rel      p_min  p_max " << std::endl;

  double total_avg_time = 0.0;

  for (unsigned int i=0; i<computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (computing_times[i], MPI_COMM_WORLD);
        total_avg_time += data.avg;
  }

  for (unsigned int i=0; i<computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (computing_times[i], MPI_COMM_WORLD);
    this->pcout << "  Step " << i+1 <<  ": " << names[i]  << std::scientific
                << std::setprecision(4) << std::setw(10) << data.min << " "
                << std::setprecision(4) << std::setw(10) << data.avg << " "
                << std::setprecision(4) << std::setw(10) << data.max << " "
                << std::setprecision(4) << std::setw(10) << data.avg/total_avg_time << "  "
                << std::setw(6) << std::left << data.min_index << " "
                << std::setw(6) << std::left << data.max_index << std::endl;
  }

  this->pcout  << "  Time in steps 1-" << computing_times.size() << ":              "
               << std::setprecision(4) << std::setw(10) << total_avg_time
               << "            "
               << std::setprecision(4) << std::setw(10) << total_avg_time/total_avg_time << std::endl;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg (this->total_time, MPI_COMM_WORLD);
  this->pcout  << "  Global time:         " << std::scientific
               << std::setprecision(4) << std::setw(10) << data.min << " "
               << std::setprecision(4) << std::setw(10) << data.avg << " "
               << std::setprecision(4) << std::setw(10) << data.max << " "
               << "          " << "  "
               << std::setw(6) << std::left << data.min_index << " "
               << std::setw(6) << std::left << data.max_index << std::endl
               << "_________________________________________________________________________________"
               << std::endl << std::endl;
}


#endif /* INCLUDE_TIMEINTBDFPRESSURECORRECTION_H_ */
