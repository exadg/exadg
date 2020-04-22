/*
 * time_int_explicit_runge_kutta.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "time_int_explicit_runge_kutta.h"

#include "../spatial_discretization/interface.h"
#include "convection_diffusion/user_interface/input_parameters.h"
#include "functionalities/print_functions.h"
#include "time_integration/time_step_calculation.h"

namespace ConvDiff
{
template<typename Number>
TimeIntExplRK<Number>::TimeIntExplRK(
  std::shared_ptr<Operator>                       operator_in,
  InputParameters const &                         param_in,
  unsigned int const                              refine_steps_time_in,
  MPI_Comm const &                                mpi_comm_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in)
  : TimeIntExplRKBase<Number>(param_in.start_time,
                              param_in.end_time,
                              param_in.max_number_of_time_steps,
                              param_in.restart_data,
                              param_in.adaptive_time_stepping,
                              mpi_comm_in),
    pde_operator(operator_in),
    param(param_in),
    refine_steps_time(refine_steps_time_in),
    time_step_diff(1.0),
    cfl(param.cfl / std::pow(2.0, refine_steps_time_in)),
    diffusion_number(param.diffusion_number / std::pow(2.0, refine_steps_time_in)),
    wall_time(0.0),
    postprocessor(postprocessor_in)
{
}

template<typename Number>
void
TimeIntExplRK<Number>::get_wall_times(std::vector<std::string> & name,
                                      std::vector<double> &      wall_time_vector) const
{
  name.resize(1);
  std::vector<std::string> names = {"Time integrator"};
  name                           = names;

  wall_time_vector.resize(1);
  wall_time_vector[0] = wall_time;
}

template<typename Number>
void
TimeIntExplRK<Number>::set_velocities_and_times(
  std::vector<VectorType const *> const & velocities_in,
  std::vector<double> const &             times_in)
{
  velocities = velocities_in;
  times      = times_in;
}

template<typename Number>
void
TimeIntExplRK<Number>::extrapolate_solution(VectorType & vector)
{
  vector.equ(1.0, this->solution_n);
}

template<typename Number>
void
TimeIntExplRK<Number>::initialize_vectors()
{
  pde_operator->initialize_dof_vector(this->solution_n);
  pde_operator->initialize_dof_vector(this->solution_np);
}

template<typename Number>
void
TimeIntExplRK<Number>::initialize_solution()
{
  pde_operator->prescribe_initial_conditions(this->solution_n, this->time);
}

template<typename Number>
void
TimeIntExplRK<Number>::calculate_time_step_size()
{
  unsigned int degree = pde_operator->get_polynomial_degree();

  if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    this->time_step = calculate_const_time_step(param.time_step_size, refine_steps_time);

    this->pcout << std::endl
                << "Calculation of time step size (user-specified):" << std::endl
                << std::endl;
    print_parameter(this->pcout, "time step size", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFL ||
          param.calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
  {
    // calculate minimum vertex distance
    double const h_min = pde_operator->calculate_minimum_element_length();

    // maximum velocity
    double max_velocity = 0.0;
    if(param.analytical_velocity_field)
    {
      max_velocity = pde_operator->calculate_maximum_velocity(this->get_time());
    }

    // max_velocity computed above might be zero depending on the initial velocity field -> dt would
    // tend to infinity
    max_velocity = std::max(max_velocity, param.max_velocity);

    double time_step_conv = calculate_time_step_cfl_global(
      cfl, max_velocity, h_min, degree, param.exponent_fe_degree_convection);

    this->pcout << std::endl
                << "Calculation of time step size according to CFL condition:" << std::endl
                << std::endl;
    print_parameter(this->pcout, "h_min", h_min);
    print_parameter(this->pcout, "U_max", max_velocity);
    print_parameter(this->pcout, "CFL", cfl);
    print_parameter(this->pcout, "Exponent fe_degree", param.exponent_fe_degree_convection);
    print_parameter(this->pcout, "Time step size (CFL global)", time_step_conv);

    // adaptive time stepping
    if(this->adaptive_time_stepping)
    {
      double time_step_adap = std::numeric_limits<double>::max();

      if(param.analytical_velocity_field)
      {
        time_step_adap = pde_operator->calculate_time_step_cfl_analytical_velocity(
          this->get_time(), cfl, param.exponent_fe_degree_convection);
      }
      else
      {
        // do nothing (the velocity field is not known at this point)
      }

      // use adaptive time step size only if it is smaller, otherwise use global time step size
      time_step_conv = std::min(time_step_conv, time_step_adap);

      // make sure that the maximum allowable time step size is not exceeded
      time_step_conv = std::min(time_step_conv, param.time_step_size_max);

      print_parameter(this->pcout, "Time step size (CFL adaptive)", time_step_conv);
    }

    // Diffusion number condition
    if(param.calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
    {
      // calculate time step according to Diffusion number condition
      time_step_diff = calculate_const_time_step_diff(
        diffusion_number, param.diffusivity, h_min, degree, param.exponent_fe_degree_diffusion);

      this->pcout << std::endl
                  << "Calculation of time step size according to Diffusion condition:" << std::endl
                  << std::endl;
      print_parameter(this->pcout, "h_min", h_min);
      print_parameter(this->pcout, "Diffusion number", diffusion_number);
      print_parameter(this->pcout, "Exponent fe_degree", param.exponent_fe_degree_diffusion);
      print_parameter(this->pcout, "Time step size (diffusion)", time_step_diff);

      time_step_conv = std::min(time_step_conv, time_step_diff);

      this->pcout << std::endl << "Use minimum time step size:" << std::endl << std::endl;
      print_parameter(this->pcout, "Time step size (CFL and diffusion)", time_step_conv);
    }

    if(this->adaptive_time_stepping == false)
    {
      time_step_conv =
        adjust_time_step_to_hit_end_time(this->start_time, this->end_time, time_step_conv);

      this->pcout << std::endl
                  << "Adjust time step size to hit end time:" << std::endl
                  << std::endl;
      print_parameter(this->pcout, "Time step size", time_step_conv);
    }

    // set the time step size
    this->time_step = time_step_conv;
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::Diffusion)
  {
    // calculate minimum vertex distance
    double const h_min = pde_operator->calculate_minimum_element_length();

    // calculate time step according to Diffusion number condition
    time_step_diff = calculate_const_time_step_diff(
      diffusion_number, param.diffusivity, h_min, degree, param.exponent_fe_degree_diffusion);

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, time_step_diff);

    this->pcout << std::endl
                << "Calculation of time step size according to Diffusion condition:" << std::endl
                << std::endl;
    print_parameter(this->pcout, "h_min", h_min);
    print_parameter(this->pcout, "Diffusion number", diffusion_number);
    print_parameter(this->pcout, "Exponent fe_degree", param.exponent_fe_degree_diffusion);
    print_parameter(this->pcout, "Time step size (diffusion)", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::MaxEfficiency)
  {
    // calculate minimum vertex distance
    double const h_min = pde_operator->calculate_minimum_element_length();

    unsigned int const order = rk_time_integrator->get_order();

    this->time_step =
      calculate_time_step_max_efficiency(param.c_eff, h_min, degree, order, refine_steps_time);

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, this->time_step);

    this->pcout << std::endl
                << "Calculation of time step size (max efficiency):" << std::endl
                << std::endl;
    print_parameter(this->pcout, "C_eff", param.c_eff / std::pow(2, refine_steps_time));
    print_parameter(this->pcout, "Time step size", this->time_step);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified type of time step calculation is not implemented."));
  }
}

template<typename Number>
double
TimeIntExplRK<Number>::recalculate_time_step_size() const
{
  AssertThrow(param.calculation_of_time_step_size == TimeStepCalculation::CFL ||
                param.calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion,
              ExcMessage(
                "Adaptive time step is not implemented for this type of time step calculation."));

  double new_time_step_size = std::numeric_limits<double>::max();
  if(param.analytical_velocity_field)
  {
    new_time_step_size = pde_operator->calculate_time_step_cfl_analytical_velocity(
      this->get_time(), cfl, param.exponent_fe_degree_convection);
  }
  else
  {
    AssertThrow(velocities[0] != nullptr, ExcMessage("Pointer velocities[0] is not initialized."));

    new_time_step_size =
      pde_operator->calculate_time_step_cfl_numerical_velocity(*velocities[0],
                                                               cfl,
                                                               param.exponent_fe_degree_convection);
  }

  // make sure that time step size does not exceed maximum allowable time step size
  new_time_step_size = std::min(new_time_step_size, param.time_step_size_max);

  // take viscous term into account
  if(param.calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
    new_time_step_size = std::min(new_time_step_size, time_step_diff);

  bool use_limiter = true;
  if(use_limiter)
  {
    double last_time_step_size = this->get_time_step_size();
    double factor              = param.adaptive_time_stepping_limiting_factor;
    limit_time_step_change(new_time_step_size, last_time_step_size, factor);
  }

  return new_time_step_size;
}

template<typename Number>
void
TimeIntExplRK<Number>::initialize_time_integrator()
{
  bool numerical_velocity_field = false;

  if(param.convective_problem())
  {
    numerical_velocity_field = (param.get_type_velocity_field() == TypeVelocityField::DoFVector);
  }

  expl_rk_operator.reset(new ExplRKOperator(pde_operator, numerical_velocity_field));

  if(param.time_integrator_rk == TimeIntegratorRK::ExplRK1Stage1)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<ExplRKOperator, VectorType>(1, expl_rk_operator));
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK2Stage2)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<ExplRKOperator, VectorType>(2, expl_rk_operator));
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK3Stage3)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<ExplRKOperator, VectorType>(3, expl_rk_operator));
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage4)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<ExplRKOperator, VectorType>(4, expl_rk_operator));
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK3Stage4Reg2C)
  {
    rk_time_integrator.reset(
      new LowStorageRK3Stage4Reg2C<ExplRKOperator, VectorType>(expl_rk_operator));
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage5Reg2C)
  {
    rk_time_integrator.reset(
      new LowStorageRK4Stage5Reg2C<ExplRKOperator, VectorType>(expl_rk_operator));
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage5Reg3C)
  {
    rk_time_integrator.reset(
      new LowStorageRK4Stage5Reg3C<ExplRKOperator, VectorType>(expl_rk_operator));
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK5Stage9Reg2S)
  {
    rk_time_integrator.reset(
      new LowStorageRK5Stage9Reg2S<ExplRKOperator, VectorType>(expl_rk_operator));
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK3Stage7Reg2)
  {
    rk_time_integrator.reset(
      new LowStorageRKTD<ExplRKOperator, VectorType>(expl_rk_operator, 3, 7));
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage8Reg2)
  {
    rk_time_integrator.reset(
      new LowStorageRKTD<ExplRKOperator, VectorType>(expl_rk_operator, 4, 8));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }
}

template<typename Number>
bool
TimeIntExplRK<Number>::print_solver_info() const
{
  return param.solver_info_data.write(this->global_timer.wall_time(),
                                      this->time,
                                      this->time_step_number);
}

template<typename Number>
void
TimeIntExplRK<Number>::solve_timestep()
{
  if(this->print_solver_info())
    this->output_solver_info_header();

  Timer timer;
  timer.restart();

  if(param.convective_problem())
  {
    if(param.get_type_velocity_field() == TypeVelocityField::DoFVector)
    {
      expl_rk_operator->set_velocities_and_times(velocities, times);
    }
  }

  rk_time_integrator->solve_timestep(this->solution_np,
                                     this->solution_n,
                                     this->time,
                                     this->time_step);

  wall_time += timer.wall_time();

  // write output
  if(print_solver_info())
  {
    this->pcout << std::endl
                << "Solve time step explicitly: Wall time in [s] = " << std::scientific
                << timer.wall_time() << std::endl;
  }
}

template<typename Number>
void
TimeIntExplRK<Number>::postprocessing() const
{
  postprocessor->do_postprocessing(this->solution_n, this->time, this->time_step_number);
}

// instantiations

template class TimeIntExplRK<float>;
template class TimeIntExplRK<double>;

} // namespace ConvDiff
