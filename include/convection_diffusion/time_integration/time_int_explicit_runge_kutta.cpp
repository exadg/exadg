/*
 * time_int_explicit_runge_kutta.cpp
 *
 *  Created on: Nov 14, 2018
 *      Author: fehn
 */

#include "time_int_explicit_runge_kutta.h"

#include "../interface_space_time/operator.h"
#include "convection_diffusion/user_interface/input_parameters.h"
#include "functionalities/print_functions.h"
#include "time_integration/time_step_calculation.h"

namespace ConvDiff
{
template<typename Number>
TimeIntExplRK<Number>::TimeIntExplRK(std::shared_ptr<Operator> operator_in,
                                     InputParameters const &   param_in,
                                     unsigned int const        n_refine_time_in)
  : TimeIntExplRKBase<Number>(param_in.start_time,
                              param_in.end_time,
                              param_in.max_number_of_time_steps,
                              param_in.restart_data,
                              param_in.adaptive_time_stepping),
    pde_operator(operator_in),
    param(param_in),
    time_step_diff(1.0),
    n_refine_time(n_refine_time_in),
    cfl(param.cfl / std::pow(2.0, n_refine_time)),
    diffusion_number(param.diffusion_number / std::pow(2.0, n_refine_time))
{
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
    this->time_step = calculate_const_time_step(param.time_step_size, n_refine_time);

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

    double const max_velocity = pde_operator->calculate_maximum_velocity(this->time);

    double time_step_conv = calculate_time_step_cfl_global(
      cfl, max_velocity, h_min, degree, param.exponent_fe_degree_convection);

    this->pcout << std::endl
                << "Calculation of time step size according to CFL condition:" << std::endl
                << std::endl;
    print_parameter(this->pcout, "h_min", h_min);
    print_parameter(this->pcout, "U_max", max_velocity);
    print_parameter(this->pcout, "CFL", cfl);
    print_parameter(this->pcout, "Exponent fe_degree", param.exponent_fe_degree_convection);
    print_parameter(this->pcout, "Time step size (global)", time_step_conv);

    // adaptive time stepping
    if(this->adaptive_time_stepping)
    {
      double time_step_adap =
        pde_operator->calculate_time_step_cfl(this->get_time(),
                                              cfl,
                                              param.exponent_fe_degree_convection);

      // use adaptive time step size only if it is smaller, otherwise use global time step size
      time_step_conv = std::min(time_step_adap, time_step_conv);

      print_parameter(this->pcout, "Time step size (adaptive)", time_step_conv);
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
      print_parameter(this->pcout, "Time step size", time_step_diff);

      time_step_conv = std::min(time_step_conv, time_step_diff);

      this->pcout << std::endl << "Use minimum time step size:" << std::endl << std::endl;
      print_parameter(this->pcout, "Time step size (combined)", time_step_conv);
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
    print_parameter(this->pcout,
                    "Exponent fe_degree (diffusion)",
                    param.exponent_fe_degree_diffusion);
    print_parameter(this->pcout, "Time step size (diffusion)", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::MaxEfficiency)
  {
    // calculate minimum vertex distance
    double const h_min = pde_operator->calculate_minimum_element_length();

    unsigned int const order = rk_time_integrator->get_order();

    this->time_step =
      calculate_time_step_max_efficiency(param.c_eff, h_min, degree, order, n_refine_time);

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, this->time_step);

    this->pcout << std::endl
                << "Calculation of time step size (max efficiency):" << std::endl
                << std::endl;
    print_parameter(this->pcout, "C_eff", param.c_eff / std::pow(2, n_refine_time));
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

  double new_time_step_size =
    pde_operator->calculate_time_step_cfl(this->get_time(),
                                          cfl,
                                          param.exponent_fe_degree_convection);

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
  if(this->param.time_integrator_rk == TimeIntegratorRK::ExplRK1Stage1)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<Operator, VectorType>(1, pde_operator));
  }
  else if(this->param.time_integrator_rk == TimeIntegratorRK::ExplRK2Stage2)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<Operator, VectorType>(2, pde_operator));
  }
  else if(this->param.time_integrator_rk == TimeIntegratorRK::ExplRK3Stage3)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<Operator, VectorType>(3, pde_operator));
  }
  else if(this->param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage4)
  {
    rk_time_integrator.reset(
      new ExplicitRungeKuttaTimeIntegrator<Operator, VectorType>(4, pde_operator));
  }
  else if(this->param.time_integrator_rk == TimeIntegratorRK::ExplRK3Stage4Reg2C)
  {
    rk_time_integrator.reset(new LowStorageRK3Stage4Reg2C<Operator, VectorType>(pde_operator));
  }
  else if(this->param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage5Reg2C)
  {
    rk_time_integrator.reset(new LowStorageRK4Stage5Reg2C<Operator, VectorType>(pde_operator));
  }
  else if(this->param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage5Reg3C)
  {
    rk_time_integrator.reset(new LowStorageRK4Stage5Reg3C<Operator, VectorType>(pde_operator));
  }
  else if(this->param.time_integrator_rk == TimeIntegratorRK::ExplRK5Stage9Reg2S)
  {
    rk_time_integrator.reset(new LowStorageRK5Stage9Reg2S<Operator, VectorType>(pde_operator));
  }
  else if(this->param.time_integrator_rk == TimeIntegratorRK::ExplRK3Stage7Reg2)
  {
    rk_time_integrator.reset(new LowStorageRKTD<Operator, VectorType>(pde_operator, 3, 7));
  }
  else if(this->param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage8Reg2)
  {
    rk_time_integrator.reset(new LowStorageRKTD<Operator, VectorType>(pde_operator, 4, 8));
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
  return this->get_time_step_number() % param.output_solver_info_every_timesteps == 0;
}

template<typename Number>
void
TimeIntExplRK<Number>::solve_timestep()
{
  Timer timer;
  timer.restart();

  rk_time_integrator->solve_timestep(this->solution_np,
                                     this->solution_n,
                                     this->time,
                                     this->time_step);

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
  pde_operator->do_postprocessing(this->solution_n, this->time, this->time_step_number);
}

template<typename Number>
void
TimeIntExplRK<Number>::analyze_computing_times() const
{
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl
              << "Computing times:          min        avg        max        rel      p_min  p_max "
              << std::endl;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(this->total_time, MPI_COMM_WORLD);
  this->pcout << "  Global time:         " << std::scientific << std::setprecision(4)
              << std::setw(10) << data.min << " " << std::setprecision(4) << std::setw(10)
              << data.avg << " " << std::setprecision(4) << std::setw(10) << data.max << " "
              << "          "
              << "  " << std::setw(6) << std::left << data.min_index << " " << std::setw(6)
              << std::left << data.max_index << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}

// instantiations
#include <navierstokes/config.h>

#if OP_FLOAT
template class TimeIntExplRK<float>;
#endif
#if OP_DOUBLE
template class TimeIntExplRK<double>;
#endif

} // namespace ConvDiff
