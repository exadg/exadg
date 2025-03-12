/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <exadg/convection_diffusion/postprocessor/postprocessor_base.h>
#include <exadg/convection_diffusion/spatial_discretization/interface.h>
#include <exadg/convection_diffusion/time_integration/time_int_explicit_runge_kutta.h>
#include <exadg/convection_diffusion/user_interface/parameters.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_functions.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace ConvDiff
{
template<typename Number>
TimeIntExplRK<Number>::TimeIntExplRK(
  std::shared_ptr<Interface::Operator<Number>>    operator_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
  Parameters const &                              param_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in)
  : TimeIntExplRKBase<Number>(param_in.start_time,
                              param_in.end_time,
                              param_in.max_number_of_time_steps,
                              param_in.restart_data,
                              param_in.adaptive_time_stepping,
                              mpi_comm_in,
                              is_test_in),
    pde_operator(operator_in),
    param(param_in),
    refine_steps_time(param_in.n_refine_time),
    time_step_diff(1.0),
    cfl(param.cfl / std::pow(2.0, refine_steps_time)),
    diffusion_number(param.diffusion_number / std::pow(2.0, refine_steps_time)),
    postprocessor(postprocessor_in)
{
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
  if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    this->time_step = calculate_const_time_step(param.time_step_size, refine_steps_time);

    this->pcout << std::endl
                << "Calculation of time step size (user-specified):" << std::endl
                << std::endl;
    print_parameter(this->pcout, "time step size", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFL or
          param.calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
  {
    double time_step_conv = pde_operator->calculate_time_step_cfl_global(this->get_time());
    time_step_conv *= cfl;

    this->pcout << std::endl
                << "Calculation of time step size according to CFL condition:" << std::endl
                << std::endl;
    print_parameter(this->pcout, "CFL", cfl);
    print_parameter(this->pcout, "Time step size (CFL global)", time_step_conv);

    // adaptive time stepping
    if(this->adaptive_time_stepping)
    {
      double time_step_adap = std::numeric_limits<double>::max();

      if(param.analytical_velocity_field)
      {
        time_step_adap =
          pde_operator->calculate_time_step_cfl_analytical_velocity(this->get_time());
        time_step_adap *= cfl;
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
      time_step_diff = pde_operator->calculate_time_step_diffusion();
      time_step_diff *= diffusion_number;

      this->pcout << std::endl
                  << "Calculation of time step size according to Diffusion condition:" << std::endl
                  << std::endl;
      print_parameter(this->pcout, "Diffusion number", diffusion_number);
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
    // calculate time step according to Diffusion number condition
    time_step_diff = pde_operator->calculate_time_step_diffusion();
    time_step_diff *= diffusion_number;

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, time_step_diff);

    this->pcout << std::endl
                << "Calculation of time step size according to Diffusion condition:" << std::endl
                << std::endl;
    print_parameter(this->pcout, "Diffusion number", diffusion_number);
    print_parameter(this->pcout, "Time step size (diffusion)", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::MaxEfficiency)
  {
    unsigned int const order = rk_time_integrator->get_order();

    this->time_step    = pde_operator->calculate_time_step_max_efficiency(order);
    double const c_eff = param.c_eff / std::pow(2., refine_steps_time);
    this->time_step *= c_eff;

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, this->time_step);

    this->pcout << std::endl
                << "Calculation of time step size (max efficiency):" << std::endl
                << std::endl;
    print_parameter(this->pcout, "C_eff", c_eff);
    print_parameter(this->pcout, "Time step size", this->time_step);
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("Specified type of time step calculation is not implemented."));
  }
}

template<typename Number>
double
TimeIntExplRK<Number>::recalculate_time_step_size() const
{
  AssertThrow(param.calculation_of_time_step_size == TimeStepCalculation::CFL or
                param.calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion,
              dealii::ExcMessage(
                "Adaptive time step is not implemented for this type of time step calculation."));

  double new_time_step_size = std::numeric_limits<double>::max();
  if(param.analytical_velocity_field)
  {
    new_time_step_size =
      pde_operator->calculate_time_step_cfl_analytical_velocity(this->get_time());
    new_time_step_size *= cfl;
  }
  else
  {
    AssertThrow(velocities[0] != nullptr,
                dealii::ExcMessage("Pointer velocities[0] is not initialized."));

    new_time_step_size = pde_operator->calculate_time_step_cfl_numerical_velocity(*velocities[0]);
    new_time_step_size *= cfl;
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

  expl_rk_operator =
    std::make_shared<OperatorExplRK<Number>>(pde_operator, numerical_velocity_field);

  if(param.time_integrator_rk == TimeIntegratorRK::ExplRK1Stage1)
  {
    rk_time_integrator =
      std::make_shared<ExplicitRungeKuttaTimeIntegrator<OperatorExplRK<Number>, VectorType>>(
        1, expl_rk_operator);
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK2Stage2)
  {
    rk_time_integrator =
      std::make_shared<ExplicitRungeKuttaTimeIntegrator<OperatorExplRK<Number>, VectorType>>(
        2, expl_rk_operator);
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK3Stage3)
  {
    rk_time_integrator =
      std::make_shared<ExplicitRungeKuttaTimeIntegrator<OperatorExplRK<Number>, VectorType>>(
        3, expl_rk_operator);
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage4)
  {
    rk_time_integrator =
      std::make_shared<ExplicitRungeKuttaTimeIntegrator<OperatorExplRK<Number>, VectorType>>(
        4, expl_rk_operator);
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK3Stage4Reg2C)
  {
    rk_time_integrator =
      std::make_shared<LowStorageRK3Stage4Reg2C<OperatorExplRK<Number>, VectorType>>(
        expl_rk_operator);
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage5Reg2C)
  {
    rk_time_integrator =
      std::make_shared<LowStorageRK4Stage5Reg2C<OperatorExplRK<Number>, VectorType>>(
        expl_rk_operator);
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage5Reg3C)
  {
    rk_time_integrator =
      std::make_shared<LowStorageRK4Stage5Reg3C<OperatorExplRK<Number>, VectorType>>(
        expl_rk_operator);
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK5Stage9Reg2S)
  {
    rk_time_integrator =
      std::make_shared<LowStorageRK5Stage9Reg2S<OperatorExplRK<Number>, VectorType>>(
        expl_rk_operator);
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK3Stage7Reg2)
  {
    rk_time_integrator =
      std::make_shared<LowStorageRKTD<OperatorExplRK<Number>, VectorType>>(expl_rk_operator, 3, 7);
  }
  else if(param.time_integrator_rk == TimeIntegratorRK::ExplRK4Stage8Reg2)
  {
    rk_time_integrator =
      std::make_shared<LowStorageRKTD<OperatorExplRK<Number>, VectorType>>(expl_rk_operator, 4, 8);
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
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
TimeIntExplRK<Number>::do_timestep_solve()
{
  dealii::Timer timer;
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

  if(print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Solve scalar convection-diffusion equation explicitly:";
    print_wall_time(this->pcout, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Solve-explicit"}, timer.wall_time());
}

template<typename Number>
void
TimeIntExplRK<Number>::postprocessing() const
{
  dealii::Timer timer;
  timer.restart();

  postprocessor->do_postprocessing(this->solution_n, this->time, this->time_step_number);

  this->timer_tree->insert({"Timeloop", "Postprocessing"}, timer.wall_time());
}

// instantiations

template class TimeIntExplRK<float>;
template class TimeIntExplRK<double>;

} // namespace ConvDiff
} // namespace ExaDG
