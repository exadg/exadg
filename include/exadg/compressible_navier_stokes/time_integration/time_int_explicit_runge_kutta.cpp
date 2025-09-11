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

#include <exadg/compressible_navier_stokes/postprocessor/postprocessor_base.h>
#include <exadg/compressible_navier_stokes/spatial_discretization/interface.h>
#include <exadg/compressible_navier_stokes/time_integration/time_int_explicit_runge_kutta.h>
#include <exadg/compressible_navier_stokes/user_interface/parameters.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace CompNS
{
template<typename Number>
TimeIntExplRK<Number>::TimeIntExplRK(
  std::shared_ptr<Operator>                       operator_in,
  Parameters const &                              param_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in)
  : TimeIntExplRKBase<Number>(param_in.start_time,
                              param_in.end_time,
                              param_in.max_number_of_time_steps,
                              param_in.restart_data,
                              false, // currently no adaptive time stepping implemented
                              mpi_comm_in,
                              is_test_in),
    pde_operator(operator_in),
    param(param_in),
    refine_steps_time(param_in.n_refine_time),
    postprocessor(postprocessor_in),
    l2_norm(0.0),
    cfl_number(param.cfl_number / std::pow(2.0, refine_steps_time)),
    diffusion_number(param.diffusion_number / std::pow(2.0, refine_steps_time))
{
}

template<typename Number>
void
TimeIntExplRK<Number>::initialize_time_integrator()
{
  // initialize Runge-Kutta time integrator
  if(this->param.temporal_discretization == TemporalDiscretization::ExplRK)
  {
    rk_time_integrator = std::make_shared<ExplicitRungeKuttaTimeIntegrator<Operator, VectorType>>(
      param.order_time_integrator, pde_operator);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK3Stage4Reg2C)
  {
    rk_time_integrator =
      std::make_shared<LowStorageRK3Stage4Reg2C<Operator, VectorType>>(pde_operator);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK4Stage5Reg2C)
  {
    rk_time_integrator =
      std::make_shared<LowStorageRK4Stage5Reg2C<Operator, VectorType>>(pde_operator);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK4Stage5Reg3C)
  {
    rk_time_integrator =
      std::make_shared<LowStorageRK4Stage5Reg3C<Operator, VectorType>>(pde_operator);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK5Stage9Reg2S)
  {
    rk_time_integrator =
      std::make_shared<LowStorageRK5Stage9Reg2S<Operator, VectorType>>(pde_operator);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK3Stage7Reg2)
  {
    rk_time_integrator = std::make_shared<LowStorageRKTD<Operator, VectorType>>(pde_operator, 3, 7);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::ExplRK4Stage8Reg2)
  {
    rk_time_integrator = std::make_shared<LowStorageRKTD<Operator, VectorType>>(pde_operator, 4, 8);
  }
  else if(this->param.temporal_discretization == TemporalDiscretization::SSPRK)
  {
    rk_time_integrator = std::make_shared<SSPRK<Operator, VectorType>>(pde_operator,
                                                                       param.order_time_integrator,
                                                                       param.stages);
  }
}

/*
 *  initialize global solution vectors (allocation)
 */
template<typename Number>
void
TimeIntExplRK<Number>::initialize_vectors()
{
  pde_operator->initialize_dof_vector(this->solution_n);
  pde_operator->initialize_dof_vector(this->solution_np);
}

/*
 *  initializes the solution by interpolation of analytical solution
 */
template<typename Number>
void
TimeIntExplRK<Number>::initialize_solution()
{
  pde_operator->prescribe_initial_conditions(this->solution_n, this->time);
}

template<typename Number>
void
TimeIntExplRK<Number>::read_restart_vectors(std::vector<VectorType *> const & vectors)
{
  pde_operator->deserialize_vectors(vectors);
}

template<typename Number>
void
TimeIntExplRK<Number>::write_restart_vectors(std::vector<VectorType const *> const & vectors) const
{
  pde_operator->serialize_vectors(vectors);
}

/*
 *  calculate time step size
 */
template<typename Number>
void
TimeIntExplRK<Number>::calculate_time_step_size()
{
  this->pcout << std::endl << "Calculation of time step size:" << std::endl << std::endl;

  if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    this->time_step = calculate_const_time_step(param.time_step_size, refine_steps_time);

    print_parameter(this->pcout, "time step size", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    // calculate time step according to CFL condition
    this->time_step = pde_operator->calculate_time_step_cfl_global();
    this->time_step *= cfl_number;

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, this->time_step);

    print_parameter(this->pcout, "CFL", cfl_number);
    print_parameter(this->pcout, "Time step size (convection)", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::Diffusion)
  {
    // calculate time step size according to diffusion number condition
    this->time_step = pde_operator->calculate_time_step_diffusion();
    this->time_step *= diffusion_number;

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, this->time_step);

    print_parameter(this->pcout, "Diffusion number", diffusion_number);
    print_parameter(this->pcout, "Time step size (diffusion)", this->time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFLAndDiffusion)
  {
    double time_step_conv = std::numeric_limits<double>::max();
    double time_step_diff = std::numeric_limits<double>::max();

    // calculate time step according to CFL condition
    time_step_conv = pde_operator->calculate_time_step_cfl_global();
    time_step_conv *= cfl_number;

    print_parameter(this->pcout, "CFL", cfl_number);
    print_parameter(this->pcout, "Time step size (convection)", time_step_conv);

    // calculate time step size according to diffusion number condition
    time_step_diff = pde_operator->calculate_time_step_diffusion();
    time_step_diff *= diffusion_number;

    print_parameter(this->pcout, "Diffusion number", diffusion_number);
    print_parameter(this->pcout, "Time step size (diffusion)", time_step_diff);

    this->time_step = std::min(time_step_conv, time_step_diff);

    this->time_step =
      adjust_time_step_to_hit_end_time(this->start_time, this->end_time, this->time_step);

    print_parameter(this->pcout, "Time step size (combined)", this->time_step);
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
  AssertThrow(false, dealii::ExcMessage("Currently no adaptive time stepping implemented."));

  return 1.0;
}

template<typename Number>
void
TimeIntExplRK<Number>::detect_instabilities() const
{
  if(this->param.detect_instabilities == true)
  {
    double const l2_norm_new = this->solution_n.l2_norm();
    if(l2_norm > 1.e-12)
    {
      AssertThrow(l2_norm_new < 10. * l2_norm,
                  dealii::ExcMessage(
                    "Instabilities detected. Norm of solution vector seems to explode."));
    }

    l2_norm = l2_norm_new;
  }
}

template<typename Number>
void
TimeIntExplRK<Number>::postprocessing() const
{
  dealii::Timer timer;
  timer.restart();

  detect_instabilities();

  postprocessor->do_postprocessing(this->solution_n, this->time, this->time_step_number);

  this->timer_tree->insert({"Timeloop", "Postprocessing"}, timer.wall_time());
}

template<typename Number>
void
TimeIntExplRK<Number>::do_timestep_solve()
{
  dealii::Timer timer;
  timer.restart();

  rk_time_integrator->solve_timestep(this->solution_np,
                                     this->solution_n,
                                     this->time,
                                     this->time_step);

  if(print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Solve compressible Navier-Stokes equations explicitly:";
    print_wall_time(this->pcout, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Solve-explicit"}, timer.wall_time());
}

template<typename Number>
bool
TimeIntExplRK<Number>::print_solver_info() const
{
  return param.solver_info_data.write(this->global_timer.wall_time(),
                                      this->time,
                                      this->time_step_number);
}

// instantiations
template class TimeIntExplRK<float>;
template class TimeIntExplRK<double>;

} // namespace CompNS
} // namespace ExaDG
