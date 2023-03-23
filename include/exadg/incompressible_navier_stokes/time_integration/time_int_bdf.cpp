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

#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor_interface.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf.h>
#include <exadg/incompressible_navier_stokes/user_interface/parameters.h>
#include <exadg/time_integration/push_back_vectors.h>
#include <exadg/time_integration/time_step_calculation.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
TimeIntBDF<dim, Number>::TimeIntBDF(
  std::shared_ptr<OperatorBase>                   operator_in,
  Parameters const &                              param_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in)
  : TimeIntBDFBase(param_in.start_time,
                   param_in.end_time,
                   param_in.max_number_of_time_steps,
                   param_in.order_time_integrator,
                   param_in.start_with_low_order,
                   param_in.adaptive_time_stepping,
                   param_in.restart_data,
                   mpi_comm_in,
                   is_test_in),
    param(param_in),
    refine_steps_time(param_in.n_refine_time),
    cfl(param.cfl / std::pow(2.0, refine_steps_time)),
    operator_base(operator_in),
    vec_convective_term(this->order),
    use_extrapolation(true),
    store_solution(false),
    postprocessor(postprocessor_in),
    vec_grid_coordinates(param_in.order_time_integrator)
{
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::allocate_vectors()
{
  // convective term
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      this->operator_base->initialize_vector_velocity(vec_convective_term[i]);

    if(param.ale_formulation == false)
    {
      this->operator_base->initialize_vector_velocity(convective_term_np);
    }
  }

  if(param.ale_formulation == true)
  {
    this->operator_base->initialize_vector_velocity(grid_velocity);

    this->operator_base->initialize_vector_velocity(grid_coordinates_np);

    for(unsigned int i = 0; i < vec_grid_coordinates.size(); ++i)
      this->operator_base->initialize_vector_velocity(vec_grid_coordinates[i]);
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::setup_derived()
{
  // In the case of an arbitrary Lagrangian-Eulerian formulation:
  if(param.ale_formulation && param.restarted_simulation == false)
  {
    // compute the grid coordinates at start time (and at previous times in case of
    // start_with_low_order == false)

    operator_base->move_grid(this->get_time());
    operator_base->fill_grid_coordinates_vector(vec_grid_coordinates[0]);

    if(this->start_with_low_order == false)
    {
      // compute grid coordinates at previous times (start with 1!)
      for(unsigned int i = 1; i < this->order; ++i)
      {
        operator_base->move_grid(this->get_previous_time(i));
        operator_base->fill_grid_coordinates_vector(vec_grid_coordinates[i]);
      }
    }
  }

  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    // vec_convective_term does not have to be initialized in ALE case (the convective
    // term is recomputed in each time step for all previous times on the new mesh).
    // vec_convective_term does not have to be initialized in case of a restart, where
    // the vectors are read from memory.
    if(this->param.ale_formulation == false && this->param.restarted_simulation == false)
    {
      initialize_vec_convective_term();
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::prepare_vectors_for_next_timestep()
{
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation == false)
    {
      push_back(this->vec_convective_term);
      vec_convective_term[0].swap(convective_term_np);
    }
  }

  if(param.ale_formulation)
  {
    push_back(vec_grid_coordinates);
    vec_grid_coordinates[0].swap(grid_coordinates_np);
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::ale_update()
{
  // and compute grid coordinates at the end of the current time step t_{n+1}
  operator_base->fill_grid_coordinates_vector(grid_coordinates_np);

  // and update grid velocity using BDF time derivative
  compute_bdf_time_derivative(
    grid_velocity, grid_coordinates_np, vec_grid_coordinates, bdf, this->get_time_step_size());

  // and hand grid velocity over to spatial discretization
  operator_base->set_grid_velocity(grid_velocity);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::advance_one_timestep_partitioned_solve(bool const use_extrapolation)
{
  this->use_extrapolation = use_extrapolation;
  this->store_solution    = true;

  Base::advance_one_timestep_solve();
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::initialize_vec_convective_term()
{
  this->operator_base->evaluate_convective_term(vec_convective_term[0],
                                                get_velocity(),
                                                this->get_time());

  if(this->param.start_with_low_order == false)
  {
    for(unsigned int i = 1; i < vec_convective_term.size(); ++i)
    {
      this->operator_base->evaluate_convective_term(vec_convective_term[i],
                                                    get_velocity(i),
                                                    this->get_previous_time(i));
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::read_restart_vectors(boost::archive::binary_iarchive & ia)
{
  for(unsigned int i = 0; i < this->order; i++)
  {
    VectorType tmp = get_velocity(i);
    ia >> tmp;
    set_velocity(tmp, i);
  }
  for(unsigned int i = 0; i < this->order; i++)
  {
    VectorType tmp = get_pressure(i);
    ia >> tmp;
    set_pressure(tmp, i);
  }

  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation == false)
    {
      for(unsigned int i = 0; i < this->order; i++)
      {
        ia >> vec_convective_term[i];
      }
    }
  }

  if(this->param.ale_formulation)
  {
    for(unsigned int i = 0; i < vec_grid_coordinates.size(); i++)
    {
      ia >> vec_grid_coordinates[i];
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::write_restart_vectors(boost::archive::binary_oarchive & oa) const
{
  for(unsigned int i = 0; i < this->order; i++)
  {
    oa << get_velocity(i);
  }
  for(unsigned int i = 0; i < this->order; i++)
  {
    oa << get_pressure(i);
  }

  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation == false)
    {
      for(unsigned int i = 0; i < this->order; i++)
      {
        oa << vec_convective_term[i];
      }
    }
  }

  if(this->param.ale_formulation)
  {
    for(unsigned int i = 0; i < vec_grid_coordinates.size(); i++)
    {
      oa << vec_grid_coordinates[i];
    }
  }
}

template<int dim, typename Number>
double
TimeIntBDF<dim, Number>::calculate_time_step_size()
{
  double time_step = 1.0;

  if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    time_step = calculate_const_time_step(param.time_step_size, refine_steps_time);

    this->pcout << std::endl << "User specified time step size:" << std::endl << std::endl;
    print_parameter(this->pcout, "time step size", time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    double time_step_global = operator_base->calculate_time_step_cfl_global();
    time_step_global *= cfl;

    this->pcout << std::endl
                << "Calculation of time step size according to CFL condition:" << std::endl
                << std::endl;
    print_parameter(this->pcout, "CFL", cfl);
    print_parameter(this->pcout, "Time step size (global)", time_step_global);

    if(this->adaptive_time_stepping == true)
    {
      // if u(x,t=0)=0, this time step size will tend to infinity
      // Note that in the ALE case there is no possibility to know the grid velocity at this point
      // and to use it for the calculation of the time step size.
      double time_step_adap = operator_base->calculate_time_step_cfl(get_velocity());
      time_step_adap *= cfl;

      // use adaptive time step size only if it is smaller, otherwise use temporary time step size
      time_step = std::min(time_step_adap, time_step_global);

      // make sure that the maximum allowable time step size is not exceeded
      time_step = std::min(time_step, param.time_step_size_max);

      print_parameter(this->pcout, "Time step size (adaptive)", time_step);
    }
    else
    {
      time_step =
        adjust_time_step_to_hit_end_time(param.start_time, param.end_time, time_step_global);

      this->pcout << std::endl
                  << "Adjust time step size to hit end time:" << std::endl
                  << std::endl;
      print_parameter(this->pcout, "Time step size", time_step);
    }
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::MaxEfficiency)
  {
    time_step          = operator_base->calculate_time_step_max_efficiency(this->order);
    double const c_eff = param.c_eff / std::pow(2., refine_steps_time);
    time_step *= c_eff;

    time_step = adjust_time_step_to_hit_end_time(param.start_time, param.end_time, time_step);

    this->pcout << std::endl
                << "Calculation of time step size (max efficiency):" << std::endl
                << std::endl;
    print_parameter(this->pcout, "C_eff", c_eff);
    print_parameter(this->pcout, "Time step size", time_step);
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage("Specified type of time step calculation is not implemented."));
  }

  return time_step;
}

template<int dim, typename Number>
double
TimeIntBDF<dim, Number>::recalculate_time_step_size() const
{
  AssertThrow(param.calculation_of_time_step_size == TimeStepCalculation::CFL,
              dealii::ExcMessage(
                "Adaptive time step is not implemented for this type of time step calculation."));

  VectorType u_relative = get_velocity();
  if(param.ale_formulation == true)
    u_relative -= grid_velocity;

  double new_time_step_size = operator_base->calculate_time_step_cfl(u_relative);
  new_time_step_size *= cfl;

  // make sure that time step size does not exceed maximum allowable time step size
  new_time_step_size = std::min(new_time_step_size, param.time_step_size_max);

  bool use_limiter = true;
  if(use_limiter)
  {
    double last_time_step_size = this->get_time_step_size();
    double factor              = param.adaptive_time_stepping_limiting_factor;
    limit_time_step_change(new_time_step_size, last_time_step_size, factor);
  }

  return new_time_step_size;
}

template<int dim, typename Number>
bool
TimeIntBDF<dim, Number>::print_solver_info() const
{
  return param.solver_info_data.write(this->global_timer.wall_time(),
                                      this->time - this->start_time,
                                      this->time_step_number);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::get_velocities_and_times(std::vector<VectorType const *> & velocities,
                                                  std::vector<double> &             times) const
{
  /*
   * the convective term is nonlinear, so we have to initialize the transport velocity
   * and the discrete time instants that can be used for interpolation
   *
   *   time t
   *  -------->   t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *               sol[2]    sol[1]   sol[0]
   *             times[2]  times[1]  times[0]
   */
  unsigned int current_order = this->order;
  if(this->time_step_number <= this->order && this->param.start_with_low_order == true)
  {
    current_order = this->time_step_number;
  }

  AssertThrow(current_order > 0 && current_order <= this->order,
              dealii::ExcMessage("Invalid parameter current_order"));

  velocities.resize(current_order);
  times.resize(current_order);

  for(unsigned int i = 0; i < current_order; ++i)
  {
    velocities.at(i) = &get_velocity(i);
    times.at(i)      = this->get_previous_time(i);
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::get_velocities_and_times_np(std::vector<VectorType const *> & velocities,
                                                     std::vector<double> &             times) const
{
  /*
   * the convective term is nonlinear, so we have to initialize the transport velocity
   * and the discrete time instants that can be used for interpolation
   *
   *   time t
   *  -------->     t_{n-2}   t_{n-1}   t_{n}     t_{n+1}
   *  _______________|_________|________|___________|___________\
   *                 |         |        |           |           /
   *               sol[3]   sol[2]    sol[1]     sol[0]
   *              times[3] times[2]  times[1]   times[0]
   */
  unsigned int current_order = this->order;
  if(this->time_step_number <= this->order && this->param.start_with_low_order == true)
  {
    current_order = this->time_step_number;
  }

  AssertThrow(current_order > 0 && current_order <= this->order,
              dealii::ExcMessage("Invalid parameter current_order"));

  velocities.resize(current_order + 1);
  times.resize(current_order + 1);

  velocities.at(0) = &get_velocity_np();
  times.at(0)      = this->get_next_time();
  for(unsigned int i = 0; i < current_order; ++i)
  {
    velocities.at(i + 1) = &get_velocity(i);
    times.at(i + 1)      = this->get_previous_time(i);
  }
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::postprocessing() const
{
  dealii::Timer timer;
  timer.restart();

  // To allow a computation of errors at start_time (= if time step number is 1 and if the
  // simulation is not a restarted one), the mesh has to be at the correct position
  if(this->param.ale_formulation && this->get_time_step_number() == 1 &&
     !this->param.restarted_simulation)
  {
    operator_base->move_grid_and_update_dependent_data_structures(this->get_time());
  }

  // We need to distribute the dofs before computing the error since
  // dealii::VectorTools::integrate_difference() does not take constraints into account
  // like MatrixFree does, hence reading the wrong values. distribute_constraint_u()
  // updates the constrained values for the velocity.
  operator_base->distribute_constraint_u(const_cast<VectorType &>(get_velocity(0)));

  bool const standard = true;
  if(standard)
  {
    postprocessor->do_postprocessing(get_velocity(0),
                                     get_pressure(0),
                                     this->get_time(),
                                     this->get_time_step_number());
  }
  else // consider velocity and pressure errors instead
  {
    VectorType velocity_error;
    operator_base->initialize_vector_velocity(velocity_error);

    VectorType pressure_error;
    operator_base->initialize_vector_pressure(pressure_error);

    operator_base->prescribe_initial_conditions(velocity_error, pressure_error, this->get_time());

    velocity_error.add(-1.0, get_velocity(0));
    pressure_error.add(-1.0, get_pressure(0));

    postprocessor->do_postprocessing(velocity_error, // error!
                                     pressure_error, // error!
                                     this->get_time(),
                                     this->get_time_step_number());
  }

  this->timer_tree->insert({"Timeloop", "Postprocessing"}, timer.wall_time());
}

// instantiations

template class TimeIntBDF<2, float>;
template class TimeIntBDF<2, double>;

template class TimeIntBDF<3, float>;
template class TimeIntBDF<3, double>;

} // namespace IncNS
} // namespace ExaDG
