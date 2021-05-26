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

#include <exadg/grid/moving_mesh_base.h>
#include <exadg/incompressible_navier_stokes/postprocessor/postprocessor_interface.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/spatial_operator_base.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf.h>
#include <exadg/incompressible_navier_stokes/user_interface/input_parameters.h>
#include <exadg/time_integration/push_back_vectors.h>
#include <exadg/time_integration/time_step_calculation.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
TimeIntBDF<dim, Number>::TimeIntBDF(
  std::shared_ptr<OperatorBase>                   operator_in,
  InputParameters const &                         param_in,
  unsigned int const                              refine_steps_time_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
  std::shared_ptr<MovingMeshBase<dim, Number>>    moving_mesh_in,
  std::shared_ptr<MatrixFree<dim, Number>>        matrix_free_in)
  : TimeIntBDFBase<Number>(param_in.start_time,
                           param_in.end_time,
                           param_in.max_number_of_time_steps,
                           param_in.order_time_integrator,
                           param_in.start_with_low_order,
                           param_in.adaptive_time_stepping,
                           param_in.restart_data,
                           mpi_comm_in,
                           is_test_in),
    param(param_in),
    refine_steps_time(refine_steps_time_in),
    cfl(param.cfl / std::pow(2.0, refine_steps_time_in)),
    cfl_oif(param_in.cfl_oif / std::pow(2.0, refine_steps_time_in)),
    operator_base(operator_in),
    vec_convective_term(this->order),
    use_extrapolation(true),
    store_solution(false),
    postprocessor(postprocessor_in),
    vec_grid_coordinates(param_in.order_time_integrator),
    moving_mesh(moving_mesh_in),
    matrix_free(matrix_free_in)
{
  if(param.ale_formulation)
  {
    AssertThrow(moving_mesh != nullptr,
                ExcMessage("Shared pointer moving_mesh is not correctly initialized."));
    AssertThrow(matrix_free != nullptr,
                ExcMessage("Shared pointer matrix_free_data is not correctly initialized."));
  }
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

    moving_mesh->update(this->get_time(), false);
    moving_mesh->fill_grid_coordinates_vector(vec_grid_coordinates[0],
                                              operator_base->get_dof_handler_u());

    if(this->start_with_low_order == false)
    {
      // compute grid coordinates at previous times (start with 1!)
      for(unsigned int i = 1; i < this->order; ++i)
      {
        moving_mesh->update(this->get_previous_time(i), false);
        moving_mesh->fill_grid_coordinates_vector(vec_grid_coordinates[i],
                                                  operator_base->get_dof_handler_u());
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
  moving_mesh->fill_grid_coordinates_vector(grid_coordinates_np,
                                            operator_base->get_dof_handler_u());

  // and update grid velocity using BDF time derivative
  compute_bdf_time_derivative(grid_velocity,
                              grid_coordinates_np,
                              vec_grid_coordinates,
                              this->bdf,
                              this->get_time_step_size());

  // and hand grid velocity over to spatial discretization
  operator_base->set_grid_velocity(grid_velocity);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::advance_one_timestep_partitioned_solve(bool const use_extrapolation,
                                                                bool const store_solution)
{
  if(this->use_extrapolation == false)
    AssertThrow(this->store_solution == true, ExcMessage("Invalid parameters."));

  this->use_extrapolation = use_extrapolation;
  this->store_solution    = store_solution;

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
TimeIntBDF<dim, Number>::initialize_oif()
{
  // Operator-integration-factor splitting
  if(param.equation_type == EquationType::NavierStokes &&
     param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    convective_operator_OIF.reset(new OperatorOIF<dim, Number>(operator_base));

    // initialize OIF time integrator
    if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK1Stage1)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<OperatorOIF<dim, Number>, VectorType>(
          1, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK2Stage2)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<OperatorOIF<dim, Number>, VectorType>(
          2, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK3Stage3)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<OperatorOIF<dim, Number>, VectorType>(
          3, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage4)
    {
      time_integrator_OIF.reset(
        new ExplicitRungeKuttaTimeIntegrator<OperatorOIF<dim, Number>, VectorType>(
          4, convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK3Stage4Reg2C)
    {
      time_integrator_OIF.reset(new LowStorageRK3Stage4Reg2C<OperatorOIF<dim, Number>, VectorType>(
        convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage5Reg2C)
    {
      time_integrator_OIF.reset(new LowStorageRK4Stage5Reg2C<OperatorOIF<dim, Number>, VectorType>(
        convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage5Reg3C)
    {
      time_integrator_OIF.reset(new LowStorageRK4Stage5Reg3C<OperatorOIF<dim, Number>, VectorType>(
        convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK5Stage9Reg2S)
    {
      time_integrator_OIF.reset(new LowStorageRK5Stage9Reg2S<OperatorOIF<dim, Number>, VectorType>(
        convective_operator_OIF));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK3Stage7Reg2)
    {
      time_integrator_OIF.reset(
        new LowStorageRKTD<OperatorOIF<dim, Number>, VectorType>(convective_operator_OIF, 3, 7));
    }
    else if(param.time_integrator_oif == IncNS::TimeIntegratorOIF::ExplRK4Stage8Reg2)
    {
      time_integrator_OIF.reset(
        new LowStorageRKTD<OperatorOIF<dim, Number>, VectorType>(convective_operator_OIF, 4, 8));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    AssertThrow(time_integrator_OIF.get() != 0,
                ExcMessage("OIF time integrator has not been initialized correctly."));
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

  unsigned int const degree_u = operator_base->get_polynomial_degree();

  if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
  {
    time_step = calculate_const_time_step(param.time_step_size, refine_steps_time);

    this->pcout << std::endl << "User specified time step size:" << std::endl << std::endl;
    print_parameter(this->pcout, "time step size", time_step);
  }
  else if(param.calculation_of_time_step_size == TimeStepCalculation::CFL)
  {
    double const h_min = operator_base->calculate_minimum_element_length();

    double time_step_global = calculate_time_step_cfl_global(
      cfl, param.max_velocity, h_min, degree_u, param.cfl_exponent_fe_degree_velocity);

    this->pcout << std::endl
                << "Calculation of time step size according to CFL condition:" << std::endl
                << std::endl;
    print_parameter(this->pcout, "h_min", h_min);
    print_parameter(this->pcout, "U_max", param.max_velocity);
    print_parameter(this->pcout, "CFL", cfl);
    print_parameter(this->pcout, "exponent fe_degree", param.cfl_exponent_fe_degree_velocity);
    print_parameter(this->pcout, "Time step size (global)", time_step_global);

    if(this->adaptive_time_stepping == true)
    {
      // if u(x,t=0)=0, this time step size will tend to infinity
      // Note that in the ALE case there is no possibility to know the grid velocity at this point
      // and to use it for the calculation of the time step size.
      double time_step_adap =
        operator_base->calculate_time_step_cfl(get_velocity(),
                                               cfl,
                                               param.cfl_exponent_fe_degree_velocity);

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
    double const h_min = operator_base->calculate_minimum_element_length();

    time_step = calculate_time_step_max_efficiency(
      param.c_eff, h_min, degree_u, this->order, refine_steps_time);

    time_step = adjust_time_step_to_hit_end_time(param.start_time, param.end_time, time_step);

    this->pcout << std::endl
                << "Calculation of time step size (max efficiency):" << std::endl
                << std::endl;
    print_parameter(this->pcout, "C_eff", param.c_eff / std::pow(2.0, refine_steps_time));
    print_parameter(this->pcout, "Time step size", time_step);
  }
  else
  {
    AssertThrow(false, ExcMessage("Specified type of time step calculation is not implemented."));
  }

  if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    // make sure that CFL condition is used for the calculation of the time step size (the aim
    // of the OIF splitting approach is to overcome limitations of the CFL condition)
    AssertThrow(param.calculation_of_time_step_size == TimeStepCalculation::CFL,
                ExcMessage(
                  "Specified type of time step calculation is not compatible with OIF approach!"));

    this->pcout << std::endl << "OIF sub-stepping for convective term:" << std::endl << std::endl;
    print_parameter(this->pcout, "CFL (OIF)", cfl_oif);
  }

  return time_step;
}

template<int dim, typename Number>
double
TimeIntBDF<dim, Number>::recalculate_time_step_size() const
{
  AssertThrow(param.calculation_of_time_step_size == TimeStepCalculation::CFL,
              ExcMessage(
                "Adaptive time step is not implemented for this type of time step calculation."));

  VectorType u_relative = get_velocity();
  if(param.ale_formulation == true)
    u_relative -= grid_velocity;

  double new_time_step_size =
    operator_base->calculate_time_step_cfl(u_relative, cfl, param.cfl_exponent_fe_degree_velocity);

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
TimeIntBDF<dim, Number>::initialize_solution_oif_substepping(VectorType & solution_tilde_m,
                                                             unsigned int i)
{
  // initialize solution: u_tilde(s=0) = u(t_{n-i})
  solution_tilde_m = get_velocity(i);
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
              ExcMessage("Invalid parameter current_order"));

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
              ExcMessage("Invalid parameter current_order"));

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
TimeIntBDF<dim, Number>::calculate_sum_alphai_ui_oif_substepping(VectorType & sum_alphai_ui,
                                                                 double const cfl,
                                                                 double const cfl_oif)
{
  std::vector<VectorType const *> velocities;
  std::vector<double>             times;

  this->get_velocities_and_times(velocities, times);

  // this is only needed for transport with interpolated/extrapolated velocity
  // as opposed to the standard nonlinear transport
  this->convective_operator_OIF->set_solutions_and_times(velocities, times);

  // call function implemented in base class for the actual OIF sub-stepping
  TimeIntBDFBase<Number>::calculate_sum_alphai_ui_oif_substepping(sum_alphai_ui, cfl, cfl_oif);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::move_mesh(double const time) const
{
  moving_mesh->update(time, false);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::move_mesh_and_update_dependent_data_structures(double const time) const
{
  moving_mesh->update(time, false);
  matrix_free->update_mapping(*moving_mesh);
  operator_base->update_after_mesh_movement();
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::update_sum_alphai_ui_oif_substepping(VectorType &       sum_alphai_ui,
                                                              VectorType const & u_tilde_i,
                                                              unsigned int       i)
{
  // calculate sum (alpha_i/dt * u_tilde_i)
  if(i == 0)
    sum_alphai_ui.equ(this->bdf.get_alpha(i) / this->get_time_step_size(), u_tilde_i);
  else // i>0
    sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(), u_tilde_i);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::do_timestep_oif_substepping(VectorType & solution_tilde_mp,
                                                     VectorType & solution_tilde_m,
                                                     double const start_time,
                                                     double const time_step_size)
{
  // solve sub-step
  time_integrator_OIF->solve_timestep(solution_tilde_mp,
                                      solution_tilde_m,
                                      start_time,
                                      time_step_size);
}

template<int dim, typename Number>
void
TimeIntBDF<dim, Number>::postprocessing() const
{
  Timer timer;
  timer.restart();

  // the mesh has to be at the correct position to allow a computation of
  // errors at start_time
  if(this->param.ale_formulation && this->get_time_step_number() == 1)
  {
    move_mesh_and_update_dependent_data_structures(this->get_time());
  }

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
