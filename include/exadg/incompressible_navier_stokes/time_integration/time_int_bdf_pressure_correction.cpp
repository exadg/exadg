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

#include <exadg/incompressible_navier_stokes/spatial_discretization/operator_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h>
#include <exadg/incompressible_navier_stokes/user_interface/input_parameters.h>
#include <exadg/time_integration/push_back_vectors.h>
#include <exadg/time_integration/time_step_calculation.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
TimeIntBDFPressureCorrection<dim, Number>::TimeIntBDFPressureCorrection(
  std::shared_ptr<Operator>                       operator_in,
  InputParameters const &                         param_in,
  unsigned int const                              refine_steps_time_in,
  MPI_Comm const &                                mpi_comm_in,
  bool const                                      is_test_in,
  std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in)
  : Base(operator_in, param_in, refine_steps_time_in, mpi_comm_in, is_test_in, postprocessor_in),
    pde_operator(operator_in),
    velocity(param_in.order_time_integrator),
    pressure(param_in.order_time_integrator),
    order_pressure_extrapolation(param_in.order_pressure_extrapolation),
    extra_pressure_gradient(param_in.order_pressure_extrapolation, param_in.start_with_low_order),
    pressure_dbc(param_in.order_pressure_extrapolation),
    iterations_momentum({0, {0, 0}}),
    iterations_pressure({0, 0}),
    iterations_projection({0, 0})
{
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  Base::update_time_integrator_constants();

  // update time integrator constants for extrapolation scheme of pressure gradient term

  // if start_with_low_order == true (no analytical solution available)
  // the pressure is unknown at t = t_0:
  // -> use no extrapolation (order=0, non-incremental) in first time step (the pressure solution is
  // calculated in the second sub step)
  // -> use first order extrapolation in second time step, second order extrapolation in third time
  // step, etc.
  if(this->adaptive_time_stepping == false)
  {
    // the "-1" indicates that the order of extrapolation of the pressure gradient is one order
    // lower than the order of the BDF time integration scheme
    extra_pressure_gradient.update(this->get_time_step_number() - 1);
  }
  else // adaptive time stepping
  {
    // the "-1" indicates that the order of extrapolation of the pressure gradient is one order
    // lower than the order of the BDF time integration scheme
    extra_pressure_gradient.update(this->get_time_step_number() - 1, this->get_time_step_vector());
  }

  // use this function to check the correctness of the time integrator constants
  //    std::cout << "Coefficients extrapolation scheme pressure: Time step = "
  //              << this->get_time_step_number() << std::endl;
  //    extra_pressure_gradient.print(this->pcout);
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::setup_derived()
{
  Base::setup_derived();

  // pressure_dbc does not have to be initialized in case of a restart, where
  // the vectors are read from memory.
  if(this->param.store_previous_boundary_values && this->param.restarted_simulation == false)
  {
    initialize_pressure_on_boundary();
  }
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::read_restart_vectors(
  boost::archive::binary_iarchive & ia)
{
  Base::read_restart_vectors(ia);

  if(this->param.store_previous_boundary_values)
  {
    for(unsigned int i = 0; i < pressure_dbc.size(); i++)
    {
      ia >> pressure_dbc[i];
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::write_restart_vectors(
  boost::archive::binary_oarchive & oa) const
{
  Base::write_restart_vectors(oa);

  if(this->param.store_previous_boundary_values)
  {
    for(unsigned int i = 0; i < pressure_dbc.size(); i++)
    {
      oa << pressure_dbc[i];
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::allocate_vectors()
{
  Base::allocate_vectors();

  // velocity
  for(unsigned int i = 0; i < velocity.size(); ++i)
    pde_operator->initialize_vector_velocity(velocity[i]);
  pde_operator->initialize_vector_velocity(velocity_np);

  // pressure
  for(unsigned int i = 0; i < pressure.size(); ++i)
    pde_operator->initialize_vector_pressure(pressure[i]);
  pde_operator->initialize_vector_pressure(pressure_np);

  if(this->param.store_previous_boundary_values)
  {
    for(unsigned int i = 0; i < pressure_dbc.size(); ++i)
      pde_operator->initialize_vector_pressure(pressure_dbc[i]);
  }
}


template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::initialize_current_solution()
{
  if(this->param.ale_formulation)
    pde_operator->move_grid(this->get_time());

  pde_operator->prescribe_initial_conditions(velocity[0], pressure[0], this->get_time());
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::initialize_former_solutions()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    if(this->param.ale_formulation)
      pde_operator->move_grid(this->get_previous_time(i));

    pde_operator->prescribe_initial_conditions(velocity[i],
                                               pressure[i],
                                               this->get_previous_time(i));
  }
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::initialize_pressure_on_boundary()
{
  // If start_with_low_order == true, no pressure extrapolation is used in the first time step
  // even for the incremental pressure-correction scheme. Hence, there is no need to initialize
  // the pressure_dbc vector in this case.

  if(this->param.start_with_low_order == false)
  {
    for(unsigned int i = 0; i < pressure_dbc.size(); ++i)
    {
      double const time = this->get_time() - double(i) * this->get_time_step_size();
      if(this->param.ale_formulation)
        pde_operator->move_grid_and_update_dependent_data_structures(time);

      pde_operator->interpolate_pressure_dirichlet_bc(pressure_dbc[i], time);
    }
  }
}

template<int dim, typename Number>
typename TimeIntBDFPressureCorrection<dim, Number>::VectorType const &
TimeIntBDFPressureCorrection<dim, Number>::get_velocity() const
{
  return velocity[0];
}

template<int dim, typename Number>
typename TimeIntBDFPressureCorrection<dim, Number>::VectorType const &
TimeIntBDFPressureCorrection<dim, Number>::get_velocity_np() const
{
  return velocity_np;
}

template<int dim, typename Number>
typename TimeIntBDFPressureCorrection<dim, Number>::VectorType const &
TimeIntBDFPressureCorrection<dim, Number>::get_pressure_np() const
{
  return pressure_np;
}

template<int dim, typename Number>
typename TimeIntBDFPressureCorrection<dim, Number>::VectorType const &
TimeIntBDFPressureCorrection<dim, Number>::get_velocity(unsigned int i) const
{
  return velocity[i];
}

template<int dim, typename Number>
typename TimeIntBDFPressureCorrection<dim, Number>::VectorType const &
TimeIntBDFPressureCorrection<dim, Number>::get_pressure(unsigned int i) const
{
  return pressure[i];
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::set_velocity(VectorType const & velocity_in,
                                                        unsigned int const i)
{
  velocity[i] = velocity_in;
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::set_pressure(VectorType const & pressure_in,
                                                        unsigned int const i)
{
  pressure[i] = pressure_in;
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::postprocessing_stability_analysis()
{
  AssertThrow(this->order == 1,
              ExcMessage("Order of BDF scheme has to be 1 for this stability analysis."));

  AssertThrow(this->param.convective_problem() == false,
              ExcMessage(
                "Stability analysis can not be performed for nonlinear convective problems."));

  AssertThrow(velocity[0].l2_norm() < 1.e-15 && pressure[0].l2_norm() < 1.e-15,
              ExcMessage("Solution vector has to be zero for this stability analysis."));

  AssertThrow(Utilities::MPI::n_mpi_processes(this->mpi_comm) == 1,
              ExcMessage("Number of MPI processes has to be 1."));

  std::cout << std::endl << "Analysis of eigenvalue spectrum:" << std::endl;

  unsigned int const size = velocity[0].locally_owned_size();

  LAPACKFullMatrix<Number> propagation_matrix(size, size);

  // loop over all columns of propagation matrix
  for(unsigned int j = 0; j < size; ++j)
  {
    // set j-th element to 1
    velocity[0].local_element(j) = 1.0;

    // solve time step
    this->solve_timestep();

    // dst-vector velocity_np is j-th column of propagation matrix
    for(unsigned int i = 0; i < size; ++i)
    {
      propagation_matrix(i, j) = velocity_np.local_element(i);
    }

    // reset j-th element to 0
    velocity[0].local_element(j) = 0.0;
  }

  // compute eigenvalues
  propagation_matrix.compute_eigenvalues();

  double norm_max = 0.0;

  std::cout << "List of all eigenvalues:" << std::endl;

  for(unsigned int i = 0; i < size; ++i)
  {
    double norm = std::abs(propagation_matrix.eigenvalue(i));
    if(norm > norm_max)
      norm_max = norm;

    // print eigenvalues
    std::cout << std::scientific << std::setprecision(5) << propagation_matrix.eigenvalue(i)
              << std::endl;
  }

  std::cout << std::endl << std::endl << "Maximum eigenvalue = " << norm_max << std::endl;
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::solve_timestep()
{
  // perform the sub-steps of the pressure-correction scheme

  momentum_step();

  VectorType pressure_increment;
  pressure_increment.reinit(pressure_np, false /* init with zero */);

  pressure_step(pressure_increment);

  projection_step(pressure_increment);

  // evaluate convective term once the final solution at time
  // t_{n+1} is known
  evaluate_convective_term();
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::momentum_step()
{
  Timer timer;
  timer.restart();

  // Extrapolate old solutionsto get a good initial estimate for the solver.
  if(this->use_extrapolation)
  {
    velocity_np = 0.0;
    for(unsigned int i = 0; i < velocity.size(); ++i)
    {
      velocity_np.add(this->extra.get_beta(i), velocity[i]);
    }
  }
  else
  {
    velocity_np = velocity_momentum_last_iter;
  }

  /*
   *  if a turbulence model is used:
   *  update turbulence model before calculating rhs_momentum
   */
  if(this->param.use_turbulence_model == true)
  {
    Timer timer_turbulence;
    timer_turbulence.restart();

    pde_operator->update_turbulence_model(velocity_np);

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Update of turbulent viscosity:";
      print_wall_time(this->pcout, timer_turbulence.wall_time());
    }
  }


  /*
   *  Calculate the right-hand side of the linear system of equations
   *  or the vector that is constant when solving the nonlinear momentum equation
   *  (where constant means that the vector does not change from one Newton iteration
   *  to the next, i.e., it does not depend on the current solution of the nonlinear solver)
   */
  VectorType rhs(velocity_np);
  rhs_momentum(rhs);

  /*
   *  Solve the linear or nonlinear problem.
   */

  bool const update_preconditioner =
    this->param.update_preconditioner_momentum &&
    ((this->time_step_number - 1) % this->param.update_preconditioner_momentum_every_time_steps ==
     0);

  if(this->param.linear_problem_has_to_be_solved())
  {
    if(this->param.viscous_problem())
    {
      // solve linear system of equations
      unsigned int n_iter = pde_operator->solve_linear_momentum_equation(
        velocity_np, rhs, update_preconditioner, this->get_scaling_factor_time_derivative_term());

      iterations_momentum.first += 1;
      std::get<1>(iterations_momentum.second) += n_iter;

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Solve momentum step:";
        print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
      }
    }
    else // Euler equations
    {
      pde_operator->apply_inverse_mass_operator(velocity_np, rhs);
      velocity_np *= this->get_time_step_size() / this->bdf.get_gamma0();

      if(this->print_solver_info() and not(this->is_test))
      {
        this->pcout << std::endl << "Explicit momentum step:";
        print_wall_time(this->pcout, timer.wall_time());
      }
    }
  }
  else // nonlinear problem
  {
    AssertThrow(this->param.nonlinear_problem_has_to_be_solved(), ExcMessage("Logical error."));

    // solve non-linear system of equations
    auto const iter = pde_operator->solve_nonlinear_momentum_equation(
      velocity_np,
      rhs,
      this->get_next_time(),
      update_preconditioner,
      this->get_scaling_factor_time_derivative_term());

    iterations_momentum.first += 1;
    std::get<0>(iterations_momentum.second) += std::get<0>(iter);
    std::get<1>(iterations_momentum.second) += std::get<1>(iter);

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve momentum step:";
      print_solver_info_nonlinear(this->pcout,
                                  std::get<0>(iter),
                                  std::get<1>(iter),
                                  timer.wall_time());
    }
  }

  if(this->store_solution)
    velocity_momentum_last_iter = velocity_np;

  this->timer_tree->insert({"Timeloop", "Momentum step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::rhs_momentum(VectorType & rhs)
{
  rhs = 0.0;

  /*
   *  Add extrapolation of pressure gradient term to the rhs in case of incremental formulation
   */
  if(extra_pressure_gradient.get_order() > 0)
  {
    for(unsigned int i = 0; i < extra_pressure_gradient.get_order(); ++i)
    {
      VectorType temp(velocity[0]);
      temp = 0.0;

      if(this->param.store_previous_boundary_values)
      {
        pde_operator->evaluate_pressure_gradient_term_dirichlet_bc_from_dof_vector(temp,
                                                                                   pressure[i],
                                                                                   pressure_dbc[i]);
      }
      else
      {
        pde_operator->evaluate_pressure_gradient_term(temp,
                                                      pressure[i],
                                                      this->get_previous_time(i));
      }

      rhs.add(-extra_pressure_gradient.get_beta(i), temp);
    }
  }

  /*
   *  Body force term
   */
  if(this->param.right_hand_side == true)
  {
    pde_operator->evaluate_add_body_force_term(rhs, this->get_next_time());
  }

  /*
   *  Convective term formulated explicitly (additive decomposition):
   *  Evaluate convective term and add extrapolation of convective term to the rhs
   */
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation)
    {
      for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
      {
        // in a general setting, we only know the boundary conditions at time t_{n+1}
        pde_operator->evaluate_convective_term(this->vec_convective_term[i],
                                               velocity[i],
                                               this->get_next_time());
      }
    }

    for(unsigned int i = 0; i < this->vec_convective_term.size(); ++i)
      rhs.add(-this->extra.get_beta(i), this->vec_convective_term[i]);
  }

  /*
   *  calculate sum (alpha_i/dt * u_i): This term is relevant for both the explicit
   *  and the implicit formulation of the convective term
   */
  VectorType sum_alphai_ui(velocity[0]);

  // calculate sum (alpha_i/dt * u_tilde_i) in case of explicit treatment of convective term
  // and operator-integration-factor (OIF) splitting
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    this->calculate_sum_alphai_ui_oif_substepping(sum_alphai_ui, this->cfl, this->cfl_oif);
  }
  // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
  else
  {
    sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->get_time_step_size(), velocity[0]);
    for(unsigned int i = 1; i < velocity.size(); ++i)
    {
      sum_alphai_ui.add(this->bdf.get_alpha(i) / this->get_time_step_size(), velocity[i]);
    }
  }

  pde_operator->apply_mass_operator_add(rhs, sum_alphai_ui);

  /*
   *  Right-hand side viscous term:
   *  If a linear system of equations has to be solved,
   *  inhomogeneous parts of boundary face integrals of the viscous operator
   *  have to be shifted to the right-hand side of the equation.
   */
  if(this->param.viscous_problem() && this->param.linear_problem_has_to_be_solved())
  {
    pde_operator->rhs_add_viscous_term(rhs, this->get_next_time());
  }
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::pressure_step(VectorType & pressure_increment)
{
  Timer timer;
  timer.restart();

  // compute right-hand side vector
  VectorType rhs(pressure_np);
  rhs_pressure(rhs);

  // calculate initial guess for pressure solve
  if(this->use_extrapolation)
  {
    // extrapolate old solution to get a good initial estimate for the
    // pressure solution p_{n+1} at time t^{n+1}
    for(unsigned int i = 0; i < pressure.size(); ++i)
    {
      pressure_increment.add(this->extra.get_beta(i), pressure[i]);
    }

    // incremental formulation
    if(extra_pressure_gradient.get_order() > 0)
    {
      // Subtract extrapolation of pressure since the PPE is solved for the
      // pressure increment phi = p_{n+1} - sum_i (beta_pressure_extra_i * pressure_i),
      // where p_{n+1} is approximated by an extrapolation of order J (=order of BDF scheme).
      // Note that the divergence correction term in case of the rotational formulation is not
      // considered when calculating a good initial guess for the solution of the PPE,
      // which will slightly increase the number of iterations compared to the standard
      // formulation of the pressure-correction scheme.
      for(unsigned int i = 0; i < extra_pressure_gradient.get_order(); ++i)
      {
        pressure_increment.add(-extra_pressure_gradient.get_beta(i), pressure[i]);
      }
    }
  }
  else
  {
    pressure_increment = pressure_increment_last_iter;
  }

  // solve linear system of equations
  bool const update_preconditioner =
    this->param.update_preconditioner_pressure_poisson &&
    ((this->time_step_number - 1) %
       this->param.update_preconditioner_pressure_poisson_every_time_steps ==
     0);

  unsigned int const n_iter =
    pde_operator->solve_pressure(pressure_increment, rhs, update_preconditioner);

  iterations_pressure.first += 1;
  iterations_pressure.second += n_iter;

  if(this->store_solution)
    pressure_increment_last_iter = pressure_increment;

  // calculate pressure p^{n+1} from pressure increment
  pressure_update(pressure_increment);

  // Special case: pressure level is undefined
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  pde_operator->adjust_pressure_level_if_undefined(pressure_np, this->get_next_time());

  // write output
  if(this->print_solver_info() and not(this->is_test))
  {
    this->pcout << std::endl << "Solve pressure step:";
    print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
  }

  this->timer_tree->insert({"Timeloop", "Pressure step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::calculate_chi(double & chi) const
{
  if(this->param.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
  {
    chi = 1.0;
  }
  else if(this->param.formulation_viscous_term == FormulationViscousTerm::DivergenceFormulation)
  {
    chi = 2.0;
  }
  else
  {
    AssertThrow(this->param.formulation_viscous_term ==
                    FormulationViscousTerm::LaplaceFormulation &&
                  this->param.formulation_viscous_term ==
                    FormulationViscousTerm::DivergenceFormulation,
                ExcMessage("Not implemented!"));
  }
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::rhs_pressure(VectorType & rhs) const
{
  /*
   *  I. calculate divergence term
   */
  VectorType temp(pressure_np);
  pde_operator->evaluate_velocity_divergence_term(temp, velocity_np, this->get_next_time());

  rhs.equ(-this->bdf.get_gamma0() / this->get_time_step_size(), temp);


  /*
   *  II. calculate terms originating from inhomogeneous parts of boundary face integrals,
   *  i.e., pressure Dirichlet boundary conditions on Gamma_N and
   *  pressure Neumann boundary conditions on Gamma_D (always h=0 for pressure-correction scheme!)
   */
  pde_operator->rhs_ppe_laplace_add(rhs, this->get_next_time());

  // incremental formulation of pressure-correction scheme
  for(unsigned int i = 0; i < extra_pressure_gradient.get_order(); ++i)
  {
    // set temp to zero since rhs_ppe_laplace_add() adds into the vector
    temp = 0.0;
    if(this->param.store_previous_boundary_values)
    {
      pde_operator->rhs_ppe_laplace_add_dirichlet_bc_from_dof_vector(temp, pressure_dbc[i]);
    }
    else
    {
      pde_operator->rhs_ppe_laplace_add(temp, this->get_previous_time(i));
    }

    rhs.add(-extra_pressure_gradient.get_beta(i), temp);
  }

  // special case: pressure level is undefined
  // Unclear if this is really necessary, because from a theoretical
  // point of view one would expect that the mean value of the rhs of the
  // presssure Poisson equation is zero if consistent Dirichlet boundary
  // conditions are prescribed.
  // In principle, it works (since the linear system of equations is consistent)
  // but we detected no convergence for some test cases and specific parameters.
  // Hence, for reasons of robustness we also solve a transformed linear system of equations
  // in case of the pressure-correction scheme.

  if(pde_operator->is_pressure_level_undefined())
    set_zero_mean_value(rhs);
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::pressure_update(VectorType const & pressure_increment)
{
  // First set pressure solution to zero.
  pressure_np = 0.0;

  // Rotational formulation only (this step is performed first in order
  // to avoid the storage of another temporary variable).
  if(this->param.rotational_formulation == true)
  {
    // Automatically sets pressure_np to zero before operator evaluation.
    pde_operator->evaluate_velocity_divergence_term(pressure_np,
                                                    velocity_np,
                                                    this->get_next_time());

    pde_operator->apply_inverse_pressure_mass_operator(pressure_np, pressure_np);

    double chi = 0.0;
    calculate_chi(chi);

    pressure_np *= -chi * this->param.viscosity;
  }

  // This is done for both the incremental and the non-incremental formulation,
  // the standard and the rotational formulation.
  pressure_np.add(1.0, pressure_increment);

  // Incremental formulation only.

  // add extrapolation of pressure to the pressure-increment solution in order to obtain
  // the pressure solution at the end of the time step, i.e.,
  // p^{n+1} = (pressure_increment)^{n+1} + sum_i (beta_pressure_extrapolation_i * p^{n-i});
  for(unsigned int i = 0; i < extra_pressure_gradient.get_order(); ++i)
  {
    pressure_np.add(extra_pressure_gradient.get_beta(i), pressure[i]);
  }
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::rhs_projection(
  VectorType &       rhs,
  VectorType const & pressure_increment) const
{
  /*
   *  I. apply mass operator
   */
  pde_operator->apply_mass_operator(rhs, velocity_np);

  /*
   *  II. calculate pressure gradient term including boundary condition g_p(t_{n+1})
   */
  VectorType temp(rhs);
  pde_operator->evaluate_pressure_gradient_term(temp, pressure_increment, this->get_next_time());

  rhs.add(-this->get_time_step_size() / this->bdf.get_gamma0(), temp);

  /*
   *  III. pressure gradient term: boundary conditions g_p(t_{n-i})
   *       in case of incremental formulation of pressure-correction scheme
   */
  if(this->param.gradp_integrated_by_parts == true && this->param.gradp_use_boundary_data == true)
  {
    for(unsigned int i = 0; i < extra_pressure_gradient.get_order(); ++i)
    {
      // evaluate inhomogeneous parts of boundary face integrals
      // note that the function rhs_...() already includes a factor of -1.0
      if(this->param.store_previous_boundary_values)
      {
        pde_operator->rhs_pressure_gradient_term_dirichlet_bc_from_dof_vector(temp,
                                                                              pressure_dbc[i]);
      }
      else
      {
        pde_operator->rhs_pressure_gradient_term(temp, this->get_previous_time(i));
      }

      rhs.add(-extra_pressure_gradient.get_beta(i) * this->get_time_step_size() /
                this->bdf.get_gamma0(),
              temp);
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::projection_step(VectorType const & pressure_increment)
{
  Timer timer;
  timer.restart();

  // compute right-hand-side vector
  VectorType rhs(velocity_np);
  rhs_projection(rhs, pressure_increment);

  // apply inverse mass operator: this is the solution if no penalty terms are applied
  // and serves as a good initial guess for the case with penalty terms
  pde_operator->apply_inverse_mass_operator(velocity_np, rhs);

  if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
  {
    // extrapolate velocity to time t_{n+1} and use this velocity field to
    // calculate the penalty parameter for the divergence and continuity penalty terms
    VectorType velocity_extrapolated;
    if(this->use_extrapolation)
    {
      velocity_extrapolated.reinit(velocity[0]);
      for(unsigned int i = 0; i < velocity.size(); ++i)
        velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);
    }
    else
    {
      velocity_extrapolated = velocity_projection_last_iter;
    }

    pde_operator->update_projection_operator(velocity_extrapolated, this->get_time_step_size());

    // add inhomogeneous contributions of continuity penalty term after computing
    // the initial guess for the linear system of equations to make sure that the initial
    // guess is as accurate as possible
    if(this->param.use_continuity_penalty && this->param.continuity_penalty_use_boundary_data)
      pde_operator->rhs_add_projection_operator(rhs, this->get_next_time());

    // solve linear system of equations
    bool const update_preconditioner =
      this->param.update_preconditioner_projection &&
      ((this->time_step_number - 1) %
         this->param.update_preconditioner_projection_every_time_steps ==
       0);

    if(this->use_extrapolation == false)
      velocity_np = velocity_projection_last_iter;

    unsigned int const n_iter =
      pde_operator->solve_projection(velocity_np, rhs, update_preconditioner);

    iterations_projection.first += 1;
    iterations_projection.second += n_iter;

    if(this->store_solution)
      velocity_projection_last_iter = velocity_np;

    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve projection step:";
      print_solver_info_linear(this->pcout, n_iter, timer.wall_time());
    }
  }
  else // no penalty terms
  {
    if(this->print_solver_info() and not(this->is_test))
    {
      this->pcout << std::endl << "Solve projection step:";
      print_wall_time(this->pcout, timer.wall_time());
    }
  }

  this->timer_tree->insert({"Timeloop", "Projection step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::evaluate_convective_term()
{
  Timer timer;
  timer.restart();

  // evaluate convective term once solution_np is known
  if(this->param.convective_problem() &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    if(this->param.ale_formulation == false) // Eulerian case
    {
      pde_operator->evaluate_convective_term(this->convective_term_np,
                                             velocity_np,
                                             this->get_next_time());
    }
  }

  this->timer_tree->insert({"Timeloop", "Momentum step"}, timer.wall_time());
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::prepare_vectors_for_next_timestep()
{
  Base::prepare_vectors_for_next_timestep();

  push_back(velocity);
  velocity[0].swap(velocity_np);

  push_back(pressure);
  pressure[0].swap(pressure_np);

  if(extra_pressure_gradient.get_order() > 0)
  {
    if(this->param.store_previous_boundary_values)
    {
      push_back(pressure_dbc);

      // no need to move the mesh here since we still have the mesh Omega_{n+1} at this point!
      pde_operator->interpolate_pressure_dirichlet_bc(pressure_dbc[0], this->get_next_time());
    }
  }
}

template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::solve_steady_problem()
{
  this->pcout << std::endl << "Starting time loop ..." << std::endl;

  // pseudo-time integration in order to solve steady-state problem
  bool converged = false;

  if(this->param.convergence_criterion_steady_problem ==
     ConvergenceCriterionSteadyProblem::SolutionIncrement)
  {
    VectorType velocity_tmp;
    VectorType pressure_tmp;

    while(!converged && this->time < (this->end_time - this->eps) &&
          this->get_time_step_number() <= this->param.max_number_of_time_steps)
    {
      // save solution from previous time step
      velocity_tmp = velocity[0];
      pressure_tmp = pressure[0];

      // calculate norm of solution
      double const norm_u = velocity_tmp.l2_norm();
      double const norm_p = pressure_tmp.l2_norm();
      double const norm   = std::sqrt(norm_u * norm_u + norm_p * norm_p);

      // solve time step
      this->do_timestep();

      // calculate increment:
      // increment = solution_{n+1} - solution_{n}
      //           = solution[0] - solution_tmp
      velocity_tmp *= -1.0;
      pressure_tmp *= -1.0;
      velocity_tmp.add(1.0, velocity[0]);
      pressure_tmp.add(1.0, pressure[0]);

      double const incr_u   = velocity_tmp.l2_norm();
      double const incr_p   = pressure_tmp.l2_norm();
      double const incr     = std::sqrt(incr_u * incr_u + incr_p * incr_p);
      double       incr_rel = 1.0;
      if(norm > 1.0e-10)
        incr_rel = incr / norm;

      // write output
      if(this->print_solver_info())
      {
        this->pcout << std::endl
                    << "Norm of solution increment:" << std::endl
                    << "  ||incr_abs|| = " << std::scientific << std::setprecision(10) << incr
                    << std::endl
                    << "  ||incr_rel|| = " << std::scientific << std::setprecision(10) << incr_rel
                    << std::endl;
      }

      // check convergence
      if(incr < this->param.abs_tol_steady || incr_rel < this->param.rel_tol_steady)
      {
        converged = true;
      }
    }
  }
  else if(this->param.convergence_criterion_steady_problem ==
          ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes)
  {
    double const initial_residual = evaluate_residual();

    while(!converged && this->time < (this->end_time - this->eps) &&
          this->get_time_step_number() <= this->param.max_number_of_time_steps)
    {
      this->do_timestep();

      // check convergence by evaluating the residual of
      // the steady-state incompressible Navier-Stokes equations
      double const residual = evaluate_residual();

      if(residual < this->param.abs_tol_steady ||
         residual / initial_residual < this->param.rel_tol_steady)
      {
        converged = true;
      }
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("not implemented."));
  }

  AssertThrow(
    converged == true,
    ExcMessage(
      "Maximum number of time steps or end time exceeded! This might be due to the fact that "
      "(i) the maximum number of time steps is simply too small to reach a steady solution, "
      "(ii) the problem is unsteady so that the applied solution approach is inappropriate, "
      "(iii) some of the solver tolerances are in conflict."));

  this->pcout << std::endl << "... done!" << std::endl;
}

template<int dim, typename Number>
double
TimeIntBDFPressureCorrection<dim, Number>::evaluate_residual()
{
  pde_operator->evaluate_nonlinear_residual_steady(
    velocity_np, pressure_np, velocity[0], pressure[0], this->get_time());

  double const norm_u = velocity_np.l2_norm();
  double const norm_p = pressure_np.l2_norm();

  double residual = std::sqrt(norm_u * norm_u + norm_p * norm_p);

  // write output
  if(this->print_solver_info())
  {
    this->pcout << std::endl
                << "Norm of residual of steady Navier-Stokes equations:" << std::endl
                << "  ||r|| = " << std::scientific << std::setprecision(10) << residual
                << std::endl;
  }

  return residual;
}


template<int dim, typename Number>
void
TimeIntBDFPressureCorrection<dim, Number>::print_iterations() const
{
  std::vector<std::string> names;
  std::vector<double>      iterations_avg;

  if(this->param.linear_problem_has_to_be_solved())
  {
    names = {"Momentum step", "Pressure step", "Projection step"};

    iterations_avg.resize(3);
    iterations_avg[0] = (double)std::get<1>(iterations_momentum.second) /
                        std::max(1., (double)iterations_momentum.first);
    iterations_avg[1] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[2] =
      (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
  }
  else // nonlinear system of equations in momentum step
  {
    names = {"Momentum (nonlinear)",
             "Momentum (linear accumulated)",
             "Momentum (linear per nonlinear)",
             "Pressure",
             "Projection"};

    iterations_avg.resize(5);
    iterations_avg[0] = (double)std::get<0>(iterations_momentum.second) /
                        std::max(1., (double)iterations_momentum.first);
    iterations_avg[1] = (double)std::get<1>(iterations_momentum.second) /
                        std::max(1., (double)iterations_momentum.first);
    if(iterations_avg[0] > std::numeric_limits<double>::min())
      iterations_avg[2] = iterations_avg[1] / iterations_avg[0];
    else
      iterations_avg[2] = iterations_avg[1];
    iterations_avg[3] =
      (double)iterations_pressure.second / std::max(1., (double)iterations_pressure.first);
    iterations_avg[4] =
      (double)iterations_projection.second / std::max(1., (double)iterations_projection.first);
  }

  print_list_of_iterations(this->pcout, names, iterations_avg);
}

// instantiations

template class TimeIntBDFPressureCorrection<2, float>;
template class TimeIntBDFPressureCorrection<2, double>;

template class TimeIntBDFPressureCorrection<3, float>;
template class TimeIntBDFPressureCorrection<3, double>;

} // namespace IncNS
} // namespace ExaDG
