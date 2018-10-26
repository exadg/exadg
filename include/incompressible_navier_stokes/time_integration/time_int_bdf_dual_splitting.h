/*
 * TimeIntBDFDualSplitting.h
 *
 *  Created on: May 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_

#include "../../incompressible_navier_stokes/time_integration/time_int_bdf_navier_stokes.h"
#include "../../time_integration/push_back_vectors.h"

namespace IncNS
{
template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
class TimeIntBDFDualSplitting
  : public TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>
{
public:
  typedef TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation> Base;

  typedef typename Base::VectorType VectorType;

  TimeIntBDFDualSplitting(std::shared_ptr<NavierStokesOperation> navier_stokes_operation_in,
                          InputParameters<dim> const &           param_in,
                          unsigned int const                     n_refine_time_in,
                          bool const                             use_adaptive_time_stepping)
    : TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>(
        navier_stokes_operation_in,
        param_in,
        n_refine_time_in,
        use_adaptive_time_stepping),
      velocity(this->order),
      pressure(this->order),
      vorticity(this->param.order_extrapolation_pressure_nbc),
      vec_convective_term(this->order),
      computing_times(4),
      iterations(4),
      extra_pressure_nbc(this->param.order_extrapolation_pressure_nbc,
                         this->param.start_with_low_order),
      navier_stokes_operation(navier_stokes_operation_in)
  {
  }

  virtual ~TimeIntBDFDualSplitting()
  {
  }

  virtual void
  analyze_computing_times() const;

protected:
  virtual void
  setup_derived();

  virtual void
  solve_timestep();

  virtual void
  initialize_vectors();

  virtual void
  prepare_vectors_for_next_timestep();

  virtual void
  convective_step();

  virtual void
  read_restart_vectors(boost::archive::binary_iarchive & ia);

  virtual void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const;

  std::vector<VectorType> velocity;

  std::vector<VectorType> pressure;

  VectorType velocity_np;

  std::vector<VectorType> vorticity;

  std::vector<VectorType> vec_convective_term;

  std::vector<double>       computing_times;
  std::vector<unsigned int> iterations;

  VectorType rhs_vec_viscous;

  // postprocessing: intermediate velocity
  VectorType intermediate_velocity;

  virtual void
  postprocessing() const;

  virtual void
  postprocessing_steady_problem() const;

private:
  virtual void
  initialize_time_integrator_constants();

  virtual void
  update_time_integrator_constants();

  virtual void
  initialize_current_solution();

  virtual void
  initialize_former_solution();

  void
  initialize_vorticity();

  void
  initialize_vec_convective_term();

  void
  initialize_intermediate_velocity();

  void
  pressure_step();

  void
  rhs_pressure();

  void
  projection_step();

  void
  rhs_projection();

  virtual void
  viscous_step();

  void
  rhs_viscous();

  virtual void
  solve_steady_problem();

  double
  evaluate_residual();

  virtual LinearAlgebra::distributed::Vector<value_type> const &
  get_velocity();

  virtual LinearAlgebra::distributed::Vector<value_type> const &
  get_velocity(unsigned int i /* t_{n-i} */);

  void
  postprocessing_stability_analysis();

  // time integrator constants: extrapolation scheme
  ExtrapolationConstants extra_pressure_nbc;

  VectorType pressure_np;

  VectorType vorticity_extrapolated;

  VectorType rhs_vec_pressure;
  VectorType rhs_vec_pressure_temp;

  VectorType rhs_vec_projection;
  VectorType rhs_vec_projection_temp;

  std::shared_ptr<NavierStokesOperation> navier_stokes_operation;

  // temporary vectors needed for pseudo-timestepping algorithm
  VectorType velocity_tmp;
  VectorType pressure_tmp;
};

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::
  initialize_time_integrator_constants()
{
  // call function of base class to initialize the standard time integrator constants
  TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
    initialize_time_integrator_constants();

  // set time integrator constants for extrapolation scheme of viscous term and convective term in
  // pressure NBC
  extra_pressure_nbc.initialize();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::
  update_time_integrator_constants()
{
  // call function of base class to update the standard time integrator constants
  TimeIntBDFNavierStokes<dim, fe_degree_u, value_type, NavierStokesOperation>::
    update_time_integrator_constants();

  // update time integrator constants for extrapolation scheme of pressure gradient term in case of
  // incremental formulation of pressure-correction scheme
  if(this->adaptive_time_stepping == false)
  {
    extra_pressure_nbc.update(this->time_step_number);
  }
  else // adaptive time stepping
  {
    extra_pressure_nbc.update(this->time_step_number, this->time_steps);
  }

  // use this function to check the correctness of the time integrator constants
  //  std::cout << "Coefficients extrapolation scheme pressure NBC:" << std::endl;
  //  extra_pressure_nbc.print();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::setup_derived()
{
  initialize_vorticity();

  initialize_intermediate_velocity();

  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.start_with_low_order == false)
    initialize_vec_convective_term();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::initialize_vectors()
{
  // velocity
  for(unsigned int i = 0; i < velocity.size(); ++i)
    navier_stokes_operation->initialize_vector_velocity(velocity[i]);
  navier_stokes_operation->initialize_vector_velocity(velocity_np);

  // pressure
  for(unsigned int i = 0; i < pressure.size(); ++i)
    navier_stokes_operation->initialize_vector_pressure(pressure[i]);
  navier_stokes_operation->initialize_vector_pressure(pressure_np);

  // vorticity
  for(unsigned int i = 0; i < vorticity.size(); ++i)
    navier_stokes_operation->initialize_vector_vorticity(vorticity[i]);
  navier_stokes_operation->initialize_vector_vorticity(vorticity_extrapolated);

  // vec_convective_term
  if(this->param.equation_type == EquationType::NavierStokes)
  {
    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      navier_stokes_operation->initialize_vector_velocity(vec_convective_term[i]);
  }

  // Sum_i (alpha_i/dt * u_i)
  navier_stokes_operation->initialize_vector_velocity(this->sum_alphai_ui);

  // rhs vector pressure
  navier_stokes_operation->initialize_vector_pressure(rhs_vec_pressure);
  navier_stokes_operation->initialize_vector_pressure(rhs_vec_pressure_temp);

  // rhs vector projection, viscous
  navier_stokes_operation->initialize_vector_velocity(rhs_vec_projection);
  navier_stokes_operation->initialize_vector_velocity(rhs_vec_projection_temp);
  navier_stokes_operation->initialize_vector_velocity(rhs_vec_viscous);

  // intermediate velocity
  if(this->param.output_data.write_divergence == true ||
     this->param.mass_data.calculate_error == true)
  {
    navier_stokes_operation->initialize_vector_velocity(intermediate_velocity);
  }
}


template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::
  initialize_current_solution()
{
  navier_stokes_operation->prescribe_initial_conditions(velocity[0], pressure[0], this->time);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::
  initialize_former_solution()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < velocity.size(); ++i)
  {
    navier_stokes_operation->prescribe_initial_conditions(
      velocity[i], pressure[i], this->time - double(i) * this->time_steps[0]);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::initialize_vorticity()
{
  navier_stokes_operation->compute_vorticity(vorticity[0], velocity[0]);

  if(this->param.start_with_low_order == false)
  {
    for(unsigned int i = 1; i < vorticity.size(); ++i)
    {
      navier_stokes_operation->compute_vorticity(vorticity[i], velocity[i]);
    }
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::
  initialize_intermediate_velocity()
{
  // intermediate velocity
  if(this->param.output_data.write_divergence == true ||
     this->param.mass_data.calculate_error == true)
  {
    intermediate_velocity = velocity[0];
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::
  initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < vec_convective_term.size(); ++i)
  {
    navier_stokes_operation->evaluate_convective_term_and_apply_inverse_mass_matrix(
      vec_convective_term[i], velocity[i], this->time - value_type(i) * this->time_steps[0]);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
LinearAlgebra::distributed::Vector<value_type> const &
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::get_velocity()
{
  return velocity[0];
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
LinearAlgebra::distributed::Vector<value_type> const &
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::get_velocity(
  unsigned int i)
{
  return velocity[i];
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::read_restart_vectors(
  boost::archive::binary_iarchive & ia)
{
  Vector<double> tmp;
  for(unsigned int i = 0; i < velocity.size(); i++)
  {
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(), velocity[i].begin());
  }
  for(unsigned int i = 0; i < pressure.size(); i++)
  {
    ia >> tmp;
    std::copy(tmp.begin(), tmp.end(), pressure[i].begin());
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::write_restart_vectors(
  boost::archive::binary_oarchive & oa) const
{
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  VectorView<double> tmp(velocity[0].local_size(), velocity[0].begin());
#pragma GCC diagnostic pop
  oa << tmp;
  for(unsigned int i = 1; i < velocity.size(); i++)
  {
    tmp.reinit(velocity[i].local_size(), velocity[i].begin());
    oa << tmp;
  }
  for(unsigned int i = 0; i < pressure.size(); i++)
  {
    tmp.reinit(pressure[i].local_size(), pressure[i].begin());
    oa << tmp;
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::postprocessing() const
{
  bool const standard = true;
  if(standard)
  {
    navier_stokes_operation->do_postprocessing(
      velocity[0], intermediate_velocity, pressure[0], this->time, this->time_step_number);
  }
  else // consider solution increment
  {
    VectorType velocity_incr;
    navier_stokes_operation->initialize_vector_velocity(velocity_incr);

    VectorType pressure_incr;
    navier_stokes_operation->initialize_vector_pressure(pressure_incr);

    velocity_incr = velocity[0];
    velocity_incr.add(-1.0, velocity[1]);
    pressure_incr = pressure[0];
    pressure_incr.add(-1.0, pressure[1]);

    navier_stokes_operation->do_postprocessing(
      velocity_incr, intermediate_velocity, pressure_incr, this->time, this->time_step_number);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::
  postprocessing_steady_problem() const
{
  navier_stokes_operation->do_postprocessing_steady_problem(velocity[0],
                                                            intermediate_velocity,
                                                            pressure[0]);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::
  postprocessing_stability_analysis()
{
  AssertThrow(this->order == 1,
              ExcMessage("Order of BDF scheme has to be 1 for this stability analysis."));

  AssertThrow(velocity[0].l2_norm() < 1.e-15 && pressure[0].l2_norm() < 1.e-15,
              ExcMessage("Solution vector has to be zero for this stability analysis."));

  AssertThrow(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD) == 1,
              ExcMessage("Number of MPI processes has to be 1."));

  std::cout << std::endl << "Analysis of eigenvalue spectrum:" << std::endl;

  const unsigned int size = velocity[0].local_size();

  LAPACKFullMatrix<value_type> propagation_matrix(size, size);

  // loop over all columns of propagation matrix
  for(unsigned int j = 0; j < size; ++j)
  {
    // set j-th element to 1
    velocity[0].local_element(j) = 1.0;

    // compute vorticity using the current velocity vector
    // (dual splitting scheme !!!)
    navier_stokes_operation->compute_vorticity(vorticity[0], velocity[0]);

    // solve time step
    solve_timestep();

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
    //    std::cout << propagation_matrix.eigenvalue(i) << std::endl;
  }

  std::cout << std::endl << std::endl << "Maximum eigenvalue = " << norm_max << std::endl;
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::solve_timestep()
{
  // perform the four substeps of the dual-splitting method
  convective_step();

  pressure_step();

  projection_step();

  viscous_step();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::convective_step()
{
  Timer timer;
  timer.restart();

  // compute body force vector
  if(this->param.right_hand_side == true)
  {
    navier_stokes_operation->evaluate_body_force_and_apply_inverse_mass_matrix(
      velocity_np, this->time + this->time_steps[0]);
  }
  else // right_hand_side == false
  {
    velocity_np = 0.0;
  }

  // compute convective term and extrapolate convective term (if not Stokes equations)
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit)
  {
    navier_stokes_operation->evaluate_convective_term_and_apply_inverse_mass_matrix(
      vec_convective_term[0], velocity[0], this->time);
    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      velocity_np.add(-this->extra.get_beta(i), vec_convective_term[i]);
  }

  // calculate sum (alpha_i/dt * u_tilde_i) in case of explicit treatment of convective term
  // and operator-integration-factor (OIF) splitting
  if(this->param.equation_type == EquationType::NavierStokes &&
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    this->calculate_sum_alphai_ui_oif_substepping();
  }
  // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
  else
  {
    this->sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->time_steps[0], velocity[0]);
    for(unsigned int i = 1; i < velocity.size(); ++i)
    {
      this->sum_alphai_ui.add(this->bdf.get_alpha(i) / this->time_steps[0], velocity[i]);
    }
  }

  // solve discrete temporal derivative term for intermediate velocity u_hat (if not STS approach)
  if(this->param.small_time_steps_stability == false)
  {
    velocity_np.add(1.0, this->sum_alphai_ui);
    velocity_np *= this->time_steps[0] / this->bdf.get_gamma0();
  }

  if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Explicit ||
     this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    // write output explicit case
    if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
    {
      this->pcout << std::endl
                  << "Solve nonlinear convective step explicitly:" << std::endl
                  << "  Iterations:        " << std::setw(6) << std::right << "-"
                  << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
    }
  }
  else // param.treatment_of_convective_term == Implicit
  {
    AssertThrow(
      this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit &&
        !(this->param.equation_type == EquationType::Stokes ||
          this->param.small_time_steps_stability),
      ExcMessage(
        "Use TreatmentOfConvectiveTerm::Explicit when solving the Stokes equations or when using the STS approach."));

    // calculate Sum_i (alpha_i/dt * u_i)
    this->sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->time_steps[0], velocity[0]);
    for(unsigned int i = 1; i < velocity.size(); ++i)
      this->sum_alphai_ui.add(this->bdf.get_alpha(i) / this->time_steps[0], velocity[i]);

    unsigned int newton_iterations;
    unsigned int linear_iterations;
    navier_stokes_operation->solve_nonlinear_convective_problem(
      velocity_np,
      this->sum_alphai_ui,
      this->time + this->time_steps[0],
      this->get_scaling_factor_time_derivative_term(),
      newton_iterations,
      linear_iterations);

    // write output implicit case
    if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
    {
      this->pcout << std::endl
                  << "Solve nonlinear convective step for intermediate velocity:" << std::endl
                  << "  Newton iterations: " << std::setw(6) << std::right << newton_iterations
                  << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl
                  << "  Linear iterations: " << std::setw(6) << std::right << std::fixed
                  << std::setprecision(2) << (double)linear_iterations / (double)newton_iterations
                  << " (avg)" << std::endl
                  << "  Linear iterations: " << std::setw(6) << std::right << std::fixed
                  << std::setprecision(2) << linear_iterations << " (tot)" << std::endl;
    }
  }

  computing_times[0] += timer.wall_time();
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::pressure_step()
{
  Timer timer;
  timer.restart();

  // compute right-hand-side vector
  rhs_pressure();

  // extrapolate old solution to get a good initial estimate for the solver
  pressure_np = 0;
  for(unsigned int i = 0; i < pressure.size(); ++i)
  {
    pressure_np.add(this->extra.get_beta(i), pressure[i]);
  }

  // solve linear system of equations
  unsigned int iterations_pressure =
    navier_stokes_operation->solve_pressure(pressure_np, rhs_vec_pressure);

  // special case: pure Dirichlet BC's
  // Adjust the pressure level in order to allow a calculation of the pressure error.
  // This is necessary because otherwise the pressure solution moves away from the exact solution.
  // For some test cases it was found that ApplyZeroMeanValue works better than
  // ApplyAnalyticalSolutionInPoint
  if(this->param.pure_dirichlet_bc)
  {
    if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalSolutionInPoint)
    {
      navier_stokes_operation->shift_pressure(pressure_np, this->time + this->time_steps[0]);
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyZeroMeanValue)
    {
      set_zero_mean_value(pressure_np);
    }
    else if(this->param.adjust_pressure_level == AdjustPressureLevel::ApplyAnalyticalMeanValue)
    {
      navier_stokes_operation->shift_pressure_mean_value(pressure_np,
                                                         this->time + this->time_steps[0]);
    }
    else
    {
      AssertThrow(false,
                  ExcMessage("Specified method to adjust pressure level is not implemented."));
    }
  }

  // write output
  if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
  {
    this->pcout << std::endl
                << "Solve Poisson equation for pressure p:" << std::endl
                << "  Iterations:        " << std::setw(6) << std::right << iterations_pressure
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  computing_times[1] += timer.wall_time();
  iterations[1] += iterations_pressure;
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::rhs_pressure()
{
  /*
   *  I. calculate divergence term
   */
  // homogeneous part of velocity divergence operator
  navier_stokes_operation->apply_velocity_divergence_term(rhs_vec_pressure, velocity_np);

  if(this->param.small_time_steps_stability == true)
    rhs_vec_pressure *= -1.0;
  else
    rhs_vec_pressure *= -this->bdf.get_gamma0() / this->time_steps[0];

  // inhomogeneous parts of boundary face integrals of velocity divergence operator
  if(this->param.divu_integrated_by_parts == true)
  {
    if(this->param.divu_use_boundary_data == true)
    {
      // sum alpha_i * u_i term
      for(unsigned int i = 0; i < velocity.size(); ++i)
      {
        double time_offset = 0.0;
        for(unsigned int k = 0; k <= i; ++k)
          time_offset += this->time_steps[k];

        rhs_vec_pressure_temp = 0;
        navier_stokes_operation->rhs_velocity_divergence_term(rhs_vec_pressure_temp,
                                                              this->time + this->time_steps[0] -
                                                                time_offset);

        // note that the minus sign related to this term is already taken into account
        // in the function .rhs() of the divergence operator
        rhs_vec_pressure.add(this->bdf.get_alpha(i) / this->time_steps[0], rhs_vec_pressure_temp);
      }

      // convective term
      if(this->param.equation_type == EquationType::NavierStokes)
      {
        for(unsigned int i = 0; i < velocity.size(); ++i)
        {
          rhs_vec_pressure_temp = 0;
          navier_stokes_operation->rhs_ppe_div_term_convective_term_add(rhs_vec_pressure_temp,
                                                                        velocity[i]);
          rhs_vec_pressure.add(this->extra.get_beta(i), rhs_vec_pressure_temp);
        }
      }

      // body force term
      navier_stokes_operation->rhs_ppe_div_term_body_forces_add(rhs_vec_pressure,
                                                                this->time + this->time_steps[0]);
    }
  }

  /*
   *  II. calculate terms originating from inhomogeneous parts of boundary face integrals of Laplace
   * operator
   */

  // II.1. inhomogeneous BC terms depending on prescribed boundary data,
  //       i.e. pressure Dirichlet boundary conditions on Gamma_N
  navier_stokes_operation->rhs_ppe_laplace_add(rhs_vec_pressure, this->time + this->time_steps[0]);
  //       and body force vector, temporal derivative of velocity on Gamma_D
  navier_stokes_operation->rhs_ppe_nbc_add(rhs_vec_pressure, this->time + this->time_steps[0]);

  // II.2. viscous term of pressure Neumann boundary condition on Gamma_D
  //       extrapolate vorticity and subsequently evaluate boundary face integral
  //       (this is possible since pressure Neumann BC is linear in vorticity)
  vorticity_extrapolated = 0;
  for(unsigned int i = 0; i < extra_pressure_nbc.get_order(); ++i)
  {
    vorticity_extrapolated.add(this->extra_pressure_nbc.get_beta(i), vorticity[i]);
  }

  navier_stokes_operation->rhs_ppe_viscous_add(rhs_vec_pressure, vorticity_extrapolated);

  // II.3. convective term of pressure Neumann boundary condition on Gamma_D
  //       (only if we do not solve the Stokes equations)
  //       evaluate convective term and subsequently extrapolate rhs vectors
  //       (the convective term is nonlinear!)
  if(this->param.equation_type == EquationType::NavierStokes)
  {
    for(unsigned int i = 0; i < extra_pressure_nbc.get_order(); ++i)
    {
      rhs_vec_pressure_temp = 0;
      navier_stokes_operation->rhs_ppe_convective_add(rhs_vec_pressure_temp, velocity[i]);
      rhs_vec_pressure.add(this->extra_pressure_nbc.get_beta(i), rhs_vec_pressure_temp);
    }
  }

  // special case: pure Dirichlet BC's
  // Set mean value of rhs to zero in order to obtain a consistent linear system of equations.
  // This is really necessary for the dual-splitting scheme in contrast to the pressure-correction
  // scheme and coupled solution approach due to the Dirichlet BC prescribed for the intermediate
  // velocity field and the pressure Neumann BC in case of the dual-splitting scheme.
  if(this->param.pure_dirichlet_bc)
    set_zero_mean_value(rhs_vec_pressure);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::projection_step()
{
  Timer timer;
  timer.restart();

  // when using the STS stability approach vector updates have to be performed to obtain the
  // intermediate velocity u_hat which is used to calculate the rhs of the projection step
  if(this->param.small_time_steps_stability == true)
  {
    velocity_np.add(1.0, this->sum_alphai_ui);
    velocity_np *= this->time_steps[0] / this->bdf.get_gamma0();
  }

  // compute right-hand-side vector
  rhs_projection();

  VectorType velocity_extrapolated;

  unsigned int iterations_projection = 0;

  // extrapolate velocity to time t_n+1 and use this velocity field to
  // caculate the penalty parameter for the divergence and continuity penalty term
  if(this->param.use_divergence_penalty == true || this->param.use_continuity_penalty == true)
  {
    velocity_extrapolated.reinit(velocity[0]);
    for(unsigned int i = 0; i < velocity.size(); ++i)
      velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);

    navier_stokes_operation->update_projection_operator(velocity_extrapolated, this->time_steps[0]);

    // solve linear system of equations
    iterations_projection =
      navier_stokes_operation->solve_projection(velocity_np, rhs_vec_projection);
  }
  else // no penalty terms, simply apply inverse mass matrix
  {
    navier_stokes_operation->apply_inverse_mass_matrix(velocity_np, rhs_vec_projection);
  }

  // write output
  if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
  {
    this->pcout << std::endl
                << "Solve projection step for intermediate velocity:" << std::endl
                << "  Iterations:        " << std::setw(6) << std::right << iterations_projection
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  // write velocity_np into intermediate_velocity which is needed for
  // postprocessing reasons
  if(this->param.output_data.write_divergence == true ||
     this->param.mass_data.calculate_error == true)
  {
    intermediate_velocity = velocity_np;
  }

  computing_times[2] += timer.wall_time();
  iterations[2] += iterations_projection;
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::rhs_projection()
{
  /*
   *  I. calculate mass matrix term
   */
  navier_stokes_operation->apply_mass_matrix(rhs_vec_projection, velocity_np);

  /*
   *  II. calculate pressure gradient term
   */
  navier_stokes_operation->evaluate_pressure_gradient_term(rhs_vec_projection_temp,
                                                           pressure_np,
                                                           this->time + this->time_steps[0]);

  rhs_vec_projection.add(-this->time_steps[0] / this->bdf.get_gamma0(), rhs_vec_projection_temp);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::viscous_step()
{
  Timer timer;
  timer.restart();

  // if a turbulence model is used:
  // update turbulence model before calculating rhs_viscous
  if(this->param.use_turbulence_model == true)
  {
    Timer timer_turbulence;
    timer_turbulence.restart();

    // extrapolate velocity to time t_n+1 and use this velocity field to
    // update the turbulence model (to recalculate the turbulent viscosity)
    VectorType velocity_extrapolated(velocity[0]);
    velocity_extrapolated = 0;
    for(unsigned int i = 0; i < velocity.size(); ++i)
      velocity_extrapolated.add(this->extra.get_beta(i), velocity[i]);

    navier_stokes_operation->update_turbulence_model(velocity_extrapolated);

    if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
    {
      this->pcout << std::endl
                  << "Update of turbulent viscosity:   Wall time [s]: " << std::scientific
                  << timer_turbulence.wall_time() << std::endl;
    }
  }

  // compute right-hand-side vector
  rhs_viscous();

  // Extrapolate old solution to get a good initial estimate for the solver.
  // Note that this has to be done after calling rhs_viscous()!
  velocity_np = 0;
  for(unsigned int i = 0; i < velocity.size(); ++i)
    velocity_np.add(this->extra.get_beta(i), velocity[i]);

  // solve linear system of equations
  unsigned int iterations_viscous =
    navier_stokes_operation->solve_viscous(velocity_np,
                                           rhs_vec_viscous,
                                           this->get_scaling_factor_time_derivative_term());

  // write output
  if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
  {
    this->pcout << std::endl
                << "Solve viscous step for velocity u:" << std::endl
                << "  Iterations:        " << std::setw(6) << std::right << iterations_viscous
                << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;
  }

  computing_times[3] += timer.wall_time();
  iterations[3] += iterations_viscous;
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::rhs_viscous()
{
  /*
   *  I. calculate mass matrix term
   */
  navier_stokes_operation->apply_mass_matrix(rhs_vec_viscous, velocity_np);
  rhs_vec_viscous *= this->bdf.get_gamma0() / this->time_steps[0];

  /*
   *  II. inhomongeous parts of boundary face integrals of viscous operator
   */
  navier_stokes_operation->rhs_add_viscous_term(rhs_vec_viscous, this->time + this->time_steps[0]);
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::
  prepare_vectors_for_next_timestep()
{
  push_back(velocity);
  velocity[0].swap(velocity_np);

  push_back(pressure);
  pressure[0].swap(pressure_np);

  push_back(vorticity);
  navier_stokes_operation->compute_vorticity(vorticity[0], velocity[0]);

  if(this->param.equation_type == EquationType::NavierStokes)
  {
    push_back(vec_convective_term);
  }
}

template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::solve_steady_problem()
{
  this->pcout << std::endl << "Starting time loop ..." << std::endl;

  // pseudo-time integration in order to solve steady-state problem
  bool converged = false;

  if(this->param.convergence_criterion_steady_problem ==
     ConvergenceCriterionSteadyProblem::SolutionIncrement)
  {
    while(!converged && this->time_step_number <= this->param.max_number_of_time_steps)
    {
      // save solution from previous time step
      velocity_tmp = this->velocity[0];
      pressure_tmp = this->pressure[0];

      // calculate normm of solution
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
      velocity_tmp.add(1.0, this->velocity[0]);
      pressure_tmp.add(1.0, this->pressure[0]);

      double const incr_u   = velocity_tmp.l2_norm();
      double const incr_p   = pressure_tmp.l2_norm();
      double const incr     = std::sqrt(incr_u * incr_u + incr_p * incr_p);
      double       incr_rel = 1.0;
      if(norm > 1.0e-10)
        incr_rel = incr / norm;

      // write output
      if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
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
    AssertThrow(this->param.convergence_criterion_steady_problem !=
                  ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes,
                ExcMessage("This option is not available for the dual splitting scheme. "
                           "Due to splitting errors the solution does not fulfill the "
                           "residual of the steady, incompressible Navier-Stokes equations."));
  }
  else
  {
    AssertThrow(false, ExcMessage("not implemented."));
  }

  AssertThrow(
    converged == true,
    ExcMessage(
      "Maximum number of time steps exceeded! This might be due to the fact that "
      "(i) the maximum number of iterations is simply too small to reach a steady solution, "
      "(ii) the problem is unsteady so that the applied solution approach is inappropriate, "
      "(iii) some of the solver tolerances are in conflict."));

  this->pcout << std::endl << "... done!" << std::endl;
}


template<int dim, int fe_degree_u, typename value_type, typename NavierStokesOperation>
void
TimeIntBDFDualSplitting<dim, fe_degree_u, value_type, NavierStokesOperation>::
  analyze_computing_times() const
{
  std::string  names[5]     = {"Convection   ", "Pressure     ", "Projection   ", "Viscous      "};
  unsigned int N_time_steps = this->time_step_number - 1;

  // iterations
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl
              << "Average number of iterations:" << std::endl;

  for(unsigned int i = 0; i < iterations.size(); ++i)
  {
    this->pcout << "  Step " << i + 1 << ": " << names[i] << std::scientific << std::setprecision(4)
                << std::setw(10) << (double)iterations[i] / (double)N_time_steps << std::endl;
  }
  this->pcout << "_________________________________________________________________________________"
              << std::endl
              << std::endl;

  // Computing times
  this->pcout << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl
              << "Computing times [s]:      min        avg        max        rel      p_min  p_max "
              << std::endl;

  double total_avg_time = 0.0;

  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data =
      Utilities::MPI::min_max_avg(computing_times[i], MPI_COMM_WORLD);
    total_avg_time += data.avg;
  }

  for(unsigned int i = 0; i < computing_times.size(); ++i)
  {
    Utilities::MPI::MinMaxAvg data =
      Utilities::MPI::min_max_avg(computing_times[i], MPI_COMM_WORLD);
    this->pcout << "  Step " << i + 1 << ": " << names[i] << std::scientific << std::setprecision(4)
                << std::setw(10) << data.min << " " << std::setprecision(4) << std::setw(10)
                << data.avg << " " << std::setprecision(4) << std::setw(10) << data.max << " "
                << std::setprecision(4) << std::setw(10) << data.avg / total_avg_time << "  "
                << std::setw(6) << std::left << data.min_index << " " << std::setw(6) << std::left
                << data.max_index << std::endl;
  }

  this->pcout << "  Time in steps 1-" << computing_times.size() << ":              "
              << std::setprecision(4) << std::setw(10) << total_avg_time << "            "
              << std::setprecision(4) << std::setw(10) << total_avg_time / total_avg_time
              << std::endl;

  this->pcout << std::endl
              << "Number of time steps =            " << std::left << N_time_steps << std::endl
              << "Average wall time per time step = " << std::scientific << std::setprecision(4)
              << total_avg_time / (double)N_time_steps << std::endl;

  // overall wall time including postprocessing
  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(this->total_time, MPI_COMM_WORLD);
  this->pcout << "Total wall time in [s] =          " << std::scientific << std::setprecision(4)
              << data.avg << std::endl;

  unsigned int N_mpi_processes = Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD);

  this->pcout << "Number of MPI processes =         " << N_mpi_processes << std::endl
              << "Computational costs in [CPUs] =   " << data.avg * (double)N_mpi_processes
              << std::endl
              << "Computational costs in [CPUh] =   " << data.avg * (double)N_mpi_processes / 3600.0
              << std::endl
              << "_________________________________________________________________________________"
              << std::endl
              << std::endl;
}



} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_BDF_DUAL_SPLITTING_H_ */
