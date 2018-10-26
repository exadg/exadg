/*
 * TimeIntBDFConvDiff.h
 *
 *  Created on: Aug 22, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_
#define INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_

#include <deal.II/lac/parallel_vector.h>

#include <deal.II/base/timer.h>

#include "time_integration/bdf_time_integration.h"
#include "time_integration/explicit_runge_kutta.h"
#include "time_integration/extrapolation_scheme.h"
#include "time_integration/push_back_vectors.h"
#include "time_integration/time_step_calculation.h"

namespace ConvDiff
{
template<int dim, int fe_degree, typename value_type>
class TimeIntBDF
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  TimeIntBDF(
    std::shared_ptr<ConvDiff::DGOperation<dim, fe_degree, value_type>> conv_diff_operation_in,
    ConvDiff::InputParameters const &                                  param_in,
    std::shared_ptr<Function<dim>>                                     velocity_in,
    unsigned int const                                                 n_refine_time_in)
    : conv_diff_operation(conv_diff_operation_in),
      param(param_in),
      velocity(velocity_in),
      n_refine_time(n_refine_time_in),
      cfl_number(param.cfl_number / std::pow(2.0, n_refine_time)),
      total_time(0.0),
      pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      time(param.start_time),
      time_step_number(1),
      order(param_in.order_time_integrator),
      time_steps(this->order),
      bdf(param_in.order_time_integrator, param_in.start_with_low_order),
      extra(param_in.order_time_integrator, param_in.start_with_low_order),
      solution(this->order),
      vec_convective_term(this->order),
      cfl_oif(param.cfl_oif / std::pow(2.0, n_refine_time)),
      M(1),
      delta_s(1.0),
      N_iter_average(0.0),
      solver_time_average(0.0)
  {
  }

  virtual ~TimeIntBDF()
  {
  }

  virtual void
  setup(bool do_restart = 0);

  virtual void
  timeloop();

  double
  get_scaling_factor_time_derivative_term()
  {
    return bdf.get_gamma0() / time_steps[0];
  }

private:
  void
  initialize_time_integrator_constants();

  void
  update_time_integrator_constants();

  void
  initialize_vectors();

  void
  initialize_solution();

  void
  initialize_vec_convective_term();

  void
  calculate_timestep();

  void
  prepare_vectors_for_next_timestep();

  void
  solve_timestep();

  void
  postprocessing() const;

  void
  analyze_computing_times() const;

  std::shared_ptr<ConvDiff::DGOperation<dim, fe_degree, value_type>> conv_diff_operation;

  ConvDiff::InputParameters const & param;

  std::shared_ptr<Function<dim>> velocity;

  unsigned int const n_refine_time;
  double const       cfl_number;

  // timer
  Timer  global_timer;
  double total_time;

  // screen output
  ConditionalOStream pcout;

  // physical time
  double time;

  // the number of the current time step starting with time_step_number = 1
  unsigned int time_step_number;

  // order of time integration scheme
  unsigned int const order;

  // vector with time step sizes
  std::vector<double> time_steps;

  // time integration constants
  BDFTimeIntegratorConstants bdf;
  ExtrapolationConstants     extra;

  // solution vectors
  VectorType              solution_np;
  std::vector<VectorType> solution;
  std::vector<VectorType> vec_convective_term;

  std::shared_ptr<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>>
    convective_operator_OIF;

  std::shared_ptr<
    ExplicitTimeIntegrator<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>, VectorType>>
    time_integrator_OIF;

  // cfl number cfl_oif for operator-integration-factor splitting
  double const cfl_oif;
  // number of substeps for operator-integration-factor splitting per time step dt
  unsigned int M;
  // substepping time step size delta_s for operator-integration-factor splitting
  double delta_s;

  VectorType solution_tilde_m;
  VectorType solution_tilde_mp;

  VectorType sum_alphai_ui;
  VectorType rhs_vector;

  double N_iter_average;
  double solver_time_average;
};

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::setup(bool /*do_restart*/)
{
  pcout << std::endl << "Setup time integrator ..." << std::endl;

  // call function of base class to initialize the time integrator constants
  this->initialize_time_integrator_constants();

  // initialize global solution vectors (allocation)
  initialize_vectors();

  // calculate time step size before initializing the solution because
  // initialization of solution depends on the time step size
  calculate_timestep();

  // initializes the solution by interpolation of analytical solution
  initialize_solution();

  // initialize vec_convective_term: Note that this function has to be called
  // after initialize_solution() because the solution is evaluated in this function
  if(this->param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Explicit &&
     (this->param.equation_type == ConvDiff::EquationType::Convection ||
      this->param.equation_type == ConvDiff::EquationType::ConvectionDiffusion) &&
     this->param.start_with_low_order == false)
  {
    initialize_vec_convective_term();
  }

  // Operator-integration-factor splitting
  if(this->param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    convective_operator_OIF.reset(
      new ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>(conv_diff_operation));

    if(this->param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK1Stage1)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                VectorType>(1, convective_operator_OIF));
    }
    else if(this->param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK2Stage2)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                VectorType>(2, convective_operator_OIF));
    }
    else if(this->param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK3Stage3)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                VectorType>(3, convective_operator_OIF));
    }
    else if(this->param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK4Stage4)
    {
      time_integrator_OIF.reset(new ExplicitRungeKuttaTimeIntegrator<
                                ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                VectorType>(4, convective_operator_OIF));
    }
    else if(this->param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK3Stage4Reg2C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK3Stage4Reg2C<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                     VectorType>(convective_operator_OIF));
    }
    else if(this->param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK4Stage5Reg2C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK4Stage5Reg2C<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                     VectorType>(convective_operator_OIF));
    }
    else if(this->param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK4Stage5Reg3C)
    {
      time_integrator_OIF.reset(
        new LowStorageRK4Stage5Reg3C<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                     VectorType>(convective_operator_OIF));
    }
    else if(this->param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK5Stage9Reg2S)
    {
      time_integrator_OIF.reset(
        new LowStorageRK5Stage9Reg2S<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>,
                                     VectorType>(convective_operator_OIF));
    }
    else if(this->param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK3Stage7Reg2)
    {
      time_integrator_OIF.reset(
        new LowStorageRKTD<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>, VectorType>(
          convective_operator_OIF, 3, 7));
    }
    else if(this->param.time_integrator_oif == ConvDiff::TimeIntegratorRK::ExplRK4Stage8Reg2)
    {
      time_integrator_OIF.reset(
        new LowStorageRKTD<ConvectiveOperatorOIFSplitting<dim, fe_degree, value_type>, VectorType>(
          convective_operator_OIF, 4, 8));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  pcout << std::endl << "... done!" << std::endl;
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::initialize_time_integrator_constants()
{
  bdf.initialize();
  extra.initialize();
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::update_time_integrator_constants()
{
  bdf.update(time_step_number);
  extra.update(time_step_number);

  // use this function to check the correctness of the time integrator constants
  //  std::cout << std::endl << "Time step " << time_step_number << std::endl << std::endl;
  //  std::cout << "Coefficients BDF time integration scheme:" << std::endl;
  //  bdf.print();
  //  std::cout << "Coefficients extrapolation scheme:" << std::endl;
  //  extra.print();
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::initialize_vectors()
{
  for(unsigned int i = 0; i < solution.size(); ++i)
    conv_diff_operation->initialize_dof_vector(solution[i]);

  conv_diff_operation->initialize_dof_vector(solution_np);

  conv_diff_operation->initialize_dof_vector(sum_alphai_ui);
  conv_diff_operation->initialize_dof_vector(rhs_vector);

  if(this->param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Explicit &&
     (this->param.equation_type == ConvDiff::EquationType::Convection ||
      this->param.equation_type == ConvDiff::EquationType::ConvectionDiffusion))
  {
    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      conv_diff_operation->initialize_dof_vector(vec_convective_term[i]);
  }

  if(this->param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    conv_diff_operation->initialize_dof_vector(solution_tilde_m);
    conv_diff_operation->initialize_dof_vector(solution_tilde_mp);
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::initialize_solution()
{
  for(unsigned int i = 0; i < solution.size(); ++i)
    conv_diff_operation->prescribe_initial_conditions(solution[i],
                                                      time - double(i) * time_steps[0]);
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::initialize_vec_convective_term()
{
  // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
  for(unsigned int i = 1; i < vec_convective_term.size(); ++i)
  {
    conv_diff_operation->evaluate_convective_term(vec_convective_term[i],
                                                  solution[i],
                                                  time - double(i) * time_steps[0]);
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::calculate_timestep()
{
  pcout << std::endl << "Calculation of time step size:" << std::endl << std::endl;

  if(param.calculation_of_time_step_size ==
     ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified)
  {
    time_steps[0] = calculate_const_time_step(param.time_step_size, n_refine_time);

    print_parameter(pcout, "time step size", time_steps[0]);
  }
  else if(param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFL)
  {
    AssertThrow(param.equation_type == ConvDiff::EquationType::Convection ||
                  param.equation_type == ConvDiff::EquationType::ConvectionDiffusion,
                ExcMessage("Time step calculation ConstTimeStepCFL does not make sense!"));

    // calculate minimum vertex distance
    double const global_min_cell_diameter = calculate_minimum_vertex_distance(
      conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    print_parameter(pcout, "h_min", global_min_cell_diameter);

    double time_step_conv = 1.0;

    double const max_velocity =
      calculate_max_velocity(conv_diff_operation->get_data().get_dof_handler().get_triangulation(),
                             velocity,
                             time);

    print_parameter(pcout, "U_max", max_velocity);
    print_parameter(pcout, "CFL", cfl_number);
    print_parameter(pcout, "Exponent fe_degree (convection)", param.exponent_fe_degree_convection);

    time_step_conv = calculate_const_time_step_cfl(cfl_number,
                                                   max_velocity,
                                                   global_min_cell_diameter,
                                                   fe_degree,
                                                   param.exponent_fe_degree_convection);

    // decrease time_step in order to exactly hit end_time
    time_steps[0] = (param.end_time - param.start_time) /
                    (1 + int((param.end_time - param.start_time) / time_step_conv));

    print_parameter(pcout, "Time step size (convection)", time_steps[0]);
  }
  else if(param.calculation_of_time_step_size ==
          ConvDiff::TimeStepCalculation::ConstTimeStepMaxEfficiency)
  {
    // calculate minimum vertex distance
    double const global_min_cell_diameter = calculate_minimum_vertex_distance(
      conv_diff_operation->get_data().get_dof_handler().get_triangulation());

    double time_step = calculate_time_step_max_efficiency(
      param.c_eff, global_min_cell_diameter, fe_degree, order, n_refine_time);

    // decrease time_step in order to exactly hit end_time
    time_steps[0] = (param.end_time - param.start_time) /
                    (1 + int((param.end_time - param.start_time) / time_step));

    pcout << "Calculation of time step size (max efficiency):" << std::endl << std::endl;
    print_parameter(pcout, "C_eff", param.c_eff / std::pow(2, n_refine_time));
    print_parameter(pcout, "Time step size", time_steps[0]);
  }
  else
  {
    AssertThrow(
      param.calculation_of_time_step_size ==
          ConvDiff::TimeStepCalculation::ConstTimeStepUserSpecified ||
        param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFL ||
        param.calculation_of_time_step_size ==
          ConvDiff::TimeStepCalculation::ConstTimeStepMaxEfficiency,
      ExcMessage(
        "Specified calculation of time step size not implemented for BDF time integrator!"));
  }

  // fill time_steps array
  for(unsigned int i = 1; i < order; ++i)
    time_steps[i] = time_steps[0];

  if(this->param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    // make sure that CFL condition is used for the calculation of the time step size (the aim
    // of the OIF splitting approach is to overcome limitations of the CFL condition)
    AssertThrow(
      param.calculation_of_time_step_size == ConvDiff::TimeStepCalculation::ConstTimeStepCFL,
      ExcMessage(
        "Specified calculation of time step size not compatible with OIF splitting approach!"));

    // calculate number of substeps M
    double tol = 1.0e-6;

    M = (int)(cfl_number / (cfl_oif - tol));

    if(cfl_oif < cfl_number / double(M) - tol)
      M += 1;

    // calculate substepping time step size delta_s
    delta_s = this->time_steps[0] / (double)M;

    // output
    pcout << std::endl
          << "Calculation of OIF substepping time step size:" << std::endl
          << std::endl;

    print_parameter(pcout, "CFL (OIF)", cfl_oif);
    print_parameter(pcout, "Number of substeps", M);
    print_parameter(pcout, "Substepping time step size", delta_s);
  }
}


template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::timeloop()
{
  pcout << std::endl << "Starting time loop ..." << std::endl;

  global_timer.restart();

  postprocessing();

  const double EPSILON = 1.0e-10;
  while(time < (param.end_time - EPSILON))
  {
    this->update_time_integrator_constants();

    solve_timestep();

    prepare_vectors_for_next_timestep();

    time += time_steps[0];
    ++time_step_number;

    postprocessing();

    // currently no write_restart implemented

    // currently no adaptive time stepping implemented
  }

  total_time += global_timer.wall_time();

  pcout << std::endl << "... done!" << std::endl;

  analyze_computing_times();
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::postprocessing() const
{
  conv_diff_operation->do_postprocessing(solution[0], time, time_step_number);
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::prepare_vectors_for_next_timestep()
{
  push_back(solution);

  solution[0].swap(solution_np);

  if(this->param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Explicit &&
     (this->param.equation_type == ConvDiff::EquationType::Convection ||
      this->param.equation_type == ConvDiff::EquationType::ConvectionDiffusion))
  {
    push_back(vec_convective_term);
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::solve_timestep()
{
  // write output
  if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
  {
    pcout << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl
          << " Number of TIME STEPS: " << std::left << std::setw(8) << this->time_step_number
          << "t_n = " << std::scientific << std::setprecision(4) << this->time
          << " -> t_n+1 = " << this->time + this->time_steps[0] << std::endl
          << "______________________________________________________________________" << std::endl
          << std::endl;
  }

  Timer timer;
  timer.restart();

  // calculate rhs (rhs-vector f and inhomogeneous boundary face integrals)
  conv_diff_operation->rhs(rhs_vector, this->time + this->time_steps[0]);

  // add the convective term to the right-hand side of the equations
  // if the convective term is treated explicitly (additive decomposition)
  if(param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Explicit)
  {
    // only if convective term is really involved
    if(this->param.equation_type == ConvDiff::EquationType::Convection ||
       this->param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
    {
      conv_diff_operation->evaluate_convective_term(vec_convective_term[0],
                                                    solution[0],
                                                    this->time);

      for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
        rhs_vector.add(-this->extra.get_beta(i), vec_convective_term[i]);
    }
  }

  // calculate sum (alpha_i/dt * u_tilde_i) in case of explicit treatment of convective term
  // and operator-integration-factor splitting
  if(this->param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::ExplicitOIF)
  {
    // start time t_{n-i} initialized with t_{n+1}
    double time_n_i = time + this->time_steps[0];

    // Loop over all previous time instants required by the BDF scheme
    // and calculate u_tilde by substepping algorithm, i.e.,
    // integrate over time interval t_{n-i} <= t <= t_{n+1}
    // using explicit Runge-Kutta methods.
    for(unsigned int i = 0; i < solution.size(); ++i)
    {
      // initialize solution: u_tilde(s=0) = u(t_{n-i})
      solution_tilde_m = solution[i];
      // calculate start time t_{n-i}
      time_n_i -= this->time_steps[i];

      // time loop substepping: t_{n-i} <= t <= t_{n+1}
      for(unsigned int m = 0; m < M * (i + 1) /*assume equidistant time step sizes*/; ++m)
      {
        time_integrator_OIF->solve_timestep(solution_tilde_mp,
                                            solution_tilde_m,
                                            time_n_i + delta_s * m,
                                            delta_s);

        solution_tilde_mp.swap(solution_tilde_m);
      }

      // calculate sum (alpha_i/dt * u_tilde_i)
      if(i == 0)
        sum_alphai_ui.equ(this->bdf.get_alpha(i) / this->time_steps[0], solution_tilde_m);
      else // i>0
        sum_alphai_ui.add(this->bdf.get_alpha(i) / this->time_steps[0], solution_tilde_m);
    }
  }
  // calculate sum (alpha_i/dt * u_i) for standard BDF discretization
  else
  {
    sum_alphai_ui.equ(this->bdf.get_alpha(0) / this->time_steps[0], solution[0]);
    for(unsigned int i = 1; i < solution.size(); ++i)
      sum_alphai_ui.add(this->bdf.get_alpha(i) / this->time_steps[0], solution[i]);
  }

  // apply mass matrix to sum_alphai_ui and add to rhs vector
  conv_diff_operation->apply_mass_matrix_add(rhs_vector, sum_alphai_ui);


  // extrapolate old solution to obtain a good initial guess for the solver
  solution_np.equ(this->extra.get_beta(0), solution[0]);
  for(unsigned int i = 1; i < solution.size(); ++i)
    solution_np.add(this->extra.get_beta(i), solution[i]);

  // solve the linear system of equations
  unsigned int iterations = conv_diff_operation->solve(solution_np,
                                                       rhs_vector,
                                                       this->bdf.get_gamma0() / this->time_steps[0],
                                                       this->time + this->time_steps[0]);

  N_iter_average += iterations;
  solver_time_average += timer.wall_time();

  // write output
  if(this->time_step_number % this->param.output_solver_info_every_timesteps == 0)
  {
    pcout << "Solve linear convection-diffusion problem:" << std::endl
          << "  Iterations: " << std::setw(6) << std::right << iterations
          << "\t Wall time [s]: " << std::scientific << timer.wall_time() << std::endl;

    if(time > param.start_time)
    {
      double const remaining_time =
        global_timer.wall_time() * (param.end_time - time) / (time - param.start_time);
      pcout << std::endl
            << "Estimated time until completion is " << remaining_time << " s / "
            << remaining_time / 3600. << " h." << std::endl;
    }
  }
}

template<int dim, int fe_degree, typename value_type>
void
TimeIntBDF<dim, fe_degree, value_type>::analyze_computing_times() const
{
  pcout << std::endl
        << "Number of time steps = " << (this->time_step_number - 1) << std::endl
        << "Average number of iterations = " << std::scientific << std::setprecision(3)
        << N_iter_average / (this->time_step_number - 1) << std::endl
        << "Average wall time per time step = " << std::scientific << std::setprecision(3)
        << solver_time_average / (this->time_step_number - 1) << std::endl;

  pcout << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl
        << "Computing times:          min        avg        max        rel      p_min  p_max"
        << std::endl;

  Utilities::MPI::MinMaxAvg data = Utilities::MPI::min_max_avg(this->total_time, MPI_COMM_WORLD);
  pcout << "  Time loop:           " << std::scientific << std::setprecision(4) << std::setw(10)
        << data.min << " " << std::setprecision(4) << std::setw(10) << data.avg << " "
        << std::setprecision(4) << std::setw(10) << data.max << " "
        << "          "
        << "  " << std::setw(6) << std::left << data.min_index << " " << std::setw(6) << std::left
        << data.max_index << std::endl
        << "_________________________________________________________________________________"
        << std::endl
        << std::endl;
}

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_TIME_INT_BDF_H_ */
