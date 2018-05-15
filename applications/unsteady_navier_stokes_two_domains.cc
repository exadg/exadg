/*
 * UnsteadyNavierStokesCoupledSolver.cc
 *
 *  Created on: Oct 10, 2016
 *      Author: fehn
 */

// deal.II
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/base/revision.h>

// postprocessor
#include "../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

// spatial discretization
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_coupled_solver.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_dual_splitting.h"
#include "../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_pressure_correction.h"

// temporal discretization
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_coupled_solver.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_dual_splitting.h"
#include "../include/incompressible_navier_stokes/time_integration/time_int_bdf_pressure_correction.h"

// Paramters, BCs, etc.
#include "../include/incompressible_navier_stokes/user_interface/input_parameters.h"
#include "../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../include/incompressible_navier_stokes/user_interface/field_functions.h"
#include "../include/incompressible_navier_stokes/user_interface/analytical_solution.h"

#include "../include/functionalities/print_general_infos.h"

using namespace dealii;

// specify the flow problem that has to be solved

//#include "incompressible_navier_stokes_test_cases/turbulent_channel_two_domains.h"
//#include "incompressible_navier_stokes_test_cases/backward_facing_step_two_domains.h"
#include "incompressible_navier_stokes_test_cases/fda_nozzle_benchmark.h"

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number=double>
class NavierStokesProblem
{
public:
  NavierStokesProblem(unsigned int const refine_steps_space1,
                      unsigned int const refine_steps_space2,
                      unsigned int const refine_steps_time = 0);

  void solve_problem(bool const do_restart);

private:
  void print_header();

  void setup_navier_stokes_operation();
  void setup_time_integrator(bool const do_restart);
  void setup_solvers();
  void setup_postprocessor();
  void run_timeloop();

  ConditionalOStream pcout;

  parallel::distributed::Triangulation<dim> triangulation_1, triangulation_2;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_faces_1, periodic_faces_2;

  const unsigned int n_refine_space_domain1, n_refine_space_domain2;

  std::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions_1, field_functions_2;
  std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > boundary_descriptor_velocity_1, boundary_descriptor_velocity_2;
  std::shared_ptr<BoundaryDescriptorNavierStokesP<dim> > boundary_descriptor_pressure_1, boundary_descriptor_pressure_2;

  std::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution_1, analytical_solution_2;

  InputParametersNavierStokes<dim> param_1, param_2;

  std::shared_ptr<DGNavierStokesBase<dim, fe_degree_u, fe_degree_p,
    fe_degree_xwall, xwall_quad_rule, Number> > navier_stokes_operation_1, navier_stokes_operation_2;

  std::shared_ptr<DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p,
    fe_degree_xwall, xwall_quad_rule, Number> > navier_stokes_operation_coupled_1, navier_stokes_operation_coupled_2;

  std::shared_ptr<DGNavierStokesDualSplitting<dim, fe_degree_u, fe_degree_p,
    fe_degree_xwall, xwall_quad_rule, Number> > navier_stokes_operation_dual_splitting_1, navier_stokes_operation_dual_splitting_2;

  std::shared_ptr<DGNavierStokesPressureCorrection<dim, fe_degree_u, fe_degree_p,
    fe_degree_xwall, xwall_quad_rule, Number> > navier_stokes_operation_pressure_correction_1, navier_stokes_operation_pressure_correction_2;

  std::shared_ptr<PostProcessorBase<dim,Number> > postprocessor_1, postprocessor_2;

  std::shared_ptr<TimeIntBDFCoupled<dim, fe_degree_u, Number,
                  DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p,
                    fe_degree_xwall, xwall_quad_rule, Number> > > time_integrator_coupled_1, time_integrator_coupled_2;

  std::shared_ptr<TimeIntBDFDualSplitting<dim, fe_degree_u, Number,
                  DGNavierStokesDualSplitting<dim, fe_degree_u, fe_degree_p,
                    fe_degree_xwall, xwall_quad_rule, Number> > > time_integrator_dual_splitting_1, time_integrator_dual_splitting_2;

  std::shared_ptr<TimeIntBDFPressureCorrection<dim, fe_degree_u, Number,
                  DGNavierStokesPressureCorrection<dim, fe_degree_u, fe_degree_p,
                    fe_degree_xwall, xwall_quad_rule, Number> > > time_integrator_pressure_correction_1, time_integrator_pressure_correction_2;
};

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
NavierStokesProblem(unsigned int const refine_steps_space1,
                    unsigned int const refine_steps_space2,
                    unsigned int const refine_steps_time)
  :
  pcout(std::cout,Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation_1(MPI_COMM_WORLD,
                  dealii::Triangulation<dim>::none,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  triangulation_2(MPI_COMM_WORLD,
                  dealii::Triangulation<dim>::none,
                  parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
  n_refine_space_domain1(refine_steps_space1),
  n_refine_space_domain2(refine_steps_space2)
{
  param_1.set_input_parameters(1);
  param_1.check_input_parameters();

  param_2.set_input_parameters(2);
  param_2.check_input_parameters();

  print_header();
  print_MPI_info(pcout);
  if(param_1.print_input_parameters == true)
  {
    pcout << std::endl << "List of input parameters for DOMAIN 1:" << std::endl;
    param_1.print(pcout);
  }
  if(param_2.print_input_parameters == true)
  {
    pcout << std::endl << "List of input parameters for DOMAIN 2:" << std::endl;
    param_2.print(pcout);
  }

  field_functions_1.reset(new FieldFunctionsNavierStokes<dim>());
  field_functions_2.reset(new FieldFunctionsNavierStokes<dim>());
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_field_functions_1(field_functions_1);
  set_field_functions_2(field_functions_2);

  analytical_solution_1.reset(new AnalyticalSolutionNavierStokes<dim>());
  analytical_solution_2.reset(new AnalyticalSolutionNavierStokes<dim>());
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  set_analytical_solution(analytical_solution_1);
  set_analytical_solution(analytical_solution_2);

  boundary_descriptor_velocity_1.reset(new BoundaryDescriptorNavierStokesU<dim>());
  boundary_descriptor_pressure_1.reset(new BoundaryDescriptorNavierStokesP<dim>());

  boundary_descriptor_velocity_2.reset(new BoundaryDescriptorNavierStokesU<dim>());
  boundary_descriptor_pressure_2.reset(new BoundaryDescriptorNavierStokesP<dim>());

  bool use_adaptive_time_stepping_1 = false, use_adaptive_time_stepping_2 = false;
  if(param_1.calculation_of_time_step_size == TimeStepCalculation::AdaptiveTimeStepCFL)
  {
    use_adaptive_time_stepping_1 = true;
    use_adaptive_time_stepping_2 = true;
  }

  AssertThrow(use_adaptive_time_stepping_1 == false && use_adaptive_time_stepping_2 == false,
      ExcMessage("Adaptive time stepping is not implemented for coupled two-domain solver. "
                 "When using adaptive time stepping for this solver, one has to make sure "
                 "that the same time step size is used for both domains."));

  AssertThrow(param_1.solver_type == SolverType::Unsteady && param_2.solver_type == SolverType::Unsteady,
      ExcMessage("This is an unsteady solver. Check input parameters."));

  // initialize navier_stokes_operation_1 (DOMAIN 1)
  if(this->param_1.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled_1.reset(
        new DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
        (triangulation_1,param_1));

    navier_stokes_operation_1 = navier_stokes_operation_coupled_1;
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting_1.reset(
        new DGNavierStokesDualSplitting<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
        (triangulation_1,param_1));

    navier_stokes_operation_1 = navier_stokes_operation_dual_splitting_1;
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction_1.reset(
        new DGNavierStokesPressureCorrection<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
        (triangulation_1,param_1));

    navier_stokes_operation_1 = navier_stokes_operation_pressure_correction_1;
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }

  // initialize navier_stokes_operation_2 (DOMAIN 2)
  if(this->param_2.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled_2.reset(
        new DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
        (triangulation_2,param_2));

    navier_stokes_operation_2 = navier_stokes_operation_coupled_2;
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting_2.reset(
        new DGNavierStokesDualSplitting<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
        (triangulation_2,param_2));

    navier_stokes_operation_2 = navier_stokes_operation_dual_splitting_2;
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction_2.reset(
        new DGNavierStokesPressureCorrection<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
        (triangulation_2,param_2));

    navier_stokes_operation_2 = navier_stokes_operation_pressure_correction_2;
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }


  // initialize postprocessor
  // this function has to be defined in the header file
  // that implements all problem specific things like
  // parameters, geometry, boundary conditions, etc.
  postprocessor_1 = construct_postprocessor<dim,Number>(param_1);
  postprocessor_2 = construct_postprocessor<dim,Number>(param_2);

  // initialize time integrator that depends on both navier_stokes_operation and postprocessor for DOMAIN 1
  if(this->param_1.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator_coupled_1.reset(new TimeIntBDFCoupled<dim, fe_degree_u, Number,
        DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> >
        (navier_stokes_operation_coupled_1,postprocessor_1,param_1,refine_steps_time,use_adaptive_time_stepping_1));
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator_dual_splitting_1.reset(new TimeIntBDFDualSplitting<dim, fe_degree_u, Number,
        DGNavierStokesDualSplitting<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> >
        (navier_stokes_operation_dual_splitting_1,postprocessor_1,param_1,refine_steps_time,use_adaptive_time_stepping_1));
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator_pressure_correction_1.reset(new TimeIntBDFPressureCorrection<dim, fe_degree_u, Number,
        DGNavierStokesPressureCorrection<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> >
        (navier_stokes_operation_pressure_correction_1,postprocessor_1,param_1,refine_steps_time,use_adaptive_time_stepping_1));
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }

  // initialize time integrator that depends on both navier_stokes_operation and postprocessor for DOMAIN 2
  if(this->param_2.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator_coupled_2.reset(new TimeIntBDFCoupled<dim, fe_degree_u, Number,
        DGNavierStokesCoupled<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> >
        (navier_stokes_operation_coupled_2,postprocessor_2,param_2,refine_steps_time,use_adaptive_time_stepping_2));
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator_dual_splitting_2.reset(new TimeIntBDFDualSplitting<dim, fe_degree_u, Number,
        DGNavierStokesDualSplitting<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> >
        (navier_stokes_operation_dual_splitting_2,postprocessor_2,param_2,refine_steps_time,use_adaptive_time_stepping_2));
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator_pressure_correction_2.reset(new TimeIntBDFPressureCorrection<dim, fe_degree_u, Number,
        DGNavierStokesPressureCorrection<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> >
        (navier_stokes_operation_pressure_correction_2,postprocessor_2,param_2,refine_steps_time,use_adaptive_time_stepping_2));
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
print_header()
{
  pcout << std::endl << std::endl << std::endl
  << "_________________________________________________________________________________" << std::endl
  << "                                                                                 " << std::endl
  << "                High-order discontinuous Galerkin solver for the                 " << std::endl
  << "                unsteady, incompressible Navier-Stokes equations                 " << std::endl
  << "                     based on a matrix-free implementation                       " << std::endl
  << "_________________________________________________________________________________" << std::endl
  << std::endl;
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup_postprocessor()
{
  AssertThrow(navier_stokes_operation_1.get() != 0, ExcMessage("Not initialized."));

  DofQuadIndexData dof_quad_index_data_1;
  dof_quad_index_data_1.dof_index_velocity = navier_stokes_operation_1->get_dof_index_velocity();
  dof_quad_index_data_1.dof_index_pressure = navier_stokes_operation_1->get_dof_index_pressure();
  dof_quad_index_data_1.quad_index_velocity = navier_stokes_operation_1->get_quad_index_velocity_linear();

  postprocessor_1->setup(navier_stokes_operation_1->get_dof_handler_u(),
                         navier_stokes_operation_1->get_dof_handler_p(),
                         navier_stokes_operation_1->get_mapping(),
                         navier_stokes_operation_1->get_data(),
                         dof_quad_index_data_1,
                         analytical_solution_1);

  AssertThrow(navier_stokes_operation_2.get() != 0, ExcMessage("Not initialized."));

  DofQuadIndexData dof_quad_index_data_2;
  dof_quad_index_data_2.dof_index_velocity = navier_stokes_operation_2->get_dof_index_velocity();
  dof_quad_index_data_2.dof_index_pressure = navier_stokes_operation_2->get_dof_index_pressure();
  dof_quad_index_data_2.quad_index_velocity = navier_stokes_operation_2->get_quad_index_velocity_linear();

  postprocessor_2->setup(navier_stokes_operation_2->get_dof_handler_u(),
                         navier_stokes_operation_2->get_dof_handler_p(),
                         navier_stokes_operation_2->get_mapping(),
                         navier_stokes_operation_2->get_data(),
                         dof_quad_index_data_2,
                         analytical_solution_2);
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup_navier_stokes_operation()
{
  AssertThrow(navier_stokes_operation_1.get() != 0, ExcMessage("Not initialized."));
  AssertThrow(navier_stokes_operation_2.get() != 0, ExcMessage("Not initialized."));

  navier_stokes_operation_1->setup(periodic_faces_1,
                                   boundary_descriptor_velocity_1,
                                   boundary_descriptor_pressure_1,
                                   field_functions_1);

  navier_stokes_operation_2->setup(periodic_faces_2,
                                   boundary_descriptor_velocity_2,
                                   boundary_descriptor_pressure_2,
                                   field_functions_2);
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup_time_integrator(bool const do_restart)
{
  // Setup time integrator and get time step size
  double time_step_size_1 = 1.0, time_step_size_2 = 1.0;
  // DOMAIN 1
  if(this->param_1.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator_coupled_1->setup(do_restart);
    time_step_size_1 = time_integrator_coupled_1->get_time_step_size();
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator_dual_splitting_1->setup(do_restart);
    time_step_size_1 = time_integrator_dual_splitting_1->get_time_step_size();
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator_pressure_correction_1->setup(do_restart);
    time_step_size_1 = time_integrator_pressure_correction_1->get_time_step_size();
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }
  // DOMAIN 2
  if(this->param_2.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator_coupled_2->setup(do_restart);
    time_step_size_2 = time_integrator_coupled_2->get_time_step_size();
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator_dual_splitting_2->setup(do_restart);
    time_step_size_2 = time_integrator_dual_splitting_2->get_time_step_size();
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator_pressure_correction_2->setup(do_restart);
    time_step_size_2 = time_integrator_pressure_correction_2->get_time_step_size();
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }

  double time_step_size = std::min(time_step_size_1,time_step_size_2);
  // decrease time_step in order to exactly hit end_time
  time_step_size = (param_1.end_time-param_1.start_time)/(1+int((param_1.end_time-param_1.start_time)/time_step_size));

  pcout << std::endl << "Combined time step size for both domains: " << time_step_size << std::endl;

  // Set the same time step size for both time integrators (for the two domains)
  // DOMAIN 1
  if(this->param_1.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator_coupled_1->set_time_step_size(time_step_size);
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator_dual_splitting_1->set_time_step_size(time_step_size);
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator_pressure_correction_1->set_time_step_size(time_step_size);
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }
  // DOMAIN 2
  if(this->param_2.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    time_integrator_coupled_2->set_time_step_size(time_step_size);
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    time_integrator_dual_splitting_2->set_time_step_size(time_step_size);
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    time_integrator_pressure_correction_2->set_time_step_size(time_step_size);
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
setup_solvers()
{
  // DOMAIN 1
  if(this->param_1.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled_1->setup_solvers(time_integrator_coupled_1->get_scaling_factor_time_derivative_term());
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting_1->setup_solvers(time_integrator_dual_splitting_1->get_time_step_size(),
                                                            time_integrator_dual_splitting_1->get_scaling_factor_time_derivative_term());
  }
  else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction_1->setup_solvers(time_integrator_pressure_correction_1->get_time_step_size(),
                                                                 time_integrator_pressure_correction_1->get_scaling_factor_time_derivative_term());
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }

  // DOMAIN 2
  if(this->param_2.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
  {
    navier_stokes_operation_coupled_2->setup_solvers(time_integrator_coupled_2->get_scaling_factor_time_derivative_term());
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
  {
    navier_stokes_operation_dual_splitting_2->setup_solvers(time_integrator_dual_splitting_2->get_time_step_size(),
                                                            time_integrator_dual_splitting_2->get_scaling_factor_time_derivative_term());
  }
  else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
  {
    navier_stokes_operation_pressure_correction_2->setup_solvers(time_integrator_pressure_correction_2->get_time_step_size(),
                                                                 time_integrator_pressure_correction_2->get_scaling_factor_time_derivative_term());
  }
  else
  {
    AssertThrow(false,ExcMessage("Not implemented."));
  }
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
run_timeloop()
{
  bool finished_1 = false, finished_2 = false;

  while(!finished_1 || !finished_2)
  {
    // advance one time step for DOMAIN 1
    if(this->param_1.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      finished_1 = time_integrator_coupled_1->advance_one_timestep(!finished_1);
    }
    else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      finished_1 = time_integrator_dual_splitting_1->advance_one_timestep(!finished_1);
    }
    else if(this->param_1.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      finished_1 = time_integrator_pressure_correction_1->advance_one_timestep(!finished_1);
    }
    else
    {
      AssertThrow(false,ExcMessage("Not implemented."));
    }

    // Note that the coupling of both solvers via the inflow boundary conditions is performed
    // in the postprocessing step of the solver for DOMAIN 1, overwriting the data global structures
    // which are subsequently used by the solver for DOMAIN 2 to evaluate the boundary conditions.

    // advance one time step for DOMAIN 2
    if(this->param_2.temporal_discretization == TemporalDiscretization::BDFCoupledSolution)
    {
      finished_2 = time_integrator_coupled_2->advance_one_timestep(!finished_2);
    }
    else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      finished_2 = time_integrator_dual_splitting_2->advance_one_timestep(!finished_2);
    }
    else if(this->param_2.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      finished_2 = time_integrator_pressure_correction_2->advance_one_timestep(!finished_2);
    }
    else
    {
      AssertThrow(false,ExcMessage("Not implemented."));
    }
  }
}

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
void NavierStokesProblem<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>::
solve_problem(bool const do_restart)
{
  // this function has to be defined in the header file that implements all
  // problem specific things like parameters, geometry, boundary conditions, etc.
  create_grid_and_set_boundary_conditions_1(triangulation_1,
                                            n_refine_space_domain1,
                                            boundary_descriptor_velocity_1,
                                            boundary_descriptor_pressure_1,
                                            periodic_faces_1);

  create_grid_and_set_boundary_conditions_2(triangulation_2,
                                            n_refine_space_domain2,
                                            boundary_descriptor_velocity_2,
                                            boundary_descriptor_pressure_2,
                                            periodic_faces_2);

  print_grid_data(pcout,
                  n_refine_space_domain1,
                  triangulation_1,
                  n_refine_space_domain2,
                  triangulation_2);

  setup_navier_stokes_operation();

  // setup time integrator before calling setup_solvers
  // (this is necessary since the setup of the solvers
  // depends on quantities such as the time_step_size or gamma0!!!)
  setup_time_integrator(do_restart);

  setup_solvers();

  setup_postprocessor();

  run_timeloop();
}

int main (int argc, char** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch "
                << DEAL_II_GIT_BRANCH << std::endl << std::endl;
    }

    deallog.depth_console(0);

    bool do_restart = false;
    if (argc > 1)
    {
      do_restart = std::atoi(argv[1]);
      if(do_restart)
      {
        // this does in principle work although it doesn't make much sense
        if(REFINE_STEPS_TIME_MIN != REFINE_STEPS_TIME_MAX && Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
          std::cout << "Warning: you are starting from a restart and refine the time steps!" << std::endl;
      }
    }

    //time refinements in order to perform temporal convergence tests
    for(unsigned int refine_steps_time = REFINE_STEPS_TIME_MIN;refine_steps_time <= REFINE_STEPS_TIME_MAX;++refine_steps_time)
    {
      NavierStokesProblem<DIMENSION, FE_DEGREE_VELOCITY, FE_DEGREE_PRESSURE, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, VALUE_TYPE>
          navier_stokes_problem(REFINE_STEPS_SPACE_DOMAIN1,
                                REFINE_STEPS_SPACE_DOMAIN2,
                                refine_steps_time);

      navier_stokes_problem.solve_problem(do_restart);
    }
  }
  catch (std::exception &exc)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  catch (...)
  {
    std::cerr << std::endl << std::endl
              << "----------------------------------------------------"
              << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------"
              << std::endl;
    return 1;
  }
  return 0;
}
