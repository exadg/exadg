/*
 * tum.h
 *
 *  Created on: Sep 8, 2017
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_


#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// single or double precision?
//typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 2;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 3;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const ProblemType PROBLEM_TYPE = ProblemType::Unsteady;
const double L = 1.0;
const double MAX_VELOCITY = 1.0;
const double VISCOSITY = 1.0e-3;

std::string OUTPUT_FOLDER = "output/tum/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "test"; //"Re1e3_l6_k21_outflow_bc_dual_splitting";

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = PROBLEM_TYPE; // PROBLEM_TYPE is also needed somewhere else
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  right_hand_side = false;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 50.0;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady; //Steady; //Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme; //BDFPressureCorrection; //BDFDualSplittingScheme; //BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit; //Explicit; //Implicit;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  adaptive_time_stepping = true;
  max_velocity = 1.0;
  // typical CFL values for BDF2, exponent = 1.5, k_u = 2, MAX_VELOCITY = 1.0
  // Re = 1e3: cfl = 0.025
  // Re = 1e4: cfl = 0.0125 and 0.003125 (l=5)
  cfl = 0.1; //TODO //0.025;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 5.0e-2;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true; // true; // false;

  // pseudo-timestepping for steady-state problems
  convergence_criterion_steady_problem = ConvergenceCriterionSteadyProblem::ResidualSteadyNavierStokes;
  abs_tol_steady = 1.e-12;
  rel_tol_steady = 1.e-10;

  // SPATIAL DISCRETIZATION

  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term - currently no parameters

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  IP_factor_viscous = 1.0;
  penalty_term_div_formulation = PenaltyTermDivergenceFormulation::NotSymmetrized;

  // gradient term
  gradp_integrated_by_parts = true;
  gradp_use_boundary_data = true;

  // divergence term
  divu_integrated_by_parts = true;
  divu_use_boundary_data = true;

  // special case: pure DBC's
  pure_dirichlet_bc = false;

  // div + conti penalty
  use_divergence_penalty = true; //true; //false;
  use_continuity_penalty = true; //true; //false;

  // PROJECTION METHODS

  // pressure Poisson equation
  IP_factor_pressure = 1.0;
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::GeometricMultigrid;
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_pressure = 1.e-12;
  rel_tol_pressure = 1.e-6;
  // stability in the limit of small time steps
  use_approach_of_ferrer = false;
  deltat_ref = 1.e0;

  // projection step
  solver_projection = SolverProjection::PCG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  abs_tol_projection = 1.e-12;
  rel_tol_projection = 1.e-6;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // convective step

  // nonlinear solver
  newton_solver_data_convective.abs_tol = 1.e-20;
  newton_solver_data_convective.rel_tol = 1.e-6;
  newton_solver_data_convective.max_iter = 100;
  // linear solver
  abs_tol_linear_convective = 1.e-20;
  rel_tol_linear_convective = 1.e-3;
  max_iter_linear_convective = 1e4;
  use_right_preconditioning_convective = true;
  max_n_tmp_vectors_convective = 100;

  // stability in the limit of small time steps and projection step
  small_time_steps_stability = false;

  // viscous step
  solver_viscous = SolverViscous::PCG;
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //GeometricMultigrid;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_viscous = 1.e-12;
  rel_tol_viscous = 1.e-6;


  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum.abs_tol = 1.e-12;
  newton_solver_data_momentum.rel_tol = 1.e-6;
  newton_solver_data_momentum.max_iter = 100;

  // linear solver
  solver_momentum = SolverMomentum::GMRES; //GMRES; //FGMRES;
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix; //InverseMassMatrix; //VelocityDiffusion; //VelocityConvectionDiffusion;
//  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  update_preconditioner_momentum = true;
  multigrid_data_momentum.smoother = MultigridSmoother::Jacobi;
  multigrid_data_momentum.jacobi_smoother_data.preconditioner = PreconditionerJacobiSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  multigrid_data_momentum.jacobi_smoother_data.number_of_smoothing_steps = 5;
  multigrid_data_momentum.jacobi_smoother_data.damping_factor = 0.7;
  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::GMRES_NoPreconditioner;
  abs_tol_momentum_linear = 1.e-12;
  rel_tol_momentum_linear = 1.e-2;
  max_iter_momentum_linear = 1e4;
  use_right_preconditioning_momentum = true;
  max_n_tmp_vectors_momentum = 100;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled.abs_tol = 1.e-12;
  newton_solver_data_coupled.rel_tol = 1.e-6;
  newton_solver_data_coupled.max_iter = 100;

  // linear solver
  solver_linearized_navier_stokes = SolverLinearizedNavierStokes::FGMRES; //FGMRES;
  abs_tol_linear = 1.e-12;
  rel_tol_linear = 1.e-2;
  max_iter_linear = 1e4;
  max_n_tmp_vectors = 1000;

  // preconditioning linear solver
  preconditioner_linearized_navier_stokes = PreconditionerLinearizedNavierStokes::BlockTriangular;
  update_preconditioner = true;

  // preconditioner velocity/momentum block
  momentum_preconditioner = MomentumPreconditioner::VelocityDiffusion; //VelocityConvectionDiffusion;
  multigrid_data_momentum_preconditioner.smoother = MultigridSmoother::Jacobi; //Jacobi; //Chebyshev; //GMRES;

  // GMRES smoother data
  multigrid_data_momentum_preconditioner.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  multigrid_data_momentum_preconditioner.gmres_smoother_data.number_of_iterations = 5;

  // Jacobi smoother data
  multigrid_data_momentum_preconditioner.jacobi_smoother_data.preconditioner = PreconditionerJacobiSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  multigrid_data_momentum_preconditioner.jacobi_smoother_data.number_of_smoothing_steps = 5;
  multigrid_data_momentum_preconditioner.jacobi_smoother_data.damping_factor = 0.7;

  multigrid_data_momentum_preconditioner.coarse_solver = MultigridCoarseGridSolver::GMRES_NoPreconditioner; //NoPreconditioner; //Chebyshev; //Chebyshev; //ChebyshevNonsymmetricOperator;

  exact_inversion_of_momentum_block = false;
  rel_tol_solver_momentum_preconditioner = 1.e-6;
  max_n_tmp_vectors_solver_momentum_preconditioner = 100;

  // preconditioner Schur-complement block
  schur_complement_preconditioner = SchurComplementPreconditioner::PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  multigrid_data_schur_complement_preconditioner.chebyshev_smoother_data.smoother_poly_degree = 5;
  multigrid_data_schur_complement_preconditioner.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  exact_inversion_of_laplace_operator = false;
  rel_tol_solver_schur_complement_preconditioner = 1.e-6;


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  print_input_parameters = true;
  output_data.write_output = true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/200;
  output_data.write_divergence = true;
  output_data.write_streamfunction = false;
  output_data.number_of_patches = FE_DEGREE_VELOCITY;

  // output of solver information
  output_solver_info_every_timesteps = 1; //1e3;
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Analytical solution velocity:
 *
 *  - This function is used to calculate the L2 error
 *
 *  - This function can be used to prescribe initial conditions for the velocity field
 *
 *  - Moreover, this function can be used (if possible for simple geometries)
 *    to prescribe Dirichlet BC's for the velocity field on Dirichlet boundaries
 */
template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity (const unsigned int  n_components = dim,
                              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~AnalyticalSolutionVelocity(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double AnalyticalSolutionVelocity<dim>::value(const Point<dim>   &p,
                                              const unsigned int component) const
{
  double t = this->get_time();
  double result = 0.0;

  // constant velocity
  if(PROBLEM_TYPE == ProblemType::Steady)
  {
    if(component == 0 && (std::abs(p[0])<1.0e-15))
      result = MAX_VELOCITY;
  }
  else if(PROBLEM_TYPE == ProblemType::Unsteady)
  {
    const double T = 0.5;
    const double pi = numbers::PI;
    if(component == 0 && (std::abs(p[0])<1.0e-15))
    {
      result = t<T ? std::sin(pi/2.*t/T) : 1.0;

      result *= MAX_VELOCITY; //*(1.0 - std::pow(p[1]/L+0.5,2.0));
    }
  }

  return result;
}

/*
 *  Analytical solution pressure
 *
 *  - It is used to calculate the L2 error
 *
 *  - It is used to adjust the pressure level in case of pure Dirichlet BC's
 *    (where the pressure is only defined up to an additive constant)
 *
 *  - This function can be used to prescribe initial conditions for the pressure field
 *
 *  - Moreover, this function can be used (if possible for simple geometries)
 *    to prescribe Dirichlet BC's for the pressure field on Neumann boundaries
 */
template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  virtual ~AnalyticalSolutionPressure(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const;
};

template<int dim>
double AnalyticalSolutionPressure<dim>::value(const Point<dim>    &p,
                                              const unsigned int  /* component */) const
{
  double result = 0.0;
  return result;
}


/*
 *  Neumann boundary conditions for velocity
 *
 *  - Laplace formulation of viscous term
 *    -> prescribe velocity gradient (grad U)*n on Gamma_N
 *
 *  - Divergence formulation of viscous term
 *    -> prescribe (grad U + (grad U) ^T)*n on Gamma_N
 */
template<int dim>
class NeumannBoundaryVelocity : public Function<dim>
{
public:
  NeumannBoundaryVelocity (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  virtual ~NeumannBoundaryVelocity(){};

  virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
};

template<int dim>
double NeumannBoundaryVelocity<dim>::value(const Point<dim> &/*p*/,const unsigned int /*component*/) const
{
  double result = 0.0;
  return result;
}

/*
 *  PressureBC_dudt:
 *
 *  This functions is only used when applying the high-order dual splitting scheme and
 *  is evaluated on Dirichlet boundaries (where the velocity is prescribed).
 *  Hence, this is the function that is set in the dirichlet_bc map of boundary_descriptor_pressure.
 *
 *  Note:
 *    When using a couples solution approach we do not have to evaluate something like
 *    pressure Neumann BC's on Dirichlet boundaries (we only have p⁺ = p⁻ on Dirichlet boundaries,
 *    i.e., no boundary data used). So it doesn't matter when writing this function into the
 *    dirichlet_bc map of boundary_descriptor_pressure because this function will never be evaluated
 *    in case of a coupled solution approach.
 *
 */
template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  virtual ~PressureBC_dudt(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double PressureBC_dudt<dim>::value(const Point<dim>   &/*p*/,
                                   const unsigned int /*component*/) const
{
  // do nothing (result = 0) since we are interested in a steady state solution
  double result = 0.0;
  return result;
}

/*
 *  Right-hand side function: Implements the body force vector occuring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations
 */
template<int dim>
 class RightHandSide : public Function<dim>
 {
 public:
   RightHandSide (const double time = 0.)
     :
     Function<dim>(dim, time)
   {}

   virtual ~RightHandSide(){};

   virtual double value (const Point<dim>    &p,
                         const unsigned int  component = 0) const;
 };

 template<int dim>
 double RightHandSide<dim>::value(const Point<dim>   &/*p*/,
                                  const unsigned int /*component*/) const
 {
   double result = 0.0;
   return result;
 }


/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

 template<int dim>
 void create_grid_and_set_boundary_conditions(
     parallel::distributed::Triangulation<dim>         &triangulation,
     unsigned int const                                n_refine_space,
     std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
     std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
     std::vector<GridTools::PeriodicFacePair<typename
       Triangulation<dim>::cell_iterator> >            &/*periodic_faces*/)
 {
  if(dim == 2)
  {
    Triangulation<dim> tria_h1, tria_h2, tria_h3, tria_v1, tria_v2, tria_v3, tria_v4, tria_v5;

    GridGenerator::subdivided_hyper_rectangle(tria_h1,
                                              std::vector<unsigned int>({4,1}),
                                              Point<dim>(0.0,0.0),
                                              Point<dim>(4.0*L,-1.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_h2,
                                              std::vector<unsigned int>({1,1}),
                                              Point<dim>(4.0*L,-4.0*L),
                                              Point<dim>(5.0*L,-5.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_h3,
                                              std::vector<unsigned int>({5,1}),
                                              Point<dim>(5.0*L,0.0*L),
                                              Point<dim>(10.0*L,-1.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_v1,
                                              std::vector<unsigned int>({1,4}),
                                              Point<dim>(1.0*L,-1.0*L),
                                              Point<dim>(2.0*L,-5.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_v2,
                                              std::vector<unsigned int>({1,4}),
                                              Point<dim>(3.0*L,-1.0*L),
                                              Point<dim>(4.0*L,-5.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_v3,
                                              std::vector<unsigned int>({1,4}),
                                              Point<dim>(5.0*L,-1.0*L),
                                              Point<dim>(6.0*L,-5.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_v4,
                                              std::vector<unsigned int>({1,4}),
                                              Point<dim>(7.0*L,-1.0*L),
                                              Point<dim>(8.0*L,-5.0*L));

    GridGenerator::subdivided_hyper_rectangle(tria_v5,
                                              std::vector<unsigned int>({1,4}),
                                              Point<dim>(9.0*L,-1.0*L),
                                              Point<dim>(10.0*L,-5.0*L));

    // merge
    Triangulation<dim> tmp1, tmp2;
    GridGenerator::merge_triangulations (tria_h1, tria_v1, tmp1);
    GridGenerator::merge_triangulations (tmp1, tria_v2, tmp2);
    GridGenerator::merge_triangulations (tmp2, tria_h2, tmp1);
    GridGenerator::merge_triangulations (tmp1, tria_v3, tmp2);
    GridGenerator::merge_triangulations (tmp2, tria_h3, tmp1);
    GridGenerator::merge_triangulations (tmp1, tria_v4, tmp2);
    GridGenerator::merge_triangulations (tmp2, tria_v5, triangulation);

    // global refinements
    triangulation.refine_global(n_refine_space);
  }
  else if(dim == 3)
  {
    AssertThrow(false, ExcMessage("NotImplemented"));
  }

  // set boundary indicator
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
      // inflow boundary
     if ((std::fabs(cell->face(face_number)->center()(0))< 1e-12))
        cell->face(face_number)->set_boundary_id (1);

     // outflow boundary
     if ((std::fabs(cell->face(face_number)->center()(1) - (-5.0*L))< 1e-12) && (cell->face(face_number)->center()(0)- 9.0*L)>=0)
        cell->face(face_number)->set_boundary_id (2);
    }
  }

  // all boundaries have ID = 0 by default -> Dirichlet boundaries

  // fill boundary descriptor velocity
  std::shared_ptr<Function<dim> > zero_velocity;
  zero_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  // walls: ID = 0
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                    (0,zero_velocity));

  std::shared_ptr<Function<dim> > analytical_solution_velocity;
  analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  // inflow: ID = 1
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                    (1,analytical_solution_velocity));

  std::shared_ptr<Function<dim> > neumann_bc_velocity;
  neumann_bc_velocity.reset(new NeumannBoundaryVelocity<dim>());
  // outflow: ID = 2
  boundary_descriptor_velocity->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                   (2,neumann_bc_velocity));

  // fill boundary descriptor pressure
  std::shared_ptr<Function<dim> > zero_pressure;
  zero_pressure.reset(new PressureBC_dudt<dim>(1));
  // walls: ID = 0
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                    (0,zero_pressure));

  // inflow: ID = 1
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                    (1,zero_pressure));

  // outflow: ID = 2
  boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                    (2,zero_pressure));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  std::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));

  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure = initial_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new Functions::ZeroFunction<dim>(1));
}

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

template<int dim, int degree_u, int degree_p, typename Number>
std::shared_ptr<PostProcessorBase<dim, degree_u, degree_p, Number> >
construct_postprocessor(InputParameters<dim> const &param)
{
  PostProcessorData<dim> pp_data;

  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;
  pp_data.kinetic_energy_data = param.kinetic_energy_data;
  pp_data.line_plot_data = param.line_plot_data;

  std::shared_ptr<PostProcessor<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessor<dim,degree_u,degree_p,Number>(pp_data));

  return pp;
}



#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TUM_H_ */
