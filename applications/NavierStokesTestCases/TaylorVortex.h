/*
 * TaylorVortex.h
 *
 *  Created on: Aug 19, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_NAVIERSTOKESTESTCASES_TAYLORVORTEX_H_
#define APPLICATIONS_NAVIERSTOKESTESTCASES_TAYLORVORTEX_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

//Taylor vortex problem (Shahbazi et al.,2007)

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 4;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1; // FE_DEGREE_VELOCITY; // FE_DEGREE_VELOCITY - 1;

// set xwall specific parameters
unsigned int const FE_DEGREE_XWALL = 1;
unsigned int const N_Q_POINTS_1D_XWALL = 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 2;
unsigned int const REFINE_STEPS_SPACE_MAX = 5;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const double VISCOSITY = 0.01;

template<int dim>
void InputParametersNavierStokes<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::DivergenceFormulation;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 1.0;
  viscosity = VISCOSITY; // VISCOSITY is also needed somewhere else


  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepUserSpecified;
  max_velocity = 1.0;
  cfl = 1.0e-1;
  time_step_size = 1.0e-2;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 3; // 1; // 2; // 3;
  start_with_low_order = false; // true; // false;


  // SPATIAL DISCRETIZATION

  // spatial discretization method
  spatial_discretization = SpatialDiscretization::DG;

  // convective term - currently no parameters

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  IP_factor_viscous = 1.0;

  // gradient term
  gradp_integrated_by_parts = true;
  gradp_use_boundary_data = true;

  // divergence term
  divu_integrated_by_parts = true;
  divu_use_boundary_data = true;

  // special case: pure DBC's
  pure_dirichlet_bc = true;

  // PROJECTION METHODS

  // pressure Poisson equation
  IP_factor_pressure = 1.0;
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::GeometricMultigrid;
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_pressure = 1.e-20;
  rel_tol_pressure = 1.e-6;

  // stability in the limit of small time steps
  use_approach_of_ferrer = false;
  deltat_ref = 1.e0;

  // projection step
  solver_projection = SolverProjection::PCG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  abs_tol_projection = 1.e-20;
  rel_tol_projection = 1.e-12;

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
  preconditioner_viscous = PreconditionerViscous::GeometricMultigrid;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_viscous = 1.e-20;
  rel_tol_viscous = 1.e-6;


  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum.abs_tol = 1.e-20;
  newton_solver_data_momentum.rel_tol = 1.e-6;
  newton_solver_data_momentum.max_iter = 100;

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  preconditioner_momentum = PreconditionerMomentum::VelocityDiffusion; //InverseMassMatrix; //VelocityConvectionDiffusion;
  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_momentum_linear = 1.e-20;
  rel_tol_momentum_linear = 1.e-4;
  max_iter_momentum_linear = 1e4;
  use_right_preconditioning_momentum = true;
  max_n_tmp_vectors_momentum = 100;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled.abs_tol = 1.e-20;
  newton_solver_data_coupled.rel_tol = 1.e-6;
  newton_solver_data_coupled.max_iter = 1e2;

  // linear solver
  solver_linearized_navier_stokes = SolverLinearizedNavierStokes::GMRES;
  abs_tol_linear = 1.e-12;
  rel_tol_linear = 1.e-6;
  max_iter_linear = 1e4;

  // preconditioning linear solver
  preconditioner_linearized_navier_stokes = PreconditionerLinearizedNavierStokes::BlockTriangular;

  // preconditioner velocity/momentum block
  momentum_preconditioner = MomentumPreconditioner::VelocityConvectionDiffusion;
  exact_inversion_of_momentum_block = false;
  rel_tol_solver_momentum_preconditioner = 1.e-6;
  max_n_tmp_vectors_solver_momentum_preconditioner = 100;

  // preconditioner Schur-complement block
  schur_complement_preconditioner = SchurComplementPreconditioner::CahouetChabard;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  exact_inversion_of_laplace_operator = false;
  rel_tol_solver_schur_complement_preconditioner = 1.e-6;


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_prefix = "taylor_vortex";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.compute_divergence = true;
  output_data.number_of_patches = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = true;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  output_solver_info_every_timesteps = 1e5;

  // restart
  write_restart = false;
  restart_interval_time = 1.e2;
  restart_interval_wall_time = 1.e6;
  restart_every_timesteps = 1e8;
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

  const double pi = numbers::PI;
  if(component == 0)
    result = (-std::cos(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
  else if(component == 1)
    result = (+std::sin(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);

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
  double t = this->get_time();
  double result = 0.0;

  const double pi = numbers::PI;
  result = -0.25*(std::cos(2*pi*p[0])+std::cos(2*pi*p[1]))*std::exp(-4.0*pi*pi*t*VISCOSITY);

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
double NeumannBoundaryVelocity<dim>::value(const Point<dim> &p,const unsigned int component) const
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
double PressureBC_dudt<dim>::value(const Point<dim>   &p,
                                   const unsigned int component) const
{
  double t = this->get_time();
  double result = 0.0;

  const double pi = numbers::PI;
  if(component == 0)
    result = + 2.0*pi*pi*VISCOSITY*std::cos(pi*p[0])*std::sin(pi*p[1])*std::exp(-2.0*pi*pi*t*VISCOSITY);
  else if(component == 1)
    result = - 2.0*pi*pi*VISCOSITY*std::sin(pi*p[0])*std::cos(pi*p[1])*std::exp(-2.0*pi*pi*t*VISCOSITY);

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
 double RightHandSide<dim>::value(const Point<dim>   &p,
                                  const unsigned int component) const
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
    parallel::distributed::Triangulation<dim>                   &triangulation,
    unsigned int const                                          n_refine_space,
    std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity,
    std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >                      &periodic_faces)
{
  const double left = -1.0, right = 1.0;
  GridGenerator::hyper_cube(triangulation,left,right);
  triangulation.refine_global(n_refine_space);

  // all boundaries have ID = 0 by default -> Dirichlet boundaries

  // fill boundary descriptor velocity
  std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
  analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  // Dirichlet boundaries: ID = 0
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (0,analytical_solution_velocity));

  // fill boundary descriptor pressure
  std_cxx11::shared_ptr<Function<dim> > pressure_bc_dudt;
  pressure_bc_dudt.reset(new PressureBC_dudt<dim>());
  // Dirichlet boundaries: ID = 0
  boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (0,pressure_bc_dudt));
}


template<int dim>
void set_field_functions(std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
  analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  std_cxx11::shared_ptr<Function<dim> > analytical_solution_pressure;
  analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());

  std_cxx11::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->initial_solution_velocity = analytical_solution_velocity;
  field_functions->initial_solution_pressure = analytical_solution_pressure;
  field_functions->analytical_solution_pressure = analytical_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_analytical_solution(std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new AnalyticalSolutionVelocity<dim>());
  analytical_solution->pressure.reset(new AnalyticalSolutionPressure<dim>());
}

template<int dim>
std_cxx11::shared_ptr<PostProcessorBase<dim> >
construct_postprocessor(InputParametersNavierStokes<dim> const &param)
{
  PostProcessorData<dim> pp_data;

  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;

  std_cxx11::shared_ptr<PostProcessor<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE> > pp;
  pp.reset(new PostProcessor<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE>(pp_data));

  return pp;
}

#endif /* APPLICATIONS_NAVIERSTOKESTESTCASES_TAYLORVORTEX_H_ */
