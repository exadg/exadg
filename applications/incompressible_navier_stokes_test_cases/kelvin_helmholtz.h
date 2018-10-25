/*
 * kelvin_helmholtz.h
 *
 *  Created on: Aug 31, 2017
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_


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
unsigned int const FE_DEGREE_VELOCITY = 11;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set xwall specific parameters
unsigned int const FE_DEGREE_XWALL = 1;
unsigned int const N_Q_POINTS_1D_XWALL = 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 1;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const double Re = 1.0e4;
const double L = 1.0;
const double DELTA_0 = 1./28.;
const double U_INF = 1.0;
const double C_N = 1.0e-3;
const double PI = numbers::PI;
const double MAX_VELOCITY = U_INF;
const double VISCOSITY = U_INF*DELTA_0/Re;
const double T = DELTA_0/U_INF;

// set output folders & names
std::string OUTPUT_FOLDER = "output/kelvin_helmholtz/test/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "Re1e4_l1_k1110";

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  right_hand_side = false;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 400.0*T;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme; //BDFDualSplittingScheme; //BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL; //ConstTimeStepUserSpecified;
  max_velocity = MAX_VELOCITY;
  cfl = 0.1;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-2;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true;


  // SPATIAL DISCRETIZATION

  // spatial discretization method
  spatial_discretization = SpatialDiscretization::DG;

  // convective term - currently no parameters

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  IP_factor_viscous = 1.0;
  penalty_term_div_formulation = PenaltyTermDivergenceFormulation::Symmetrized;

  // gradient term
  gradp_integrated_by_parts = true;
  gradp_use_boundary_data = true;

  // divergence term
  divu_integrated_by_parts = true;
  divu_use_boundary_data = true;

  // special case: pure DBC's
  pure_dirichlet_bc = true;

  // divergence and continuity penalty terms
  use_divergence_penalty = true;
  divergence_penalty_factor = 1.0e0;
  use_continuity_penalty = true;
  continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  type_penalty_parameter = TypePenaltyParameter::ConvectiveTerm;
  continuity_penalty_factor = divergence_penalty_factor;

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
  preconditioner_viscous = PreconditionerViscous::GeometricMultigrid;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_viscous = 1.e-12;
  rel_tol_viscous = 1.e-6;


  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum.abs_tol = 1.e-14;
  newton_solver_data_momentum.rel_tol = 1.e-6;
  newton_solver_data_momentum.max_iter = 100;

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  abs_tol_momentum_linear = 1.e-20;
  rel_tol_momentum_linear = 1.e-6;
  max_iter_momentum_linear = 1e4;
  use_right_preconditioning_momentum = true;
  max_n_tmp_vectors_momentum = 100;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled.abs_tol = 1.e-10;
  newton_solver_data_coupled.rel_tol = 1.e-6;
  newton_solver_data_coupled.max_iter = 1e2;

  // linear solver
  solver_linearized_navier_stokes = SolverLinearizedNavierStokes::GMRES;
  abs_tol_linear = 1.e-12;
  rel_tol_linear = 1.e-6;
  max_iter_linear = 1e4;
  max_n_tmp_vectors = 200;

  // preconditioning linear solver
  preconditioner_linearized_navier_stokes = PreconditionerLinearizedNavierStokes::BlockTriangular;
  update_preconditioner = true;

  // preconditioner velocity/momentum block
  momentum_preconditioner = MomentumPreconditioner::InverseMassMatrix; //VelocityConvectionDiffusion;
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
  output_data.write_output = false;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/200;
  output_data.write_divergence = true;
  output_data.write_vorticity_magnitude = true;
  output_data.write_processor_id = true;
  output_data.number_of_patches = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  output_solver_info_every_timesteps = 1; //1e2;

  // kinetic energy
  kinetic_energy_data.calculate = true;
  kinetic_energy_data.calculate_every_time_steps = 1;
  kinetic_energy_data.viscosity = VISCOSITY;
  kinetic_energy_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;
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
  double result = 0.0;

  double const x1 = p[0];
  double const x2 = p[1];

  if(component == 0)
  {
    double const dpsidx2 = - C_N*U_INF*std::exp(-std::pow(x2-0.5,2.0)/std::pow(DELTA_0,2.0))*2.0*(x2-0.5)/std::pow(DELTA_0,2.0)
                             *(std::cos(8.0*PI*x1) + std::cos(20.0*PI*x1));

    result = U_INF*std::tanh((2.0*x2-1.0)/DELTA_0) + dpsidx2;
  }
  else if(component == 1)
  {
    double const dpsidx1 = - C_N*U_INF*std::exp(-std::pow(x2-0.5,2.0)/std::pow(DELTA_0,2.0))*PI*(8.0*std::sin(8.0*PI*x1)+20.0*std::sin(20.0*PI*x1));

    result = - dpsidx1;
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
double AnalyticalSolutionPressure<dim>::value(const Point<dim>    &/*p*/,
                                              const unsigned int  /* component */) const
{
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
    parallel::distributed::Triangulation<dim>         &triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  std::vector<unsigned int> repetitions({1,1});
  Point<dim> point1(0.0,0.0), point2(L,L);
  GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,point1,point2);

//  std::vector<unsigned int> repetitions({2,2});
//  Point<dim> point1(0.0,0.0), point2(L,L);
//  GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,point1,point2);
//    for (Triangulation<2>::cell_iterator cell = triangulation.begin(); cell != triangulation.end(); ++cell)
//    {
//      for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
//      {
//        Point<2> &vertex = cell->vertex(v);
//        if (std::abs(vertex[0] - 0.5*L) < 1.e-8 && std::abs(vertex[1] - 0.5*L) < 1.e-8)
//        {
//          vertex = Point<2>(0.6*L, 0.7*L);
//        }
//      }
//    }

  //periodicity in x-direction
  //add 10 to avoid conflicts with dirichlet boundary, which is 0
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
     if ((std::fabs(cell->face(face_number)->center()(0) - 0.0)< 1e-12))
         cell->face(face_number)->set_boundary_id (0+10);
     if ((std::fabs(cell->face(face_number)->center()(0) - L)< 1e-12))
        cell->face(face_number)->set_boundary_id (1+10);
    }
  }
  GridTools::collect_periodic_faces(triangulation, 0+10, 1+10, 0, periodic_faces);
  triangulation.add_periodicity(periodic_faces);

  triangulation.refine_global(n_refine_space);

  // fill boundary descriptor velocity
  std::shared_ptr<Function<dim> > zero_function;
  zero_function.reset(new Functions::ZeroFunction<dim>(dim));
  // Symmetry boundaries: ID = 0
  boundary_descriptor_velocity->symmetry_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                   (0,zero_function));

  // fill boundary descriptor pressure
  std::shared_ptr<Function<dim> > pressure_bc_dudt;
  pressure_bc_dudt.reset(new Functions::ZeroFunction<dim>(dim));
  // Neumann boundaries: ID = 0
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                     (0,pressure_bc_dudt));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());

  std::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());

  std::shared_ptr<Function<dim> > analytical_solution_pressure;
  analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());

  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  field_functions->analytical_solution_pressure = analytical_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new AnalyticalSolutionVelocity<dim>());
  analytical_solution->pressure.reset(new AnalyticalSolutionPressure<dim>());
}

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, FE_DEGREE_VELOCITY, FE_DEGREE_PRESSURE, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, Number> >
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

  std::shared_ptr<PostProcessor<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE,FE_DEGREE_XWALL,N_Q_POINTS_1D_XWALL,Number> > pp;
  pp.reset(new PostProcessor<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE,FE_DEGREE_XWALL,N_Q_POINTS_1D_XWALL,Number>(pp_data));

  return pp;
}


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_ */
