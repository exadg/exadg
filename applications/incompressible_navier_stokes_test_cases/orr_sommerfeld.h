/*
 * orr_sommerfeld.h
 *
 *  Created on: Aug 31, 2017
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_


#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include "../include/incompressible_navier_stokes/postprocessor/orr_sommerfeld_equation.h"

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
unsigned int const FE_DEGREE_VELOCITY = 5;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;  // FE_DEGREE_VELOCITY; // FE_DEGREE_VELOCITY - 1;

// set xwall specific parameters
unsigned int const FE_DEGREE_XWALL = 1;
unsigned int const N_Q_POINTS_1D_XWALL = 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 5;
unsigned int const REFINE_STEPS_SPACE_MAX = 5; //REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const ProblemType PROBLEM_TYPE = ProblemType::Unsteady;

const double Re = 7500.0;

const double H = 1.0;
const double PI = numbers::PI;
const double L = 2.0*PI*H;

const double MAX_VELOCITY = 1.0;
const double VISCOSITY = MAX_VELOCITY*H/Re;
const double ALPHA = 1.0;
const double EPSILON = 1.0e-5; //perturbations are small (<< 1, linearization)

// Orr-Sommerfeld solver: calculates unstable eigenvalue (OMEGA) of
// Orr-Sommerfeld equation for Poiseuille flow and corresponding
// eigenvector (EIG_VEC).
const unsigned int DEGREE_OS_SOLVER = 200; // use not more than 300 due to conditioning of polynomials
FE_DGQ<1> FE(DEGREE_OS_SOLVER);
std::complex<double> OMEGA;
std::vector<std::complex<double> > EIG_VEC(DEGREE_OS_SOLVER+1);

std::string OUTPUT_FOLDER = "output/orr_sommerfeld/test/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "Re7500_l5_21_div_conti";
std::string FILENAME_ENERGY = "perturbation_energy_ku2_kp1";

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // solve Orr-Sommerfeld equation
  compute_eigenvector(EIG_VEC,OMEGA,Re,ALPHA,FE);

  // MATHEMATICAL MODEL
  problem_type = PROBLEM_TYPE; // PROBLEM_TYPE is also needed somewhere else
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  right_hand_side = true; //prescribe body force in x-direction in case of perodic BC's


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  // the time the T-S-waves need to travel through the domain
  double t0 = 2.0*PI*ALPHA/OMEGA.real();
  end_time = 2.0*t0;
  viscosity = VISCOSITY; // VISCOSITY is also needed somewhere else


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFCoupledSolution; //BDFDualSplittingScheme; //BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL; //ConstTimeStepUserSpecified;
  max_velocity = MAX_VELOCITY; // MAX_VELOCITY is also needed somewhere else
  // approx. cfl_crit = 0.9 for BDF2 (and exponent = 1.5)
  // approx. cfl_crit = 0.4 for BDF3 (and exponent = 1.5)
  cfl = 0.6;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-2;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true; // true; // false;


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
  continuity_penalty_use_boundary_data = false;
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
  output_data.write_output = true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.write_divergence = true;
  output_data.number_of_patches = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = false; //true;

  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  output_solver_info_every_timesteps = 1e2; //1e5;

  // perturbation energy
  perturbation_energy_data.calculate = true;
  perturbation_energy_data.calculate_every_time_steps = 1;
  perturbation_energy_data.filename_prefix = OUTPUT_FOLDER + FILENAME_ENERGY;
  perturbation_energy_data.U_max = MAX_VELOCITY;
  perturbation_energy_data.h = H;
  perturbation_energy_data.omega_i = OMEGA.imag();
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

  double const x = p[0]/H;
  // transform from interval [-H,H] (-> y) to unit interval [0,1] (-> eta)
  double const eta = 0.5*(p[1]/H + 1.0);
  double const tol = 1.e-12;
  AssertThrow(eta<=1.0+tol and eta>=0.0-tol, ExcMessage("Point in reference coordinates is invalid."));

  double cos = std::cos(ALPHA*x-OMEGA.real()*t);
  double sin = std::sin(ALPHA*x-OMEGA.real()*t);
  double amplification = std::exp(OMEGA.imag()*t);
  std::complex<double> exp(cos,sin);

  if(component == 0)
  {
    double base = MAX_VELOCITY * (1.0 - pow(p[1]/H,2.0));

    // d(psi)/dy = d(psi)/d(eta) * d(eta)/dy
    // evaluate derivative d(psi)/d(eta) in eta(y)
    std::complex<double> dpsi = 0;
    for (unsigned int i=0; i<FE.get_degree()+1; ++i)
      dpsi += EIG_VEC[i] * FE.shape_grad(i,Point<1>(eta))[0];

    // multiply by d(eta)/dy to obtain derivative d(psi)/dy in physical space
    dpsi *= 0.5/H;

    std::complex<double> perturbation_complex = dpsi*exp*amplification;
    double perturbation = perturbation_complex.real();

    result = base + EPSILON*perturbation;
  }
  else if(component == 1)
  {
    // evaluate function psi in y
    std::complex<double> psi = 0;
    for (unsigned int i=0; i<FE.get_degree()+1; ++i)
      psi += EIG_VEC[i] * FE.shape_value(i,Point<1>(eta));

    std::complex<double> i(0,1);
    std::complex<double> perturbation_complex = -i*ALPHA*psi*exp*amplification;
    double perturbation = perturbation_complex.real();

    result = EPSILON*perturbation;
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

   // mean flow is driven by a constant body force since we use
   // periodic BC's in streamwise direction.
   // Body force is derived by a balance of forces in streamwise direction
   //   f * L * 2H = tau * 2 * L (2H = height, L = length, factor 2 = upper and lower wall)
   // with tau = nu du/dy|_{y=-H} = nu * U_max * (-2y/H^2)|_{y=-H} = 2 * nu * U_max / H
   if(component==0)
     result = 2.*VISCOSITY*MAX_VELOCITY/(H*H);

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
  Point<dim> point1(0.0,-H), point2(L,H);
  GridGenerator::subdivided_hyper_rectangle(triangulation,repetitions,point1,point2);

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
  std::shared_ptr<Function<dim> > analytical_solution_velocity;
  analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  // Dirichlet boundaries: ID = 0
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                     (0,analytical_solution_velocity));

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
#include "../../include/incompressible_navier_stokes/postprocessor/perturbation_energy_orr_sommerfeld.h"

template<int dim>
struct PostProcessorDataOrrSommerfeld
{
  PostProcessorData<dim> pp_data;
  PerturbationEnergyData energy_data;
};

template<int dim, int fe_degree_u, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename Number>
class PostProcessorOrrSommerfeld : public PostProcessor<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>
{
public:
  typedef PostProcessor<dim, fe_degree_u, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> Base;

  typedef typename Base::NavierStokesOperator NavierStokesOperator;

  PostProcessorOrrSommerfeld(PostProcessorDataOrrSommerfeld<dim> const & pp_data_os)
    :
    Base(pp_data_os.pp_data),
    energy_data(pp_data_os.energy_data)
  {}

  void setup(NavierStokesOperator const                &navier_stokes_operator_in,
             DoFHandler<dim> const                     &dof_handler_velocity_in,
             DoFHandler<dim> const                     &dof_handler_pressure_in,
             Mapping<dim> const                        &mapping_in,
             MatrixFree<dim,Number> const              &matrix_free_data_in,
             DofQuadIndexData const                    &dof_quad_index_data_in,
             std::shared_ptr<AnalyticalSolution<dim> > analytical_solution_in)
  {
    // call setup function of base class
    Base::setup(
        navier_stokes_operator_in,
        dof_handler_velocity_in,
        dof_handler_pressure_in,
        mapping_in,
        matrix_free_data_in,
        dof_quad_index_data_in,
        analytical_solution_in);

    energy_calculator.setup(matrix_free_data_in,
                            dof_quad_index_data_in,
                            energy_data);
  }

  void do_postprocessing(parallel::distributed::Vector<Number> const &velocity,
                         parallel::distributed::Vector<Number> const &intermediate_velocity,
                         parallel::distributed::Vector<Number> const &pressure,
                         double const                                time,
                         int const                                   time_step_number)
  {
    Base::do_postprocessing(
        velocity,
        intermediate_velocity,
        pressure,
        time,
        time_step_number);

    energy_calculator.evaluate(velocity,time,time_step_number);
  }

  PerturbationEnergyData energy_data;
  PerturbationEnergyCalculator<dim,fe_degree_u,Number> energy_calculator;
};

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

  PostProcessorDataOrrSommerfeld<dim> pp_data_os;
  pp_data_os.pp_data = pp_data;
  pp_data_os.energy_data = param.perturbation_energy_data;

  std::shared_ptr<PostProcessorBase<dim, FE_DEGREE_VELOCITY, FE_DEGREE_PRESSURE, FE_DEGREE_XWALL, N_Q_POINTS_1D_XWALL, Number> > pp;
  pp.reset(new PostProcessorOrrSommerfeld<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE,FE_DEGREE_XWALL,N_Q_POINTS_1D_XWALL,Number>(pp_data_os));

  return pp;
}


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_ORR_SOMMERFELD_H_ */
