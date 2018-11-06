/*
 * TurbulentChannel.h
 *
 *  Created on: Oct 14, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_

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
unsigned int const DIMENSION = 3;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 3;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY - 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 3;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// can only be used for GridStretchType::TransformGridCells, otherwise coarse grid consists of 1 cell
const unsigned int N_CELLS_1D_COARSE_GRID = 1;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

std::string OUTPUT_FOLDER = "output_comp_ns/turbulent_channel/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME = "Re180_l3_k3";

// set problem specific parameters like physical dimensions, etc.
double const DIMENSIONS_X1 = 2.0*numbers::PI;
double const DIMENSIONS_X2 = 2.0;
double const DIMENSIONS_X3 = numbers::PI;

// nu = 1/180  coarsest meshes: l2_ku3 or l3_ku2
// nu = 1/395
// nu = 1/590
// nu = 1/950
double const VISCOSITY = 1./180.; // critical value: 1./50. - 1./75.

double const MAX_VELOCITY = 18.3; //18.3 for Re_tau = 180; //22.0;

// flow-through time based on mean centerline velocity
const double CHARACTERISTIC_TIME = DIMENSIONS_X1/MAX_VELOCITY;

double const START_TIME = 0.0;
double const END_TIME = 200.0*CHARACTERISTIC_TIME; // 50.0;

double const SAMPLE_START_TIME = 100.0*CHARACTERISTIC_TIME; // 30.0;
double const SAMPLE_END_TIME = END_TIME;

// use a negative GRID_STRETCH_FAC to deactivate grid stretching
const double GRID_STRETCH_FAC = 1.8;

enum class GridStretchType{ TransformGridCells, VolumeManifold };
GridStretchType GRID_STRETCH_TYPE = GridStretchType::TransformGridCells; //VolumeManifold;

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation; //LaplaceFormulation; //DivergenceFormulation;
  right_hand_side = true;


  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = END_TIME;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme; // BDFDualSplittingScheme; //BDFPressureCorrection; //BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit; //Explicit;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL; // AdaptiveTimeStepCFL
  max_velocity = MAX_VELOCITY;
  cfl = 1.7;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-1;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true;


  // SPATIAL DISCRETIZATION

  degree_mapping = FE_DEGREE_VELOCITY;

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

  // div-div and continuity penalty
  use_divergence_penalty = true;
  divergence_penalty_factor = 1.0e0;
  use_continuity_penalty = true;
  continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  type_penalty_parameter = TypePenaltyParameter::ConvectiveTerm;
  continuity_penalty_factor = divergence_penalty_factor;

  // TURBULENCE
  use_turbulence_model = false;
  turbulence_model = TurbulenceEddyViscosityModel::Sigma;
  // Smagorinsky: 0.165
  // Vreman: 0.28
  // WALE: 0.50
  // Sigma: 1.35
  turbulence_model_constant = 1.35;

  // PROJECTION METHODS

  // pressure Poisson equation
  IP_factor_pressure = 1.0;
  solver_pressure_poisson = SolverPressurePoisson::PCG;
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::GeometricMultigrid;
  multigrid_data_pressure_poisson.smoother = MultigridSmoother::Chebyshev; //Chebyshev; //Jacobi; //GMRES;

  //Chebyshev
//  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::AMG_ML;

  //GMRES
//  multigrid_data_pressure_poisson.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::None; //None; //PointJacobi; //BlockJacobi;
//  multigrid_data_pressure_poisson.gmres_smoother_data.number_of_iterations = 4;
//  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::GMRES_NoPreconditioner;

  //Jacobi
//  multigrid_data_pressure_poisson.jacobi_smoother_data.preconditioner = PreconditionerJacobiSmoother::PointJacobi;
//  multigrid_data_pressure_poisson.jacobi_smoother_data.number_of_smoothing_steps = 5;
//  multigrid_data_pressure_poisson.jacobi_smoother_data.damping_factor = 0.7;
//  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::GMRES_NoPreconditioner; //Chebyshev; //PCG_NoPreconditioner; //GMRES_NoPreconditioner;

  abs_tol_pressure = 1.e-12;
  rel_tol_pressure = 1.e-3; //1.e-6;

  // stability in the limit of small time steps
  use_approach_of_ferrer = false;
  deltat_ref = 1.e0;

  // projection step
  solver_projection = SolverProjection::PCG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix; //BlockJacobi; //PointJacobi; //InverseMassMatrix;
  update_preconditioner_projection = true;
  abs_tol_projection = 1.e-12;
  rel_tol_projection = 1.e-3; //1.e-6;



  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // convective step

  // nonlinear solver
  newton_solver_data_convective.abs_tol = 1.e-12;
  newton_solver_data_convective.rel_tol = 1.e-6;
  newton_solver_data_convective.max_iter = 100;
  // linear solver
  abs_tol_linear_convective = 1.e-12;
  rel_tol_linear_convective = 1.e-6;
  max_iter_linear_convective = 1e4;
  use_right_preconditioning_convective = true;
  max_n_tmp_vectors_convective = 100;

  // stability in the limit of small time steps and projection step
  small_time_steps_stability = false;

  // viscous step
  solver_viscous = SolverViscous::PCG;
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix; //GeometricMultigrid;
  multigrid_data_viscous.smoother = MultigridSmoother::GMRES;
  // MG smoother data: GMRES smoother
  multigrid_data_viscous.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::BlockJacobi; //None; //PointJacobi; //BlockJacobi;
  multigrid_data_viscous.gmres_smoother_data.number_of_iterations = 4;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::PCG_NoPreconditioner; //Chebyshev;

  abs_tol_viscous = 1.e-12;
  rel_tol_viscous = 1.e-3; // 1.e-6;


  // PRESSURE-CORRECTION SCHEME

  // formulation
  order_pressure_extrapolation = 1; // use 0 for non-incremental formulation
  rotational_formulation = true;

  // momentum step

  // Newton solver
  newton_solver_data_momentum.abs_tol = 1.e-12;
  newton_solver_data_momentum.rel_tol = 1.e-6;
  newton_solver_data_momentum.max_iter = 100;

  // linear solver
  abs_tol_momentum_linear = 1.e-12;
  rel_tol_momentum_linear = 1.e-6;
  max_iter_momentum_linear = 1e4;
  use_right_preconditioning_momentum = true;
  max_n_tmp_vectors_momentum = 100;
  update_preconditioner_momentum = false;

  solver_momentum = SolverMomentum::GMRES;
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix; //InverseMassMatrix; //VelocityDiffusion; //VelocityConvectionDiffusion;
  multigrid_data_momentum.smoother = MultigridSmoother::Jacobi;

  // MG smoother data: GMRES smoother
//  multigrid_data_momentum.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::BlockJacobi; //None; //PointJacobi; //BlockJacobi;
//  multigrid_data_momentum.gmres_smoother_data.number_of_iterations = 4;

  // MG smoother data: Jacobi smoother
  multigrid_data_momentum.jacobi_smoother_data.preconditioner = PreconditionerJacobiSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  multigrid_data_momentum.jacobi_smoother_data.number_of_smoothing_steps = 4;
  multigrid_data_momentum.jacobi_smoother_data.damping_factor = 0.7;

  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::GMRES_NoPreconditioner;



  // COUPLED NAVIER-STOKES SOLVER
  use_scaling_continuity = false;
  scaling_factor_continuity = 1.0;

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled.abs_tol = 1.e-12;
  newton_solver_data_coupled.rel_tol = 1.e-6;
  newton_solver_data_coupled.max_iter = 1e2;

  // linear solver
  solver_linearized_navier_stokes = SolverLinearizedNavierStokes::GMRES; //GMRES; //FGMRES;
  abs_tol_linear = 1.e-12;
  rel_tol_linear = 1.e-6;
  max_iter_linear = 1e3;
  max_n_tmp_vectors = 100;

  // preconditioning linear solver
  preconditioner_linearized_navier_stokes = PreconditionerLinearizedNavierStokes::BlockTriangular;
  update_preconditioner = false;

  // preconditioner velocity/momentum block
  momentum_preconditioner = MomentumPreconditioner::InverseMassMatrix; //VelocityDiffusion; //VelocityConvectionDiffusion;
  multigrid_data_momentum_preconditioner.smoother = MultigridSmoother::Chebyshev;

  // MG smoother data: Chebyshev smoother
  multigrid_data_momentum_preconditioner.coarse_solver = MultigridCoarseGridSolver::Chebyshev;

  // MG smoother data: GMRES smoother
//  multigrid_data_momentum_preconditioner.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::BlockJacobi;
//  multigrid_data_momentum_preconditioner.gmres_smoother_data.number_of_iterations = 4;
//  multigrid_data_momentum_preconditioner.coarse_solver = MultigridCoarseGridSolver::GMRES_NoPreconditioner;

  // MG smoother data: Jacobi smoother
//  multigrid_data_momentum_preconditioner.jacobi_smoother_data.preconditioner = PreconditionerJacobiSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
//  multigrid_data_momentum_preconditioner.jacobi_smoother_data.number_of_smoothing_steps = 4;
//  multigrid_data_momentum_preconditioner.jacobi_smoother_data.damping_factor = 0.7;
//  multigrid_data_momentum_preconditioner.coarse_solver = MultigridCoarseGridSolver::GMRES_BlockJacobi; //GMRES_NoPreconditioner;

  exact_inversion_of_momentum_block = false;
  rel_tol_solver_momentum_preconditioner = 1.e-3;
  max_n_tmp_vectors_solver_momentum_preconditioner = 100;

  // preconditioner Schur-complement block
  schur_complement_preconditioner = SchurComplementPreconditioner::CahouetChabard; //PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;

  // Chebyshev moother
  multigrid_data_schur_complement_preconditioner.smoother = MultigridSmoother::Chebyshev;
  multigrid_data_schur_complement_preconditioner.coarse_solver = MultigridCoarseGridSolver::Chebyshev;

  // GMRES smoother
//  multigrid_data_schur_complement_preconditioner.smoother = MultigridSmoother::GMRES;
//  multigrid_data_schur_complement_preconditioner.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::None; //PointJacobi; //BlockJacobi;
//  multigrid_data_schur_complement_preconditioner.gmres_smoother_data.number_of_iterations = 4;
//  multigrid_data_schur_complement_preconditioner.coarse_solver = MultigridCoarseGridSolver::GMRES_NoPreconditioner; //PCG_NoPreconditioner;

  exact_inversion_of_laplace_operator = false;
  rel_tol_solver_schur_complement_preconditioner = 1.e-6;


  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_folder = OUTPUT_FOLDER_VTU;
  output_data.output_name = OUTPUT_NAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = 1.0;
  output_data.write_divergence = true;
  output_data.number_of_patches = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  output_solver_info_every_timesteps = 1e2; //1e3; //1e4;

  // restart
  write_restart = false;
  restart_interval_time = 1.e2;
  restart_interval_wall_time = 1.e6;
  restart_every_timesteps = 1e8;

  // calculate div and mass error
  mass_data.calculate_error = false; //true;
  mass_data.start_time = START_TIME;
  mass_data.sample_every_time_steps = 1e0; //1e2;
  mass_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;
  mass_data.reference_length_scale = 1.0;

  // turbulent channel statistics
  turb_ch_data.calculate_statistics = true;
  turb_ch_data.cells_are_stretched = false;
  if(GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
    turb_ch_data.cells_are_stretched = true;
  turb_ch_data.sample_start_time = SAMPLE_START_TIME;
  turb_ch_data.sample_end_time = SAMPLE_END_TIME;
  turb_ch_data.sample_every_timesteps = 10;
  turb_ch_data.viscosity = VISCOSITY;
  turb_ch_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME;
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
  {
    /*
     * Use the following line to obtain different initial velocity fields.
     * Without this line, the initial field is always the same as long as the
     * same number of processors is used.
     */
//    srand(std::time(NULL));
  }

  virtual ~AnalyticalSolutionVelocity(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double AnalyticalSolutionVelocity<dim>::value(const Point<dim>   &p,
                                              const unsigned int component) const
{
  double result = 0.0;

  const double tol = 1.e-12;
  AssertThrow(std::abs(p[1])<DIMENSIONS_X2/2.0+tol,ExcMessage("Invalid geometry parameters."));

  AssertThrow(dim==3, ExcMessage("Dimension has to be dim==3."));

  // TODO
  if(component == 0)
    result = -MAX_VELOCITY*(pow(p[1],6.0)-1.0)*(1.0+((double)rand()/RAND_MAX-1.0)*0.5-2./MAX_VELOCITY*std::sin(p[2]*8.));
  else if(component == 2)
    result = (pow(p[1],6.0)-1.0)*std::sin(p[0]*8.)*2.;

//  if(component == 0)
//  {
//    double factor = 1.0;
//    result = -MAX_VELOCITY*(pow(p[1],6.0)-1.0)*(1.0+((double)rand()/RAND_MAX-0.5)*factor);
//  }

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
double AnalyticalSolutionPressure<dim>::value(const Point<dim>    &/* p */,
                                              const unsigned int  /* component */) const
{
  double result = 0.0;

  // For this flow problem no analytical solution is available.

  return result;
}


/*
 *  Neumann boundary conditions for velocity
 *
 *  - Laplace formulation of viscous term
 *    -> prescribe velocity gradient (grad U)*n on Gamma_N
 *
 *  - Divergence formulation of viscous term
 *    -> prescribe (grad U + (grad U)^T)*n on Gamma_N
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
double NeumannBoundaryVelocity<dim>::value(const Point<dim> &/* p */,const unsigned int /* component */) const
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
double PressureBC_dudt<dim>::value(const Point<dim>   &/* p */,
                                   const unsigned int /* component */) const
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
 double RightHandSide<dim>::value(const Point<dim>   &/* p */,
                                  const unsigned int component) const
 {
   double result = 0.0;

   //channel flow with periodic bc
   if(component==0)
     return 1.0;
   else
     return 0.0;

   return result;
 }


/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

#include "../incompressible_navier_stokes_test_cases/grid_functions_turbulent_channel.h"

template<int dim>
void create_grid_and_set_boundary_conditions(
    parallel::distributed::Triangulation<dim>         &triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &periodic_faces)
{
  /* --------------- Generate grid ------------------- */
  if(GRID_STRETCH_TYPE == GridStretchType::TransformGridCells)
  {
    Point<dim> coordinates;
    coordinates[0] = DIMENSIONS_X1;
    coordinates[1] = DIMENSIONS_X2/2.0; // dimension in y-direction is 2.0, see also function grid_transform() that maps the y-coordinate from [0,1] to [-1,1]
    if (dim == 3)
     coordinates[2] = DIMENSIONS_X3;

    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
    std::vector<unsigned int> refinements(dim, N_CELLS_1D_COARSE_GRID);
    GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(),coordinates);

    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
        // x-direction
        if((std::fabs(cell->face(face_number)->center()(0) - 0.0)< 1e-12))
          cell->face(face_number)->set_all_boundary_ids (0+10);
        else if((std::fabs(cell->face(face_number)->center()(0) - DIMENSIONS_X1)< 1e-12))
          cell->face(face_number)->set_all_boundary_ids (1+10);
        // z-direction
        else if((std::fabs(cell->face(face_number)->center()(2) - 0-0)< 1e-12))
          cell->face(face_number)->set_all_boundary_ids (2+10);
        else if((std::fabs(cell->face(face_number)->center()(2) - DIMENSIONS_X3)< 1e-12))
          cell->face(face_number)->set_all_boundary_ids (3+10);
      }
    }
  }
  else if (GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
  {
    Tensor<1,dim> dimensions;
    dimensions[0] = DIMENSIONS_X1;
    dimensions[1] = DIMENSIONS_X2;
    if(dim==3)
      dimensions[2] = DIMENSIONS_X3;

    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
    AssertThrow(N_CELLS_1D_COARSE_GRID == 1,
        ExcMessage("Only N_CELLS_1D_COARSE_GRID=1 possible for curvilinear grid."));

    std::vector<unsigned int> refinements(dim, N_CELLS_1D_COARSE_GRID);
    GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(-dimensions/2.0),Point<dim>(dimensions/2.0));

    // manifold
    unsigned int manifold_id = 1;
    for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin(); cell != triangulation.end(); ++cell)
    {
      cell->set_all_manifold_ids(manifold_id);
    }

    // apply mesh stretching towards no-slip boundaries in y-direction
    static const ManifoldTurbulentChannel<dim> manifold(dimensions);
    triangulation.set_manifold(manifold_id, manifold);

    //periodicity in x- and z-direction
    //add 10 to avoid conflicts with dirichlet boundary, which is 0
    triangulation.begin()->face(0)->set_all_boundary_ids(0+10);
    triangulation.begin()->face(1)->set_all_boundary_ids(1+10);
    //periodicity in z-direction
    if (dim == 3)
    {
      triangulation.begin()->face(4)->set_all_boundary_ids(2+10);
      triangulation.begin()->face(5)->set_all_boundary_ids(3+10);
    }
  }

   GridTools::collect_periodic_faces(triangulation, 0+10, 1+10, 0, periodic_faces);
   if (dim == 3)
     GridTools::collect_periodic_faces(triangulation, 2+10, 3+10, 2, periodic_faces);

   triangulation.add_periodicity(periodic_faces);

   // perform global refinements
   triangulation.refine_global(n_refine_space);

   if(GRID_STRETCH_TYPE == GridStretchType::TransformGridCells)
   {
     // perform grid transform
     GridTools::transform (&grid_transform<dim>, triangulation);
   }

   // fill boundary descriptor velocity
   std::shared_ptr<Function<dim> > analytical_solution_velocity;
   analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(dim));
   // Dirichlet boundaries: ID = 0
   boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                     (0,analytical_solution_velocity));

   // fill boundary descriptor pressure
   std::shared_ptr<Function<dim> > pressure_bc_dudt;
   pressure_bc_dudt.reset(new PressureBC_dudt<dim>());
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

// Postprocessor

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"
#include "../../include/incompressible_navier_stokes/postprocessor/statistics_manager.h"

template<int dim>
struct PostProcessorDataTurbulentChannel
{
  PostProcessorData<dim> pp_data;
  TurbulentChannelData turb_ch_data;
};

template<int dim, int degree_u, int degree_p, typename Number>
class PostProcessorTurbulentChannel : public PostProcessor<dim, degree_u, degree_p, Number>
{
public:
  typedef PostProcessor<dim, degree_u, degree_p, Number> Base;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::NavierStokesOperator NavierStokesOperator;

  PostProcessorTurbulentChannel(PostProcessorDataTurbulentChannel<dim> const & pp_data_turb_channel)
    :
    Base(pp_data_turb_channel.pp_data),
    turb_ch_data(pp_data_turb_channel.turb_ch_data)
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

    // perform setup of turbulent channel related things
    statistics_turb_ch.reset(new StatisticsManager<dim>(dof_handler_velocity_in,mapping_in));
    statistics_turb_ch->setup(&grid_transform_y,turb_ch_data);
  }

  void do_postprocessing(VectorType const &velocity,
                         VectorType const &intermediate_velocity,
                         VectorType const &pressure,
                         double const     time,
                         int const        time_step_number)
  {
    Base::do_postprocessing(
	      velocity,
        intermediate_velocity,
        pressure,
        time,
        time_step_number);
   
    statistics_turb_ch->evaluate(velocity,time,time_step_number);
  }

  TurbulentChannelData turb_ch_data;
  std::shared_ptr<StatisticsManager<dim> > statistics_turb_ch;
};

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

  PostProcessorDataTurbulentChannel<dim> pp_data_turb_ch;
  pp_data_turb_ch.pp_data = pp_data;
  pp_data_turb_ch.turb_ch_data = param.turb_ch_data;

  std::shared_ptr<PostProcessorBase<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessorTurbulentChannel<dim,degree_u,degree_p,Number>(pp_data_turb_ch));

  return pp;
}


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
