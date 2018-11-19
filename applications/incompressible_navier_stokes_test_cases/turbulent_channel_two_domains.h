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
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set the number of refine levels for DOMAIN 1
unsigned int const REFINE_STEPS_SPACE_DOMAIN1 = 2;

// set the number of refine levels for DOMAIN 2
unsigned int const REFINE_STEPS_SPACE_DOMAIN2 = 2;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
double const DIMENSIONS_X1 = 2.0*numbers::PI;
double const DIMENSIONS_X2 = 2.0;
double const DIMENSIONS_X3 = numbers::PI;

double const MAX_VELOCITY = 22.0;
double const VISCOSITY = 1./180.; // critical value: 1./50. - 1./75. //1./180.; //1./395.; //1./590.; //1./950;

double const START_TIME = 0.0;
double const SAMPLE_START_TIME = 30.0;
double const END_TIME = 50.0;

// use a negative GRID_STRETCH_FAC to deactivate grid stretching
const double GRID_STRETCH_FAC = 1.8;

enum class GridStretchType{ TransformGridCells, VolumeManifold };
GridStretchType GRID_STRETCH_TYPE = GridStretchType::VolumeManifold;

std::string OUTPUT_FOLDER = "output/bfs/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME_1 = "bfs1";
std::string OUTPUT_NAME_2 = "bfs2";

// DOMAIN 1: turbulent channel problem: used to generate inflow data for the BFS
// DOMAIN 2: backward facing step (using results of the turbulent channel flow as velocity inflow profile)

// data structures that we need to apply the velocity inflow profile
// we currently use global variables for this purpose
unsigned int N_POINTS_Y = 101;
unsigned int N_POINTS_Z = N_POINTS_Y;
std::vector<double> Y_VALUES(N_POINTS_Y);
std::vector<double> Z_VALUES(N_POINTS_Z);
std::vector<Tensor<1,DIMENSION,double> > VELOCITY_VALUES(N_POINTS_Y*N_POINTS_Z);

// initial vectors
void initialize_y_and_z_values()
{
  AssertThrow(N_POINTS_Y >= 2, ExcMessage("Variable N_POINTS_Y is invalid"));
  AssertThrow(N_POINTS_Z >= 2, ExcMessage("Variable N_POINTS_Z is invalid"));

  for(unsigned int i=0; i<N_POINTS_Y; ++i)
    Y_VALUES[i] = -DIMENSIONS_X2/2.0 + double(i)/double(N_POINTS_Y-1)*DIMENSIONS_X2;

  for(unsigned int i=0; i<N_POINTS_Z; ++i)
    Z_VALUES[i] = -DIMENSIONS_X3/2.0 + double(i)/double(N_POINTS_Z-1)*DIMENSIONS_X3;
}

void initialize_velocity_values()
{
  AssertThrow(N_POINTS_Y >= 2, ExcMessage("Variable N_POINTS_Y is invalid"));
  AssertThrow(N_POINTS_Z >= 2, ExcMessage("Variable N_POINTS_Z is invalid"));

  for(unsigned int iy=0; iy<N_POINTS_Y; ++iy)
  {
    for(unsigned int iz=0; iz<N_POINTS_Z; ++iz)
    {
      Tensor<1,DIMENSION,double> velocity;
      VELOCITY_VALUES[iy*N_POINTS_Y + iz] = velocity;
    }
  }
}

// we do not need this function here (but have to implement it)
template<int dim>
void InputParameters<dim>::set_input_parameters()
{

}

/*
 *  To set input parameters for DOMAIN 1 and DOMAIN 2, use
 *
 *  if(domain_id == 1){}
 *  else if(domain_id == 2){}
 *
 *  Most of the input parameters are the same for both domains!
 */
template<int dim>
void InputParameters<dim>::set_input_parameters(unsigned int const domain_id)
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
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  max_velocity = MAX_VELOCITY;
  cfl = 0.5;
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
  if(domain_id == 1)
    pure_dirichlet_bc = true;
  else if(domain_id == 2)
    pure_dirichlet_bc = false;

  // div-div and continuity penalty
  use_divergence_penalty = true;
  divergence_penalty_factor = 1.0e0;
  use_continuity_penalty = true;
  continuity_penalty_components = ContinuityPenaltyComponents::Normal;
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
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::Chebyshev;

  abs_tol_pressure = 1.e-12;
  rel_tol_pressure = 1.e-6;

  // stability in the limit of small time steps
  use_approach_of_ferrer = false;
  deltat_ref = 1.e0;

  // projection step
  solver_projection = SolverProjection::PCG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix; //BlockJacobi; //PointJacobi; //InverseMassMatrix;
  update_preconditioner_projection = true;
  abs_tol_projection = 1.e-12;
  rel_tol_projection = 1.e-6;



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
  abs_tol_viscous = 1.e-12;
  rel_tol_viscous = 1.e-6;


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
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;


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
  momentum_preconditioner = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  schur_complement_preconditioner = SchurComplementPreconditioner::CahouetChabard; //PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;

  // Chebyshev moother
  multigrid_data_schur_complement_preconditioner.smoother = MultigridSmoother::Chebyshev;
  multigrid_data_schur_complement_preconditioner.coarse_solver = MultigridCoarseGridSolver::Chebyshev;


  if(domain_id == 1)
  {
    // OUTPUT AND POSTPROCESSING
    print_input_parameters = true;

    // write output for visualization of results
    output_data.write_output = true;
    output_data.output_folder = OUTPUT_FOLDER_VTU;
    output_data.output_name = OUTPUT_NAME_1;
    output_data.output_start_time = start_time;
    output_data.output_interval_time = 1.0;
    output_data.write_divergence = true;
    output_data.number_of_patches = FE_DEGREE_VELOCITY;

    // output of solver information
    output_solver_info_every_timesteps = 1e3; //1e4;

    // turbulent channel statistics
    turb_ch_data.calculate_statistics = true;
    turb_ch_data.cells_are_stretched = false;
    if(GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
      turb_ch_data.cells_are_stretched = true;
    turb_ch_data.sample_start_time = SAMPLE_START_TIME;
    turb_ch_data.sample_end_time = END_TIME;
    turb_ch_data.sample_every_timesteps = 10;
    turb_ch_data.viscosity = VISCOSITY;
    turb_ch_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME_1;

    inflow_data.write_inflow_data = true;
    inflow_data.n_points_y = N_POINTS_Y;
    inflow_data.n_points_z = N_POINTS_Z;
    inflow_data.y_values = &Y_VALUES;
    inflow_data.z_values = &Z_VALUES;
    inflow_data.array = &VELOCITY_VALUES;
  }
  else if(domain_id == 2)
  {
    // OUTPUT AND POSTPROCESSING
    print_input_parameters = true;

    // write output for visualization of results
    output_data.write_output = true;
    output_data.output_folder = OUTPUT_FOLDER_VTU;
    output_data.output_name = OUTPUT_NAME_2;
    output_data.output_start_time = start_time;
    output_data.output_interval_time = 1.0;
    output_data.write_divergence = true;
    output_data.number_of_patches = FE_DEGREE_VELOCITY;

    // output of solver information
    output_solver_info_every_timesteps = 1e3; //1e4;

    // turbulent channel statistics
    turb_ch_data.calculate_statistics = true;
    turb_ch_data.cells_are_stretched = false;
    if(GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
      turb_ch_data.cells_are_stretched = true;
    turb_ch_data.sample_start_time = SAMPLE_START_TIME;
    turb_ch_data.sample_end_time = END_TIME;
    turb_ch_data.sample_every_timesteps = 10;
    turb_ch_data.viscosity = VISCOSITY;
    turb_ch_data.filename_prefix = OUTPUT_FOLDER + OUTPUT_NAME_2;
  }
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity (const unsigned int  n_components = dim,
                           const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {
  }

  virtual ~InitialSolutionVelocity(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double InitialSolutionVelocity<dim>::value(const Point<dim>   &p,
                                           const unsigned int component) const
{
  double result = 0.0;

  const double tol = 1.e-12;
  AssertThrow(std::abs(p[1])<DIMENSIONS_X2/2.0+tol,ExcMessage("Invalid geometry parameters."));

  if(dim==3)
  {
    if(component == 0)
    {
      double factor = 1.0;
      result = -MAX_VELOCITY*(pow(p[1],2.0)-1.0)*(1.0+((double)rand()/RAND_MAX-0.5)*factor);
    }
  }
  else
  {
    AssertThrow(false, ExcMessage("Dimension has to be dim==3."));
  }

  return result;
}

#include "../../include/incompressible_navier_stokes/postprocessor/inflow_data_calculator.h"

template<int dim>
class InflowProfile : public Function<dim>
{
public:
  InflowProfile (const unsigned int  n_components = dim,
                 const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {
    initialize_y_and_z_values();
    initialize_velocity_values();
  }

  virtual ~InflowProfile(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double InflowProfile<dim>::value(const Point<dim>   &p,
                                              const unsigned int component) const
{
  double result = linear_interpolation_2d_cartesian(p,Y_VALUES,Z_VALUES,VELOCITY_VALUES,component);

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

 /*
  *  maps eta in [0,1] --> y in [-1,1]*length_y/2.0 (using a hyperbolic mesh stretching)
  */
double grid_transform_y(const double &eta)
{
  double y = 0.0;

  if(GRID_STRETCH_FAC >= 0)
    y = DIMENSIONS_X2/2.0*std::tanh(GRID_STRETCH_FAC*(2.*eta-1.))/std::tanh(GRID_STRETCH_FAC);
  else // use a negative GRID_STRETCH_FACto deactivate grid stretching
    y = DIMENSIONS_X2/2.0*(2.*eta-1.);

  return y;
}

/*
 * inverse mapping:
 *
 *  maps y in [-1,1]*length_y/2.0 --> eta in [0,1]
 */
double inverse_grid_transform_y(const double &y)
{
  double eta = 0.0;

  if(GRID_STRETCH_FAC >= 0)
    eta = (std::atanh(y*std::tanh(GRID_STRETCH_FAC)*2.0/DIMENSIONS_X2)/GRID_STRETCH_FAC+1.0)/2.0;
  else // use a negative GRID_STRETCH_FACto deactivate grid stretching
    eta = (2.*y/DIMENSIONS_X2+1.)/2.0;

  return eta;
}

template <int dim>
Point<dim> grid_transform (const Point<dim> &in)
{
  Point<dim> out = in;

  out[0] = in(0)-numbers::PI;
  out[1] = grid_transform_y(in[1]);

  if(dim==3)
    out[2] = in(2)-0.5*numbers::PI;
  return out;
}

#include <deal.II/grid/manifold_lib.h>

template <int dim>
class ManifoldTurbulentChannel : public ChartManifold<dim,dim,dim>
{
public:
  ManifoldTurbulentChannel(Tensor<1,dim> const &dimensions_in)
  {
    dimensions = dimensions_in;
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  Point<dim> push_forward(const Point<dim> &xi) const
  {
    Point<dim> x;

    x[0] = xi[0]*dimensions[0]-dimensions[0]/2.0;
    x[1] = grid_transform_y(xi[1]);

    if(dim==3)
      x[2] = xi[2]*dimensions[2]-dimensions[2]/2.0;

    return x;
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates [0,1]^d
   */
  Point<dim> pull_back(const Point<dim> &x) const
  {
    Point<dim> xi;

    xi[0] = x[0]/dimensions[0]+0.5;
    xi[1] = inverse_grid_transform_y(x[1]);

    if(dim==3)
      xi[2] = x[2]/dimensions[2]+0.5;

    return xi;
  }

  std::unique_ptr<Manifold<dim>>
  clone() const override
  {
    return std_cxx14::make_unique<ManifoldTurbulentChannel<dim>>(dimensions);
  }

private:
 Tensor<1,dim> dimensions;
};

template<int dim>
void create_grid_and_set_boundary_conditions_1(
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
    coordinates[0] = 2.0*numbers::PI;
    coordinates[1] = 1.0; // dimension in y-direction is 2.0, see also function grid_transform() that maps the y-coordinate from [0,1] to [-1,1]
    if (dim == 3)
     coordinates[2] = numbers::PI;

    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
    std::vector<unsigned int> refinements(dim, 1);
    GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(),coordinates);
  }
  else if (GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
  {
    Tensor<1,dim> dimensions;
    dimensions[0] = DIMENSIONS_X1;
    dimensions[1] = DIMENSIONS_X2;
    if(dim==3)
      dimensions[2] = DIMENSIONS_X3;

    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
    std::vector<unsigned int> refinements(dim, 1);
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
  }

   //periodicity in x- and z-direction (add 10 to avoid conflicts with dirichlet boundary, which is 0)
   triangulation.begin()->face(0)->set_all_boundary_ids(0+10);
   triangulation.begin()->face(1)->set_all_boundary_ids(1+10);
   //periodicity in z-direction
   if (dim == 3)
   {
     triangulation.begin()->face(4)->set_all_boundary_ids(2+10);
     triangulation.begin()->face(5)->set_all_boundary_ids(3+10);
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
   // no slip boundaries at lower and upper wall with ID=0
   std::shared_ptr<Function<dim> > zero_function_velocity;
   zero_function_velocity.reset(new Functions::ZeroFunction<dim>(dim));
   boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                     (0,zero_function_velocity));

   // fill boundary descriptor pressure
   // no slip boundaries at lower and upper wall with ID=0
   std::shared_ptr<Function<dim> > pressure_bc_dudt;
   pressure_bc_dudt.reset(new PressureBC_dudt<dim>());
   boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                     (0,pressure_bc_dudt));
}

template<int dim>
void create_grid_and_set_boundary_conditions_2(
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
    coordinates[0] = 2.0*numbers::PI;
    coordinates[1] = 1.0; // dimension in y-direction is 2.0, see also function grid_transform() that maps the y-coordinate from [0,1] to [-1,1]
    if (dim == 3)
     coordinates[2] = numbers::PI;

    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
    std::vector<unsigned int> refinements(dim, 1);
    GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(),coordinates);
  }
  else if (GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
  {
    Tensor<1,dim> dimensions;
    dimensions[0] = DIMENSIONS_X1;
    dimensions[1] = DIMENSIONS_X2;
    if(dim==3)
      dimensions[2] = DIMENSIONS_X3;

    Tensor<1,dim> offset;
    offset[0] = 1.2*DIMENSIONS_X1;

    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
    std::vector<unsigned int> refinements(dim, 1);
    GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(-dimensions/2.0+offset),Point<dim>(dimensions/2.0+offset));

    // set boundary ID's
    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
      for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
      {
        // outflow boundary on the right has ID = 1
         if ((std::fabs(cell->face(face_number)->center()(0) - (dimensions/2.0+offset)[0])< 1e-12))
           cell->face(face_number)->set_boundary_id (1);

         // inflow boundary on the left has ID = 2
          if ((std::fabs(cell->face(face_number)->center()(0) - (-dimensions/2.0+offset)[0])< 1e-12))
            cell->face(face_number)->set_boundary_id (2);
      }
    }

    // manifold
    unsigned int manifold_id = 1;
    for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin(); cell != triangulation.end(); ++cell)
    {
      cell->set_all_manifold_ids(manifold_id);
    }

    // apply mesh stretching towards no-slip boundaries in y-direction
    static const ManifoldTurbulentChannel<dim> manifold(dimensions);
    triangulation.set_manifold(manifold_id, manifold);
  }

  // periodicity in z-direction
  // add 10 to avoid conflicts with dirichlet boundary
  triangulation.begin()->face(4)->set_all_boundary_ids(2+10);
  triangulation.begin()->face(5)->set_all_boundary_ids(3+10);

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
  // no slip boundaries at the upper and lower wall with ID=0
  std::shared_ptr<Function<dim> > zero_function_velocity;
  zero_function_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                   (0,zero_function_velocity));

  // inflow boundary condition at left boundary with ID=2: prescribe velocity profile which
  // is obtained as the results of the simulation on DOMAIN 1
  std::shared_ptr<Function<dim> > inflow_profile;
  inflow_profile.reset(new InflowProfile<dim>(dim));
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                   (2,inflow_profile));

  // outflow boundary condition at right boundary with ID=1
  boundary_descriptor_velocity->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                   (1,zero_function_velocity));

  // fill boundary descriptor pressure
  // no slip boundaries at the upper and lower wall with ID=0
  std::shared_ptr<Function<dim> > pressure_bc_dudt;
  pressure_bc_dudt.reset(new PressureBC_dudt<dim>());
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                   (0,pressure_bc_dudt));

  // inflow boundary condition at left boundary with ID=2
  // the inflow boundary condition is time dependent (du/dt != 0) but, for simplicity,
  // we assume that this is negligible when using the dual splitting scheme
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                   (2,pressure_bc_dudt));

  // outflow boundary condition at right boundary with ID=1: set pressure to zero
  std::shared_ptr<Function<dim> > zero_function_pressure;
  zero_function_pressure.reset(new Functions::ZeroFunction<dim>(1));
  boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >
                                                   (1,zero_function_pressure));
}


template<int dim>
void set_field_functions_1(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  std::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));

  // use a constant body force for the turbulent channel (DOMAIN 1)
  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure = initial_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_field_functions_2(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
//  initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  std::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));

  // no body forces for the second domain
  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));

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
  InflowData<dim> inflow_data;
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
    write_final_output(true),
    turb_ch_data(pp_data_turb_channel.turb_ch_data)
  {
    inflow_data_calculator.reset(new InflowDataCalculator<dim,Number>(pp_data_turb_channel.inflow_data));
  }

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

    // inflow data
    inflow_data_calculator->setup(dof_handler_velocity_in,mapping_in);
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

    // turbulent channel statistics
    statistics_turb_ch->evaluate(velocity,time,time_step_number);

    // inflow data
    inflow_data_calculator->calculate(velocity);
  }

  bool write_final_output;
  TurbulentChannelData turb_ch_data;
  std::shared_ptr<StatisticsManager<dim> > statistics_turb_ch;
  std::shared_ptr<InflowDataCalculator<dim, Number> > inflow_data_calculator;
};

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

  PostProcessorDataTurbulentChannel<dim> pp_data_turb_ch;
  pp_data_turb_ch.pp_data = pp_data;
  pp_data_turb_ch.turb_ch_data = param.turb_ch_data;
  pp_data_turb_ch.inflow_data = param.inflow_data;

  std::shared_ptr<PostProcessorBase<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessorTurbulentChannel<dim,degree_u,degree_p,Number>(pp_data_turb_ch));

  return pp;
}


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
