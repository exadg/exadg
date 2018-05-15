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
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1; // FE_DEGREE_VELOCITY; // FE_DEGREE_VELOCITY - 1;

// set xwall specific parameters
unsigned int const FE_DEGREE_XWALL = 1;
unsigned int const N_Q_POINTS_1D_XWALL = 1;

// set the number of refine levels for DOMAIN 1
unsigned int const REFINE_STEPS_SPACE_DOMAIN1 = 0;

// set the number of refine levels for DOMAIN 2
unsigned int const REFINE_STEPS_SPACE_DOMAIN2 = 0;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.

// radius
double const R = 0.002;
double const R_INNER = R;
double const R_OUTER = 3.0*R;

// lengths (dimensions in flow direction z)
double const LENGTH_PRECURSOR = 4.0*R_OUTER;
double const LENGTH_INFLOW = 8.0*R_OUTER;
double const LENGTH_CONE = (R_OUTER-R_INNER)/std::tan(20.0/2.0*numbers::PI/180.0);
double const LENGTH_THROAT = 0.04;
double const LENGTH_OUTFLOW = 8.0*R_OUTER;
double const OFFSET = 2.0*R_OUTER;

// z-coordinates
double const Z2_OUTFLOW = LENGTH_OUTFLOW;
double const Z1_OUTFLOW = 0.0;

double const Z2_THROAT = 0.0;
double const Z1_THROAT = - LENGTH_THROAT;

double const Z2_CONE = - LENGTH_THROAT;
double const Z1_CONE = - LENGTH_THROAT - LENGTH_CONE;

double const Z2_INFLOW = - LENGTH_THROAT - LENGTH_CONE;
double const Z1_INFLOW = - LENGTH_THROAT - LENGTH_CONE - LENGTH_INFLOW;

double const Z2_PRECURSOR = - LENGTH_THROAT - LENGTH_CONE - LENGTH_INFLOW - OFFSET;
double const Z1_PRECURSOR = - LENGTH_THROAT - LENGTH_CONE - LENGTH_INFLOW - OFFSET - LENGTH_PRECURSOR;

// set target flow rate according to the desired Reynolds number
// if (Re_t == 500)
double const TARGET_FLOW_RATE = 5.21e-6;
// else if (Re_t = 2000)
//double const TARGET_FLOW_RATE = 2.08e-5;
// else if (Re_t = 3500)
//double const TARGET_FLOW_RATE = 3.64e-5;
// else if (Re_t = 5000)
//double const TARGET_FLOW_RATE = 5.21e-5;
// else if (Re_t = 6500)
//double const TARGET_FLOW_RATE = 6.77e-5;

double const AREA_INFLOW = R_OUTER*R_OUTER*numbers::PI;
double const MAX_VELOCITY = 2.0*TARGET_FLOW_RATE/AREA_INFLOW;

// kinematic viscosity
// same viscosity for all Reynolds numbers
double const VISCOSITY = 3.31e-6;

// data structures that we need to control the mass flow rate
// NOTA BENE: these variables will be modified by the postprocessor!
double MEAN_VELOCITY = 0.0;
double TIME_STEP_FLOW_RATE_CONTROLLER = 1.0;

// mesh parameters
unsigned int const N_CELLS_AXIAL_PRECURSOR = 2;
unsigned int const N_CELLS_AXIAL_INFLOW = 4;
unsigned int const N_CELLS_AXIAL_CONE = 2;
unsigned int const N_CELLS_AXIAL_THROAT = 4;
unsigned int const N_CELLS_AXIAL_OUTFLOW = 4;

unsigned int const MANIFOLD_ID_CYLINDER = 1234;
unsigned int const MANIFOLD_ID_OFFSET_CONE = 7890;

double const START_TIME = 0.0;
// estimation of flow-through time T_0 based on the mean velocity (i.e. velocity averaged over cross section)
double const MEAN_VELOCITY_TARGET = TARGET_FLOW_RATE/AREA_INFLOW;
double const T_0 = (LENGTH_INFLOW+LENGTH_CONE+LENGTH_THROAT+LENGTH_OUTFLOW)/MEAN_VELOCITY_TARGET;
double const END_TIME = 1.0*T_0;


// output folders
std::string OUTPUT_FOLDER = "output/fda/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string OUTPUT_NAME_1 = "precursor";
std::string OUTPUT_NAME_2 = "nozzle";

// DOMAIN 1: precursor (used to generate inflow data)
// DOMAIN 2: nozzle (the actual domain of interest)

// data structures that we need to apply the velocity inflow profile
// we currently use global variables for this purpose
unsigned int N_POINTS_R = 101;
unsigned int N_POINTS_PHI = N_POINTS_R;
std::vector<double> R_VALUES(N_POINTS_R);
std::vector<double> PHI_VALUES(N_POINTS_PHI);
std::vector<Tensor<1,DIMENSION,double> > VELOCITY_VALUES(N_POINTS_R*N_POINTS_PHI);

// initial vectors
void initialize_r_and_phi_values()
{
  AssertThrow(N_POINTS_R >= 2, ExcMessage("Variable N_POINTS_R is invalid"));
  AssertThrow(N_POINTS_PHI >= 2, ExcMessage("Variable N_POINTS_PHI is invalid"));

  for(unsigned int i=0; i<N_POINTS_R; ++i)
    R_VALUES[i] = double(i)/double(N_POINTS_R-1)*R_OUTER;

  for(unsigned int i=0; i<N_POINTS_PHI; ++i)
    PHI_VALUES[i] = -numbers::PI + double(i)/double(N_POINTS_PHI-1)*2.0*numbers::PI;
}

void initialize_velocity_values()
{
  AssertThrow(N_POINTS_R >= 2, ExcMessage("Variable N_POINTS_R is invalid"));
  AssertThrow(N_POINTS_PHI >= 2, ExcMessage("Variable N_POINTS_PHI is invalid"));

  for(unsigned int iy=0; iy<N_POINTS_R; ++iy)
  {
    for(unsigned int iz=0; iz<N_POINTS_PHI; ++iz)
    {
      Tensor<1,DIMENSION,double> velocity;
      // flow in z-direction
      // TODO: initialize with zeros
      velocity[2] = MAX_VELOCITY*(1.0-std::pow(R_VALUES[iy]/R_OUTER,2.0));
      VELOCITY_VALUES[iy*N_POINTS_R + iz] = velocity;
    }
  }
}

// we do not need this function here (but have to implement it)
template<int dim>
void InputParametersNavierStokes<dim>::set_input_parameters()
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
void InputParametersNavierStokes<dim>::set_input_parameters(unsigned int const domain_id)
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
  cfl = 0.15;
  cfl_exponent_fe_degree_velocity = 1.5;
  time_step_size = 1.0e-1;
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
  if(domain_id == 1)
    pure_dirichlet_bc = true;
  else if(domain_id == 2)
    pure_dirichlet_bc = false;

  // div-div and continuity penalty
  use_divergence_penalty = true;
  divergence_penalty_factor = 1.0e0;
  use_continuity_penalty = true;
  continuity_penalty_components = ContinuityPenaltyComponents::Normal;
  continuity_penalty_use_boundary_data = false;
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
    output_data.output_interval_time = 0.1;
    output_data.write_divergence = true;
    output_data.number_of_patches = FE_DEGREE_VELOCITY;

    // output of solver information
    output_solver_info_every_timesteps = 1; //1e5;

    // inflow data
    inflow_data.write_inflow_data = true;
    inflow_data.inflow_geometry = InflowGeometry::Cylindrical;
    inflow_data.normal_direction = 2;
    inflow_data.normal_coordinate = Z2_PRECURSOR;
    inflow_data.n_points_y = N_POINTS_R;
    inflow_data.n_points_z = N_POINTS_PHI;
    inflow_data.y_values = &R_VALUES;
    inflow_data.z_values = &PHI_VALUES;
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
    output_data.output_interval_time = 0.1;
    output_data.write_divergence = true;
    output_data.number_of_patches = FE_DEGREE_VELOCITY;

    // output of solver information
    output_solver_info_every_timesteps = 1; //1e5;

    // measure mean velocity at inflow boundary of the nozzle domain
    // (since matrix-free implementation does not allow to integrate
    // over one of the periodic boundaries of the precursor domain)
    mean_velocity_data.calculate = true;
    mean_velocity_data.boundary_IDs.insert(1); // left boundary has ID 1 (see below)
    Tensor<1,dim> normal; normal[2] = 1.0;
  }
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

double radius_function(double const z)
{
  double radius = R_OUTER;

  if(z >= Z1_INFLOW && z <= Z2_INFLOW)
    radius = R_OUTER;
  else if(z >= Z1_CONE && z <= Z2_CONE)
    radius = R_OUTER * (1.0 - (z-Z1_CONE)/(Z2_CONE-Z1_CONE)*(R_OUTER-R_INNER)/R_OUTER);
  else if(z >= Z1_THROAT && z <= Z2_THROAT)
    radius = R_INNER;
  else if(z > Z1_OUTFLOW && z <= Z2_OUTFLOW)
    radius = R_OUTER;

  return radius;
}

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
  AssertThrow(dim==3, ExcMessage("Dimension has to be dim==3."));

  double result = 0.0;

  // flow in z-direction
  // TODO: initialize with zero function
  if(component == 2)
  {
    double radius = std::sqrt(p[0]*p[0]+p[1]*p[1]);

    // assume parabolic profile u(r) = u_max * [1-(r/R)^2]
    //  -> u_max = 2 * u_mean = 2 * flow_rate / area
    double const RADIUS = radius_function(p[2]);
    if(radius > RADIUS)
      radius = RADIUS;

    double const max_velocity_z = MAX_VELOCITY * std::pow(R_OUTER/RADIUS,2.0);
    result = max_velocity_z*(1.0-pow(radius/RADIUS,2.0));
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
    initialize_r_and_phi_values();
    initialize_velocity_values();
  }

  virtual ~InflowProfile(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const
  {
    // compute polar coordinates (r, phi) from point p
    // given in Cartesian coordinates (x, y) = inflow plane
    double const r = std::sqrt(p[0]*p[0] + p[1]*p[1]);
    double const phi = std::atan2(p[1],p[0]);

    double const result = linear_interpolation_2d_cylindrical(r,
                                                              phi,
                                                              R_VALUES,
                                                              PHI_VALUES,
                                                              VELOCITY_VALUES,
                                                              component);

    return result;
  }
};


/*
 *  Right-hand side function: Implements the body force vector occurring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations
 */
 template<int dim>
 class RightHandSide : public Function<dim>
 {
 public:
   RightHandSide (const double time = 0.)
     :
     Function<dim>(dim, time),
     f(0.0) // f(t=t_0) = f_0
   {}

   virtual ~RightHandSide(){};

   virtual double value (const Point<dim>    & /*p*/,
                         const unsigned int  component = 0) const
   {
     double result = 0.0;

     //channel flow with periodic bc
     if(component==2)
     {
       // dimensional analysis: [k] = 1/(m^2 s^2) -> k = const * nu^2 / A_inflow^3
       double const k = 1.0*std::pow(VISCOSITY,2.0)/std::pow(AREA_INFLOW,3.0);
       // mean velocity is negative since the flow rate is measured at the
       // inflow boundary (normal vector points in upstream direction)
       f += k*(TARGET_FLOW_RATE - AREA_INFLOW*(-MEAN_VELOCITY))*TIME_STEP_FLOW_RATE_CONTROLLER;
       result = f;
     }

     return result;
   }

private:
   mutable double f;
 };


/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

#include "../../include/functionalities/one_sided_cylindrical_manifold.h"

template<int dim>
void create_grid_and_set_boundary_conditions_1(
    parallel::distributed::Triangulation<dim>              &triangulation,
    unsigned int const                                     n_refine_space,
    std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorNavierStokesP<dim> > boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >                 &periodic_faces)
{
  /*
   *   PRECURSOR
   */
  Triangulation<2> tria_2d;
  GridGenerator::hyper_ball(tria_2d, Point<2>(), R_OUTER);
  GridGenerator::extrude_triangulation(tria_2d,N_CELLS_AXIAL_PRECURSOR+1,LENGTH_PRECURSOR,triangulation);
  Tensor<1,dim> offset = Tensor<1,dim>();
  offset[2] = Z1_PRECURSOR;
  GridTools::shift(offset,triangulation);

  /*
   *  MANIFOLDS
   */
  triangulation.set_all_manifold_ids(0);

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids;
  std::vector<unsigned int> face_ids;

  for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin();cell != triangulation.end(); ++cell)
  {
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
    {
      bool face_at_sphere_boundary = true;
      for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
      {
        Point<dim> point = Point<dim>(0,0,cell->face(f)->vertex(v)[2]);

        if (std::abs((cell->face(f)->vertex(v)-point).norm()-R_OUTER) > 1e-12)
          face_at_sphere_boundary = false;
      }
      if (face_at_sphere_boundary)
      {
        face_ids.push_back(f);
        unsigned int manifold_id = manifold_ids.size() + 1;
        cell->set_all_manifold_ids(manifold_id);
        manifold_ids.push_back(manifold_id);
      }
    }
  }

  // generate vector of manifolds and apply manifold to all cells that have been marked
  static std::vector<std::shared_ptr<Manifold<dim> > > manifold_vec;
  manifold_vec.resize(manifold_ids.size());

  for(unsigned int i=0;i<manifold_ids.size();++i)
  {
    for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin(); cell != triangulation.end(); ++cell)
    {
      if(cell->manifold_id() == manifold_ids[i])
      {
        manifold_vec[i] = std::shared_ptr<Manifold<dim> >(
            static_cast<Manifold<dim>*>(new OneSidedCylindricalManifold<dim>(cell,face_ids[i],Point<dim>())));
        triangulation.set_manifold(manifold_ids[i],*(manifold_vec[i]));
      }
    }
  }

  /*
   *  BOUNDARY ID's
   */
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0; face_number < GeometryInfo<dim>::faces_per_cell; ++face_number)
    {
      // left boundary
      if ((std::fabs(cell->face(face_number)->center()[2] - Z1_PRECURSOR) < 1e-12))
      {
        cell->face(face_number)->set_boundary_id (0+10);
      }

      // right boundary
      if ((std::fabs(cell->face(face_number)->center()[2] - Z2_PRECURSOR) < 1e-12))
      {
        cell->face(face_number)->set_boundary_id (1+10);
      }
    }
  }

  GridTools::collect_periodic_faces(triangulation, 0+10, 1+10, 2, periodic_faces);
  triangulation.add_periodicity(periodic_faces);

  // perform global refinements
  triangulation.refine_global(n_refine_space);

  /*
   *  FILL BOUNDARY DESCRIPTORS
   */
  // fill boundary descriptor velocity
  // no slip boundaries at lower and upper wall with ID=0
  std::shared_ptr<Function<dim> > zero_function_velocity;
  zero_function_velocity.reset(new ZeroFunction<dim>(dim));
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_velocity));

  // fill boundary descriptor pressure
  // no slip boundaries at lower and upper wall with ID=0
  std::shared_ptr<Function<dim> > pressure_bc_dudt;
  pressure_bc_dudt.reset(new ZeroFunction<dim>(dim));
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,pressure_bc_dudt));
}

template<int dim>
void create_grid_and_set_boundary_conditions_2(
    parallel::distributed::Triangulation<dim>              &triangulation,
    unsigned int const                                     n_refine_space,
    std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorNavierStokesP<dim> > boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >                 &/*periodic_faces*/)
{
  /*
   *   Inflow
   */
  Triangulation<2> tria_2d_inflow;
  Triangulation<dim> tria_inflow;
  GridGenerator::hyper_ball(tria_2d_inflow, Point<2>(), R_OUTER);

  GridGenerator::extrude_triangulation(tria_2d_inflow,N_CELLS_AXIAL_INFLOW+1,LENGTH_INFLOW,tria_inflow);
  Tensor<1,dim> offset_inflow; offset_inflow[2] = Z1_INFLOW;
  GridTools::shift(offset_inflow,tria_inflow);

  Triangulation<dim> * current_tria = &tria_inflow;

  /*
   *   Cone
   */
  Triangulation<2> tria_2d_cone;
  Triangulation<dim> tria_cone;
  GridGenerator::hyper_ball(tria_2d_cone, Point<2>(), R_OUTER);

  GridGenerator::extrude_triangulation(tria_2d_cone,N_CELLS_AXIAL_CONE+1,LENGTH_CONE,tria_cone);
  Tensor<1,dim> offset_cone; offset_cone[2] = Z1_CONE;
  GridTools::shift(offset_cone,tria_cone);

  // apply conical geometry: stretch vertex positions according to z-coordinate
  for (typename Triangulation<dim>::cell_iterator cell = tria_cone.begin(); cell != tria_cone.end(); ++cell)
  {
    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      if(cell->vertex(v)[2] > Z1_CONE+1.e-10)
      {
        Point<dim> point_2d;
        double const z = cell->vertex(v)[2];
        point_2d[2] = z;

        if(std::abs((cell->vertex(v) - point_2d).norm() - 2.485281374239e-03) < 1.e-10 ||
           std::abs((cell->vertex(v) - point_2d).norm() - R_OUTER) < 1.e-10)
        {
          cell->vertex(v)[0] *= 1.0 - (cell->vertex(v)[2]-Z1_CONE)/(Z2_CONE-Z1_CONE)*(R_OUTER-R_INNER)/R_OUTER;
          cell->vertex(v)[1] *= 1.0 - (cell->vertex(v)[2]-Z1_CONE)/(Z2_CONE-Z1_CONE)*(R_OUTER-R_INNER)/R_OUTER;
        }
      }
    }
  }

  /*
   *   Throat
   */
  Triangulation<2> tria_2d_throat;
  Triangulation<dim> tria_throat;
  GridGenerator::hyper_ball(tria_2d_throat, Point<2>(), R_INNER);

  GridGenerator::extrude_triangulation(tria_2d_throat,N_CELLS_AXIAL_THROAT+1,LENGTH_THROAT,tria_throat);
  Tensor<1,dim> offset_throat; offset_throat[2] = Z1_THROAT;
  GridTools::shift(offset_throat,tria_throat);

  /*
   *   OUTFLOW
   */
  const unsigned int n_cells_circle = 4;
  double const R_1 = R_INNER + 1.0/3.0*(R_OUTER-R_INNER);
  double const R_2 = R_INNER + 2.0/3.0*(R_OUTER-R_INNER);

  Triangulation<2> tria_2d_outflow_inner, circle_1, circle_2, circle_3, tria_tmp_2d_1, tria_tmp_2d_2, tria_2d_outflow;
  GridGenerator::hyper_ball(tria_2d_outflow_inner, Point<2>(), R_INNER);

  GridGenerator::hyper_shell(circle_1, Point<2>(), R_INNER, R_1, n_cells_circle, true);
  GridTools::rotate(numbers::PI/4, circle_1);
  GridGenerator::hyper_shell(circle_2, Point<2>(), R_1, R_2, n_cells_circle, true);
  GridTools::rotate(numbers::PI/4, circle_2);
  GridGenerator::hyper_shell(circle_3, Point<2>(), R_2, R_OUTER, n_cells_circle, true);
  GridTools::rotate(numbers::PI/4, circle_3);

  // merge 2d triangulations
  GridGenerator::merge_triangulations (tria_2d_outflow_inner, circle_1, tria_tmp_2d_1);
  GridGenerator::merge_triangulations (circle_2, circle_3, tria_tmp_2d_2);
  GridGenerator::merge_triangulations (tria_tmp_2d_1, tria_tmp_2d_2, tria_2d_outflow);

  // extrude in z-direction
  Triangulation<dim> tria_outflow;
  GridGenerator::extrude_triangulation(tria_2d_outflow,N_CELLS_AXIAL_OUTFLOW+1,LENGTH_OUTFLOW,tria_outflow);
  Tensor<1,dim> offset_outflow; offset_outflow[2] = Z1_OUTFLOW;
  GridTools::shift(offset_outflow,tria_outflow);

  /*
   *  MERGE TRIANGULATIONS
   */
  Triangulation<dim> tria_tmp, tria_tmp2;
  GridGenerator::merge_triangulations (tria_inflow, tria_cone, tria_tmp);
  GridGenerator::merge_triangulations (tria_tmp, tria_throat, tria_tmp2);
  GridGenerator::merge_triangulations (tria_tmp2, tria_outflow, triangulation);

  /*
   *  MANIFOLDS
   */
  current_tria = &triangulation;
  current_tria->set_all_manifold_ids(0);

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids;
  std::vector<unsigned int> face_ids;

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids_cone;
  std::vector<unsigned int> face_ids_cone;
  std::vector<double> radius_0_cone;
  std::vector<double> radius_1_cone;

  for (typename Triangulation<dim>::cell_iterator cell = current_tria->begin();cell != current_tria->end(); ++cell)
  {
    // INFLOW
    if(cell->center()[2] < Z2_INFLOW)
    {
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
        {
          Point<dim> point = Point<dim>(0,0,cell->face(f)->vertex(v)[2]);
          if (std::abs((cell->face(f)->vertex(v)-point).norm()-R_OUTER) > 1e-12)
          {
            face_at_sphere_boundary = false;
          }
        }
        if (face_at_sphere_boundary)
        {
          face_ids.push_back(f);
          unsigned int manifold_id = manifold_ids.size() + 1;
          cell->set_all_manifold_ids(manifold_id);
          manifold_ids.push_back(manifold_id);
        }
      }
    }
    // CONE
    else if(cell->center()[2] > Z1_CONE && cell->center()[2] < Z2_CONE)
    {
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        double min_z = std::numeric_limits<double>::max();
        double max_z = - std::numeric_limits<double>::max();

        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
        {
          double const z = cell->face(f)->vertex(v)[2];
          if(z > max_z)
            max_z = z;
          if(z < min_z)
            min_z = z;

          Point<dim> point = Point<dim>(0,0,z);
          if (std::abs((cell->face(f)->vertex(v)-point).norm()-radius_function(z)) > 1e-12)
          {
            face_at_sphere_boundary = false;
          }
        }
        if (face_at_sphere_boundary)
        {
          face_ids_cone.push_back(f);
          unsigned int manifold_id = MANIFOLD_ID_OFFSET_CONE + manifold_ids_cone.size() + 1;
          cell->set_all_manifold_ids(manifold_id);
          manifold_ids_cone.push_back(manifold_id);
          radius_0_cone.push_back(radius_function(min_z));
          radius_1_cone.push_back(radius_function(max_z));
        }
      }
    }
    // THROAT
    else if(cell->center()[2] > Z1_THROAT && cell->center()[2] < Z2_THROAT)
    {
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
        {
          Point<dim> point = Point<dim>(0,0,cell->face(f)->vertex(v)[2]);
          if (std::abs((cell->face(f)->vertex(v)-point).norm()-R_INNER) > 1e-12)
          {
            face_at_sphere_boundary = false;
          }
        }
        if (face_at_sphere_boundary)
        {
          face_ids.push_back(f);
          unsigned int manifold_id = manifold_ids.size() + 1;
          cell->set_all_manifold_ids(manifold_id);
          manifold_ids.push_back(manifold_id);
        }
      }
    }
    // OUTFLOW
    else if(cell->center()[2] > Z1_OUTFLOW && cell->center()[2] < Z2_OUTFLOW)
    {
      Point<dim> point2 = Point<dim>(0,0,cell->center()[2]);

      // cylindrical manifold for outer cell layers
      if((cell->center()-point2).norm() > R_INNER/std::sqrt(2.0))
        cell->set_all_manifold_ids(MANIFOLD_ID_CYLINDER);

      // one-sided cylindrical manifold for core region
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
        {
          Point<dim> point = Point<dim>(0,0,cell->face(f)->vertex(v)[2]);
          if (std::abs((cell->face(f)->vertex(v)-point).norm()-R_INNER) > 1e-12 ||
              (cell->center()-point2).norm() > R_INNER/std::sqrt(2.0))
          {
            face_at_sphere_boundary = false;
          }
        }
        if (face_at_sphere_boundary)
        {
          face_ids.push_back(f);
          unsigned int manifold_id = manifold_ids.size() + 1;
          cell->set_all_manifold_ids(manifold_id);
          manifold_ids.push_back(manifold_id);
        }
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Should not arrive here."));
    }
  }

  // one-sided spherical manifold
  // generate vector of manifolds and apply manifold to all cells that have been marked
  static std::vector<std::shared_ptr<Manifold<dim> > > manifold_vec;
  manifold_vec.resize(manifold_ids.size());

  for(unsigned int i=0;i<manifold_ids.size();++i)
  {
    for (typename Triangulation<dim>::cell_iterator cell = current_tria->begin(); cell != current_tria->end(); ++cell)
    {
      if(cell->manifold_id() == manifold_ids[i])
      {
        Point<dim> center = Point<dim>();
        manifold_vec[i] = std::shared_ptr<Manifold<dim> >(
            static_cast<Manifold<dim>*>(new OneSidedCylindricalManifold<dim>(cell,face_ids[i],center)));
        current_tria->set_manifold(manifold_ids[i],*(manifold_vec[i]));
      }
    }
  }

  // conical manifold
  static std::vector<std::shared_ptr<Manifold<dim> > > manifold_vec_cone;
  manifold_vec_cone.resize(manifold_ids_cone.size());

  for(unsigned int i=0;i<manifold_ids_cone.size();++i)
  {
    for (typename Triangulation<dim>::cell_iterator cell = current_tria->begin(); cell != current_tria->end(); ++cell)
    {
      if(cell->manifold_id() == manifold_ids_cone[i])
      {
        Point<dim> center = Point<dim>();
        manifold_vec_cone[i] = std::shared_ptr<Manifold<dim> >(
            static_cast<Manifold<dim>*>(new OneSidedConicalManifold<dim>(cell,face_ids_cone[i],center,radius_0_cone[i],radius_1_cone[i])));
        current_tria->set_manifold(manifold_ids_cone[i],*(manifold_vec_cone[i]));
      }
    }
  }

  // set cylindrical manifold
  static std::shared_ptr<Manifold<dim> > cylinder_manifold;
  cylinder_manifold = std::shared_ptr<Manifold<dim> >(static_cast<Manifold<dim>*>(new MyCylindricalManifold<dim>(Point<dim>())));
  current_tria->set_manifold(MANIFOLD_ID_CYLINDER, *cylinder_manifold);



  /*
   *  BOUNDARY ID's
   */
  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int f=0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      // inflow boundary on the left has ID = 1
      if ((std::fabs(cell->face(f)->center()[2] - Z1_INFLOW)< 1e-12))
      {
        cell->face(f)->set_boundary_id (1);
      }

      // outflow boundary on the right has ID = 2
      if ((std::fabs(cell->face(f)->center()[2] - Z2_OUTFLOW)< 1e-12))
      {
        cell->face(f)->set_boundary_id (2);
      }
    }
  }

  // perform global refinements
  triangulation.refine_global(n_refine_space);

  /*
   *  FILL BOUNDARY DESCRIPTORS
   */
  // fill boundary descriptor velocity
  // no slip boundaries at the upper and lower wall with ID=0
  std::shared_ptr<Function<dim> > zero_function_velocity;
  zero_function_velocity.reset(new ZeroFunction<dim>(dim));
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_velocity));

  // inflow boundary condition at left boundary with ID=1: prescribe velocity profile which
  // is obtained as the results of the simulation on DOMAIN 1
  std::shared_ptr<Function<dim> > inflow_profile;
  inflow_profile.reset(new InflowProfile<dim>(dim));
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(1,inflow_profile));

  // outflow boundary condition at right boundary with ID=2
  boundary_descriptor_velocity->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(2,zero_function_velocity));

  // fill boundary descriptor pressure
  // no slip boundaries at the upper and lower wall with ID=0
  std::shared_ptr<Function<dim> > pressure_bc_dudt;
  pressure_bc_dudt.reset(new ZeroFunction<dim>(dim));
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,pressure_bc_dudt));

  // inflow boundary condition at left boundary with ID=1
  // the inflow boundary condition is time dependent (du/dt != 0) but, for simplicity,
  // we assume that this is negligible when using the dual splitting scheme
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(1,pressure_bc_dudt));

  // outflow boundary condition at right boundary with ID=2: set pressure to zero
  std::shared_ptr<Function<dim> > zero_function_pressure;
  zero_function_pressure.reset(new ZeroFunction<dim>(1));
  boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(2,zero_function_pressure));
}


template<int dim>
void set_field_functions_1(std::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  std::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new ZeroFunction<dim>(1));

  // prescribe body force for the turbulent channel (DOMAIN 1) to
  // adjust the desired flow rate
  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure = initial_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_field_functions_2(std::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new InitialSolutionVelocity<dim>());
  std::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new ZeroFunction<dim>(1));

  // no body forces for the second domain
  std::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new ZeroFunction<dim>(dim));

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure = initial_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new ZeroFunction<dim>(1));
}

// Postprocessor

#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

template<int dim>
struct PostProcessorDataFDA
{
  PostProcessorData<dim> pp_data;
  InflowData<dim> inflow_data;
  MeanVelocityCalculatorData<dim> mean_velocity_data;
};

template<int dim, int fe_degree_u, int fe_degree_p, typename Number>
class PostProcessorFDA : public PostProcessor<dim, fe_degree_u, fe_degree_p, Number>
{
public:
  PostProcessorFDA(PostProcessorDataFDA<dim> const & pp_data_in)
    :
    PostProcessor<dim,fe_degree_u,fe_degree_p, Number>(pp_data_in.pp_data),
    pp_data_fda(pp_data_in),
    time_old(START_TIME)
  {
    inflow_data_calculator.reset(new InflowDataCalculator<dim,Number>(pp_data_in.inflow_data));
  }

  void setup(DoFHandler<dim> const                                  &dof_handler_velocity_in,
             DoFHandler<dim> const                                  &dof_handler_pressure_in,
             Mapping<dim> const                                     &mapping_in,
             MatrixFree<dim,Number> const                           &matrix_free_data_in,
             DofQuadIndexData const                                 &dof_quad_index_data_in,
             std::shared_ptr<AnalyticalSolutionNavierStokes<dim> >  analytical_solution_in)
  {
    // call setup function of base class
    PostProcessor<dim,fe_degree_u,fe_degree_p,Number>::setup(
        dof_handler_velocity_in,
        dof_handler_pressure_in,
        mapping_in,
        matrix_free_data_in,
        dof_quad_index_data_in,
        analytical_solution_in);

    // inflow data
    inflow_data_calculator->setup(dof_handler_velocity_in,mapping_in);

    // calculation of mean velocity
    mean_velocity_calculator.reset(new MeanVelocityCalculator<dim,fe_degree_u,Number>(
        matrix_free_data_in, dof_quad_index_data_in, pp_data_fda.mean_velocity_data));
  }

  void do_postprocessing(parallel::distributed::Vector<Number> const   &velocity,
                         parallel::distributed::Vector<Number> const   &intermediate_velocity,
                         parallel::distributed::Vector<Number> const   &pressure,
                         parallel::distributed::Vector<Number> const   &vorticity,
                         std::vector<SolutionField<dim,Number> > const &additional_fields,
                         double const                                  time,
                         int const                                     time_step_number)
  {
    PostProcessor<dim,fe_degree_u,fe_degree_p,Number>::do_postprocessing(
	      velocity,
        intermediate_velocity,
        pressure,
        vorticity,
        additional_fields,
        time,
        time_step_number);

    // inflow data
    inflow_data_calculator->calculate(velocity);

    // calculation of mean velocity
    MEAN_VELOCITY = mean_velocity_calculator->evaluate(velocity);
    // set time step size for flow rate controller
    TIME_STEP_FLOW_RATE_CONTROLLER = time-time_old;
    time_old = time;
  }

private:
  PostProcessorDataFDA<dim> pp_data_fda;
  std::shared_ptr<InflowDataCalculator<dim, Number> > inflow_data_calculator;
  std::shared_ptr<MeanVelocityCalculator<dim,fe_degree_u,Number> > mean_velocity_calculator;
  double time_old;
};

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim,Number> >
construct_postprocessor(InputParametersNavierStokes<dim> const &param)
{
  PostProcessorData<dim> pp_data;
  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;

  PostProcessorDataFDA<dim> pp_data_fda;
  pp_data_fda.pp_data = pp_data;
  pp_data_fda.inflow_data = param.inflow_data;
  pp_data_fda.mean_velocity_data = param.mean_velocity_data;

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessorFDA<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE,Number>(pp_data_fda));

  return pp;
}


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
