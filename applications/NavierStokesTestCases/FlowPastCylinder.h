/*
 * FlowPastCylinder.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_NAVIERSTOKESTESTCASES_FLOWPASTCYLINDER_H_
#define APPLICATIONS_NAVIERSTOKESTESTCASES_FLOWPASTCYLINDER_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/manifold_lib.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 10;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1; // FE_DEGREE_VELOCITY; // FE_DEGREE_VELOCITY - 1;

// set xwall specific parameters
unsigned int const FE_DEGREE_XWALL = 1;
unsigned int const N_Q_POINTS_1D_XWALL = 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 3;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
ProblemType PROBLEM_TYPE = ProblemType::Unsteady;
const unsigned int TEST_CASE = 3; // 1, 2 or 3
const double Um = (DIMENSION == 2 ? (TEST_CASE==1 ? 0.3 : 1.5) : (TEST_CASE==1 ? 0.45 : 2.25));

// pyhsical dimensions
const double D = 0.1;
const double R = D/2.0;
const double H = 0.41;
const double L1 = 0.3;
const double L2 = 2.5;
const double X_0 = 0.0;
const double Y_0 = 0.0;
const double X_1 = L1;
const double X_2 = 0.7;
const double X_C = 0.5; // center
const double Y_C = 0.2; // center

// ManifoldType
// Surface manifold: when refining the mesh only the cells close to the manifold-surface are curved
// Volume manifold: all child cells are curved and subject to the manifold since it is a volume manifold
enum class ManifoldType{ SurfaceManifold, VolumeManifold };
const ManifoldType MANIFOLD_TYPE = ManifoldType::VolumeManifold;

// MeshType
// Type1: no refinement around cylinder surface
// Type2: two layers of spherical cells around cylinder
// Type3: coarse mesh has only one element in direction perpendicular to flow direction,
//        one layer of spherical cells around cylinder for coarsest mesh
enum class MeshType{ Type1, Type2, Type3 };
const MeshType MESH_TYPE = MeshType::Type3;

const double END_TIME = 8.0;
std::string OUTPUT_PREFIX = "2D_3_cfl_0-2";
std::string OUTPUT_FOLDER = "/paper/dual_splitting/"; //"/comparison_lehrenfeld/pressure_correction/"; // "/paper/pressure_correction";

template<int dim>
void InputParametersNavierStokes<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = PROBLEM_TYPE;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  right_hand_side = false;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = END_TIME; //END_TIME is also needed somewhere else
  viscosity = 1.e-3;


  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme; //BDFPressureCorrection; //BDFDualSplittingScheme; //BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit; //Explicit;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL; //ConstTimeStepUserSpecified; //ConstTimeStepCFL;
  max_velocity = Um;
  cfl = 0.2;//0.6;//2.5e-1;
  cfl_exponent_fe_degree_velocity = 1.0;
  time_step_size = 1.0e-3;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 2; //2; // 1; // 2; // 3;
  start_with_low_order = true; // true; // false;


  // SPATIAL DISCRETIZATION

  // spatial discretization method
  spatial_discretization = SpatialDiscretization::DG;

  // convective term - currently no parameters

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  IP_factor_viscous = 1.0;
  penalty_term_div_formulation = PenaltyTermDivergenceFormulation::NotSymmetrized; //Symmetrized;

  // gradient term
  gradp_integrated_by_parts = true;
  gradp_use_boundary_data = true;

  // divergence term
  divu_integrated_by_parts = true;
  divu_use_boundary_data = true;

  // special case: pure DBC's
  pure_dirichlet_bc = false;

  // PROJECTION METHODS

  // pressure Poisson equation
  IP_factor_pressure = 1.0;
  solver_pressure_poisson = SolverPressurePoisson::FGMRES; //PCG; //FGMRES;
  max_n_tmp_vectors_pressure_poisson = 60;
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::GeometricMultigrid; //Jacobi; //GeometricMultigrid;
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::PCG_Jacobi; //Chebyshev;
  abs_tol_pressure = 1.e-12;
  rel_tol_pressure = 1.e-8;

  // stability in the limit of small time steps
  use_approach_of_ferrer = false;
  deltat_ref = 1.e0;

  // projection step
  projection_type = ProjectionType::NoPenalty; //NoPenalty; //DivergencePenalty;
  penalty_factor_divergence = 1.0e0;
  penalty_factor_continuity = 1.0e0;
  solver_projection = SolverProjection::PCG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  abs_tol_projection = 1.e-20;
  rel_tol_projection = 1.e-12;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = 2; //order_time_integrator <=2 ? order_time_integrator : 2;

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
  solver_viscous = SolverViscous::PCG; //PCG;
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::PCG_Jacobi; //PCG_Jacobi; //Chebyshev;
  abs_tol_viscous = 1.e-12;
  rel_tol_viscous = 1.e-8;


  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver
  newton_solver_data_momentum.abs_tol = 1.e-12;
  newton_solver_data_momentum.rel_tol = 1.e-8;
  newton_solver_data_momentum.max_iter = 100;

  // linear solver
  solver_momentum = SolverMomentum::FGMRES; //GMRES; //FGMRES;
  preconditioner_momentum = PreconditionerMomentum::VelocityDiffusion; //InverseMassMatrix; //VelocityDiffusion;
  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::GMRES_Jacobi; //Chebyshev;
  abs_tol_momentum_linear = 1.e-12;
  rel_tol_momentum_linear = 1.e-8;
  max_iter_momentum_linear = 1e4;
  use_right_preconditioning_momentum = true;
  max_n_tmp_vectors_momentum = 100;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled.abs_tol = 1.e-12;
  newton_solver_data_coupled.rel_tol = 1.e-8;
  newton_solver_data_coupled.max_iter = 1e2;

  // linear solver
  solver_linearized_navier_stokes = SolverLinearizedNavierStokes::FGMRES; //GMRES; //FGMRES;
  abs_tol_linear = 1.e-12;
  rel_tol_linear = 1.e-8;
  max_iter_linear = 1e4;
  max_n_tmp_vectors = 100;

  // preconditioning linear solver
  preconditioner_linearized_navier_stokes = PreconditionerLinearizedNavierStokes::BlockTriangular;

  // preconditioner velocity/momentum block
  momentum_preconditioner = MomentumPreconditioner::VelocityDiffusion; //InverseMassMatrix; //VelocityDiffusion;
  multigrid_data_momentum_preconditioner.coarse_solver = MultigridCoarseGridSolver::GMRES_Jacobi;
  exact_inversion_of_momentum_block = false;
  rel_tol_solver_momentum_preconditioner = 1.e-3;
  max_n_tmp_vectors_solver_momentum_preconditioner = 100;

  // preconditioner Schur-complement block
  schur_complement_preconditioner = SchurComplementPreconditioner::PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  multigrid_data_schur_complement_preconditioner.coarse_solver = MultigridCoarseGridSolver::PCG_Jacobi;
  exact_inversion_of_laplace_operator = false;
  rel_tol_solver_schur_complement_preconditioner = 1.e-6;


  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;

  // write output for visualization of results
  output_data.write_output = true; //false; //true;
  output_data.output_prefix = OUTPUT_PREFIX;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.compute_divergence = true;
  output_data.number_of_patches = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  output_solver_info_every_timesteps = 1e5;

  // restart
  write_restart = false;
  restart_interval_time = 1.e2;
  restart_interval_wall_time = 1.e6;
  restart_every_timesteps = 1e8;

  // lift and drag
  lift_and_drag_data.calculate_lift_and_drag = true;
  lift_and_drag_data.viscosity = viscosity;
  const double U = Um * (DIMENSION == 2 ? 2./3. : 4./9.);
  if(DIMENSION == 2)
    lift_and_drag_data.reference_value = 1.0/2.0*pow(U,2.0)*D;
  else if(DIMENSION == 3)
    lift_and_drag_data.reference_value = 1.0/2.0*pow(U,2.0)*D*H;

  // surfaces for calculation of lift and drag coefficients have boundary_ID = 2
  lift_and_drag_data.boundary_IDs.insert(2);

  lift_and_drag_data.filename_prefix_lift = OUTPUT_FOLDER + output_data.output_prefix; //"paper/pressure_correction/" + output_data.output_prefix;
  lift_and_drag_data.filename_prefix_drag = OUTPUT_FOLDER + output_data.output_prefix; //"paper/pressure_correction/" + output_data.output_prefix;

  // pressure difference
  pressure_difference_data.calculate_pressure_difference = true;
  if(DIMENSION == 2)
  {
    Point<dim> point_1_2D((X_C-D/2.0),Y_C), point_2_2D((X_C+D/2.0),Y_C);
    pressure_difference_data.point_1 = point_1_2D;
    pressure_difference_data.point_2 = point_2_2D;
  }
  else if(DIMENSION == 3)
  {
    Point<dim> point_1_3D((X_C-D/2.0),Y_C,H/2.0), point_2_3D((X_C+D/2.0),Y_C,H/2.0);
    pressure_difference_data.point_1 = point_1_3D;
    pressure_difference_data.point_2 = point_2_3D;
  }

  pressure_difference_data.filename_prefix_pressure_difference = OUTPUT_FOLDER + output_data.output_prefix; //"paper/pressure_correction/" + output_data.output_prefix;

  mass_data.calculate_error = false; //true;
  mass_data.start_time = 0.0;
  mass_data.sample_every_time_steps = 1;
  mass_data.filename_prefix = OUTPUT_PREFIX;
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

  if(component == 0 && std::abs(p[0]-(dim==2 ? L1: 0.0))<1.e-12)
  {
    const double pi = numbers::PI;
    const double T = 1.0;
    double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
    if(TEST_CASE < 3)
    {
      if(PROBLEM_TYPE == ProblemType::Steady)
      {
        result = coefficient * p[1] * (H-p[1]);
      }
      else if(PROBLEM_TYPE == ProblemType::Unsteady)
      {
        result = coefficient * p[1] * (H-p[1]) * ( (t/T)<1.0 ? std::sin(pi/2.*t/T) : 1.0);
      }
    }
    if(TEST_CASE == 3)
      result = coefficient * p[1] * (H-p[1]) * std::sin(pi*t/END_TIME);
    if (dim == 3)
      result *= p[2] * (H-p[2]);
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

  // For this flow problem no analytical solution is available.
  // Set the pressure to zero at the outflow boundary. This is
  // already done since result is initialized with a value of 0.0.
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
double NeumannBoundaryVelocity<dim>::value(const Point<dim> &/*p*/,
                                           const unsigned int /*component*/) const
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

  if(component == 0 && std::abs(p[0]-(dim==2 ? L1 : 0.0))<1.e-12)
  {
    const double pi = numbers::PI;
    const double T = 1.0;
    double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
    if(TEST_CASE < 3)
      result = coefficient * p[1] * (H-p[1]) * ( (t/T)<1.0 ? (pi/2./T)*std::cos(pi/2.*t/T) : 0.0);
    if(TEST_CASE == 3)
      result = coefficient * p[1] * (H-p[1]) * std::cos(pi*t/END_TIME)*pi/END_TIME;
    if (dim == 3)
      result *= p[2] * (H-p[2]);
  }

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

 // needed for mesh type 2 with two layers of spherical cells around cylinder
const double R_1 = 1.2*R;
const double R_2 = 1.7*R;
const double R_3 = 1.75*R;

// manifold ID of spherical manifold
const unsigned int MANIFOLD_ID = 10;

#include "../../include/OneSidedSphericalManifold.h"

// vectors of manifold_ids and face_ids
std::vector<unsigned int> manifold_ids;
std::vector<unsigned int> face_ids;

template<int dim>
void set_boundary_ids(Triangulation<dim> &tria, bool compute_in_2d)
{
  // Set the cylinder boundary to 2, outflow to 1, the rest to 0.
  for (typename Triangulation<dim>::active_cell_iterator cell=tria.begin(); cell != tria.end(); ++cell)
  {
    for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)// loop over cells
    {
      if (cell->face(f)->at_boundary())
      {
        Point<dim> point_on_centerline;
        point_on_centerline[0] = X_C;
        point_on_centerline[1] = Y_C;
        if(dim==3)
          point_on_centerline[dim-1] = cell->face(f)->center()[2];

        if (std::abs(cell->face(f)->center()[0] - (compute_in_2d ? L1 : X_0)) < 1e-12)
          cell->face(f)->set_all_boundary_ids(0);
        else if (std::abs(cell->face(f)->center()[0]-L2) < 1e-12)
          cell->face(f)->set_all_boundary_ids(1);
        else if (point_on_centerline.distance(cell->face(f)->center()) <= R)
          cell->face(f)->set_all_boundary_ids(2);
        else
          cell->face(f)->set_all_boundary_ids(0);
      }
    }
  }
}

void create_triangulation(Triangulation<2> &tria, const bool compute_in_2d = true)
{
  AssertThrow(std::abs((X_2-X_1) - 2.0*(X_C-X_1))<1.0e-12, ExcMessage("Geometry parameters X_1, X_2, X_C invalid!"));

  Point<2> center = Point<2>(X_C,Y_C);

  if(MESH_TYPE == MeshType::Type1)
  {
    SphericalManifold<2> boundary(center);
    Triangulation<2> left, middle, right, tmp, tmp2;
    std::vector<unsigned int> ref_1(2, 2);
    ref_1[1] = 2;

    GridGenerator::subdivided_hyper_rectangle(left, ref_1 ,Point<2>(X_0,Y_0), Point<2>(X_1, H), false);
    std::vector<unsigned int> ref_2(2, 9);
    ref_2[1] = 2;

    GridGenerator::subdivided_hyper_rectangle(right, ref_2,Point<2>(X_2,Y_0), Point<2>(L2, H), false);

    // create middle part first as a hyper shell
    /*const double outer_radius = (X_2-X_1)/2.0;*/
    const unsigned int n_cells = 4;
    // use value of 0.2 in the following line instead of outer_radius since this yields
    // different results for the pressure-difference --> TODO
    GridGenerator::hyper_shell(middle, center, R, 0.2, n_cells, true);
    middle.set_manifold(0, boundary);
    middle.refine_global(1);

    // then move the vertices to the points where we want them to be to create a slightly asymmetric cube with a hole
    for (Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
    {
     for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
     {
       Point<2> &vertex = cell->vertex(v);
       if (std::abs(vertex[0] - 0.7) < 1e-10 && std::abs(vertex[1] - 0.2) < 1e-10)
         vertex = Point<2>(0.7, 0.205);
       else if (std::abs(vertex[0] - 0.6) < 1e-10 && std::abs(vertex[1] - 0.3) < 1e-10)
         vertex = Point<2>(0.7, 0.41);
       else if (std::abs(vertex[0] - 0.6) < 1e-10 && std::abs(vertex[1] - 0.1) < 1e-10)
         vertex = Point<2>(0.7, 0);
       else if (std::abs(vertex[0] - 0.5) < 1e-10 && std::abs(vertex[1] - 0.4) < 1e-10)
         vertex = Point<2>(0.5, 0.41);
       else if (std::abs(vertex[0] - 0.5) < 1e-10 && std::abs(vertex[1] - 0.0) < 1e-10)
         vertex = Point<2>(0.5, 0.0);
       else if (std::abs(vertex[0] - 0.4) < 1e-10 && std::abs(vertex[1] - 0.3) < 1e-10)
         vertex = Point<2>(0.3, 0.41);
       else if (std::abs(vertex[0] - 0.4) < 1e-10 && std::abs(vertex[1] - 0.1) < 1e-10)
         vertex = Point<2>(0.3, 0);
       else if (std::abs(vertex[0] - 0.3) < 1e-10 && std::abs(vertex[1] - 0.2) < 1e-10)
         vertex = Point<2>(0.3, 0.205);
       else if (std::abs(vertex[0] - 0.56379) < 1e-4 && std::abs(vertex[1] - 0.13621) < 1e-4)
         vertex = Point<2>(0.59, 0.11);
       else if (std::abs(vertex[0] - 0.56379) < 1e-4 && std::abs(vertex[1] - 0.26379) < 1e-4)
         vertex = Point<2>(0.59, 0.29);
       else if (std::abs(vertex[0] - 0.43621) < 1e-4 && std::abs(vertex[1] - 0.13621) < 1e-4)
         vertex = Point<2>(0.41, 0.11);
       else if (std::abs(vertex[0] - 0.43621) < 1e-4 && std::abs(vertex[1] - 0.26379) < 1e-4)
         vertex = Point<2>(0.41, 0.29);
     }
    }


    // must copy the triangulation because we cannot merge triangulations with
    // refinement...
    GridGenerator::flatten_triangulation(middle, tmp2);

    if (compute_in_2d)
    {
     GridGenerator::merge_triangulations (tmp2, right, tria);
    }
    else
    {
     GridGenerator::merge_triangulations (left, tmp2, tmp);
     GridGenerator::merge_triangulations (tmp, right, tria);
    }

    if (compute_in_2d)
    {
      // set manifold ID's
      tria.set_all_manifold_ids(0);

      for (Triangulation<2>::active_cell_iterator cell=tria.begin(); cell != tria.end(); ++cell)
      {
        if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
        {
          for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
          {
            if (cell->face(f)->at_boundary() && center.distance(cell->face(f)->center())<=R)
            {
              cell->face(f)->set_all_manifold_ids(MANIFOLD_ID);
            }
          }
        }
        else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
        {
          for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
          {
            bool face_at_sphere_boundary = true;
            for (unsigned int v=0; v<GeometryInfo<2-1>::vertices_per_cell; ++v)
            {
              if (std::abs(center.distance(cell->face(f)->vertex(v)) - R) > 1e-12)
                face_at_sphere_boundary = false;
            }
            if (face_at_sphere_boundary)
            {
              face_ids.push_back(f);
              unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
              cell->set_all_manifold_ids(manifold_id);
              manifold_ids.push_back(manifold_id);
            }
          }
        }
        else
        {
          AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold || MANIFOLD_TYPE == ManifoldType::VolumeManifold,
              ExcMessage("Specified manifold type not implemented"));
        }
      }
    }
  }
  else if(MESH_TYPE == MeshType::Type2)
  {
    SphericalManifold<2> spherical_manifold(center);

    Triangulation<2> left, circle_1, circle_2, circle_tmp, middle, middle_tmp, middle_tmp2, right, tmp_3D;
    std::vector<unsigned int> ref_1(2, 2);
    ref_1[1] = 2;

    GridGenerator::subdivided_hyper_rectangle(left, ref_1 ,Point<2>(X_0,Y_0), Point<2>(X_1, H), false);
    std::vector<unsigned int> ref_2(2, 9);
    ref_2[1] = 2;

    GridGenerator::subdivided_hyper_rectangle(right, ref_2, Point<2>(X_2, Y_0), Point<2>(L2, H), false);

    // create middle part first as a hyper shell
    const double outer_radius = (X_2-X_1)/2.0;
    const unsigned int n_cells = 4;
    GridGenerator::hyper_shell(middle, center, R_2, outer_radius, n_cells, true);
    middle.set_all_manifold_ids(MANIFOLD_ID);
    middle.set_manifold(MANIFOLD_ID, spherical_manifold);
    middle.refine_global(1);

    // two inner circles in order to refine towards the cylinder surface
    const unsigned int n_cells_circle = 8;
    GridGenerator::hyper_shell(circle_1, center, R, R_1, n_cells_circle, true);
    GridGenerator::hyper_shell(circle_2, center, R_1, R_2, n_cells_circle, true);

    // then move the vertices to the points where we want them to be to create a slightly asymmetric cube with a hole
    for (Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
    {
      for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
      {
        Point<2> &vertex = cell->vertex(v);
        if (std::abs(vertex[0] - X_2) < 1e-10 && std::abs(vertex[1] - Y_C) < 1e-10)
        {
          vertex = Point<2>(X_2, H/2.0);
        }
        else if (std::abs(vertex[0] - (X_C + (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10 && std::abs(vertex[1] - (Y_C + (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10)
        {
          vertex = Point<2>(X_2, H);
        }
        else if (std::abs(vertex[0] - (X_C + (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10 && std::abs(vertex[1] - (Y_C - (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10)
        {
          vertex = Point<2>(X_2, Y_0);
        }
        else if (std::abs(vertex[0] - X_C) < 1e-10 && std::abs(vertex[1] - (Y_C +(X_2-X_1)/2.0)) < 1e-10)
        {
          vertex = Point<2>(X_C, H);
        }
        else if (std::abs(vertex[0] - X_C) < 1e-10 && std::abs(vertex[1] - (Y_C-(X_2-X_1)/2.0)) < 1e-10)
        {
          vertex = Point<2>(X_C, Y_0);
        }
        else if (std::abs(vertex[0] - (X_C - (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10 && std::abs(vertex[1] - (Y_C + (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10)
        {
          vertex = Point<2>(X_1, H);
        }
        else if (std::abs(vertex[0] - (X_C - (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10 && std::abs(vertex[1] - (Y_C - (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10)
        {
          vertex = Point<2>(X_1, Y_0);
        }
        else if (std::abs(vertex[0] - X_1) < 1e-10 && std::abs(vertex[1] - Y_C) < 1e-10)
        {
          vertex = Point<2>(X_1, H/2.0);
        }
      }
    }

    // must copy the triangulation because we cannot merge triangulations with refinement...
    GridGenerator::flatten_triangulation(middle, middle_tmp);

    GridGenerator::merge_triangulations(circle_1,circle_2,circle_tmp);
    GridGenerator::merge_triangulations(middle_tmp,circle_tmp,middle_tmp2);

    if (compute_in_2d)
    {
      GridGenerator::merge_triangulations(middle_tmp2,right,tria);
    }
    else // 3D
    {
      GridGenerator::merge_triangulations (left, middle_tmp2, tmp_3D);
      GridGenerator::merge_triangulations (tmp_3D, right, tria);
    }

    if (compute_in_2d)
    {
      // set manifold ID's
      tria.set_all_manifold_ids(0);

      for (Triangulation<2>::active_cell_iterator cell=tria.begin(); cell != tria.end(); ++cell)
      {
        if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
        {
          if(center.distance(cell->center())<= R_2)
            cell->set_all_manifold_ids(MANIFOLD_ID);
        }
        else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
        {
          if(center.distance(cell->center())<= R_2)
            cell->set_all_manifold_ids(MANIFOLD_ID);
          else
          {
            for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
            {
              bool face_at_sphere_boundary = true;
              for (unsigned int v=0; v<GeometryInfo<2-1>::vertices_per_cell; ++v)
              {
                if (std::abs(center.distance(cell->face(f)->vertex(v)) - R_2) > 1e-12)
                  face_at_sphere_boundary = false;
              }
              if (face_at_sphere_boundary)
              {
                face_ids.push_back(f);
                unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
                cell->set_all_manifold_ids(manifold_id);
                manifold_ids.push_back(manifold_id);
              }
            }
          }
        }
      }
    }
  }
  else if(MESH_TYPE == MeshType::Type3)
  {
    SphericalManifold<2> spherical_manifold(center);

    Triangulation<2> left, middle, circle, middle_tmp, right, tmp_3D;

    // left part (only needed for 3D problem)
    std::vector<unsigned int> ref_1(2, 1);
    GridGenerator::subdivided_hyper_rectangle(left, ref_1 ,Point<2>(X_0,Y_0), Point<2>(X_1, H), false);

    // right part (2D and 3D)
    std::vector<unsigned int> ref_2(2, 5);
    ref_2[1] = 1;
    GridGenerator::subdivided_hyper_rectangle(right, ref_2, Point<2>(X_2, Y_0), Point<2>(L2, H), false);

    // middle part
    const double outer_radius = (X_2-X_1)/2.0;
    const unsigned int n_cells = 4;
    Point<2> origin;

    // inner circle around cylinder
    GridGenerator::hyper_shell(circle, origin, R, R_3, n_cells, true);
    GridTools::rotate(numbers::PI/4, circle);
    GridTools::shift(Point<2>(outer_radius+X_1,outer_radius),circle);

    // create middle part first as a hyper shell
    GridGenerator::hyper_shell(middle, origin, R_3, outer_radius*std::sqrt(2.0), n_cells, true);
    GridTools::rotate(numbers::PI/4, middle);
    GridTools::shift(Point<2>(outer_radius+X_1,outer_radius),middle);

    // then move the vertices to the points where we want them to be
    for (Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
    {
      for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
      {
        Point<2> &vertex = cell->vertex(v);
        if (std::abs(vertex[0] - X_1) < 1e-10 && std::abs(vertex[1] - (X_2-X_1)) < 1e-10)
        {
          vertex = Point<2>(X_1, H);
        }
        else if (std::abs(vertex[0] - X_2) < 1e-10 && std::abs(vertex[1] - (X_2-X_1)) < 1e-10)
        {
          vertex = Point<2>(X_2, H);
        }
      }
    }

    GridGenerator::merge_triangulations(circle,middle,middle_tmp);

    if (compute_in_2d)
    {
      GridGenerator::merge_triangulations(middle_tmp,right,tria);
    }
    else // 3D
    {
      GridGenerator::merge_triangulations (left, middle_tmp, tmp_3D);
      GridGenerator::merge_triangulations (tmp_3D, right, tria);
    }

    if (compute_in_2d)
    {
      // set manifold ID's
      tria.set_all_manifold_ids(0);

      for (Triangulation<2>::active_cell_iterator cell=tria.begin(); cell != tria.end(); ++cell)
      {
        if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
        {
          if(center.distance(cell->center())<= R_3)
            cell->set_all_manifold_ids(MANIFOLD_ID);
          else
          {
            for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
            {
              bool face_at_sphere_boundary = true;
              for (unsigned int v=0; v<GeometryInfo<2-1>::vertices_per_cell; ++v)
              {
                if (std::abs(center.distance(cell->face(f)->vertex(v)) - R_3) > 1e-12)
                  face_at_sphere_boundary = false;
              }
              if (face_at_sphere_boundary)
              {
                face_ids.push_back(f);
                unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
                cell->set_all_manifold_ids(manifold_id);
                manifold_ids.push_back(manifold_id);
              }
            }
          }
        }
        else
        {
          AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold, ExcMessage("Specified manifold type not implemented."));
        }
      }
    }
  }

  // Set boundary ID's
  // Set the cylinder boundary to 2, outflow to 1, the rest to 0.
//  for (Triangulation<2>::active_cell_iterator cell=tria.begin(); cell != tria.end(); ++cell)
//  {
//    for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)// loop over cells
//    {
//      if (cell->face(f)->at_boundary())
//      {
//        if (std::abs(cell->face(f)->center()[0] - (compute_in_2d ? L1 : X_0)) < 1e-12)
//          cell->face(f)->set_all_boundary_ids(0);
//        else if (std::abs(cell->face(f)->center()[0]-L2) < 1e-12)
//          cell->face(f)->set_all_boundary_ids(1);
//        else if (center.distance(cell->face(f)->center()) <= R)
//          cell->face(f)->set_all_boundary_ids(2);
//        else
//          cell->face(f)->set_all_boundary_ids(0);
//      }
//    }
//  }

  if(compute_in_2d == true)
  {
    // Set boundary ID's
    set_boundary_ids<2>(tria, compute_in_2d);
  }
}


void create_triangulation(Triangulation<3> &tria)
{
 Triangulation<2> tria_2d;
 create_triangulation(tria_2d, false);

 if(MESH_TYPE == MeshType::Type1)
 {
   GridGenerator::extrude_triangulation(tria_2d, 3, H, tria);

   // set manifold ID's
   tria.set_all_manifold_ids(0);

   if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
   {
     for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
     {
       for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
       {
         if (cell->face(f)->at_boundary() && Point<3>(X_C,Y_C,cell->face(f)->center()[2]).distance(cell->face(f)->center()) <= R)
         {
           cell->face(f)->set_all_manifold_ids(MANIFOLD_ID);
         }
       }
     }
   }
   else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
   {
     for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
     {
       for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
       {
         bool face_at_sphere_boundary = true;
         for (unsigned int v=0; v<GeometryInfo<3-1>::vertices_per_cell; ++v)
         {
           if (std::abs(Point<3>(X_C,Y_C,cell->face(f)->vertex(v)[2]).distance(cell->face(f)->vertex(v)) - R) > 1e-12)
             face_at_sphere_boundary = false;
         }
         if (face_at_sphere_boundary)
         {
           face_ids.push_back(f);
           unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
           cell->set_all_manifold_ids(manifold_id);
           manifold_ids.push_back(manifold_id);
         }
       }
     }
   }
   else
   {
     AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold || MANIFOLD_TYPE == ManifoldType::VolumeManifold,
         ExcMessage("Specified manifold type not implemented"));
   }
 }
 else if(MESH_TYPE == MeshType::Type2)
 {
   GridGenerator::extrude_triangulation(tria_2d, 3, H, tria);

   // set manifold ID's
   tria.set_all_manifold_ids(0);

   if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
   {
     for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
     {
       if(Point<3>(X_C,Y_C,cell->center()[2]).distance(cell->center()) <= R_2)
         cell->set_all_manifold_ids(MANIFOLD_ID);
     }
   }
   else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
   {
     for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
     {
       if(Point<3>(X_C,Y_C,cell->center()[2]).distance(cell->center())<= R_2)
         cell->set_all_manifold_ids(MANIFOLD_ID);
       else
       {
         for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
         {
           bool face_at_sphere_boundary = true;
           for (unsigned int v=0; v<GeometryInfo<3-1>::vertices_per_cell; ++v)
           {
             if (std::abs(Point<3>(X_C,Y_C,cell->face(f)->vertex(v)[2]).distance(cell->face(f)->vertex(v)) - R_2) > 1e-12)
               face_at_sphere_boundary = false;
           }
           if (face_at_sphere_boundary)
           {
             face_ids.push_back(f);
             unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
             cell->set_all_manifold_ids(manifold_id);
             manifold_ids.push_back(manifold_id);
           }
         }
       }
     }
   }
   else
   {
     AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold || MANIFOLD_TYPE == ManifoldType::VolumeManifold,
         ExcMessage("Specified manifold type not implemented"));
   }
 }
 else if(MESH_TYPE == MeshType::Type3)
 {
   GridGenerator::extrude_triangulation(tria_2d, 2, H, tria);

   // set manifold ID's
   tria.set_all_manifold_ids(0);

   if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
   {
    for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
    {
      if(Point<3>(X_C,Y_C,cell->center()[2]).distance(cell->center())<= R_3)
        cell->set_all_manifold_ids(MANIFOLD_ID);
      else
      {
        for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
        {
          bool face_at_sphere_boundary = true;
          for (unsigned int v=0; v<GeometryInfo<3-1>::vertices_per_cell; ++v)
          {
            if (std::abs(Point<3>(X_C,Y_C,cell->face(f)->vertex(v)[2]).distance(cell->face(f)->vertex(v)) - R_3) > 1e-12)
              face_at_sphere_boundary = false;
          }
          if (face_at_sphere_boundary)
          {
            face_ids.push_back(f);
            unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
            cell->set_all_manifold_ids(manifold_id);
            manifold_ids.push_back(manifold_id);
          }
        }
      }
    }
  }
  else
  {
    AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold, ExcMessage("Specified manifold type not implemented"));
  }
 }
 else
 {
   AssertThrow(MESH_TYPE == MeshType::Type1 || MESH_TYPE == MeshType::Type2 || MESH_TYPE == MeshType::Type3,
       ExcMessage("Specified mesh type not implemented"));
 }

// // Set boundary ID's
// // Set the cylinder boundary to 2, outflow to 1, the rest to 0.
// for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
// {
//   for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
//   {
//     if (cell->face(f)->at_boundary())
//     {
//       if (std::abs(cell->face(f)->center()[0] - X_0) < 1e-12)
//         cell->face(f)->set_all_boundary_ids(0);
//       else if (std::abs(cell->face(f)->center()[0]-L2) < 1e-12)
//         cell->face(f)->set_all_boundary_ids(1);
//       else if (Point<3>(X_C,Y_C,cell->face(f)->center()[2]).distance(cell->face(f)->center()) <= R)
//         cell->face(f)->set_all_boundary_ids(2);
//       else
//         cell->face(f)->set_all_boundary_ids(0);
//     }
//   }
// }

  // Set boundary ID's
  set_boundary_ids<3>(tria, false);
}

template<int dim>
void create_grid_and_set_boundary_conditions(
   parallel::distributed::Triangulation<dim>                   &triangulation,
   unsigned int const                                          n_refine_space,
   std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity,
   std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure,
   std::vector<GridTools::PeriodicFacePair<typename
     Triangulation<dim>::cell_iterator> >                      &/*periodic_faces*/)
{
 Point<dim> center;
 center[0] = X_C;
 center[1] = Y_C;

 // apply this manifold for all mesh types
 Point<dim> direction;
 direction[dim-1] = 1.;

 static std_cxx11::shared_ptr<Manifold<dim> > cylinder_manifold;

 if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
 {
   cylinder_manifold = std_cxx11::shared_ptr<Manifold<dim> >(dim == 2 ? static_cast<Manifold<dim>*>(new SphericalManifold<dim>(center)) :
                                           static_cast<Manifold<dim>*>(new CylindricalManifold<dim>(direction, center)));
 }
 else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
 {
   cylinder_manifold = std_cxx11::shared_ptr<Manifold<dim> >(static_cast<Manifold<dim>*>(new MyCylindricalManifold<dim>(center)));
 }
 else
 {
   AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold || MANIFOLD_TYPE == ManifoldType::VolumeManifold,
       ExcMessage("Specified manifold type not implemented"));
 }

 create_triangulation(triangulation);
 triangulation.set_manifold(MANIFOLD_ID, *cylinder_manifold);

 // generate vector of manifolds and apply manifold to all cells that have been marked
 static std::vector<std_cxx11::shared_ptr<Manifold<dim> > > manifold_vec;
 manifold_vec.resize(manifold_ids.size());

 for(unsigned int i=0;i<manifold_ids.size();++i)
 {
   for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin(); cell != triangulation.end(); ++cell)
   {
     if(cell->manifold_id() == manifold_ids[i])
     {
       manifold_vec[i] = std_cxx11::shared_ptr<Manifold<dim> >(
           static_cast<Manifold<dim>*>(new OneSidedCylindricalManifold<dim>(cell,face_ids[i],center)));
       triangulation.set_manifold(manifold_ids[i],*(manifold_vec[i]));
     }
   }
 }

 triangulation.refine_global(n_refine_space);

 // fill boundary descriptor velocity
 std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
 analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
 // Dirichlet boundaries: ID = 0, 2
 boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (0,analytical_solution_velocity));
 boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (2,analytical_solution_velocity));

 std_cxx11::shared_ptr<Function<dim> > neumann_bc_velocity;
 neumann_bc_velocity.reset(new NeumannBoundaryVelocity<dim>());
 // Neumann boundaris: ID = 1
 boundary_descriptor_velocity->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                   (1,neumann_bc_velocity));

 // fill boundary descriptor pressure
 std_cxx11::shared_ptr<Function<dim> > pressure_bc_dudt;
 pressure_bc_dudt.reset(new PressureBC_dudt<dim>());
 // Dirichlet boundaries: ID = 0, 2
 boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (0,pressure_bc_dudt));
 boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (2,pressure_bc_dudt));

 std_cxx11::shared_ptr<Function<dim> > analytical_solution_pressure;
 analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
 // Neumann boundaries: ID = 1
 boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                  (1,analytical_solution_pressure));
}


template<int dim>
void set_field_functions(std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std_cxx11::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new ZeroFunction<dim>(dim));
  std_cxx11::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new ZeroFunction<dim>(1));

  std_cxx11::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure = initial_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_analytical_solution(std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new ZeroFunction<dim>(1));
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



#endif /* APPLICATIONS_NAVIERSTOKESTESTCASES_FLOWPASTCYLINDER_H_ */
