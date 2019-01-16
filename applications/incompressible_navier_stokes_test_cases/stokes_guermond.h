/*
 * StokesGuermond.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_GUERMOND_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_GUERMOND_H_


#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>

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
unsigned int const FE_DEGREE_VELOCITY = 5;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 0;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const double VISCOSITY = 1.0e-6;

enum class MeshType{ UniformCartesian, Complex };
const MeshType MESH_TYPE = MeshType::UniformCartesian;

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::Stokes;
  formulation_viscous_term = FormulationViscousTerm::LaplaceFormulation;
  right_hand_side = true;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 1.0;
  viscosity = VISCOSITY; // VISCOSITY is also needed somewhere else


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFCoupledSolution; //BDFDualSplittingScheme; //BDFPressureCorrection;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
  max_velocity = 2.65;
  cfl = 2.0e-1;
  time_step_size = 1.e-2;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 3; // 1; // 2; // 3;
  start_with_low_order = false; // true; // false;

  // SPATIAL DISCRETIZATION

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  pure_dirichlet_bc = true;
  adjust_pressure_level = AdjustPressureLevel::ApplyZeroMeanValue;

  // divergence and continuity penalty terms
  use_divergence_penalty = false;
  use_continuity_penalty = false;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_pressure_poisson = SolverPressurePoisson::CG;
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-8);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::Chebyshev;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, 1.e-20, 1.e-12);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator <=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,1.e-12,1.e-8);
  preconditioner_viscous = PreconditionerViscous::Multigrid;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::Chebyshev;


  // PRESSURE-CORRECTION SCHEME

  // momentum step

  // Newton solver

  // linear solver
  solver_momentum = SolverMomentum::GMRES;
  solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-8, 100);
  preconditioner_momentum = MomentumPreconditioner::InverseMassMatrix;
  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::Chebyshev;
  update_preconditioner_momentum = false;

  // formulation
  order_pressure_extrapolation = 1;
  rotational_formulation = true;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)

  // linear solver
  solver_coupled = SolverCoupled::GMRES;
  solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-8, 100);

  // preconditioning linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

  // preconditioner velocity/momentum block
  preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_data.write_output = false; //true;
  output_data.output_folder = "output/stokes_guermond/";
  output_data.output_name = "stokes_guermond";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/10;
  output_data.write_divergence = false;
  output_data.degree = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = true;
  error_data.calculate_relative_errors = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  output_solver_info_every_timesteps = 1e5;
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity (const unsigned int  n_components = dim,
                              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double t = this->get_time();
    double result = 0.0;

    const double pi = numbers::PI;
    double sint = std::sin(t);
    double sinx = std::sin(pi*p[0]);
    double siny = std::sin(pi*p[1]);
    double sin2x = std::sin(2.*pi*p[0]);
    double sin2y = std::sin(2.*pi*p[1]);
    if (component == 0)
      result = pi*sint*sin2y*std::pow(sinx,2.);
    else if (component == 1)
      result = -pi*sint*sin2x*std::pow(siny,2.);

    return result;
  }
};

template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  double value (const Point<dim>   &p,
                const unsigned int /*component*/) const
  {
    double t = this->get_time();
    double result = 0.0;

    const double pi = numbers::PI;
    double sint = std::sin(t);
    double siny = std::sin(pi*p[1]);
    double cosx = std::cos(pi*p[0]);
    result = cosx*siny*sint;

    return result;
  }
};

template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double t = this->get_time();
    double result = 0.0;

    const double pi = numbers::PI;
    double cost = std::cos(t);
    double sinx = std::sin(pi*p[0]);
    double siny = std::sin(pi*p[1]);
    double sin2x = std::sin(2.*pi*p[0]);
    double sin2y = std::sin(2.*pi*p[1]);
    if (component == 0)
      result = pi*cost*sin2y*std::pow(sinx,2.);
    else if (component == 1)
      result = -pi*cost*sin2x*std::pow(siny,2.);

    return result;
  }
};

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

   double value (const Point<dim>    &p,
                 const unsigned int  component = 0) const
   {
     double t = this->get_time();
     double result = 0.0;

     const double pi = numbers::PI;
     double sint = std::sin(t);
     double cost = std::cos(t);
     double sinx = std::sin(pi*p[0]);
     double siny = std::sin(pi*p[1]);
     double cosx = std::cos(pi*p[0]);
     double cosy = std::cos(pi*p[1]);
     double sin2x = std::sin(2.*pi*p[0]);
     double sin2y = std::sin(2.*pi*p[1]);
     if (component == 0)
     {
       result = + pi*cost*sin2y*std::pow(sinx,2.)
                - 2.*std::pow(pi,3.)*sint*sin2y*(1.-4.*std::pow(sinx,2.))*VISCOSITY
                - pi*sint*sinx*siny;
     }
     else if (component == 1)
     {
       result = - pi*cost*sin2x*std::pow(siny,2.)
                + 2.*std::pow(pi,3.)*sint*sin2x*(1.-4.*std::pow(siny,2.))*VISCOSITY
                + pi*sint*cosx*cosy;
     }

     return result;
   }
 };

/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

#include "../../include/functionalities/one_sided_cylindrical_manifold.h"

template<int dim>
void create_grid_and_set_boundary_conditions(
    parallel::distributed::Triangulation<dim>         &triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &/*periodic_faces*/)
{
  if(MESH_TYPE == MeshType::UniformCartesian)
  {
    // Uniform Cartesian grid
    const double left = 0.0, right = 1.0;
    GridGenerator::hyper_cube(triangulation,left,right);

    /****** test one-sided spherical manifold *********/
//    Point<dim> center = Point<dim>();
//    center[0] = 1.15;
//    center[1] = 0.5;
//    typename Triangulation<dim>::cell_iterator cell = triangulation.begin();
//    cell->set_all_manifold_ids(10);
//    //    cell->vertex(0)[1] = -1.0;
//    //    cell->vertex(2)[1] = 2.0;
//    //    cell->vertex(4)[1] = -1.0;
//    //    cell->vertex(6)[1] = 2.0;
//
//    static std::shared_ptr<Manifold<dim> > my_manifold =
//      std::shared_ptr<Manifold<dim> >(static_cast<Manifold<dim>*>(new OneSidedCylindricalManifold<dim>(cell,1,center)));
//    triangulation.set_manifold(10,*my_manifold);
    /****** test one-sided spherical manifold *********/

    triangulation.refine_global(n_refine_space);
  }
//  else if(MESH_TYPE == MeshType::Complex)
//  {
//    // Complex Geometry
//    Triangulation<dim> tria1, tria2;
//    GridGenerator::hyper_shell(tria1, Point<dim>(), 0.4, std::sqrt(dim), 2*dim);
//    if (dim == 2)
//      GridTools::rotate(numbers::PI/4, tria1);
//    GridGenerator::hyper_ball(tria2, Point<dim>(), 0.4);
//    GridGenerator::merge_triangulations(tria1, tria2, triangulation);
//    triangulation.set_all_manifold_ids(0);
//    for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin();cell != triangulation.end(); ++cell)
//    {
//      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
//      {
//        bool face_at_sphere_boundary = true;
//        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
//        {
//          if (std::abs(cell->face(f)->vertex(v).norm()-0.4) > 1e-12)
//            face_at_sphere_boundary = false;
//        }
//        if (face_at_sphere_boundary)
//        {
//          cell->face(f)->set_all_manifold_ids(1);
//        }
//      }
//      if (cell->center().norm() > 0.4)
//        cell->set_material_id(1);
//      else
//        cell->set_material_id(0);
//    }
//    static const SphericalManifold<dim> spherical_manifold;
//    triangulation.set_manifold(1, spherical_manifold);
//    triangulation.set_boundary(0);
//    triangulation.refine_global(n_refine_space);
//  }


  // test case with pure Dirichlet boundary conditions for velocity
  // all boundaries have ID = 0 by default

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new AnalyticalSolutionVelocity<dim>()));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new PressureBC_dudt<dim>()));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->right_hand_side.reset(new RightHandSide<dim>());
}

template<int dim>
void set_analytical_solution(std::shared_ptr<AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new AnalyticalSolutionVelocity<dim>());
  analytical_solution->pressure.reset(new AnalyticalSolutionPressure<dim>());
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

  std::shared_ptr<PostProcessor<dim,degree_u,degree_p,Number> > pp;
  pp.reset(new PostProcessor<dim,degree_u,degree_p,Number>(pp_data));

  return pp;
}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_STOKES_GUERMOND_H_ */
