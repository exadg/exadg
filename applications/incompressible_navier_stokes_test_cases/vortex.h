/*
 * Vortex.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_

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
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 5;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 1;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const double U_X_MAX = 1.0;
const double VISCOSITY = 2.5e-2; //1.e-2; //2.5e-2;
const FormulationViscousTerm FORMULATION_VISCOUS_TERM = FormulationViscousTerm::LaplaceFormulation;

enum class MeshType{ UniformCartesian, ComplexSurfaceManifold, ComplexVolumeManifold };
const MeshType MESH_TYPE = MeshType::UniformCartesian;

template<int dim>
void InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FORMULATION_VISCOUS_TERM;
  formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  right_hand_side = false;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 1.0;
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  solver_type = SolverType::Unsteady;
  temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  time_integrator_oif = TimeIntegratorOIF::ExplRK3Stage7Reg2;
  calculation_of_time_step_size = TimeStepCalculation::CFL;
  adaptive_time_stepping = false;
  max_velocity = 1.4 * U_X_MAX;
  cfl = 0.1;
  cfl_oif = cfl/1.0;
  cfl_exponent_fe_degree_velocity = 1.5;
  c_eff = 8.0;
  time_step_size = 5.e-5;
  order_time_integrator = 2;
  start_with_low_order = false;


  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // mapping
  degree_mapping = FE_DEGREE_VELOCITY;

  // convective term
  upwind_factor = 1.0;

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  penalty_term_div_formulation = PenaltyTermDivergenceFormulation::Symmetrized;

  // special case: pure DBC's
  pure_dirichlet_bc = false;

  // divergence and continuity penalty terms
  add_penalty_terms_to_monolithic_system = false;

  // NUMERICAL PARAMETERS
  implement_block_diagonal_preconditioner_matrix_free = true;
  use_cell_based_face_loops = true;

  // PROJECTION METHODS

  // pressure Poisson equation
  solver_pressure_poisson = SolverPressurePoisson::CG;
  solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
  multigrid_data_pressure_poisson.smoother_data.preconditioner = PreconditionerSmoother::PointJacobi;

  // projection step
  solver_projection = SolverProjection::CG;
  solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  preconditioner_block_diagonal_projection = PreconditionerBlockDiagonal::InverseMassMatrix;
  solver_data_block_diagonal_projection = SolverData(1000,1.e-12,1.e-2,1000);

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  order_extrapolation_pressure_nbc = order_time_integrator<=2 ? order_time_integrator : 2;

  // viscous step
  solver_viscous = SolverViscous::CG;
  solver_data_viscous = SolverData(1000,1.e-12,1.e-6);
  preconditioner_viscous = PreconditionerViscous::Multigrid; //InverseMassMatrix; //Multigrid;
  update_preconditioner_viscous = true;


  // PRESSURE-CORRECTION SCHEME
  // formulation
  order_pressure_extrapolation = order_time_integrator-1;
  rotational_formulation = true;

  // momentum step

  // Newton solver
  newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  solver_momentum = SolverMomentum::FGMRES;
  solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);
  preconditioner_momentum = MomentumPreconditioner::Multigrid; //BlockJacobi; //InverseMassMatrix;
  multigrid_operator_type_momentum = MultigridOperatorType::ReactionConvectionDiffusion;
  multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
  update_preconditioner_momentum = true;

  // Jacobi smoother data
  multigrid_data_momentum.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  multigrid_data_momentum.smoother_data.iterations = 5;
  multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // Chebyshev smoother data
//  multigrid_data_momentum.smoother = MultigridSmoother::Chebyshev;
//  multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::Chebyshev;

  // GMRES smoother data
//    multigrid_data_momentum.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::BlockJacobi;


  // COUPLED NAVIER-STOKES SOLVER
  use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  solver_coupled = SolverCoupled::FGMRES; //FGMRES; //GMRES;
  solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-6, 100);

  // preconditioner linear solver
  preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  update_preconditioner_coupled = true;

  // preconditioner momentum block
  preconditioner_velocity_block = MomentumPreconditioner::Multigrid; //InverseMassMatrix;
  multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;
  multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Jacobi; //Jacobi; //Chebyshev; //GMRES;
  multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi; //PointJacobi; //BlockJacobi;
  multigrid_data_velocity_block.smoother_data.iterations = 5;
  multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
  // coarse grid solver
  multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // preconditioner Schur-complement block
  preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;


  // OUTPUT AND POSTPROCESSING

  // print input parameters
  print_input_parameters = true; //false;

  // write output for visualization of results
  output_data.write_output = false;
  output_data.output_folder = "output/vortex/vtu/";
  output_data.output_name = "vortex";
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.write_vorticity = true;
  output_data.write_divergence = true;
  output_data.write_velocity_magnitude = true;
  output_data.write_vorticity_magnitude = true;
  output_data.write_processor_id = true;
  output_data.mean_velocity.calculate = true;
  output_data.mean_velocity.sample_start_time = start_time;
  output_data.mean_velocity.sample_end_time = end_time;
  output_data.mean_velocity.sample_every_timesteps = 1;
  output_data.degree = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = true;
  error_data.calculate_relative_errors = true;
  error_data.calculate_H1_seminorm_velocity = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = end_time - start_time;
  error_data.write_errors_to_file = false;
  error_data.filename_prefix = "output/vortex/error";

  // analysis of mass conservation error
  mass_data.calculate_error = false;
  mass_data.start_time = 0.0;
  mass_data.sample_every_time_steps = 1;
  mass_data.filename_prefix = "test";

  // output of solver information
  output_solver_info_every_timesteps = 1e5;

  // restart
  restart_data.write_restart = false;
  restart_data.interval_time = 0.75;
  restart_data.interval_wall_time = 1.e6;
  restart_data.interval_time_steps = 1e8;
  restart_data.filename = "output/vortex/vortex";
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
    if(component == 0)
      result = -U_X_MAX*std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
    else if(component == 1)
      result = U_X_MAX*std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);

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
    result = -U_X_MAX*std::cos(2*pi*p[0])*std::cos(2*pi*p[1])*std::exp(-8.0*pi*pi*VISCOSITY*t);

    return result;
  }
};

template<int dim>
class NeumannBoundaryVelocity : public Function<dim>
{
public:
  NeumannBoundaryVelocity (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  double value (const Point<dim> &p,const unsigned int component = 0) const
  {
    double t = this->get_time();
    double result = 0.0;

    if(FORMULATION_VISCOUS_TERM == FormulationViscousTerm::LaplaceFormulation)
    {
      const double pi = numbers::PI;
      if(component==0)
      {
        if( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
          result = U_X_MAX*2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
        else if( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
          result = -U_X_MAX*2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
      }
      else if(component==1)
      {
        if( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
          result = -U_X_MAX*2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
        else if((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
          result = U_X_MAX*2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
      }
    }
    else if(FORMULATION_VISCOUS_TERM == FormulationViscousTerm::DivergenceFormulation)
    {
      const double pi = numbers::PI;
      if(component==0)
      {
        if( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
          result = -U_X_MAX*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
        else if( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
          result = U_X_MAX*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
      }
      else if(component==1)
      {
        if( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
          result = -U_X_MAX*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
        else if((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
          result = U_X_MAX*2.0*pi*(std::cos(2.0*pi*p[0]) - std::cos(2.0*pi*p[1]))*std::exp(-4.0*pi*pi*VISCOSITY*t);
      }
    }
    else
    {
      AssertThrow(FORMULATION_VISCOUS_TERM == FormulationViscousTerm::LaplaceFormulation ||
                  FORMULATION_VISCOUS_TERM == FormulationViscousTerm::DivergenceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented!"));
    }

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
    if(component == 0)
      result = U_X_MAX*4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
    else if(component == 1)
      result = -U_X_MAX*4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);

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
    std::shared_ptr<parallel::Triangulation<dim>>     triangulation,
    unsigned int const                                n_refine_space,
    std::shared_ptr<BoundaryDescriptorU<dim> >        boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> >        boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >            &/*periodic_faces*/)
{
  const double left = -0.5, right = 0.5;

  if(MESH_TYPE == MeshType::UniformCartesian)
  {
    // Uniform Cartesian grid
    GridGenerator::subdivided_hyper_cube(*triangulation,2,left,right);
  }
  else if(MESH_TYPE == MeshType::ComplexSurfaceManifold)
  {
    // Complex Geometry
    Triangulation<dim> tria1, tria2;
    const double radius = (right-left)*0.25;
    const double width = right-left;
    GridGenerator::hyper_shell(tria1, Point<dim>(), radius, 0.5*width*std::sqrt(dim), 2*dim);
    tria1.reset_all_manifolds();
    if (dim == 2)
    {
      GridTools::rotate(numbers::PI/4, tria1);
    }
    GridGenerator::hyper_ball(tria2, Point<dim>(), radius);
    GridGenerator::merge_triangulations(tria1, tria2, *triangulation);
    triangulation->set_all_manifold_ids(0);
    for (typename Triangulation<dim>::cell_iterator cell = triangulation->begin();cell != triangulation->end(); ++cell)
    {
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
        {
          if (std::abs(cell->face(f)->vertex(v).norm()-radius) > 1e-12)
            face_at_sphere_boundary = false;
        }
        if (face_at_sphere_boundary)
        {
          cell->face(f)->set_all_manifold_ids(1);
        }
      }
    }
    static const SphericalManifold<dim> spherical_manifold;
    triangulation->set_manifold(1, spherical_manifold);

    // refine globally due to boundary conditions for vortex problem
    triangulation->refine_global(1);
  }
  else if(MESH_TYPE == MeshType::ComplexVolumeManifold)
  {
    // Complex Geometry
    Triangulation<dim> tria1, tria2;
    const double radius = (right-left)*0.25;
    const double width = right-left;
    Point<dim> center = Point<dim>();

    GridGenerator::hyper_shell(tria1, Point<dim>(), radius, 0.5*width*std::sqrt(dim), 2*dim);
    tria1.reset_all_manifolds();
    if (dim == 2)
    {
      GridTools::rotate(numbers::PI/4, tria1);
    }
    GridGenerator::hyper_ball(tria2, Point<dim>(), radius);
    GridGenerator::merge_triangulations(tria1, tria2, *triangulation);
    triangulation->set_all_manifold_ids(0);

    // first fill vectors of manifold_ids and face_ids
    std::vector<unsigned int> manifold_ids;
    std::vector<unsigned int> face_ids;

    for (typename Triangulation<dim>::cell_iterator cell = triangulation->begin();cell != triangulation->end(); ++cell)
    {
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for (unsigned int v=0; v<GeometryInfo<dim-1>::vertices_per_cell; ++v)
        {
          if (std::abs(cell->face(f)->vertex(v).norm()-radius) > 1e-12)
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
      for (typename Triangulation<dim>::cell_iterator cell = triangulation->begin(); cell != triangulation->end(); ++cell)
      {
        if(cell->manifold_id() == manifold_ids[i])
        {
          manifold_vec[i] = std::shared_ptr<Manifold<dim> >(
              static_cast<Manifold<dim>*>(new OneSidedCylindricalManifold<dim>(cell,face_ids[i],center)));
          triangulation->set_manifold(manifold_ids[i],*(manifold_vec[i]));
        }
      }
    }

    // refine globally due to boundary conditions for vortex problem
    triangulation->refine_global(1);
  }

  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
  for(;cell!=endc;++cell)
  {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
       if (((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12) && (cell->face(face_number)->center()(1)<0))||
           ((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12) && (cell->face(face_number)->center()(1)>0))||
           ((std::fabs(cell->face(face_number)->center()(1) - left)< 1e-12) && (cell->face(face_number)->center()(0)<0))||
           ((std::fabs(cell->face(face_number)->center()(1) - right)< 1e-12) && (cell->face(face_number)->center()(0)>0)))
         cell->face(face_number)->set_boundary_id (1);
    }
  }
  triangulation->refine_global(n_refine_space);

  typedef typename std::pair<types::boundary_id,std::shared_ptr<Function<dim> > > pair;

  // fill boundary descriptor velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new AnalyticalSolutionVelocity<dim>()));
  boundary_descriptor_velocity->neumann_bc.insert(pair(1,new NeumannBoundaryVelocity<dim>()));

  // fill boundary descriptor pressure
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new PressureBC_dudt<dim>()));
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(1,new AnalyticalSolutionPressure<dim>()));
}


template<int dim>
void set_field_functions(std::shared_ptr<FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  field_functions->initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
  field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
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

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_ */
