/*
 * vortex.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_

#include "../../include/functionalities/one_sided_cylindrical_manifold.h"
#include "../grid_tools/deformed_cube_manifold.h"
#include "../grid_tools/dealii_extensions.h"
#include "../../include/incompressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 4;
unsigned int const DEGREE_MAX = 4;

unsigned int const REFINE_SPACE_MIN = 1;
unsigned int const REFINE_SPACE_MAX = 1;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;


// set problem specific parameters like physical dimensions, etc.
const double U_X_MAX = 1.0;
const double VISCOSITY = 2.5e-2; //1.e-2; //2.5e-2;
const FormulationViscousTerm FORMULATION_VISCOUS_TERM = FormulationViscousTerm::LaplaceFormulation;

enum class MeshType{ UniformCartesian, ComplexSurfaceManifold, ComplexVolumeManifold, Curvilinear };
const MeshType MESH_TYPE = MeshType::Curvilinear; //UniformCartesian;

namespace IncNS
{
void set_input_parameters(InputParameters &param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.problem_type = ProblemType::Unsteady;
  param.equation_type = EquationType::NavierStokes;
  param.formulation_viscous_term = FORMULATION_VISCOUS_TERM;
  param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
  param.right_hand_side = false;


  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 1.0;
  param.viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  param.solver_type = SolverType::Unsteady;
  param.temporal_discretization = TemporalDiscretization::BDFDualSplittingScheme;
  param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Explicit;
  param.time_integrator_oif = TimeIntegratorOIF::ExplRK3Stage7Reg2;
  param.calculation_of_time_step_size = TimeStepCalculation::CFL;
  param.adaptive_time_stepping = false;
  param.max_velocity = 1.4 * U_X_MAX;
  param.cfl = 0.4;
  param.cfl_oif = param.cfl/1.0;
  param.cfl_exponent_fe_degree_velocity = 1.5;
  param.c_eff = 8.0;
  param.time_step_size = 5.e-5;
  param.order_time_integrator = 2;
  param.start_with_low_order = false;
  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = 0.1;

  // restart
  param.restarted_simulation = false;
  param.restart_data.write_restart = true;
  param.restart_data.interval_time = 0.25;
  param.restart_data.interval_wall_time = 1.e6;
  param.restart_data.interval_time_steps = 1e8;
  param.restart_data.filename = "output/vortex/vortex";


  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree_u = DEGREE_MIN;
  param.degree_p = DegreePressure::MixedOrder;
  param.mapping = MappingType::Isoparametric;
  param.h_refinements = REFINE_SPACE_MIN;

  // convective term
  param.upwind_factor = 1.0;

  // viscous term
  param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

  // special case: pure DBC's
  param.pure_dirichlet_bc = false;

  // NUMERICAL PARAMETERS
  param.implement_block_diagonal_preconditioner_matrix_free = false;
  param.use_cell_based_face_loops = false;
  param.quad_rule_linearization = QuadratureRuleLinearization::Overintegration32k;

  // PROJECTION METHODS

  // pressure Poisson equation
  param.solver_pressure_poisson = SolverPressurePoisson::CG;
  param.solver_data_pressure_poisson = SolverData(1000,1.e-12,1.e-6,100);
  param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;
  param.multigrid_data_pressure_poisson.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.multigrid_data_pressure_poisson.smoother_data.preconditioner = PreconditionerSmoother::PointJacobi;

  // projection step
  param.solver_projection = SolverProjection::CG;
  param.solver_data_projection = SolverData(1000, 1.e-12, 1.e-6);
  param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  param.preconditioner_block_diagonal_projection = Elementwise::Preconditioner::InverseMassMatrix;
  param.solver_data_block_diagonal_projection = SolverData(1000,1.e-12,1.e-2,1000);

  // HIGH-ORDER DUAL SPLITTING SCHEME

  // formulations
  param.order_extrapolation_pressure_nbc = param.order_time_integrator<=2 ? param.order_time_integrator : 2;

  // viscous step
  param.solver_viscous = SolverViscous::CG;
  param.solver_data_viscous = SolverData(1000,1.e-12,1.e-6);
  param.preconditioner_viscous = PreconditionerViscous::Multigrid;
  param.multigrid_data_viscous.type = MultigridType::hMG;
  param.multigrid_data_viscous.smoother_data.smoother = MultigridSmoother::Chebyshev;
  param.update_preconditioner_viscous = false;


  // PRESSURE-CORRECTION SCHEME

  // formulation
  param.order_pressure_extrapolation = param.order_time_integrator-1;
  param.rotational_formulation = true;

  // momentum step

  // Newton solver
  param.newton_solver_data_momentum = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  param.solver_momentum = SolverMomentum::FGMRES;
  param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-2, 100);
  param.preconditioner_momentum = MomentumPreconditioner::Multigrid;
  param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionConvectionDiffusion;
  param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
  param.update_preconditioner_momentum = true;

  // Jacobi smoother data
  param.multigrid_data_momentum.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  param.multigrid_data_momentum.smoother_data.iterations = 5;
  param.multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

  // Chebyshev smoother data
//  param.multigrid_data_momentum.smoother = MultigridSmoother::Chebyshev;
//  param.multigrid_data_momentum.coarse_solver = MultigridCoarseGridSolver::Chebyshev;

  // GMRES smoother data
//    param.multigrid_data_momentum.gmres_smoother_data.preconditioner = PreconditionerGMRESSmoother::BlockJacobi;


  // COUPLED NAVIER-STOKES SOLVER
  param.use_scaling_continuity = false;

  // nonlinear solver (Newton solver)
  param.newton_solver_data_coupled = NewtonSolverData(100,1.e-12,1.e-6);

  // linear solver
  param.solver_coupled = SolverCoupled::FGMRES; //FGMRES; //GMRES;
  param.solver_data_coupled = SolverData(1e4, 1.e-12, 1.e-2, 100);

  // preconditioner linear solver
  param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;
  param.update_preconditioner_coupled = true;

  // preconditioner momentum block
  param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;
  param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionConvectionDiffusion;
  param.multigrid_data_velocity_block.type = MultigridType::phMG;
  param.multigrid_data_velocity_block.smoother_data.smoother = MultigridSmoother::Jacobi; //Chebyshev; //GMRES;
  param.multigrid_data_velocity_block.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
  param.multigrid_data_velocity_block.smoother_data.iterations = 5;
  param.multigrid_data_velocity_block.smoother_data.relaxation_factor = 0.7;
  // coarse grid solver
  param.multigrid_data_velocity_block.coarse_problem.solver = MultigridCoarseGridSolver::GMRES; //Chebyshev; //GMRES;

  // preconditioner Schur-complement block
  param.preconditioner_pressure_block = SchurComplementPreconditioner::PressureConvectionDiffusion;
  param.discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
}
}


/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void
create_grid_and_set_boundary_ids(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                                 unsigned int const                            n_refine_space,
                                 std::vector<GridTools::PeriodicFacePair<typename
                                   Triangulation<dim>::cell_iterator> >        &periodic_faces)
{
  (void)periodic_faces;

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
  else if(MESH_TYPE == MeshType::Curvilinear)
  {
    GridGenerator::subdivided_hyper_cube(*triangulation,2,left,right);
    triangulation->set_all_manifold_ids(1);
    double const deformation = 0.1;
    unsigned int const frequency = 2;
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequency);
    triangulation->set_manifold(1, manifold);
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
       {
         cell->face(face_number)->set_boundary_id (1);
       }
    }
  }
  triangulation->refine_global(n_refine_space);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

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

namespace IncNS
{

template<int dim>
void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim> > boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim> > boundary_descriptor_pressure)
{
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

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<PostProcessorBase<dim, Number> >
construct_postprocessor(InputParameters const &param)
{
  PostProcessorData<dim> pp_data;

  // write output for visualization of results
  pp_data.output_data.write_output = true;
  pp_data.output_data.output_folder = "output/vortex/vtu/";
  pp_data.output_data.output_name = "vortex_curvilinear";
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/20;
  pp_data.output_data.write_vorticity = true;
  pp_data.output_data.write_divergence = true;
  pp_data.output_data.write_velocity_magnitude = true;
  pp_data.output_data.write_vorticity_magnitude = true;
  pp_data.output_data.write_processor_id = true;
  pp_data.output_data.mean_velocity.calculate = true;
  pp_data.output_data.mean_velocity.sample_start_time = param.start_time;
  pp_data.output_data.mean_velocity.sample_end_time = param.end_time;
  pp_data.output_data.mean_velocity.sample_every_timesteps = 1;
  pp_data.output_data.write_higher_order = true;
  pp_data.output_data.degree = param.degree_u;

  // calculation of velocity error
  pp_data.error_data_u.analytical_solution_available = true;
  pp_data.error_data_u.analytical_solution.reset(new AnalyticalSolutionVelocity<dim>());
  pp_data.error_data_u.calculate_relative_errors = true;
  pp_data.error_data_u.error_calc_start_time = param.start_time;
  pp_data.error_data_u.error_calc_interval_time = (param.end_time - param.start_time)/20;
  pp_data.error_data_u.name = "velocity";

  // ... pressure error
  pp_data.error_data_p.analytical_solution_available = true;
  pp_data.error_data_p.analytical_solution.reset(new AnalyticalSolutionPressure<dim>());
  pp_data.error_data_p.calculate_relative_errors = true;
  pp_data.error_data_p.error_calc_start_time = param.start_time;
  pp_data.error_data_p.error_calc_interval_time = (param.end_time - param.start_time)/20;
  pp_data.error_data_p.name = "pressure";

  std::shared_ptr<PostProcessorBase<dim,Number> > pp;
  pp.reset(new PostProcessor<dim,Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_VORTEX_H_ */
