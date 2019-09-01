/*
 * flow_past_cylinder.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_

#include "../../include/compressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 3;
unsigned int const DEGREE_MAX = 3;

unsigned int const REFINE_SPACE_MIN = 0;
unsigned int const REFINE_SPACE_MAX = 0;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// problem specific parameters
unsigned int const TEST_CASE = 3; // 1, 2 or 3
unsigned int const DIMENSION = 2;
double const Um = (DIMENSION == 2 ? (TEST_CASE==1 ? 0.3 : 1.5) : (TEST_CASE==1 ? 0.45 : 2.25));

// physical quantities
double const VISCOSITY = 1.e-3;
double const GAMMA = 1.4;
double const LAMBDA = 0.0262;
double const GAS_CONSTANT = 287.058;
double const U_0 = Um;
double const MACH = 0.2;
double const SPEED_OF_SOUND = U_0/MACH;
double const RHO_0 = 1.0;
double const T_0 = SPEED_OF_SOUND*SPEED_OF_SOUND/GAMMA/GAS_CONSTANT;
double const E_0 = GAS_CONSTANT/(GAMMA-1.0)*T_0;

// physical dimensions
double const Y_C = 0.2; // center of cylinder (y-coordinate)
double const D = 0.1; // cylinder diameter

// end time
double const START_TIME = 0.0;
double const END_TIME = 8.0;

std::string const OUTPUT_FOLDER = "output_comp_ns/flow_past_cylinder/";
std::string const FILENAME = "test";

namespace CompNS
{
void set_input_parameters(InputParameters & param)
{
  // MATHEMATICAL MODEL
  param.dim = DIMENSION;
  param.equation_type = EquationType::NavierStokes;
  param.right_hand_side = false;

  // PHYSICAL QUANTITIES
  param.start_time = START_TIME;
  param.end_time = END_TIME;
  param.dynamic_viscosity = VISCOSITY;
  param.reference_density = RHO_0;
  param.heat_capacity_ratio = GAMMA;
  param.thermal_conductivity = LAMBDA;
  param.specific_gas_constant = GAS_CONSTANT;
  param.max_temperature = T_0;

  // TEMPORAL DISCRETIZATION
  param.temporal_discretization = TemporalDiscretization::ExplRK3Stage7Reg2;
  param.order_time_integrator = 3;
  param.stages = 7;
  param.calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
  param.time_step_size = 1.0e-3;
  param.max_velocity = U_0;
  param.cfl_number = 1.0;
  param.diffusion_number = 0.1;
  param.exponent_fe_degree_cfl = 1.5;
  param.exponent_fe_degree_viscous = 3.0;
  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/20;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree = DEGREE_MIN;
  param.mapping = MappingType::Isoparametric;
  param.n_q_points_convective = QuadratureRule::Overintegration32k;
  param.n_q_points_viscous = QuadratureRule::Overintegration32k;
  param.h_refinements = REFINE_SPACE_MIN;

  // viscous term
  param.IP_factor = 1.0;
}

}

/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

#include "../grid_tools/mesh_flow_past_cylinder.h"
#include "../../include/functionalities/one_sided_cylindrical_manifold.h"

template<int dim>
void create_grid_and_set_boundary_ids(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                                      unsigned int const                            n_refine_space,
                                      std::vector<GridTools::PeriodicFacePair<typename
                                        Triangulation<dim>::cell_iterator> >        &periodic_faces)
{
  (void)periodic_faces;

  Point<dim> center;
  center[0] = X_C;
  center[1] = Y_C;

  // apply this manifold for all mesh types
  Point<dim> direction;
  direction[dim-1] = 1.;

  static std::shared_ptr<Manifold<dim> > cylinder_manifold;

  if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
  {
   cylinder_manifold = std::shared_ptr<Manifold<dim> >(dim == 2 ? static_cast<Manifold<dim>*>(new SphericalManifold<dim>(center)) :
                                           static_cast<Manifold<dim>*>(new CylindricalManifold<dim>(direction, center)));
  }
  else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
  {
   cylinder_manifold = std::shared_ptr<Manifold<dim> >(static_cast<Manifold<dim>*>(new MyCylindricalManifold<dim>(center)));
  }
  else
  {
   AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold || MANIFOLD_TYPE == ManifoldType::VolumeManifold,
       ExcMessage("Specified manifold type not implemented"));
  }

  create_triangulation(*triangulation);
  triangulation->set_manifold(MANIFOLD_ID, *cylinder_manifold);

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

  triangulation->refine_global(n_refine_space);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
class InitialSolution : public Function<dim>
{
public:
  InitialSolution (const unsigned int  n_components = dim + 2,
                   const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>   &p,
                const unsigned int component = 0) const
  {
    (void)p;

    double result = 0.0;

    if(component == 0)
    {
      result = RHO_0;
    }
    // prescribe zero velocity field at t=0
    else if (component == dim+1)
    {
      result = RHO_0 * E_0;
    }

    return result;
  }

};

template<int dim>
class VelocityBC : public Function<dim>
{
public:
  VelocityBC (const unsigned int  n_components = dim,
              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    double t = this->get_time();
    const double pi = numbers::PI;
    const double T = 1.0;

    double result = 0.0;

    if(component == 0)
    {
      // values unequal zero only at inflow boundary
      if(std::abs(p[0]-(dim==2 ? L1 : X_0)) <1.e-12)
      {
        double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
        if(TEST_CASE < 3)
          result = coefficient * p[1] * (H-p[1]) * ( (t/T)<1.0 ? std::sin(pi/2.*t/T) : 1.0);
        else if(TEST_CASE == 3)
          result = coefficient * p[1] * (H-p[1]) * std::sin(pi*t/END_TIME);

        if (dim == 3)
          result *= p[2] * (H-p[2]);
      }
    }

    return result;
  }
};

/*
 *  prescribe a constant pressure at the outflow boundary
 */
template<int dim>
class PressureBC : public Function<dim>
{
public:
  PressureBC (const double time = 0.)
    :
    Function<dim>(1, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    (void)p;
    (void)component;

    double result = RHO_0 * GAS_CONSTANT * T_0;

    return result;
  }
};

/*
 *  prescribe a constant temperature at the walls and at the inflow
 */
template<int dim>
class EnergyBC : public Function<dim>
{
public:
  EnergyBC (const double time = 0.)
    :
    Function<dim>(1, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    (void)p;
    (void)component;

    double result = T_0;

    return result;
  }
};

/*
 *  prescribe a constant temperature at the walls and at the inflow
 */
template<int dim>
class TempGradientBC : public Function<dim>
{
public:
  TempGradientBC (const double time = 0.)
    :
    Function<dim>(1, time)
  {}

  double value (const Point<dim>    &p,
                const unsigned int  component = 0) const
  {
    (void)p;
    (void)component;

    double result = -0.1;

    return result;
  }
};

namespace CompNS
{

template<int dim>
void set_boundary_conditions(
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_density,
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_velocity,
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_pressure,
  std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim> >  boundary_descriptor_energy)
{
  typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
  typedef typename std::pair<types::boundary_id, EnergyBoundaryVariable> pair_variable;

  //inflow and upper/lower walls: 0, outflow: 1, cylinder: 2

  // density
  // For Neumann boundaries, no value is prescribed (only first derivative of density occurs in equations).
  // Hence the specified function is irrelevant (i.e., it is not used).
  boundary_descriptor_density->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(1)));
  boundary_descriptor_density->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(1)));
  boundary_descriptor_density->neumann_bc.insert(pair(2,new Functions::ZeroFunction<dim>(1)));

  // velocity
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new VelocityBC<dim>()));
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(2,new VelocityBC<dim>()));
  boundary_descriptor_velocity->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(dim)));

  // pressure
  boundary_descriptor_pressure->dirichlet_bc.insert(pair(1,new PressureBC<dim>()));
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(1)));
  boundary_descriptor_pressure->neumann_bc.insert(pair(2,new Functions::ZeroFunction<dim>(1)));

  // energy: prescribe temperature
  boundary_descriptor_energy->boundary_variable.insert(pair_variable(0,CompNS::EnergyBoundaryVariable::Temperature));
  boundary_descriptor_energy->boundary_variable.insert(pair_variable(1,CompNS::EnergyBoundaryVariable::Temperature));
  boundary_descriptor_energy->boundary_variable.insert(pair_variable(2,CompNS::EnergyBoundaryVariable::Temperature));

  boundary_descriptor_energy->dirichlet_bc.insert(pair(0,new EnergyBC<dim>()));
  boundary_descriptor_energy->dirichlet_bc.insert(pair(2,new EnergyBC<dim>()));
  boundary_descriptor_energy->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(1)));

  // alternative: test temperature boundary conditions with heat flux
//  boundary_descriptor_energy->dirichlet_bc.insert(pair(0,new EnergyBC<dim>()));
//  boundary_descriptor_energy->neumann_bc.insert(pair(2,new TempGradientBC<dim>()));
//  boundary_descriptor_energy->neumann_bc.insert(pair(1,new Functions::ZeroFunction<dim>(1)));
}

template<int dim>
void set_field_functions(std::shared_ptr<CompNS::FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution.reset(new InitialSolution<dim>());
  field_functions->right_hand_side_density.reset(new Functions::ZeroFunction<dim>(1));
  field_functions->right_hand_side_velocity.reset(new Functions::ZeroFunction<dim>(dim));
  field_functions->right_hand_side_energy.reset(new Functions::ZeroFunction<dim>(1));
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                              POSTPROCESSOR                                               */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim, typename Number>
std::shared_ptr<CompNS::PostProcessorBase<dim, Number> >
construct_postprocessor(CompNS::InputParameters const &param)
{
  CompNS::PostProcessorData<dim> pp_data;

  pp_data.calculate_velocity = true;
  pp_data.calculate_pressure = true;
  pp_data.output_data.write_output = true;
  pp_data.output_data.write_pressure = true;
  pp_data.output_data.write_velocity = true;
  pp_data.output_data.write_temperature = true;
  pp_data.output_data.write_vorticity = true;
  pp_data.output_data.write_divergence = true;
  pp_data.output_data.output_folder = OUTPUT_FOLDER;
  pp_data.output_data.output_name = FILENAME;
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/20;
  pp_data.output_data.degree = param.degree;

  // lift and drag
  pp_data.lift_and_drag_data.calculate_lift_and_drag = true;
  pp_data.lift_and_drag_data.viscosity = param.dynamic_viscosity;
  const double U = Um * (param.dim == 2 ? 2./3. : 4./9.);
  if(param.dim == 2)
    pp_data.lift_and_drag_data.reference_value = RHO_0/2.0*pow(U,2.0)*D;
  else if(param.dim == 3)
    pp_data.lift_and_drag_data.reference_value = RHO_0/2.0*pow(U,2.0)*D*H;

  // surfaces for calculation of lift and drag coefficients have boundary_ID = 2
  pp_data.lift_and_drag_data.boundary_IDs.insert(2);

  pp_data.lift_and_drag_data.filename_lift = OUTPUT_FOLDER + FILENAME + "_lift";
  pp_data.lift_and_drag_data.filename_drag = OUTPUT_FOLDER + FILENAME + "_drag";

  // pressure difference
  pp_data.pressure_difference_data.calculate_pressure_difference = true;
  if(param.dim == 2)
  {
    Point<dim> point_1_2D((X_C-D/2.0),Y_C), point_2_2D((X_C+D/2.0),Y_C);
    pp_data.pressure_difference_data.point_1 = point_1_2D;
    pp_data.pressure_difference_data.point_2 = point_2_2D;
  }
  else if(param.dim == 3)
  {
    Point<dim> point_1_3D((X_C-D/2.0),Y_C,H/2.0), point_2_3D((X_C+D/2.0),Y_C,H/2.0);
    pp_data.pressure_difference_data.point_1 = point_1_3D;
    pp_data.pressure_difference_data.point_2 = point_2_3D;
  }

  pp_data.pressure_difference_data.filename = OUTPUT_FOLDER + FILENAME + "_pressure_difference";

  std::shared_ptr<CompNS::PostProcessorBase<dim, Number> > pp;
  pp.reset(new CompNS::PostProcessor<dim, Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_ */
