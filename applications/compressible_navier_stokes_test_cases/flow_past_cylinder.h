/*
 * FlowPastCylinder.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/manifold_lib.h>
#include "../../include/functionalities/one_sided_cylindrical_manifold.h"


/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions
unsigned int const FE_DEGREE = 2;

//number of quadrature points in 1D
//const unsigned int QPOINTS_CONV = FE_DEGREE + 1;
const unsigned int QPOINTS_CONV = FE_DEGREE + (FE_DEGREE+2)/2; // 3/2-overintegration
const unsigned int QPOINTS_VIS = QPOINTS_CONV;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 1;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// mesh
#include "../grid_tools/mesh_flow_past_cylinder.h"

// set problem specific parameters like physical dimensions, etc.
const unsigned int TEST_CASE = 3; // 1, 2 or 3
const double Um = (DIMENSION == 2 ? (TEST_CASE==1 ? 0.3 : 1.5) : (TEST_CASE==1 ? 0.45 : 2.25));

// physical quantities
const double VISCOSITY = 1.e-3;
const double GAMMA = 1.4;
const double LAMBDA = 0.0262;
const double GAS_CONSTANT = 287.058;
const double U_0 = Um;
const double MACH = 0.2;
const double SPEED_OF_SOUND = U_0/MACH;
const double RHO_0 = 1.0;
const double T_0 = SPEED_OF_SOUND*SPEED_OF_SOUND/GAMMA/GAS_CONSTANT;
const double E_0 = GAS_CONSTANT/(GAMMA-1.0)*T_0;

// end time
const double END_TIME = 8.0;

std::string OUTPUT_FOLDER = "output_comp_ns/flow_past_cylinder/";
std::string FILENAME = "new";

template<int dim>
void CompNS::InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  equation_type = EquationType::NavierStokes;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = END_TIME; //END_TIME is also needed somewhere else
  dynamic_viscosity = VISCOSITY;
  reference_density = RHO_0;
  heat_capacity_ratio = GAMMA;
  thermal_conductivity = LAMBDA;
  specific_gas_constant = GAS_CONSTANT;
  max_temperature = T_0;


  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::ExplRK3Stage7Reg2;
  order_time_integrator = 3;
  stages = 7;
  calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
  time_step_size = 1.0e-3;
  max_velocity = U_0;
  cfl_number = 0.6;
  diffusion_number = 0.02;
  exponent_fe_degree_cfl = 1.5; //2.0;
  exponent_fe_degree_viscous = 3.0; //4.0;


  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  degree = FE_DEGREE;
  degree_mapping = FE_DEGREE;
  n_q_points_conv = QPOINTS_CONV;
  n_q_points_vis = QPOINTS_VIS;

  // viscous term
  IP_factor = 1.0;


  // COUPLED NAVIER-STOKES SOLVER

  // SOLVER

  // OUTPUT AND POSTPROCESSING
  calculate_velocity = true;
  calculate_pressure = true;
  output_data.write_output = false;
  output_data.write_pressure = true;
  output_data.write_velocity = true;
  output_data.write_temperature = true;
  output_data.write_vorticity = true;
  output_data.write_divergence = true;
  output_data.output_folder = OUTPUT_FOLDER;
  output_data.output_name = FILENAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.degree = FE_DEGREE;

  // calculation of error
  error_data.analytical_solution_available = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (end_time-start_time)/20;

  // lift and drag
  lift_and_drag_data.calculate_lift_and_drag = true;
  lift_and_drag_data.viscosity = dynamic_viscosity;
  const double U = Um * (DIMENSION == 2 ? 2./3. : 4./9.);
  if(DIMENSION == 2)
    lift_and_drag_data.reference_value = RHO_0/2.0*pow(U,2.0)*D;
  else if(DIMENSION == 3)
    lift_and_drag_data.reference_value = RHO_0/2.0*pow(U,2.0)*D*H;

  // surfaces for calculation of lift and drag coefficients have boundary_ID = 2
  lift_and_drag_data.boundary_IDs.insert(2);

  lift_and_drag_data.filename_lift = OUTPUT_FOLDER + output_data.output_name + "_lift";
  lift_and_drag_data.filename_drag = OUTPUT_FOLDER + output_data.output_name + "_drag";

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

  pressure_difference_data.filename = OUTPUT_FOLDER + output_data.output_name + "_pressure_difference";
}

/**************************************************************************************/
/*                                                                                    */
/*                        GENERATE GRID AND SET BOUNDARY INDICATORS                   */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_ids(
  std::shared_ptr<parallel::Triangulation<dim>>            triangulation,
  unsigned int const                                       n_refine_space,
  std::vector<GridTools::PeriodicFacePair<typename
    Triangulation<dim>::cell_iterator> >                   &/*periodic_faces*/)
{
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

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
class InitialSolution : public Function<dim>
{
public:
  InitialSolution (const unsigned int  n_components = dim + 2,
                   const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~InitialSolution(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const;
};

template<int dim>
double InitialSolution<dim>::value(const Point<dim>    &/*p*/,
                                   const unsigned int  component) const
{
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


template<int dim>
class VelocityBC : public Function<dim>
{
public:
  VelocityBC (const unsigned int  n_components = dim,
              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~VelocityBC(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double VelocityBC<dim>::value(const Point<dim>   &p,
                              const unsigned int component) const
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

  virtual ~PressureBC(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double PressureBC<dim>::value(const Point<dim>   &/*p*/,
                              const unsigned int /*component*/) const
{
  double result = 0.0;
  result = RHO_0 * GAS_CONSTANT * T_0;

  return result;
}

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

  virtual ~EnergyBC(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double EnergyBC<dim>::value(const Point<dim>   &/*p*/,
                            const unsigned int /*component*/) const
{
  double result = T_0;

  return result;
}

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

  virtual ~TempGradientBC(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double TempGradientBC<dim>::value(const Point<dim>   &/*p*/,
                                  const unsigned int /*component*/) const
{
  double result = -0.1;

  return result;
}


namespace CompNS
{

template<int dim>
void set_boundary_conditions(
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_density,
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_velocity,
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_pressure,
  std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim> >  boundary_descriptor_energy)
{
  //inflow and upper/lower walls: 0, outflow: 1, cylinder: 2

  // zero function scalar
  std::shared_ptr<Function<dim> > zero_function_scalar;
  zero_function_scalar.reset(new Functions::ZeroFunction<dim>(1));

  // zero function vectorial
  std::shared_ptr<Function<dim> > zero_function_vectorial;
  zero_function_vectorial.reset(new Functions::ZeroFunction<dim>(dim));

  // density

  // For Neumann boundaries, no value is prescribed (only first derivative of density occurs in equations).
  // Hence the specified function is irrelevant (i.e., it is not used).
  boundary_descriptor_density->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_scalar));
  boundary_descriptor_density->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(1,zero_function_scalar));
  boundary_descriptor_density->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(2,zero_function_scalar));

  // velocity
  std::shared_ptr<Function<dim> > velocity_bc;
  velocity_bc.reset(new VelocityBC<dim>());
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,velocity_bc));
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(2,velocity_bc));
  boundary_descriptor_velocity->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(1,zero_function_vectorial));

  // pressure
  std::shared_ptr<Function<dim> > pressure_bc;
  pressure_bc.reset(new PressureBC<dim>());
  boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(1,pressure_bc));
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_scalar));
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(2,zero_function_scalar));

  // energy: prescribe temperature
  boundary_descriptor_energy->boundary_variable.insert(std::pair<types::boundary_id,CompNS::EnergyBoundaryVariable>(0,CompNS::EnergyBoundaryVariable::Temperature));
  boundary_descriptor_energy->boundary_variable.insert(std::pair<types::boundary_id,CompNS::EnergyBoundaryVariable>(1,CompNS::EnergyBoundaryVariable::Temperature));
  boundary_descriptor_energy->boundary_variable.insert(std::pair<types::boundary_id,CompNS::EnergyBoundaryVariable>(2,CompNS::EnergyBoundaryVariable::Temperature));

  std::shared_ptr<Function<dim> > energy_bc;
  energy_bc.reset(new EnergyBC<dim>());
  boundary_descriptor_energy->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,energy_bc));
  boundary_descriptor_energy->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(2,energy_bc));
  boundary_descriptor_energy->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(1,zero_function_scalar));

 // test alternative temperature boundary conditions with heat flux
// std::shared_ptr<Function<dim> > energy_bc;
// energy_bc.reset(new EnergyBC<dim>());
// std::shared_ptr<Function<dim> > temp_grad_bc;
// temp_grad_bc.reset(new TempGradientBC<dim>());
// boundary_descriptor_energy->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,energy_bc));
// boundary_descriptor_energy->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(2,temp_grad_bc));
// boundary_descriptor_energy->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(1,zero_function_scalar));
}

template<int dim>
void set_field_functions(std::shared_ptr<CompNS::FieldFunctions<dim> > field_functions)
{
  // zero function scalar
  std::shared_ptr<Function<dim> > zero_function_scalar;
  zero_function_scalar.reset(new Functions::ZeroFunction<dim>(1));

  // zero function vectorial
  std::shared_ptr<Function<dim> > zero_function_vectorial;
  zero_function_vectorial.reset(new Functions::ZeroFunction<dim>(dim));

  // initial solution
  std::shared_ptr<Function<dim> > initial_solution;
  initial_solution.reset(new InitialSolution<dim>());
  field_functions->initial_solution = initial_solution;

  // rhs density
  field_functions->right_hand_side_density = zero_function_scalar;

  // rhs velocity
  field_functions->right_hand_side_velocity = zero_function_vectorial;

  // rhs energy
  field_functions->right_hand_side_energy = zero_function_scalar;
}

template<int dim>
void set_analytical_solution(std::shared_ptr<CompNS::AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->solution.reset(new InitialSolution<dim>());
}

template<int dim, typename Number>
std::shared_ptr<CompNS::PostProcessor<dim, Number> >
construct_postprocessor(CompNS::InputParameters<dim> const &param)
{
  CompNS::PostProcessorData<dim> pp_data;

  pp_data.calculate_velocity = param.calculate_velocity;
  pp_data.calculate_pressure = param.calculate_pressure;
  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.kinetic_energy_data = param.kinetic_energy_data;
  pp_data.kinetic_energy_spectrum_data = param.kinetic_energy_spectrum_data;

  std::shared_ptr<CompNS::PostProcessor<dim, Number> > pp;
  pp.reset(new CompNS::PostProcessor<dim, Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_ */
