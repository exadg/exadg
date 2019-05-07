/*
 * test_comp_NS.h
 *
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COMP_NS_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COMP_NS_H_

#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/function.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

/*
 *  This 2D test case is a quasi one-dimensional problem with periodic boundary
 *  conditions in x_2-direction. The velocity u_2 is zero. The energy is constant.
 *  The density and the velocity u_1 are a function of x_1 and time t.
 */

// set the number of space dimensions: DIMENSION = 2, 3
const unsigned int DIMENSION = 2;

// set the polynomial degree of the shape functions
const unsigned int FE_DEGREE = 7;

//number of quadrature points in 1D
const unsigned int QPOINTS_CONV = FE_DEGREE + 1;
const unsigned int QPOINTS_VIS = QPOINTS_CONV;

// set the number of refine levels for spatial convergence tests
const unsigned int REFINE_STEPS_SPACE_MIN = 3;
const unsigned int REFINE_STEPS_SPACE_MAX = 3;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;

const double DYN_VISCOSITY = 0.0;
const double GAMMA = 1.4;
const double LAMBDA = 0.0;
const double R = 1.0;
const double U_0 = 1.0;
const double MACH = 0.5;
const double SPEED_OF_SOUND = U_0/MACH;
const double T_0 = SPEED_OF_SOUND*SPEED_OF_SOUND/GAMMA/R;

const double X_0 = 0.0;
const double Y_0 = 0.0;
const double H = 10.0;
const double L = 10.0;
const double BETA = 5.0;

std::string OUTPUT_FOLDER = "output_comp_ns/";
std::string FILENAME = "euler_vortex_flow";

template<int dim>
void CompNS::InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  equation_type = EquationType::Euler;
  right_hand_side = true;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 1.0;
  dynamic_viscosity = DYN_VISCOSITY;
  reference_density = 1.0;
  heat_capacity_ratio = GAMMA;
  thermal_conductivity = LAMBDA;
  specific_gas_constant = R;
  max_temperature = T_0;

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::ExplRK;
  order_time_integrator = 2;
  calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
  time_step_size = 1.0e-3;
  max_velocity = U_0;
  cfl_number = 0.05;
  diffusion_number = 0.01;
  exponent_fe_degree_cfl = 2.0;
  exponent_fe_degree_viscous = 4.0;

  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  degree = FE_DEGREE;
  degree_mapping = FE_DEGREE;
  n_q_points_conv = QPOINTS_CONV;
  n_q_points_vis = QPOINTS_VIS;

  // viscous term
  IP_factor = 1.0e0;

  // SOLVER

  // NUMERICAL PARAMETERS
  use_combined_operator = false;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  calculate_velocity = true;
  output_data.write_output = true; //false;
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

  error_data.analytical_solution_available = true;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  solver_info_data.print_to_screen = true;
  solver_info_data.interval_time = (end_time-start_time)/20;

  lift_and_drag_data.calculate_lift_and_drag = false;
  pressure_difference_data.calculate_pressure_difference = false;
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
   std::vector<unsigned int> repetitions({1,1});
   Point<dim> point1(X_0-L/2.0,Y_0-H/2.0), point2(X_0+L/2.0,Y_0+H/2.0);
   GridGenerator::subdivided_hyper_rectangle(*triangulation,repetitions,point1,point2);

   triangulation->refine_global(n_refine_space);
 }

 /**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Analytical solutions for initial field functions
 */
double get_r_square(double const x, double const y, double const t)
{
  return (x-t-X_0)*(x-t-X_0)+(y-Y_0)*(y-Y_0);
}

double get_rho(double const r_sq)
{
  const double pi = numbers::PI;
  return std::pow(1.0 - ((GAMMA-1.0)/(16.0*GAMMA*pi*pi) * BETA*BETA * std::exp(2.0*(1.0-r_sq))), 1/(GAMMA-1.0));
}

double get_u(double const y, double const r_sq)
{
  const double pi = numbers::PI;
  return 1.0 - BETA * std::exp(1.0-r_sq) * (y-Y_0) / (2.0*pi);
}

double get_v(double const x, double const t, double const r_sq)
{
  const double pi = numbers::PI;
  return BETA * std::exp(1.0-r_sq) * (x-t-X_0) / (2.0*pi);
}

double get_energy(double const rho, double const u, double const v)
{
  const double pressure = std::pow(rho,GAMMA);

  return pressure / (rho*(GAMMA-1.0)) + 0.5 * (u*u +v*v);
}

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution (const unsigned int  n_components = dim + 2,
            const double        time = 0.)
    :
  Function<dim>(n_components, time)
  {}

  virtual ~Solution(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const;
};

template<int dim>
double Solution<dim>::value(const Point<dim>    &p,
                            const unsigned int  component) const
{
  const double t = this->get_time();
  const double r_sq = get_r_square(p[0],p[1],t);
  const double rho =  get_rho(r_sq);
  const double u = get_u(p[1],r_sq);
  const double v = get_v(p[0],t,r_sq);

  double result = 0.0;
  if(component==0)
  {
    result =  rho;
  }
  else if (component==1)
  {
    result =  rho*u;
  }
  else if (component==2)
  {
    result =  rho*v;
  }
  else if (component==1+dim)
  {
    result = rho * get_energy(rho,u,v);
  }

  return result;
}


/*
*  prescribe a parabolic velocity profile at the inflow and
*  zero velocity at the wall boundaries
*/
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
 const double t = this->get_time();
 const double r_sq = get_r_square(p[0],p[1],t);

 double result = 0.0;
 if (component==0)
   result = get_u(p[1],r_sq);
 else if (component==1)
   result = get_v(p[0],t,r_sq);

 return result;
}

/*
*  prescribe a constant temperature at the channel walls
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
double EnergyBC<dim>::value(const Point<dim>   &p,
                           const unsigned int /*component*/) const
{
 const double t = this->get_time();
 const double r_sq = get_r_square(p[0],p[1],t);
 const double rho =  get_rho(r_sq);
 const double u = get_u(p[1],r_sq);
 const double v = get_v(p[0],t,r_sq);
 double energy = get_energy(rho,u,v);

 return energy;
}

template<int dim>
class DensityBC : public Function<dim>
{
public:
 DensityBC (const double time = 0.)
   :
   Function<dim>(1, time)
 {}

 virtual ~DensityBC(){};

 virtual double value (const Point<dim>    &p,
                       const unsigned int  component = 0) const;
};

template<int dim>
double DensityBC<dim>::value(const Point<dim>   &p,
                            const unsigned int /*component*/) const
{
 const double t = this->get_time();
 const double r_sq = get_r_square(p[0],p[1],t);
 const double rho =  get_rho(r_sq);

 return rho;
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
  // zero function scalar
  std::shared_ptr<Function<dim> > zero_function_scalar;
  zero_function_scalar.reset(new Functions::ZeroFunction<dim>(1));

  // density
  std::shared_ptr<Function<dim> > density_bc;
  density_bc.reset(new DensityBC<dim>());
  boundary_descriptor_density->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,density_bc));

  // velocity
  std::shared_ptr<Function<dim> > velocity_bc;
  velocity_bc.reset(new VelocityBC<dim>());
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,velocity_bc));

  // pressure
  boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_scalar));

  // energy: prescribe energy
  boundary_descriptor_energy->boundary_variable.insert(std::pair<types::boundary_id,CompNS::EnergyBoundaryVariable>(0,CompNS::EnergyBoundaryVariable::Energy));

  std::shared_ptr<Function<dim> > energy_bc;
  energy_bc.reset(new EnergyBC<dim>());
  boundary_descriptor_energy->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,energy_bc));
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
  initial_solution.reset(new Solution<dim>());
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
  analytical_solution->solution.reset(new Solution<dim>());
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

#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COMP_NS_H_ */
