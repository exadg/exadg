/*
 * steady_shear_flow.h
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COMP_NS_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COMP_NS_H_

#include "../../include/compressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 6;
unsigned int const DEGREE_MAX = 6;

unsigned int const REFINE_SPACE_MIN = 2;
unsigned int const REFINE_SPACE_MAX = 2;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// problem specific parameters
const double DYN_VISCOSITY = 0.01;
const double GAMMA = 1.5;
const double LAMBDA = 0.0;
const double R = 1.0;
const double U_0 = 1.0;
const double MACH = 0.2;
const double SPEED_OF_SOUND = U_0/MACH;
const double RHO_0 = 1.0;
const double T_0 = SPEED_OF_SOUND*SPEED_OF_SOUND/GAMMA/R;

const double H = 1.0;
const double L = 1.0;

std::string OUTPUT_FOLDER = "output_comp_ns/steady_shear_flow/";
std::string FILENAME = "output";

namespace CompNS
{
void set_input_parameters(InputParameters & param)
{
  // MATHEMATICAL MODEL
  param.dim = 2;
  param.equation_type = EquationType::NavierStokes;
  param.right_hand_side = true;

  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 10.0;
  param.dynamic_viscosity = DYN_VISCOSITY;
  param.reference_density = RHO_0;
  param.heat_capacity_ratio = GAMMA;
  param.thermal_conductivity = LAMBDA;
  param.specific_gas_constant = R;
  param.max_temperature = T_0;

  // TEMPORAL DISCRETIZATION
  param.temporal_discretization = TemporalDiscretization::ExplRK;
  param.order_time_integrator = 2;
  param.calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
  param.time_step_size = 1.0e-3;
  param.max_velocity = U_0;
  param.cfl_number = 0.1;
  param.diffusion_number = 0.01;
  param.exponent_fe_degree_cfl = 2.0;
  param.exponent_fe_degree_viscous = 4.0;
  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/20;

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree = DEGREE_MIN;
  param.mapping = MappingType::Isoparametric;
  param.n_q_points_convective = QuadratureRule::Standard;
  param.n_q_points_viscous = QuadratureRule::Standard;
  param.h_refinements = REFINE_SPACE_MIN;

  // viscous term
  param.IP_factor = 1.0e0;

  // NUMERICAL PARAMETERS
  param.use_combined_operator = false;
}
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_ids(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                                      unsigned int const                            n_refine_space,
                                      std::vector<GridTools::PeriodicFacePair<typename
                                        Triangulation<dim>::cell_iterator> >        &periodic_faces)
{
  (void)periodic_faces;

  std::vector<unsigned int> repetitions({1,1});
  Point<dim> point1(0.0,0.0), point2(L,H);
  GridGenerator::subdivided_hyper_rectangle(*triangulation,repetitions,point1,point2);

  triangulation->refine_global(n_refine_space);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

/*
 *  Analytical solutions
 */
template<int dim>
class Solution : public Function<dim>
{
public:
  Solution (const unsigned int  n_components = dim + 2,
            const double        time = 0.)
    :
  Function<dim>(n_components, time)
  {}

  double value (const Point<dim>   &p,
                const unsigned int component = 0) const
  {
    double result = 0.0;

    if(component==0)
      result = RHO_0;
    else if (component==1)
      result = RHO_0 * p[1] * p[1];
    else if (component==2)
      result = 0.0;
    else if (component==1+dim)
      result = RHO_0 * (2.0 * DYN_VISCOSITY * p[0] + 10.0)/(GAMMA-1.0) + std::pow(p[1],4.0)/2.0;

    return result;
  }
};

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

 double value (const Point<dim>    &p,
               const unsigned int  component = 0) const
 {
   double result = 0.0;

   // copied from analytical solution
   if (component==0)
     result = RHO_0 * p[1] * p[1];
   else if (component==1)
     result = 0.0;

   return result;
 }
};

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

 double value (const Point<dim>    &p,
               const unsigned int  component = 0) const
 {
   (void)component;

   double result = (2.0 * DYN_VISCOSITY * p[0] + 10.0)/(GAMMA-1.0) + std::pow(p[1],4.0)/2.0;

   return result;
 }
};

template<int dim>
class DensityBC : public Function<dim>
{
public:
 DensityBC (const double time = 0.)
   :
   Function<dim>(1, time)
 {}

 double value (const Point<dim>    &p,
               const unsigned int  component = 0) const
 {
   (void)p;
   (void)component;

   return RHO_0;
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

  boundary_descriptor_density->dirichlet_bc.insert(pair(0,new DensityBC<dim>()));
  boundary_descriptor_velocity->dirichlet_bc.insert(pair(0,new VelocityBC<dim>()));
  boundary_descriptor_pressure->neumann_bc.insert(pair(0,new Functions::ZeroFunction<dim>(1)));
  // energy: prescribe energy
  boundary_descriptor_energy->boundary_variable.insert(pair_variable(0,CompNS::EnergyBoundaryVariable::Energy));
  boundary_descriptor_energy->dirichlet_bc.insert(pair(0,new EnergyBC<dim>()));
}

template<int dim>
void set_field_functions(std::shared_ptr<CompNS::FieldFunctions<dim> > field_functions)
{
  field_functions->initial_solution.reset(new Solution<dim>());
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

  pp_data.error_data.analytical_solution_available = true;
  pp_data.error_data.analytical_solution.reset(new Solution<dim>());
  pp_data.error_data.error_calc_start_time = param.start_time;
  pp_data.error_data.error_calc_interval_time = (param.end_time-param.start_time)/20;

  std::shared_ptr<CompNS::PostProcessorBase<dim, Number> > pp;
  pp.reset(new CompNS::PostProcessor<dim, Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COMP_NS_H_ */
