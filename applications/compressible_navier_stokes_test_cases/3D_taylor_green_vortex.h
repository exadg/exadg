/*
 * 3D_taylor_green_vortex.h
 *
 *  Created on: Mar 7, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_

#include "../grid_tools/periodic_box.h"
#include "../../include/compressible_navier_stokes/postprocessor/postprocessor.h"

/************************************************************************************************************/
/*                                                                                                          */
/*                                              INPUT PARAMETERS                                            */
/*                                                                                                          */
/************************************************************************************************************/

// convergence studies in space or time
unsigned int const DEGREE_MIN = 3;
unsigned int const DEGREE_MAX = 3;

unsigned int const REFINE_SPACE_MIN = 3;
unsigned int const REFINE_SPACE_MAX = 3;

unsigned int const REFINE_TIME_MIN = 0;
unsigned int const REFINE_TIME_MAX = 0;

// problem specific parameters

enum class MeshType{ Cartesian, Curvilinear };
const MeshType MESH_TYPE = MeshType::Cartesian;

// set parameters according to Wiart et al. ("Assessment of discontinuous Galerkin method
// for the simulation of vortical flows at high Reynolds number"):

// set Re, rho_0, V_0, L -> calculate viscosity mu
const double Re = 1600.0;
const double RHO_0 = 1.0;
const double V_0 = 1.0;
const double L = 1.0;
const double DYN_VISCOSITY = RHO_0*V_0*L/Re;

// set R, gamma, Pr -> calculate c_p, lambda
const double R = 287.0;
const double GAMMA = 1.4;
const double C_P = GAMMA/(GAMMA-1.0)*R;
const double PRANDTL = 0.71; // Pr = mu * c_p / lambda
const double LAMBDA = DYN_VISCOSITY * C_P / PRANDTL;

// set Ma number -> calculate speed of sound c_0, temperature T_0, pressure p_0
const double MACH = 0.1;
const double SPEED_OF_SOUND = V_0/MACH;
const double T_0 = SPEED_OF_SOUND*SPEED_OF_SOUND/GAMMA/R;
const double p_0 = RHO_0*R*T_0;

const double MAX_VELOCITY = V_0;
const double CHARACTERISTIC_TIME = L/V_0;

// output folders and filenames
std::string OUTPUT_FOLDER = "output_comp_ns/taylor_green_vortex/";
std::string FILENAME = "test";

namespace CompNS
{
void set_input_parameters(InputParameters & param)
{
  // MATHEMATICAL MODEL
  param.dim = 3;
  param.equation_type = EquationType::NavierStokes;
  param.right_hand_side = false;

  // PHYSICAL QUANTITIES
  param.start_time = 0.0;
  param.end_time = 20.0*CHARACTERISTIC_TIME;
  param.dynamic_viscosity = DYN_VISCOSITY;
  param.reference_density = RHO_0;
  param.heat_capacity_ratio = GAMMA;
  param.thermal_conductivity = LAMBDA;
  param.specific_gas_constant = R;
  param.max_temperature = T_0;

  // TEMPORAL DISCRETIZATION

  // l=4, k=3 (64³), exponent_fe_degree_cfl = 2:
  // Classical RK:
  // classical RK1 (1 stage): CFL_crit = 0.04 - 0.05, costs = stages / CFL = 20 - 25
  // classical RK2 (2 stages):CFL_crit = 0.31  - 0.32, costs = stages / CFL =  6.3 - 6.5
  // classical RK3 (3 stages):CFL_crit = 0.35 - 0.4, costs = stages / CFL =  7.5 - 8.5
  // classical RK4 (4 stages):CFL_crit = 0.4  - 0.45, costs = stages / CFL =  8.9 - 10

  // Kennedy et al.:
  // LowStorageRK3Stage3Reg2: CFL_crit = 0.3 - 0.4, costs = stages / CFL =  7.5 - 10
  // LowStorageRK3Stage4Reg2: CFL_crit = 0.4 - 0.5, costs = stages / CFL =  8 - 10
  // LowStorageRK4Stage5Reg2: CFL_crit = 0.725 - 0.75, costs = stages / CFL =  6.7 - 6.9
  // LowStorageRK4Stage5Reg3: CFL_crit = 0.7 - 0.725, costs = stages / CFL =  6.9 - 7.1
  // LowStorageRK5Stage9Reg2: CFL_crit = 0.8 - 0.9, costs = stages / CFL =  10  - 11.3

  // Toulorge & Desmet: 3rd order scheme with 7 stages currently the most efficient scheme
  // LowStorageRK3Stage7Reg2: CFL_crit = 1.3 - 1.35, costs = stages / CFL =  5.2 - 5.4, computational costs = 0.69 CPUh
  // LowStorageRK4Stage8Reg2: CFL_crit = 1.25 - 1.3, costs = stages / CFL =  6.2 - 6.4 

  // Kubatko et al.:
  // SSPRK(7,3): CFL_crit = 1.2 - 1.25, costs = stages / CFL = 5.6 - 5.8
  // SSPRK(8,3): CFL_crit = 1.5 - 1.6, costs = stages / CFL = 5.0 - 5.3, computational costs = 0.77 CPUh
  // SSPRK(8,4): CFL_crit = 1.25 - 1.3, costs = stages / CFL = 6.2 - 6.4

  param.temporal_discretization = TemporalDiscretization::ExplRK3Stage7Reg2;
  param.order_time_integrator = 3;
  param.stages = 7;
  param.calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
  param.time_step_size = 1.0e-3;
  param.max_velocity = MAX_VELOCITY;
  param.cfl_number = 0.6;
  param.diffusion_number = 0.02;
  param.exponent_fe_degree_cfl = 1.5;
  param.exponent_fe_degree_viscous = 3.0;
  param.dt_refinements = REFINE_TIME_MIN;

  // output of solver information
  param.solver_info_data.print_to_screen = true;
  param.solver_info_data.interval_time = (param.end_time-param.start_time)/20;

  // restart
  param.restart_data.write_restart = false;
  param.restart_data.interval_time = 1.0;
  param.restart_data.filename = OUTPUT_FOLDER + FILENAME + "_restart";

  // SPATIAL DISCRETIZATION
  param.triangulation_type = TriangulationType::Distributed;
  param.degree = DEGREE_MIN;
  param.mapping = MappingType::Affine;
  param.n_q_points_convective = QuadratureRule::Overintegration32k;
  param.n_q_points_viscous = QuadratureRule::Overintegration32k;
  param.h_refinements = REFINE_SPACE_MIN;

  // viscous term
  param.IP_factor = 1.0;

  // NUMERICAL PARAMETERS
  param.use_combined_operator = true;
}
}

/************************************************************************************************************/
/*                                                                                                          */
/*                                       CREATE GRID AND SET BOUNDARY IDs                                   */
/*                                                                                                          */
/************************************************************************************************************/

template<int dim>
void create_grid_and_set_boundary_ids(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
                                      unsigned int const                                n_refine_space,
                                      std::vector<GridTools::PeriodicFacePair<typename
                                        Triangulation<dim>::cell_iterator> >            &periodic_faces,
                                      unsigned int const                                n_subdivisions = 1)
{
  double const pi = numbers::PI;
  double const left = - pi * L, right = pi * L;
  double const deformation = 0.1;

  bool curvilinear_mesh = false;
  if(MESH_TYPE == MeshType::Cartesian)
  {
    // do nothing
  }
  else if(MESH_TYPE == MeshType::Curvilinear)
  {
    curvilinear_mesh = true;
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  create_periodic_box(triangulation,
                      n_refine_space,
                      periodic_faces,
                      n_subdivisions,
                      left,
                      right,
                      curvilinear_mesh,
                      deformation);
}

/************************************************************************************************************/
/*                                                                                                          */
/*                         FUNCTIONS (INITIAL/BOUNDARY CONDITIONS, RIGHT-HAND SIDE, etc.)                   */
/*                                                                                                          */
/************************************************************************************************************/

/*
 *  Analytical solution
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

  double value (const Point<dim>   &x,
                const unsigned int component = 0) const
  {
    double const u1 = V_0*std::sin(x[0]/L)*std::cos(x[1]/L)*std::cos(x[2]/L);
    double const u2 = -V_0*std::cos(x[0]/L)*std::sin(x[1]/L)*std::cos(x[2]/L);
    double const p = p_0 + RHO_0 * V_0 * V_0 / 16.0 * (std::cos(2.0*x[0]/L) + std::cos(2.0*x[1]/L)) * (std::cos(2.0*x[2]/L) + 2.0);
    double const T = T_0;
    double const rho = p/(R*T);
    double const E = R/(GAMMA-1.0)*T /* e = c_v * T */
                     + 0.5*(u1*u1 + u2*u2 /* + u3*u3 with u3=0*/);

    double result = 0.0;

    if(component==0)
      result = rho;
    else if (component==1)
      result = rho * u1;
    else if (component==2)
      result = rho * u2;
    else if (component==3)
      result = 0.0;
    else if (component==1+dim)
      result = rho * E;

    return result;
  }
};

namespace CompNS
{

template<int dim>
void set_boundary_conditions(
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        /*boundary_descriptor_density*/,
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        /*boundary_descriptor_velocity*/,
  std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        /*boundary_descriptor_pressure*/,
  std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim> >  /*boundary_descriptor_energy*/)
{
  // test case with periodic BC -> boundary descriptors remain empty
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

  pp_data.calculate_velocity = true; // activate this for kinetic energy calculations (see below)
  pp_data.output_data.write_output = false;
  pp_data.output_data.write_pressure = true;
  pp_data.output_data.write_velocity = true;
  pp_data.output_data.write_temperature = true;
  pp_data.output_data.write_vorticity = true;
  pp_data.output_data.write_divergence = true;
  pp_data.output_data.output_folder = OUTPUT_FOLDER + "vtu/";
  pp_data.output_data.output_name = FILENAME;
  pp_data.output_data.output_start_time = param.start_time;
  pp_data.output_data.output_interval_time = (param.end_time-param.start_time)/20;
  pp_data.output_data.degree = param.degree;

  // kinetic energy
  pp_data.kinetic_energy_data.calculate = true;
  pp_data.kinetic_energy_data.calculate_every_time_steps = 1;
  pp_data.kinetic_energy_data.viscosity = DYN_VISCOSITY/RHO_0;
  pp_data.kinetic_energy_data.filename_prefix = OUTPUT_FOLDER + FILENAME;

  // kinetic energy spectrum
  pp_data.kinetic_energy_spectrum_data.calculate = true;
  pp_data.kinetic_energy_spectrum_data.calculate_every_time_steps = -1;
  pp_data.kinetic_energy_spectrum_data.calculate_every_time_interval = 0.5;
  pp_data.kinetic_energy_spectrum_data.filename_prefix = OUTPUT_FOLDER + "spectrum_" + FILENAME;
  pp_data.kinetic_energy_spectrum_data.degree = param.degree;
  pp_data.kinetic_energy_spectrum_data.evaluation_points_per_cell = (param.degree+1)*1;
  pp_data.kinetic_energy_spectrum_data.output_tolerance = 1.e-12;

  std::shared_ptr<CompNS::PostProcessorBase<dim, Number> > pp;
  pp.reset(new CompNS::PostProcessor<dim, Number>(pp_data));

  return pp;
}

}

#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_ */
