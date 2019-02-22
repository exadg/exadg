/*
 * 3D_taylor_green_vortex.h
 *
 *  Created on: Mar 7, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_


#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/base/function.h>

/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// set the number of space dimensions: DIMENSION = 2, 3
const unsigned int DIMENSION = 3;

// set the polynomial degree of the shape functions
const unsigned int FE_DEGREE = 3;

//number of quadrature points in 1D
//const unsigned int QPOINTS_CONV = FE_DEGREE + 1;
const unsigned int QPOINTS_CONV = FE_DEGREE + (FE_DEGREE+2)/2; // 3/2-overintegration
//const unsigned int QPOINTS_CONV = 2*FE_DEGREE+1;

const unsigned int QPOINTS_VIS = QPOINTS_CONV;
//const unsigned int QPOINTS_VIS = FE_DEGREE + 1;

// set the number of refine levels for spatial convergence tests
const unsigned int REFINE_STEPS_SPACE_MIN = 3;
const unsigned int REFINE_STEPS_SPACE_MAX = 3;

enum class MeshType{ Cartesian, Curvilinear };
const MeshType MESH_TYPE = MeshType::Cartesian;

// only relevant for Cartesian mesh
const unsigned int N_CELLS_1D_COARSE_GRID = 1;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;

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
std::string FILENAME = "after_refactoring_runge_kutta"; // "Re1600_l2_k15_overint";

template<int dim>
void CompNS::InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  equation_type = EquationType::NavierStokes;
  right_hand_side = false;

  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = 20.0*CHARACTERISTIC_TIME;
  dynamic_viscosity = DYN_VISCOSITY;
  reference_density = RHO_0;
  heat_capacity_ratio = GAMMA;
  thermal_conductivity = LAMBDA;
  specific_gas_constant = R;
  max_temperature = T_0;

  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::ExplRK3Stage7Reg2; //ExplRK; //SSPRK; //ExplRK4Stage5Reg3C; //ExplRK3Stage7Reg2;
  order_time_integrator = 3;
  stages = 7;
  calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
  time_step_size = 1.0e-3;
  max_velocity = MAX_VELOCITY;

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

  cfl_number = 0.6;
  diffusion_number = 0.02;
  exponent_fe_degree_cfl = 1.5; //2.0;
  exponent_fe_degree_viscous = 3.0; //4.0;

  // SPATIAL DISCRETIZATION

  // triangulation
  triangulation_type = TriangulationType::Distributed;

  // viscous term
  IP_factor = 1.0;

  // SOLVER

  // NUMERICAL PARAMETERS
  use_combined_operator = true;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  calculate_velocity = true; // activate this for kinetic energy calculations (see below)
  output_data.write_output = true; //false;
  output_data.write_pressure = true;
  output_data.write_velocity = true;
  output_data.write_temperature = true;
  output_data.write_vorticity = true;
  output_data.write_divergence = true;
  output_data.output_folder = OUTPUT_FOLDER + "vtu/";
  output_data.output_name = FILENAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/20;
  output_data.degree = FE_DEGREE;

  // kinetic energy
  kinetic_energy_data.calculate = true;
  kinetic_energy_data.calculate_every_time_steps = 1;
  kinetic_energy_data.viscosity = DYN_VISCOSITY/RHO_0;
  kinetic_energy_data.filename_prefix = OUTPUT_FOLDER + FILENAME;

  // kinetic energy spectrum
  kinetic_energy_spectrum_data.calculate = true;
  kinetic_energy_spectrum_data.calculate_every_time_steps = -1;
  kinetic_energy_spectrum_data.calculate_every_time_interval = 0.5;
  kinetic_energy_spectrum_data.filename_prefix = OUTPUT_FOLDER + "spectrum_" + FILENAME;
  kinetic_energy_spectrum_data.evaluation_points_per_cell = (FE_DEGREE+1)*1;
  kinetic_energy_spectrum_data.output_tolerance = 1.e-12;

  output_solver_info_every_timesteps = 1e3; //1e6;

  // restart
  restart_data.write_restart = false;
  restart_data.interval_time = 8.0;
  restart_data.filename = OUTPUT_FOLDER + FILENAME + "_restart";
}


/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Analytical solutions for initial field functions
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

  virtual ~Solution(){};

  virtual double value (const Point<dim>   &x,
                        const unsigned int component = 0) const;
};

template<int dim>
double Solution<dim>::value(const Point<dim>    &x,
                            const unsigned int  component) const
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

 /**************************************************************************************/
 /*                                                                                    */
 /*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
 /*                                                                                    */
 /**************************************************************************************/

#include "../incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"

 template<int dim>
 void create_grid_and_set_boundary_conditions(
   std::shared_ptr<parallel::Triangulation<dim>>            triangulation,
   unsigned int const                                       n_refine_space,
   std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        /*boundary_descriptor_density*/,
   std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        /*boundary_descriptor_velocity*/,
   std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        /*boundary_descriptor_pressure*/,
   std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim> >  /*boundary_descriptor_energy*/,
   std::vector<GridTools::PeriodicFacePair<typename
     Triangulation<dim>::cell_iterator> >                   &periodic_faces)
 {
   const double pi = numbers::PI;
   const double left = - pi * L, right = pi * L;
   std::vector<unsigned int> repetitions({N_CELLS_1D_COARSE_GRID,
                                          N_CELLS_1D_COARSE_GRID,
                                          N_CELLS_1D_COARSE_GRID});

   Point<dim> point1(left,left,left), point2(right,right,right);
   GridGenerator::subdivided_hyper_rectangle(*triangulation,repetitions,point1,point2);
   
   if(MESH_TYPE == MeshType::Cartesian)
   {
     // do nothing
   }
   else if(MESH_TYPE == MeshType::Curvilinear)
   {
     AssertThrow(N_CELLS_1D_COARSE_GRID == 1,
         ExcMessage("Only N_CELLS_1D_COARSE_GRID=1 possible for curvilinear grid."));

     triangulation->set_all_manifold_ids(1);
     double const deformation = 0.5;
     unsigned const frequency = 2;
     static DeformedCubeManifold<dim> manifold(left, right, deformation, frequency);
     triangulation->set_manifold(1, manifold);
   }

   // set boundary indicators
   AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim==3!"));

   typename Triangulation<dim>::cell_iterator cell = triangulation->begin(), endc = triangulation->end();
   for(;cell!=endc;++cell)
   {
     for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
     {
       // x-direction
       if((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12))
         cell->face(face_number)->set_all_boundary_ids (0);
       else if((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
         cell->face(face_number)->set_all_boundary_ids (1);
       // y-direction
       else if((std::fabs(cell->face(face_number)->center()(1) - left)< 1e-12))
         cell->face(face_number)->set_all_boundary_ids (2);
       else if((std::fabs(cell->face(face_number)->center()(1) - right)< 1e-12))
         cell->face(face_number)->set_all_boundary_ids (3);
       // z-direction
       else if((std::fabs(cell->face(face_number)->center()(2) - left)< 1e-12))
         cell->face(face_number)->set_all_boundary_ids (4);
       else if((std::fabs(cell->face(face_number)->center()(2) - right)< 1e-12))
         cell->face(face_number)->set_all_boundary_ids (5);
     }
   }

   auto tria = dynamic_cast<Triangulation<dim>*>(&*triangulation);
   GridTools::collect_periodic_faces(*tria, 0, 1, 0 /*x-direction*/, periodic_faces);
   GridTools::collect_periodic_faces(*tria, 2, 3, 1 /*y-direction*/, periodic_faces);
   GridTools::collect_periodic_faces(*tria, 4, 5, 2 /*z-direction*/, periodic_faces);

   triangulation->add_periodicity(periodic_faces);

   // perform global refinements
   triangulation->refine_global(n_refine_space);

   // test case with periodics BC -> boundary descriptors remain empty
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

template<int dim, int fe_degree, int n_q_points_conv, int n_q_points_vis, typename value_type>
std::shared_ptr<CompNS::PostProcessor<dim,fe_degree, n_q_points_conv, n_q_points_vis, value_type> >
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

  std::shared_ptr<CompNS::PostProcessor<dim,fe_degree, n_q_points_conv, n_q_points_vis, value_type> > pp;
  pp.reset(new CompNS::PostProcessor<dim,fe_degree, n_q_points_conv, n_q_points_vis, value_type>(pp_data));

  return pp;
}


#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_ */
