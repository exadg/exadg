/*
 * turbulent_channel.h
 *
 *  Created on: Apr 20, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_




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
const unsigned int REFINE_STEPS_SPACE_MIN = 2;
const unsigned int REFINE_STEPS_SPACE_MAX = 2;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;

std::string OUTPUT_FOLDER = "output_comp_ns/turbulent_channel/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string FILENAME = "test";

// set problem specific parameters like physical dimensions, etc.
double const DIMENSIONS_X1 = 2.0*numbers::PI;
double const DIMENSIONS_X2 = 2.0;
double const DIMENSIONS_X3 = numbers::PI;

// set Re = u_tau * delta / nu, density=1, u_tau=1, delta=1 -> calculate kinematic and dynamic viscosities
const double Re = 180.0;
const double RHO_0 = 1.0;
const double nu = 1.0/Re;
const double DYN_VISCOSITY = RHO_0*nu;

// set R, gamma, Pr -> calculate c_p, lambda
const double R = 287.0;
const double GAMMA = 1.4;
const double C_P = GAMMA/(GAMMA-1.0)*R;
const double PRANDTL = 0.71; // Pr = mu * c_p / lambda
const double LAMBDA = DYN_VISCOSITY * C_P / PRANDTL;

// set Ma number -> calculate speed of sound c_0, temperature T_0, pressure p_0
const double MACH = 0.1;
const double MAX_VELOCITY = 22.0;
const double SPEED_OF_SOUND = MAX_VELOCITY/MACH;
const double T_0 = SPEED_OF_SOUND*SPEED_OF_SOUND/GAMMA/R;
const double p_0 = RHO_0*R*T_0;

// flow-through time based on mean centerline velocity
const double CHARACTERISTIC_TIME = DIMENSIONS_X1/MAX_VELOCITY;

double const START_TIME = 0.0;
double const END_TIME = 200.0*CHARACTERISTIC_TIME;

double const SAMPLE_START_TIME = 100.0*CHARACTERISTIC_TIME;
double const SAMPLE_END_TIME = END_TIME;

// use a negative GRID_STRETCH_FAC to deactivate grid stretching
const double GRID_STRETCH_FAC = 1.8;

enum class GridStretchType{ TransformGridCells, VolumeManifold };
GridStretchType GRID_STRETCH_TYPE = GridStretchType::VolumeManifold;

template<int dim>
void CompNS::InputParameters<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  right_hand_side = true;

  // PHYSICAL QUANTITIES
  start_time = START_TIME;
  end_time = END_TIME;
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
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFLAndDiffusion;
  time_step_size = 1.0e-3;
  max_velocity = MAX_VELOCITY;
  cfl_number = 1.0;
  diffusion_number = 0.02;
  exponent_fe_degree_cfl = 1.5;
  exponent_fe_degree_viscous = 3.0;

  // SPATIAL DISCRETIZATION

  // viscous term
  IP_factor = 1.0;

  // SOLVER

  // NUMERICAL PARAMETERS
  use_combined_operator = true;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  calculate_velocity = true; // activate this for kinetic energy calculations (see below)
  output_data.write_output = true;
  output_data.write_pressure = true;
  output_data.write_velocity = true;
  output_data.write_temperature = true;
  output_data.write_vorticity = false;
  output_data.write_divergence = false;
  output_data.output_folder = OUTPUT_FOLDER + "vtu/";
  output_data.output_name = FILENAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = (end_time-start_time)/100;
  output_data.number_of_patches = FE_DEGREE;

  output_solver_info_every_timesteps = 1e3;

  // turbulent channel statistics
  turb_ch_data.calculate_statistics = true;
  turb_ch_data.cells_are_stretched = false;
  if(GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
    turb_ch_data.cells_are_stretched = true;
  turb_ch_data.sample_start_time = SAMPLE_START_TIME;
  turb_ch_data.sample_end_time = SAMPLE_END_TIME;
  turb_ch_data.sample_every_timesteps = 10;
  turb_ch_data.viscosity = DYN_VISCOSITY;
  turb_ch_data.density = RHO_0;
  turb_ch_data.filename_prefix = OUTPUT_FOLDER + FILENAME;
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
class AnalyticalSolution : public Function<dim>
{
public:
  AnalyticalSolution (const unsigned int  n_components = dim + 2,
                      const double        time = 0.)
    :
  Function<dim>(n_components, time)
  {}

  virtual ~AnalyticalSolution(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const
  {
    const double tol = 1.e-12;
    AssertThrow(std::abs(p[1])<DIMENSIONS_X2/2.0+tol,ExcMessage("Invalid geometry parameters."));

    double factor = 0.2;
    double velocity = -MAX_VELOCITY*(pow(p[1],6.0)-1.0)*(1.0+((double)rand()/RAND_MAX-0.5)*factor);

    double const u1 = velocity;
    double const u2 = 0.0;
    double const u3 = 0.0;
    double const rho = RHO_0;
    double const T = T_0;
    double const E = R/(GAMMA-1.0)*T /* e = c_v * T */
                     + 0.5*(u1*u1 + u2*u2 + u3*u3);

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
                        const unsigned int  component = 0) const
  {
    return T_0;
  }
};

template<int dim>
 class RightHandSideVelocity : public Function<dim>
 {
 public:
   RightHandSideVelocity (const unsigned int   n_components = dim,
                          const double         time = 0.)
     :
     Function<dim>(n_components, time)
   {}

   virtual ~RightHandSideVelocity(){};

   virtual double value (const Point<dim>    &p,
                         const unsigned int  component = 0) const;
 };

 template<int dim>
 double RightHandSideVelocity<dim>::value(const Point<dim>   &p,
                                          const unsigned int component) const
 {
   double result = 0.0;

   if(component==0)
   {
     result = RHO_0;
   }

   return result;
 }

 /**************************************************************************************/
 /*                                                                                    */
 /*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
 /*                                                                                    */
 /**************************************************************************************/

#include "../incompressible_navier_stokes_test_cases/grid_functions_turbulent_channel.h"

 template<int dim>
 void create_grid_and_set_boundary_conditions(
   parallel::distributed::Triangulation<dim>                &triangulation,
   unsigned int const                                       n_refine_space,
   std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_density,
   std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_velocity,
   std::shared_ptr<CompNS::BoundaryDescriptor<dim> >        boundary_descriptor_pressure,
   std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim> >  boundary_descriptor_energy,
   std::vector<GridTools::PeriodicFacePair<typename
     Triangulation<dim>::cell_iterator> >                   &periodic_faces)
 {
    /* --------------- Generate grid ------------------- */
    if(GRID_STRETCH_TYPE == GridStretchType::TransformGridCells)
    {
      Point<dim> coordinates;
      coordinates[0] = DIMENSIONS_X1;
      coordinates[1] = DIMENSIONS_X2/2.0; // dimension in y-direction is 2.0, see also function grid_transform() that maps the y-coordinate from [0,1] to [-1,1]
      if (dim == 3)
        coordinates[2] = DIMENSIONS_X3;

      // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
      std::vector<unsigned int> refinements(dim, 1);
      GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(),coordinates);
    }
    else if (GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
    {
      Tensor<1,dim> dimensions;
      dimensions[0] = DIMENSIONS_X1;
      dimensions[1] = DIMENSIONS_X2;
      if(dim==3)
        dimensions[2] = DIMENSIONS_X3;

      // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
      std::vector<unsigned int> refinements(dim, 1);
      GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(-dimensions/2.0),Point<dim>(dimensions/2.0));

      // manifold
      unsigned int manifold_id = 1;
      for (typename Triangulation<dim>::cell_iterator cell = triangulation.begin(); cell != triangulation.end(); ++cell)
      {
        cell->set_all_manifold_ids(manifold_id);
      }

      // apply mesh stretching towards no-slip boundaries in y-direction
      static const ManifoldTurbulentChannel<dim> manifold(dimensions);
      triangulation.set_manifold(manifold_id, manifold);
    }

    //periodicity in x- and z-direction
    //add 10 to avoid conflicts with dirichlet boundary, which is 0
    triangulation.begin()->face(0)->set_all_boundary_ids(0+10);
    triangulation.begin()->face(1)->set_all_boundary_ids(1+10);
    //periodicity in z-direction
    if (dim == 3)
    {
      triangulation.begin()->face(4)->set_all_boundary_ids(2+10);
      triangulation.begin()->face(5)->set_all_boundary_ids(3+10);
    }

    GridTools::collect_periodic_faces(triangulation, 0+10, 1+10, 0, periodic_faces);
    if (dim == 3)
      GridTools::collect_periodic_faces(triangulation, 2+10, 3+10, 2, periodic_faces);

    triangulation.add_periodicity(periodic_faces);

    // perform global refinements
    triangulation.refine_global(n_refine_space);

    if(GRID_STRETCH_TYPE == GridStretchType::TransformGridCells)
    {
      // perform grid transform
      GridTools::transform (&grid_transform<dim>, triangulation);
    }

    // fill boundary descriptors

    // zero function scalar
    std::shared_ptr<Function<dim> > zero_function_scalar;
    zero_function_scalar.reset(new ZeroFunction<dim>(1));

    // zero function vectorial
    std::shared_ptr<Function<dim> > zero_function_vectorial;
    zero_function_vectorial.reset(new ZeroFunction<dim>(dim));

    // For Neumann boundaries, no value is prescribed (only first derivative of density occurs in equations).
    // Hence the specified function is irrelevant (i.e., it is not used).
    boundary_descriptor_density->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_scalar));

    // velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_vectorial));

    // pressure
    boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_scalar));

    // energy: prescribe temperature
    boundary_descriptor_energy->boundary_variable.insert(std::pair<types::boundary_id,CompNS::EnergyBoundaryVariable>(0,CompNS::EnergyBoundaryVariable::Temperature));

    std::shared_ptr<Function<dim> > energy_bc;
    energy_bc.reset(new EnergyBC<dim>());
    boundary_descriptor_energy->dirichlet_bc.insert(std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,energy_bc));
 }

template<int dim>
void set_field_functions(std::shared_ptr<CompNS::FieldFunctions<dim> > field_functions)
{
  // zero function scalar
  std::shared_ptr<Function<dim> > zero_function_scalar;
  zero_function_scalar.reset(new ZeroFunction<dim>(1));

  // zero function vectorial
  std::shared_ptr<Function<dim> > zero_function_vectorial;
  zero_function_vectorial.reset(new ZeroFunction<dim>(dim));

  // initial solution
  std::shared_ptr<Function<dim> > initial_solution;
  initial_solution.reset(new AnalyticalSolution<dim>());
  field_functions->initial_solution = initial_solution;

  // rhs density
  field_functions->right_hand_side_density = zero_function_scalar;

  // rhs velocity
  std::shared_ptr<Function<dim> > right_hand_side_velocity;
  right_hand_side_velocity.reset(new RightHandSideVelocity<dim>());
  field_functions->right_hand_side_velocity = right_hand_side_velocity;

  // rhs energy
  field_functions->right_hand_side_energy = zero_function_scalar;
}

template<int dim>
void set_analytical_solution(std::shared_ptr<CompNS::AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->solution.reset(new AnalyticalSolution<dim>());
}

// postprocessor
#include "../../include/incompressible_navier_stokes/postprocessor/statistics_manager.h"

template<int dim>
struct PostProcessorDataTurbulentChannel
{
  CompNS::PostProcessorData<dim> pp_data;
  TurbulentChannelData turb_ch_data;
};

template<int dim, int fe_degree>
class PostProcessorTurbulentChannel : public CompNS::PostProcessor<dim,fe_degree>
{
public:
  PostProcessorTurbulentChannel(PostProcessorDataTurbulentChannel<dim> const & pp_data_turb_channel)
    :
    CompNS::PostProcessor<dim,fe_degree>(pp_data_turb_channel.pp_data),
    turb_ch_data(pp_data_turb_channel.turb_ch_data)
  {}

  void setup(DoFHandler<dim> const                             &dof_handler_in,
             DoFHandler<dim> const                             &dof_handler_vector_in,
             DoFHandler<dim> const                             &dof_handler_scalar_in,
             Mapping<dim> const                                &mapping_in,
             MatrixFree<dim,double> const                      &matrix_free_data_in,
             DofQuadIndexData const                            &dof_quad_index_data_in,
             std::shared_ptr<CompNS::AnalyticalSolution<dim> > analytical_solution_in)
  {
    // call setup function of base class
    CompNS::PostProcessor<dim,fe_degree>::setup(
        dof_handler_in,
        dof_handler_vector_in,
        dof_handler_scalar_in,
        mapping_in,
        matrix_free_data_in,
        dof_quad_index_data_in,
        analytical_solution_in);

    // perform setup of turbulent channel related things
    statistics_turb_ch.reset(new StatisticsManager<dim>(dof_handler_vector_in,mapping_in));
    statistics_turb_ch->setup(&grid_transform_y,turb_ch_data);
  }

  void do_postprocessing(parallel::distributed::Vector<double> const   &solution,
                         parallel::distributed::Vector<double> const   &velocity,
                         parallel::distributed::Vector<double> const   &pressure,
                         std::vector<SolutionField<dim,double> > const &additional_fields,
                         double const                                  time,
                         int const                                     time_step_number)
  {
    CompNS::PostProcessor<dim,fe_degree>::do_postprocessing(
        solution,
        velocity,
        pressure,
        additional_fields,
        time,
        time_step_number);

    statistics_turb_ch->evaluate(velocity,time,time_step_number);
  }

  TurbulentChannelData turb_ch_data;
  std::shared_ptr<StatisticsManager<dim> > statistics_turb_ch;
};

template<int dim, int fe_degree>
std::shared_ptr<CompNS::PostProcessor<dim,fe_degree> >
construct_postprocessor(CompNS::InputParameters<dim> const &param)
{
  CompNS::PostProcessorData<dim> pp_data;

  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.kinetic_energy_data = param.kinetic_energy_data;
  pp_data.kinetic_energy_spectrum_data = param.kinetic_energy_spectrum_data;

  PostProcessorDataTurbulentChannel<dim> pp_data_turb_ch;
  pp_data_turb_ch.pp_data = pp_data;
  pp_data_turb_ch.turb_ch_data = param.turb_ch_data;

  std::shared_ptr<CompNS::PostProcessor<dim,fe_degree> > pp;
  pp.reset(new PostProcessorTurbulentChannel<dim,fe_degree>(pp_data_turb_ch));

  return pp;
}


#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
