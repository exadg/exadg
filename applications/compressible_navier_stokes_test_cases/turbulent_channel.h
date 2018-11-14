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
const unsigned int REFINE_STEPS_SPACE_MIN = 3;
const unsigned int REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// can only be used for GridStretchType::TransformGridCells, otherwise coarse grid consists of 1 cell
const unsigned int N_CELLS_1D_COARSE_GRID = 1;

// set the number of refine levels for temporal convergence tests
const unsigned int REFINE_STEPS_TIME_MIN = 0;
const unsigned int REFINE_STEPS_TIME_MAX = 0;

std::string OUTPUT_FOLDER = "output_comp_ns/turbulent_channel/";
std::string OUTPUT_FOLDER_VTU = OUTPUT_FOLDER + "vtu/";
std::string FILENAME = "Re180_l3_k3";

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

// set Ma number -> calculate speed of sound c_0, temperature T_0
const double MACH = 0.1;
const double MAX_VELOCITY =  18.3; //18.3 for Re_tau = 180; //22.0;
const double SPEED_OF_SOUND = MAX_VELOCITY/MACH;
const double T_0 = SPEED_OF_SOUND*SPEED_OF_SOUND/GAMMA/R;

// flow-through time based on mean centerline velocity
const double CHARACTERISTIC_TIME = DIMENSIONS_X1/MAX_VELOCITY;

double const START_TIME = 0.0;
double const END_TIME = 200.0*CHARACTERISTIC_TIME;

double const SAMPLE_START_TIME = 100.0*CHARACTERISTIC_TIME;
double const SAMPLE_END_TIME = END_TIME;

// use a negative GRID_STRETCH_FAC to deactivate grid stretching
const double GRID_STRETCH_FAC = 1.8;

enum class GridStretchType{ TransformGridCells, VolumeManifold };
GridStretchType GRID_STRETCH_TYPE = GridStretchType::TransformGridCells;

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
  cfl_number = 1.9;
  diffusion_number = 0.17;
  exponent_fe_degree_cfl = 1.5;
  exponent_fe_degree_viscous = 3.0;

  // SPATIAL DISCRETIZATION
  degree_mapping = FE_DEGREE;

  // viscous term
  IP_factor = 1.0;

  // SOLVER

  // NUMERICAL PARAMETERS
  use_combined_operator = true;

  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;
  calculate_velocity = true; // activate this for kinetic energy calculations (see below)
  output_data.write_output = false;
  output_data.write_pressure = true;
  output_data.write_velocity = true;
  output_data.write_temperature = true;
  output_data.write_vorticity = false;
  output_data.write_divergence = false;
  output_data.output_folder = OUTPUT_FOLDER + "vtu/";
  output_data.output_name = FILENAME;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = 1.0;
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
                        const unsigned int component = 0) const
  {
    const double tol = 1.e-12;
    AssertThrow(std::abs(p[1])<DIMENSIONS_X2/2.0+tol,ExcMessage("Invalid geometry parameters."));

    double velocity1 = -MAX_VELOCITY*(pow(p[1],6.0)-1.0)*(1.0+((double)rand()/RAND_MAX-1.0)*0.5-2./MAX_VELOCITY*std::sin(p[2]*8.));
    double velocity3 = (pow(p[1],6.0)-1.0)*std::sin(p[0]*8.)*2.;

    // viscous time step limitations: consider a laminar test case with a large viscosity
//    double velocity1 =  -MAX_VELOCITY*(pow(p[1],2.0)-1.0);
//    double velocity3 = 0.0;

    double const u1 = velocity1;
    double const u2 = 0.0;
    double const u3 = velocity3;
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

  virtual double value (const Point<dim>    &/*p*/,
                        const unsigned int  /*component = 0*/) const
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
 double RightHandSideVelocity<dim>::value(const Point<dim>   &/*p*/,
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
      std::vector<unsigned int> refinements(dim, N_CELLS_1D_COARSE_GRID);
      GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(),coordinates);

      typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
      for(;cell!=endc;++cell)
      {
        for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
        {
          // x-direction
          if((std::fabs(cell->face(face_number)->center()(0) - 0.0)< 1e-12))
            cell->face(face_number)->set_all_boundary_ids (0+10);
          else if((std::fabs(cell->face(face_number)->center()(0) - DIMENSIONS_X1)< 1e-12))
            cell->face(face_number)->set_all_boundary_ids (1+10);
          // z-direction
          else if((std::fabs(cell->face(face_number)->center()(2) - 0-0)< 1e-12))
            cell->face(face_number)->set_all_boundary_ids (2+10);
          else if((std::fabs(cell->face(face_number)->center()(2) - DIMENSIONS_X3)< 1e-12))
            cell->face(face_number)->set_all_boundary_ids (3+10);
        }
      }
    }
    else if (GRID_STRETCH_TYPE == GridStretchType::VolumeManifold)
    {
      Tensor<1,dim> dimensions;
      dimensions[0] = DIMENSIONS_X1;
      dimensions[1] = DIMENSIONS_X2;
      if(dim==3)
        dimensions[2] = DIMENSIONS_X3;

      // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
      AssertThrow(N_CELLS_1D_COARSE_GRID == 1,
          ExcMessage("Only N_CELLS_1D_COARSE_GRID=1 possible for curvilinear grid."));

      std::vector<unsigned int> refinements(dim, N_CELLS_1D_COARSE_GRID);
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
    zero_function_scalar.reset(new Functions::ZeroFunction<dim>(1));

    // zero function vectorial
    std::shared_ptr<Function<dim> > zero_function_vectorial;
    zero_function_vectorial.reset(new Functions::ZeroFunction<dim>(dim));

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
  std::shared_ptr<Function<dim> > right_hand_side_velocity;
  right_hand_side_velocity.reset(new RightHandSideVelocity<dim>());
  field_functions->right_hand_side_velocity = right_hand_side_velocity;

  // rhs energy
  field_functions->right_hand_side_energy = zero_function_scalar;
}

template<int dim>
void set_analytical_solution(std::shared_ptr<CompNS::AnalyticalSolution<dim> > analytical_solution)
{
  analytical_solution->solution.reset(new Solution<dim>());
}

// postprocessor
#include "../../include/incompressible_navier_stokes/postprocessor/statistics_manager.h"

template<int dim>
struct PostProcessorDataTurbulentChannel
{
  CompNS::PostProcessorData<dim> pp_data;
  TurbulentChannelData turb_ch_data;
};

template<int dim, int fe_degree, int n_q_points_conv, int n_q_points_vis, typename value_type>
class PostProcessorTurbulentChannel : public CompNS::PostProcessor<dim,fe_degree, n_q_points_conv, n_q_points_vis, value_type>
{
public:
  typedef CompNS::PostProcessor<dim,fe_degree, n_q_points_conv, n_q_points_vis, value_type> Base;

  typedef LinearAlgebra::distributed::Vector<double> VectorType;

  typedef typename Base::NavierStokesOperator NavierStokesOperator;

  PostProcessorTurbulentChannel(PostProcessorDataTurbulentChannel<dim> const & pp_data_turb_channel)
    :
    Base(pp_data_turb_channel.pp_data),
    turb_ch_data(pp_data_turb_channel.turb_ch_data)
  {}

  void setup(NavierStokesOperator const                        &navier_stokes_operator_in,
             DoFHandler<dim> const                             &dof_handler_in,
             DoFHandler<dim> const                             &dof_handler_vector_in,
             DoFHandler<dim> const                             &dof_handler_scalar_in,
             Mapping<dim> const                                &mapping_in,
             MatrixFree<dim,double> const                      &matrix_free_data_in,
             DofQuadIndexData const                            &dof_quad_index_data_in,
             std::shared_ptr<CompNS::AnalyticalSolution<dim> > analytical_solution_in)
  {
    // call setup function of base class
    Base::setup(
        navier_stokes_operator_in,
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

  void do_postprocessing(VectorType const &solution,
                         double const     time,
                         int const        time_step_number)
  {
    Base::do_postprocessing(
        solution,
        time,
        time_step_number);

    statistics_turb_ch->evaluate(this->velocity,time,time_step_number);
  }

  TurbulentChannelData turb_ch_data;
  std::shared_ptr<StatisticsManager<dim> > statistics_turb_ch;
};

template<int dim, int fe_degree, int n_q_points_conv, int n_q_points_vis, typename value_type>
std::shared_ptr<CompNS::PostProcessor<dim, fe_degree, n_q_points_conv, n_q_points_vis, value_type> >
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

  PostProcessorDataTurbulentChannel<dim> pp_data_turb_ch;
  pp_data_turb_ch.pp_data = pp_data;
  pp_data_turb_ch.turb_ch_data = param.turb_ch_data;

  std::shared_ptr<CompNS::PostProcessor<dim, fe_degree, n_q_points_conv, n_q_points_vis, value_type> > pp;
  pp.reset(new PostProcessorTurbulentChannel<dim, fe_degree, n_q_points_conv, n_q_points_vis, value_type>(pp_data_turb_ch));

  return pp;
}


#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TURBULENT_CHANNEL_H_ */
