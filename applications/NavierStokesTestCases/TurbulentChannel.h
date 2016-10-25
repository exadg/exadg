/*
 * TurbulentChannel.h
 *
 *  Created on: Oct 14, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_NAVIERSTOKESTESTCASES_TURBULENTCHANNEL_H_
#define APPLICATIONS_NAVIERSTOKESTESTCASES_TURBULENTCHANNEL_H_



/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 3;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 3;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1; // FE_DEGREE_VELOCITY; // FE_DEGREE_VELOCITY - 1;

// set xwall specific parameters
unsigned int const FE_DEGREE_XWALL = 1;
unsigned int const N_Q_POINTS_1D_XWALL = 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 3;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.

double const MAX_VELOCITY = 22.0;

const double GRID_STRETCH_FAC = 1.8;

// nu = 180  l2p4 or l3p3 with GRID_STRETCH_FAC = 1.8
// nu = 395
// nu = 590
// nu = 950
double const VISCOSITY = 1./180.;
double const END_TIME = 50.0;

std::string OUTPUT_PREFIX = "turb_ch_coupled_nu_180_l3_p3_CFL1";

template<int dim>
void InputParametersNavierStokes<dim>::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::DivergenceFormulation;
  right_hand_side = true;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = END_TIME; //END_TIME is also needed somewhere else
  viscosity = VISCOSITY;


  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::BDFCoupledSolution; //BDFDualSplittingScheme;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit; //Explicit;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL; // AdaptiveTimeStepCFL
  max_velocity = MAX_VELOCITY;
  cfl = 1.0e0; //1.0 if ConstTimeStepCFL
  time_step_size = 1.0e-1;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true;


  // SPATIAL DISCRETIZATION

  // spatial discretization method
  spatial_discretization = SpatialDiscretization::DG;

  // convective term - currently no parameters

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  IP_factor_viscous = 1.0;

  // gradient term
  gradp_integrated_by_parts = true;
  gradp_use_boundary_data = true; //false;

  // divergence term
  divu_integrated_by_parts = true;
  divu_use_boundary_data = true; //false;

  // special case: pure DBC's
  pure_dirichlet_bc = true;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // pressure Poisson equation
  IP_factor_pressure = 1.0;
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::GeometricMultigrid;
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::ChebyshevSmoother;
  abs_tol_pressure = 1.e-12;
  rel_tol_pressure = 1.e-6; //1.e-4;

  // stability in the limit of small time steps and projection step
  small_time_steps_stability = false;
  use_approach_of_ferrer = false;
  deltat_ref = 1.e0;

  // projection step
  projection_type = ProjectionType::DivergencePenalty;
  penalty_factor_divergence = 1.0e0;//1.0e0;
  penalty_factor_continuity = 1.0e0;
  solver_projection = SolverProjection::PCG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  abs_tol_projection = 1.e-20;
  rel_tol_projection = 1.e-12; //1.e-6;

  // viscous step
  solver_viscous = SolverViscous::PCG;
  preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::ChebyshevSmoother;
  abs_tol_viscous = 1.e-12;
  rel_tol_viscous = 1.e-6; //1.e-4;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  abs_tol_newton = 1.e-12;
  rel_tol_newton = 1.e-6;
  max_iter_newton = 1e2;

  // linear solver
  solver_linearized_navier_stokes = SolverLinearizedNavierStokes::GMRES;
  abs_tol_linear = 1.e-12;
  rel_tol_linear = 1.e-3;
  max_iter_linear = 1e4;
  max_n_tmp_vectors = 100;

  // preconditioning linear solver
  preconditioner_linearized_navier_stokes = PreconditionerLinearizedNavierStokes::BlockTriangular;

  // preconditioner velocity/momentum block
  momentum_preconditioner = MomentumPreconditioner::InverseMassMatrix; //VelocityDiffusion;
  solver_momentum_preconditioner = SolverMomentumPreconditioner::GeometricMultigridVCycle;
  rel_tol_solver_momentum_preconditioner = 1.e-3;

  // preconditioner Schur-complement block
  schur_complement_preconditioner = SchurComplementPreconditioner::PressureConvectionDiffusion;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  solver_schur_complement_preconditioner = SolverSchurComplementPreconditioner::GeometricMultigridVCycle;
  rel_tol_solver_schur_complement_preconditioner = 1.e-6;


  // OUTPUT AND POSTPROCESSING
  print_input_parameters = true;

  // write output for visualization of results
  output_data.write_output = true;
  output_data.output_prefix = OUTPUT_PREFIX;
  output_data.output_start_time = start_time;
  output_data.output_interval_time = 1.0;
  output_data.compute_divergence = true;
  output_data.number_of_patches = FE_DEGREE_VELOCITY;

  // calculation of error
  error_data.analytical_solution_available = false;
  error_data.error_calc_start_time = start_time;
  error_data.error_calc_interval_time = output_data.output_interval_time;

  // output of solver information
  output_solver_info_every_timesteps = 1e2;

  // restart
  write_restart = false;
  restart_interval_time = 1.e2;
  restart_interval_wall_time = 1.e6;
  restart_every_timesteps = 1e8;

  // calculate div and mass error
  mass_data.calculate_error = true;
  mass_data.start_time = 0.0;
  mass_data.sample_every_time_steps = 1e2;
  mass_data.filename_prefix = OUTPUT_PREFIX;

  // turbulent channel statistics
  turb_ch_data.calculate_statistics = true;
  turb_ch_data.sample_start_time = 30.0;
  turb_ch_data.sample_end_time = END_TIME;
  turb_ch_data.sample_every_timesteps = 10;
  turb_ch_data.viscosity = VISCOSITY;
  turb_ch_data.filename_prefix = output_data.output_prefix;
}

/**************************************************************************************/
/*                                                                                    */
/*    FUNCTIONS (ANALYTICAL SOLUTION, BOUNDARY CONDITIONS, VELOCITY FIELD, etc.)      */
/*                                                                                    */
/**************************************************************************************/

/*
 *  Analytical solution velocity:
 *
 *  - This function is used to calculate the L2 error
 *
 *  - This function can be used to prescribe initial conditions for the velocity field
 *
 *  - Moreover, this function can be used (if possible for simple geometries)
 *    to prescribe Dirichlet BC's for the velocity field on Dirichlet boundaries
 */

// TODO
template<int dim>
class AnalyticalSolutionVelocity : public Function<dim>
{
public:
  AnalyticalSolutionVelocity (const unsigned int  n_components = dim,
                              const double        time = 0.)
    :
    Function<dim>(n_components, time)
  {}

  virtual ~AnalyticalSolutionVelocity(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double AnalyticalSolutionVelocity<dim>::value(const Point<dim>   &p,
                                              const unsigned int component) const
{
  double result = 0.0;

  if(p[1]<0.9999 && p[1]>-0.9999)
  {
    if(dim==3)
    {
      if(component == 0)
        result = -MAX_VELOCITY*(pow(p[1],6.0)-1.0)*(1.0+((double)rand()/RAND_MAX-1.0)*0.5-2./MAX_VELOCITY*std::sin(p[2]*8.));
      else if(component ==2)
        result = (pow(p[1],6.0)-1.0)*std::sin(p[0]*8.)*2.;
    }
    else if(component == 0)
      result = -MAX_VELOCITY*(pow(p[1],6.0)-1.0);
  }

  return result;
}

/*
 *  Analytical solution pressure
 *
 *  - It is used to calculate the L2 error
 *
 *  - It is used to adjust the pressure level in case of pure Dirichlet BC's
 *    (where the pressure is only defined up to an additive constant)
 *
 *  - This function can be used to prescribe initial conditions for the pressure field
 *
 *  - Moreover, this function can be used (if possible for simple geometries)
 *    to prescribe Dirichlet BC's for the pressure field on Neumann boundaries
 */


template<int dim>
class AnalyticalSolutionPressure : public Function<dim>
{
public:
  AnalyticalSolutionPressure (const double time = 0.)
    :
    Function<dim>(1 /*n_components*/, time)
  {}

  virtual ~AnalyticalSolutionPressure(){};

  virtual double value (const Point<dim>   &p,
                        const unsigned int component = 0) const;
};

template<int dim>
double AnalyticalSolutionPressure<dim>::value(const Point<dim>    &p,
                                              const unsigned int  /* component */) const
{
  double result = 0.0;

  // For this flow problem no analytical solution is available.
  return result;
}


/*
 *  Neumann boundary conditions for velocity
 *
 *  - Laplace formulation of viscous term
 *    -> prescribe velocity gradient (grad U)*n on Gamma_N
 *
 *  - Divergence formulation of viscous term
 *    -> prescribe (grad U + (grad U)^T)*n on Gamma_N
 */
template<int dim>
class NeumannBoundaryVelocity : public Function<dim>
{
public:
  NeumannBoundaryVelocity (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  virtual ~NeumannBoundaryVelocity(){};

  virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
};

template<int dim>
double NeumannBoundaryVelocity<dim>::value(const Point<dim> &p,const unsigned int component) const
{
  double result = 0.0;
  return result;
}

/*
 *  PressureBC_dudt:
 *
 *  This functions is only used when applying the high-order dual splitting scheme and
 *  is evaluated on Dirichlet boundaries (where the velocity is prescribed).
 *  Hence, this is the function that is set in the dirichlet_bc map of boundary_descriptor_pressure.
 *
 *  Note:
 *    When using a couples solution approach we do not have to evaluate something like
 *    pressure Neumann BC's on Dirichlet boundaries (we only have p⁺ = p⁻ on Dirichlet boundaries,
 *    i.e., no boundary data used). So it doesn't matter when writing this function into the
 *    dirichlet_bc map of boundary_descriptor_pressure because this function will never be evaluated
 *    in case of a coupled solution approach.
 *
 */

template<int dim>
class PressureBC_dudt : public Function<dim>
{
public:
  PressureBC_dudt (const double time = 0.)
    :
    Function<dim>(dim, time)
  {}

  virtual ~PressureBC_dudt(){};

  virtual double value (const Point<dim>    &p,
                        const unsigned int  component = 0) const;
};

template<int dim>
double PressureBC_dudt<dim>::value(const Point<dim>   &p,
                                   const unsigned int component) const
{
  double result = 0.0;

  return result;
}

/*
 *  Right-hand side function: Implements the body force vector occuring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations
 */

// TODO
template<int dim>
 class RightHandSide : public Function<dim>
 {
 public:
   RightHandSide (const double time = 0.)
     :
     Function<dim>(dim, time)
   {}

   virtual ~RightHandSide(){};

   virtual double value (const Point<dim>    &p,
                         const unsigned int  component = 0) const;
 };

 template<int dim>
 double RightHandSide<dim>::value(const Point<dim>   &p,
                                  const unsigned int component) const
 {
   double result = 0.0;

   //channel flow with periodic bc
   if(component==0)
     return 1.0;
   else
     return 0.0;

   return result;
 }


/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

 template <int dim>
Point<dim> grid_transform (const Point<dim> &in)
{
  Point<dim> out = in;

  out[0] = in(0)-numbers::PI;
  out[1] =  std::tanh(GRID_STRETCH_FAC*(2.*in(1)-1.))/std::tanh(GRID_STRETCH_FAC);
  if(dim==3)
    out[2] = in(2)-0.5*numbers::PI;
  return out;
}

template<int dim>
void create_grid_and_set_boundary_conditions(
    parallel::distributed::Triangulation<dim>                   &triangulation,
    unsigned int const                                          n_refine_space,
    std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity,
    std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure,
    std::vector<GridTools::PeriodicFacePair<typename
      Triangulation<dim>::cell_iterator> >                      &periodic_faces)
{
  /* --------------- Generate grid ------------------- */
   //turbulent channel flow
   Point<dim> coordinates;
   coordinates[0] = 2.0*numbers::PI;
   coordinates[1] = 1.0;
   if (dim == 3)
     coordinates[2] = numbers::PI;
   // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
   std::vector<unsigned int> refinements(dim, 1);
   GridGenerator::subdivided_hyper_rectangle (triangulation, refinements,Point<dim>(),coordinates);

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
   triangulation.refine_global(n_refine_space);

   GridTools::transform (&grid_transform<dim>, triangulation);

   // fill boundary descriptor velocity
   std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
   analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>(dim));
   // Dirichlet boundaries: ID = 0
   boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                     (0,analytical_solution_velocity));

   std_cxx11::shared_ptr<Function<dim> > neumann_bc_velocity;
   neumann_bc_velocity.reset(new NeumannBoundaryVelocity<dim>());
   // Neumann boundaris: ID = 1
   boundary_descriptor_velocity->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                   (1,neumann_bc_velocity));

   // fill boundary descriptor pressure
   std_cxx11::shared_ptr<Function<dim> > pressure_bc_dudt;
   pressure_bc_dudt.reset(new PressureBC_dudt<dim>());
   // Dirichlet boundaries: ID = 0
   boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                     (0,pressure_bc_dudt));

   std_cxx11::shared_ptr<Function<dim> > analytical_solution_pressure;
   analytical_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());
   // Neumann boundaries: ID = 1
   boundary_descriptor_pressure->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                   (1,analytical_solution_pressure));

}


template<int dim>
void set_field_functions(std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> > field_functions)
{
  // initialize functions (analytical solution, rhs, boundary conditions)
  std_cxx11::shared_ptr<Function<dim> > initial_solution_velocity;
  initial_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  std_cxx11::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new AnalyticalSolutionPressure<dim>());

  std_cxx11::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure = initial_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}

template<int dim>
void set_analytical_solution(std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution)
{
  analytical_solution->velocity.reset(new ZeroFunction<dim>(dim));
  analytical_solution->pressure.reset(new ZeroFunction<dim>(1));
}

// Postprocessor

#include "../../include/PostProcessor.h"
#include "../../include/statistics_manager.h"

template<int dim>
struct PostProcessorDataTurbulentChannel
{
  PostProcessorData<dim> pp_data;
  TurbulentChannelData turb_ch_data;
};

template<int dim, int fe_degree_u, int fe_degree_p>
class PostProcessorTurbulentChannel : public PostProcessor<dim, fe_degree_u, fe_degree_p>
{
public:
  PostProcessorTurbulentChannel(PostProcessorDataTurbulentChannel<dim> const & pp_data_turb_channel)
    :
    PostProcessor<dim,fe_degree_u,fe_degree_p>(pp_data_turb_channel.pp_data),
    write_final_output(true),
    turb_ch_data(pp_data_turb_channel.turb_ch_data)
  {}

  void setup(DoFHandler<dim> const                                        &dof_handler_velocity_in,
             DoFHandler<dim> const                                        &dof_handler_pressure_in,
             Mapping<dim> const                                           &mapping_in,
             MatrixFree<dim,double> const                                 &matrix_free_data_in,
             DofQuadIndexData const                                       &dof_quad_index_data_in,
             std_cxx11::shared_ptr<AnalyticalSolutionNavierStokes<dim> >  analytical_solution_in)
  {
    // call setup function of base class
    PostProcessor<dim,fe_degree_u,fe_degree_p>::setup(
        dof_handler_velocity_in,
        dof_handler_pressure_in,
        mapping_in,
        matrix_free_data_in,
        dof_quad_index_data_in,
        analytical_solution_in);

    // perform setup of turbulent channel related things
    statistics_turb_ch.reset(new StatisticsManager<dim>(dof_handler_velocity_in));
    statistics_turb_ch->setup(&grid_transform<dim>);
  }

  void do_postprocessing(parallel::distributed::Vector<double> const &velocity,
                         parallel::distributed::Vector<double> const &intermediate_velocity,
                         parallel::distributed::Vector<double> const &pressure,
                         parallel::distributed::Vector<double> const &vorticity,
                         parallel::distributed::Vector<double> const &divergence,
                         double const                                time,
                         int const                                   time_step_number = -1)
  {
    PostProcessor<dim,fe_degree_u,fe_degree_p>::do_postprocessing(
	      velocity,
        intermediate_velocity,
        pressure,
        vorticity,
        divergence,
        time,
        time_step_number);
   
    // EPSILON: small number which is much smaller than the time step size
    const double EPSILON = 1.0e-10;
    if((time > turb_ch_data.sample_start_time-EPSILON) &&
       (time < turb_ch_data.sample_end_time+EPSILON) && 
       (time_step_number % turb_ch_data.sample_every_timesteps == 0))
    {
      // evaluate statistics
      statistics_turb_ch->evaluate(velocity);
     
      // write intermediate output
      if(time_step_number % (turb_ch_data.sample_every_timesteps * 100) == 0)
      {
        statistics_turb_ch->write_output(turb_ch_data.filename_prefix,
                                         turb_ch_data.viscosity);
      }
    }
    // write final output
    if((time > turb_ch_data.sample_end_time-EPSILON) && write_final_output)
    {
      statistics_turb_ch->write_output(turb_ch_data.filename_prefix,
                                       turb_ch_data.viscosity);
      write_final_output = false;
    }
  }

  bool write_final_output;
  TurbulentChannelData turb_ch_data;
  std_cxx11::shared_ptr<StatisticsManager<dim> > statistics_turb_ch;
};

template<int dim>
std_cxx11::shared_ptr<PostProcessorBase<dim> >
construct_postprocessor(InputParametersNavierStokes<dim> const &param)
{
  PostProcessorData<dim> pp_data;
  pp_data.output_data = param.output_data;
  pp_data.error_data = param.error_data;
  pp_data.lift_and_drag_data = param.lift_and_drag_data;
  pp_data.pressure_difference_data = param.pressure_difference_data;
  pp_data.mass_data = param.mass_data;

  PostProcessorDataTurbulentChannel<dim> pp_data_turb_ch;
  pp_data_turb_ch.pp_data = pp_data;
  pp_data_turb_ch.turb_ch_data = param.turb_ch_data;

  std_cxx11::shared_ptr<PostProcessorBase<dim> > pp;
  pp.reset(new PostProcessorTurbulentChannel<dim,FE_DEGREE_VELOCITY,FE_DEGREE_PRESSURE>(pp_data_turb_ch));

  return pp;
}




#endif /* APPLICATIONS_NAVIERSTOKESTESTCASES_TURBULENTCHANNEL_H_ */
