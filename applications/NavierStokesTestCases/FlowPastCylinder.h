/*
 * FlowPastCylinder.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_NAVIERSTOKESTESTCASES_FLOWPASTCYLINDER_H_
#define APPLICATIONS_NAVIERSTOKESTESTCASES_FLOWPASTCYLINDER_H_


/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/


// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 2;

// set the polynomial degree of the shape functions for velocity and pressure
unsigned int const FE_DEGREE_VELOCITY = 2;
unsigned int const FE_DEGREE_PRESSURE = FE_DEGREE_VELOCITY-1; // FE_DEGREE_VELOCITY; // FE_DEGREE_VELOCITY - 1;

// set xwall specific parameters
unsigned int const FE_DEGREE_XWALL = 1;
unsigned int const N_Q_POINTS_1D_XWALL = 1;

// set the number of refine levels for spatial convergence tests
unsigned int const REFINE_STEPS_SPACE_MIN = 1;
unsigned int const REFINE_STEPS_SPACE_MAX = REFINE_STEPS_SPACE_MIN;

// set the number of refine levels for temporal convergence tests
unsigned int const REFINE_STEPS_TIME_MIN = 0;
unsigned int const REFINE_STEPS_TIME_MAX = REFINE_STEPS_TIME_MIN;

// set problem specific parameters like physical dimensions, etc.
const unsigned int TEST_CASE = 1; // 1, 2 or 3
const double Um = (DIMENSION == 2 ? (TEST_CASE==1 ? 0.3 : 1.5) : (TEST_CASE==1 ? 0.45 : 2.25));
const double D = 0.1;
const double H = 0.41;
const double L1 = 0.3;
const double L2 = 2.5;
const double X_C = 0.5;
const double Y_C = 0.2;
const double END_TIME = 8.0;

void InputParametersNavierStokes::set_input_parameters()
{
  // MATHEMATICAL MODEL
  problem_type = ProblemType::Unsteady;
  equation_type = EquationType::NavierStokes;
  formulation_viscous_term = FormulationViscousTerm::DivergenceFormulation;
  right_hand_side = false;


  // PHYSICAL QUANTITIES
  start_time = 0.0;
  end_time = END_TIME; //END_TIME is also needed somewhere else
  viscosity = 0.001;


  // TEMPORAL DISCRETIZATION
  temporal_discretization = TemporalDiscretization::BDFCoupledSolution;
  treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
  calculation_of_time_step_size = TimeStepCalculation::ConstTimeStepCFL;
  max_velocity = Um;
  cfl = 3.0e-1;
  time_step_size = 1.0e-1;
  max_number_of_time_steps = 1e8;
  order_time_integrator = 2; // 1; // 2; // 3;
  start_with_low_order = true; // true; // false;


  // SPATIAL DISCRETIZATION

  // spatial discretization method
  spatial_discretization = SpatialDiscretization::DG;

  // convective term - currently no parameters

  // viscous term
  IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;
  IP_factor_viscous = 1.0;

  // gradient term
  gradp_integrated_by_parts = true;
  gradp_use_boundary_data = true;

  // divergence term
  divu_integrated_by_parts = true;
  divu_use_boundary_data = true;

  // special case: pure DBC's
  pure_dirichlet_bc = false;


  // HIGH-ORDER DUAL SPLITTING SCHEME

  // pressure Poisson equation
  IP_factor_pressure = 1.0;
  preconditioner_pressure_poisson = PreconditionerPressurePoisson::GeometricMultigrid;
  multigrid_data_pressure_poisson.coarse_solver = MultigridCoarseGridSolver::coarse_chebyshev_smoother;
  abs_tol_pressure = 1.e-12;
  rel_tol_pressure = 1.e-6;

  // stability in the limit of small time steps and projection step
  small_time_steps_stability = false;
  use_approach_of_ferrer = false;
  deltat_ref = 1.e0;

  // projection step
  projection_type = ProjectionType::DivergencePenalty;
  penalty_factor_divergence = 1.0e1;
  penalty_factor_continuity = 1.0e1;
  solver_projection = SolverProjection::PCG;
  preconditioner_projection = PreconditionerProjection::InverseMassMatrix;
  abs_tol_projection = 1.e-20;
  rel_tol_projection = 1.e-12;

  // viscous step
  solver_viscous = SolverViscous::PCG;
  preconditioner_viscous = PreconditionerViscous::GeometricMultigrid;
  multigrid_data_viscous.coarse_solver = MultigridCoarseGridSolver::coarse_chebyshev_smoother;
  abs_tol_viscous = 1.e-12;
  rel_tol_viscous = 1.e-6;


  // COUPLED NAVIER-STOKES SOLVER

  // nonlinear solver (Newton solver)
  abs_tol_newton = 1.e-12;
  rel_tol_newton = 1.e-6;
  max_iter_newton = 1e2;

  // linear solver
  solver_linearized_navier_stokes = SolverLinearizedNavierStokes::GMRES;
  abs_tol_linear = 1.e-12;
  rel_tol_linear = 1.e-6;
  max_iter_linear = 1e4;

  // preconditioning linear solver
  preconditioner_linearized_navier_stokes = PreconditionerLinearizedNavierStokes::BlockTriangular;

  // preconditioner velocity/momentum block
  momentum_preconditioner = MomentumPreconditioner::VelocityConvectionDiffusion;
  solver_momentum_preconditioner = SolverMomentumPreconditioner::GeometricMultigridVCycle;
  rel_tol_solver_momentum_preconditioner = 1.e-6;

  // preconditioner Schur-complement block
  schur_complement_preconditioner = SchurComplementPreconditioner::CahouetChabard;
  discretization_of_laplacian =  DiscretizationOfLaplacian::Classical;
  solver_schur_complement_preconditioner = SolverSchurComplementPreconditioner::GeometricMultigridVCycle;
  rel_tol_solver_schur_complement_preconditioner = 1.e-6;


  // OUTPUT AND POSTPROCESSING

  // write output for visualization of results
  output_prefix = "flow_past_cylinder";
  output_start_time = start_time;
  output_interval_time = (end_time-start_time)/20;
  compute_divergence = true;

  // calculation of error
  analytical_solution_available = false;
  error_calc_start_time = start_time;
  error_calc_interval_time = output_interval_time;

  // output of solver information
  output_solver_info_every_timesteps = 1e5;

  // restart
  write_restart = false;
  restart_interval_time = 1.e2;
  restart_interval_wall_time = 1.e6;
  restart_every_timesteps = 1e8;
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
  double t = this->get_time();
  double result = 0.0;

  if(component == 0 && std::abs(p[0]-(dim==2 ? L1: 0.0))<1.e-12)
  {
    const double pi = numbers::PI;
    const double T = 1.0;
    double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
    if(TEST_CASE < 3)
      result = coefficient * p[1] * (H-p[1]) * ( (t/T)<1.0 ? std::sin(pi/2.*t/T) : 1.0);
    if(TEST_CASE == 3)
      result = coefficient * p[1] * (H-p[1]) * std::sin(pi*t/END_TIME);
    if (dim == 3)
      result *= p[2] * (H-p[2]);
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
  // Set the pressure to zero at the outflow boundary. This is
  // already done since result is initialized with a value of 0.0.
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
  double t = this->get_time();
  double result = 0.0;

  if(component == 0 && std::abs(p[0]-(dim==2 ? L1 : 0.0))<1.e-12)
  {
    const double pi = numbers::PI;
    const double T = 1.0;
    double coefficient = Utilities::fixed_power<dim-1>(4.) * Um / Utilities::fixed_power<2*dim-2>(H);
    if(TEST_CASE < 3)
      result = coefficient * p[1] * (H-p[1]) * ( (t/T)<1.0 ? (pi/2./T)*std::cos(pi/2.*t/T) : 0.0);
    if(TEST_CASE == 3)
      result = coefficient * p[1] * (H-p[1]) * std::cos(pi*t/END_TIME)*pi/END_TIME;
    if (dim == 3)
      result *= p[2] * (H-p[2]);
  }

  return result;
}

/*
 *  Right-hand side function: Implements the body force vector occuring on the
 *  right-hand side of the momentum equation of the Navier-Stokes equations
 */
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
   return result;
 }


/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

void create_triangulation(Triangulation<2> &tria, const bool compute_in_2d = true)
{
   HyperBallBoundary<2> boundary(Point<2>(0.5,0.2), 0.05);
   Triangulation<2> left, middle, right, tmp, tmp2;
   std::vector<unsigned int> ref_1(2, 2);
   ref_1[1] = 2;

   GridGenerator::subdivided_hyper_rectangle(left, ref_1 ,Point<2>(), Point<2>(0.3, 0.41), false);
   std::vector<unsigned int> ref_2(2, 9);
   ref_2[1] = 2;

   GridGenerator::subdivided_hyper_rectangle(right, ref_2,Point<2>(0.7, 0), Point<2>(2.5, 0.41), false);

   // create middle part first as a hyper shell
   GridGenerator::hyper_shell(middle, Point<2>(0.5, 0.2), 0.05, 0.2, 4, true);
   middle.set_manifold(0, boundary);
   middle.refine_global(1);

   //for (unsigned int v=0; v<middle.get_vertices().size(); ++v)
   //  const_cast<Point<dim> &>(middle.get_vertices()[v]) = 0.4 / 3. * middle.get_vertices()[v];

   // then move the vertices to the points where we want them to be to create a
   // slightly asymmetric cube with a hole
   for (Triangulation<2>::cell_iterator cell = middle.begin();
      cell != middle.end(); ++cell)
   {
     for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
     {
       Point<2> &vertex = cell->vertex(v);
       if (std::abs(vertex[0] - 0.7) < 1e-10 &&
         std::abs(vertex[1] - 0.2) < 1e-10)
       vertex = Point<2>(0.7, 0.205);
       else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
            std::abs(vertex[1] - 0.3) < 1e-10)
       vertex = Point<2>(0.7, 0.41);
       else if (std::abs(vertex[0] - 0.6) < 1e-10 &&
            std::abs(vertex[1] - 0.1) < 1e-10)
       vertex = Point<2>(0.7, 0);
       else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
            std::abs(vertex[1] - 0.4) < 1e-10)
       vertex = Point<2>(0.5, 0.41);
       else if (std::abs(vertex[0] - 0.5) < 1e-10 &&
            std::abs(vertex[1] - 0.0) < 1e-10)
       vertex = Point<2>(0.5, 0.0);
       else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
            std::abs(vertex[1] - 0.3) < 1e-10)
       vertex = Point<2>(0.3, 0.41);
       else if (std::abs(vertex[0] - 0.4) < 1e-10 &&
            std::abs(vertex[1] - 0.1) < 1e-10)
       vertex = Point<2>(0.3, 0);
       else if (std::abs(vertex[0] - 0.3) < 1e-10 &&
            std::abs(vertex[1] - 0.2) < 1e-10)
       vertex = Point<2>(0.3, 0.205);
       else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
            std::abs(vertex[1] - 0.13621) < 1e-4)
       vertex = Point<2>(0.59, 0.11);
       else if (std::abs(vertex[0] - 0.56379) < 1e-4 &&
            std::abs(vertex[1] - 0.26379) < 1e-4)
       vertex = Point<2>(0.59, 0.29);
       else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
            std::abs(vertex[1] - 0.13621) < 1e-4)
       vertex = Point<2>(0.41, 0.11);
       else if (std::abs(vertex[0] - 0.43621) < 1e-4 &&
            std::abs(vertex[1] - 0.26379) < 1e-4)
       vertex = Point<2>(0.41, 0.29);
     }
   }

   // must copy the triangulation because we cannot merge triangulations with
   // refinement...
   GridGenerator::flatten_triangulation(middle, tmp2);

   if (compute_in_2d)
   {
     GridGenerator::merge_triangulations (tmp2, right, tria);
   }
   else
   {
   GridGenerator::merge_triangulations (left, tmp2, tmp);
   GridGenerator::merge_triangulations (tmp, right, tria);
   }

   // Set the cylinder boundary  to 2, outflow to 1, the rest to 0.
   for (Triangulation<2>::active_cell_iterator cell=tria.begin() ;
      cell != tria.end(); ++cell)
   {
     for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
     {
       if (cell->face(f)->at_boundary())
       {
         if (std::abs(cell->face(f)->center()[0] - (compute_in_2d ? 0.3 : 0)) < 1e-12)
           cell->face(f)->set_all_boundary_ids(0);
         else if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
           cell->face(f)->set_all_boundary_ids(1);
         else if (Point<2>(0.5,0.2).distance(cell->face(f)->center())<=0.05)
         {
           cell->face(f)->set_all_manifold_ids(10);
           cell->face(f)->set_all_boundary_ids(2);
         }
         else
           cell->face(f)->set_all_boundary_ids(0);
       }
     }
   }
}

void create_triangulation(Triangulation<3> &tria)
{
  Triangulation<2> tria_2d;
  create_triangulation(tria_2d, false);
  GridGenerator::extrude_triangulation(tria_2d, 3, 0.41, tria);

  // Set the cylinder boundary  to 2, outflow to 1, the rest to 0.
  for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
  {
    for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
    {
      if (cell->face(f)->at_boundary())
      {
        if (std::abs(cell->face(f)->center()[0]) < 1e-12)
          cell->face(f)->set_all_boundary_ids(0);
        else if (std::abs(cell->face(f)->center()[0]-2.5) < 1e-12)
          cell->face(f)->set_all_boundary_ids(1);
        else if (Point<3>(0.5,0.2,cell->face(f)->center()[2]).distance(cell->face(f)->center())<=0.05)
        {
          cell->face(f)->set_all_manifold_ids(10);
          cell->face(f)->set_all_boundary_ids(2);
        }
        else
          cell->face(f)->set_all_boundary_ids(0);
      }
    }
  }
}
template<int dim>
void create_grid_and_set_boundary_conditions(
    parallel::distributed::Triangulation<dim>                   &triangulation,
    unsigned int const                                          n_refine_space,
    std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity,
    std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure)
{

  Point<dim> direction;
  direction[dim-1] = 1.;

  Point<dim> center;
  center[0] = 0.5;
  center[1] = 0.2;

  static std_cxx11::shared_ptr<Manifold<dim> > cylinder_manifold =
    std_cxx11::shared_ptr<Manifold<dim> >(dim == 2 ? static_cast<Manifold<dim>*>(new HyperBallBoundary<dim>(center, 0.05)) :
                                          static_cast<Manifold<dim>*>(new CylindricalManifold<dim>(direction, center)));
  create_triangulation(triangulation);
  triangulation.set_manifold(10, *cylinder_manifold);

  triangulation.refine_global(n_refine_space);

  // fill boundary descriptor velocity
  std_cxx11::shared_ptr<Function<dim> > analytical_solution_velocity;
  analytical_solution_velocity.reset(new AnalyticalSolutionVelocity<dim>());
  // Dirichlet boundaries: ID = 0, 2
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                     (0,analytical_solution_velocity));
  boundary_descriptor_velocity->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                     (2,analytical_solution_velocity));

  std_cxx11::shared_ptr<Function<dim> > neumann_bc_velocity;
  neumann_bc_velocity.reset(new NeumannBoundaryVelocity<dim>());
  // Neumann boundaris: ID = 1
  boundary_descriptor_velocity->neumann_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                    (1,neumann_bc_velocity));

  // fill boundary descriptor pressure
  std_cxx11::shared_ptr<Function<dim> > pressure_bc_dudt;
  pressure_bc_dudt.reset(new PressureBC_dudt<dim>());
  // Dirichlet boundaries: ID = 0, 2
  boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                     (0,pressure_bc_dudt));
  boundary_descriptor_pressure->dirichlet_bc.insert(std::pair<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >
                                                     (2,pressure_bc_dudt));

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
  initial_solution_velocity.reset(new ZeroFunction<dim>(dim));
  std_cxx11::shared_ptr<Function<dim> > initial_solution_pressure;
  initial_solution_pressure.reset(new ZeroFunction<dim>(1));

  std_cxx11::shared_ptr<Function<dim> > right_hand_side;
  right_hand_side.reset(new RightHandSide<dim>());

  field_functions->initial_solution_velocity = initial_solution_velocity;
  field_functions->initial_solution_pressure = initial_solution_pressure;
  // This function will not be used since no analytical solution is available for this flow problem
  field_functions->analytical_solution_pressure = initial_solution_pressure;
  field_functions->right_hand_side = right_hand_side;
}


#endif /* APPLICATIONS_NAVIERSTOKESTESTCASES_FLOWPASTCYLINDER_H_ */
