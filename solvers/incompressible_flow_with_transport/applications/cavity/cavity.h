/*
 * cavity.h
 *
 *  Created on: Nov 26, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_

namespace ExaDG
{
namespace FTI
{
namespace Cavity
{
using namespace dealii;

template<int dim>
class DirichletBC : public Function<dim>
{
public:
  DirichletBC(double const L) : Function<dim>(dim, 0.0), L(L)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double result = 0.0;
    (void)p;

    if(component == 0)
    {
      // Variation of boundary condition: avoid velocity jumps in corners
      // (reduces Newton iterations considerably for convection-dominated problems).
      result = 4. / (L * L) * (-(p[0] - L / 2.0) * (p[0] - L / 2.0) + L * L / 4.0);

      // might not converge
      //      result = 1.0;
    }

    return result;
  }

private:
  double const L;
};

template<int dim, typename Number>
class Application : public FTI::ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file) : FTI::ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  // Length of cavity
  double const L = 1.0;

  // start and end time
  double const start_time = 0.0;
  double const end_time   = 10.0;

  // Explicit:    CFL_crit = 0.3-0.35 (DivergenceFormulation, upwind_factor=0.5!),
  //              CFL_crit = 0.5-0.6  (ConvectiveFormulation) for BDF2 with adaptive time stepping
  // ExplicitOIF: CFL_crit,oif = 3.0 (3.5 unstable) for ExplRK3Stage7Reg2
  double const CFL_OIF                = 0.3;
  double const CFL                    = CFL_OIF;
  double const max_velocity           = 1.0;
  bool const   adaptive_time_stepping = true;

  // vtu output
  double const output_interval_time = (end_time - start_time) / 100.0;

  // restart
  bool const   write_restart         = false;
  double const restart_interval_time = 10.0;

  void
  set_input_parameters(IncNS::InputParameters & param)
  {
    using namespace IncNS;

    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    param.right_hand_side             = false;


    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = 1.0e-5;


    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.time_integrator_oif             = TimeIntegratorOIF::ExplRK3Stage7Reg2;
    param.adaptive_time_stepping          = adaptive_time_stepping;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.max_velocity                    = max_velocity;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.cfl_oif                         = CFL_OIF;
    param.cfl                             = CFL;
    param.time_step_size                  = 1.0e-1;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // restart
    param.restart_data.write_restart = write_restart;
    param.restart_data.interval_time = restart_interval_time;
    param.restart_data.filename      = this->output_directory + this->output_name + "_fluid";

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = true;
    param.use_cell_based_face_loops                           = true;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_data_pressure_poisson    = SolverData(1000, 1.e-12, 1.e-6, 100);
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::Multigrid;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulation
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, 1.e-12, 1.e-6);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;


    // PRESSURE-CORRECTION SCHEME

    // formulation
    param.order_pressure_extrapolation = 1;
    param.rotational_formulation       = true;

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-20, 1.e-6);

    // linear solver
    // use FGMRES for matrix-free BlockJacobi or Multigrid with Krylov methods as smoother/coarse
    // grid solver
    param.solver_momentum      = SolverMomentum::FGMRES;
    param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.preconditioner_momentum =
      MomentumPreconditioner::InverseMassMatrix; // BlockJacobi; //Multigrid;
    param.multigrid_operator_type_momentum = MultigridOperatorType::ReactionConvectionDiffusion;
    param.update_preconditioner_momentum   = true;
    param.multigrid_data_momentum.smoother_data.smoother = MultigridSmoother::Jacobi;
    param.multigrid_data_momentum.smoother_data.preconditioner =
      PreconditionerSmoother::BlockJacobi;
    param.multigrid_data_momentum.smoother_data.iterations        = 5;
    param.multigrid_data_momentum.smoother_data.relaxation_factor = 0.7;
    param.multigrid_data_momentum.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;
    param.multigrid_data_momentum.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::None;


    // COUPLED NAVIER-STOKES SOLVER

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-12, 1.e-6);

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES; // FGMRES;
    param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

    // preconditioner linear solver
    param.preconditioner_coupled        = PreconditionerCoupled::BlockTriangular;
    param.update_preconditioner_coupled = false;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block          = MomentumPreconditioner::Multigrid;
    param.multigrid_operator_type_velocity_block = MultigridOperatorType::ReactionDiffusion;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::PressureConvectionDiffusion;
    param.discretization_of_laplacian = DiscretizationOfLaplacian::Classical;
  }

  void
  set_input_parameters_scalar(ConvDiff::InputParameters & param, unsigned int const scalar_index)
  {
    using namespace ConvDiff;

    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::ConvectionDiffusion;
    param.analytical_velocity_field   = false;
    param.right_hand_side             = false;
    param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;

    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    if(scalar_index == 0)
    {
      param.diffusivity = 1.e-5;
    }
    else
    {
      param.diffusivity = 1.e-3;
    }

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization       = TemporalDiscretization::BDF;
    param.time_integrator_rk            = TimeIntegratorRK::ExplRK3Stage7Reg2;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit;
    param.adaptive_time_stepping        = adaptive_time_stepping;
    param.order_time_integrator         = 2;
    param.time_integrator_oif           = TimeIntegratorRK::ExplRK3Stage7Reg2;
    param.start_with_low_order          = true;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.time_step_size                = 1.0e-2;
    param.cfl_oif                       = CFL_OIF;
    param.cfl                           = CFL;
    param.max_velocity                  = max_velocity;
    param.exponent_fe_degree_convection = 1.5;
    param.exponent_fe_degree_diffusion  = 3.0;
    param.diffusion_number              = 0.01;

    // restart
    param.restart_data.write_restart = write_restart;
    param.restart_data.interval_time = restart_interval_time;
    param.restart_data.filename =
      this->output_directory + this->output_name + "_scalar_" + std::to_string(scalar_index);

    // SPATIAL DISCRETIZATION

    // triangulation
    param.triangulation_type = TriangulationType::Distributed;

    // mapping
    param.mapping = MappingType::Affine;

    // convective term
    param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    param.IP_factor = 1.0;

    // SOLVER
    param.solver      = ConvDiff::Solver::GMRES;
    param.solver_data = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.preconditioner =
      Preconditioner::Multigrid; // InverseMassMatrix; //BlockJacobi; //Multigrid;
    param.implement_block_diagonal_preconditioner_matrix_free = true;
    param.solver_block_diagonal                               = Elementwise::Solver::GMRES;
    param.preconditioner_block_diagonal = Elementwise::Preconditioner::InverseMassMatrix;
    param.solver_data_block_diagonal    = SolverData(1000, 1.e-12, 1.e-2, 1000);
    param.use_cell_based_face_loops     = true;
    param.update_preconditioner         = true;

    param.multigrid_data.type = MultigridType::phMG;
    param.mg_operator_type    = MultigridOperatorType::ReactionConvectionDiffusion;
    // MG smoother
    param.multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi;
    // MG smoother data
    param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
    param.multigrid_data.smoother_data.iterations     = 5;

    // MG coarse grid solver
    param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 10.;

    // NUMERICAL PARAMETERS
    param.use_combined_operator = true;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    const double left = 0.0, right = L;
    GridGenerator::hyper_cube(*triangulation, left, right);
    triangulation->refine_global(n_refine_space);

    // set boundary IDs: 0 by default, set upper boundary to 1
    typename Triangulation<dim>::cell_iterator cell;
    for(cell = triangulation->begin(); cell != triangulation->end(); ++cell)
    {
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        if((std::fabs(cell->face(face_number)->center()(1) - L) < 1e-12))
        {
          cell->face(face_number)->set_boundary_id(1);
        }
      }
    }
  }

  void
  set_boundary_conditions(
    std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // fill boundary descriptor velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_velocity->dirichlet_bc.insert(pair(1, new DirichletBC<dim>(L)));

    // fill boundary descriptor pressure
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
  }

  void
  set_field_functions(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.output_folder        = this->output_directory + "vtu/";
    pp_data.output_data.output_name          = this->output_name + "_fluid";
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = output_interval_time;
    pp_data.output_data.write_processor_id   = true;
    pp_data.output_data.degree               = degree;

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new IncNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }

  void
  set_boundary_conditions_scalar(
    std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor,
    unsigned int                                       scalar_index = 0)
  {
    (void)scalar_index; // only one scalar quantity considered

    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor->dirichlet_bc.insert(pair(1, new Functions::ConstantFunction<dim>(1.0, 1)));
  }

  void
  set_field_functions_scalar(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions,
                             unsigned int                                   scalar_index = 0)
  {
    (void)scalar_index; // only one scalar quantity considered

    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor_scalar(unsigned int const degree,
                                 MPI_Comm const &   mpi_comm,
                                 unsigned int const scalar_index)
  {
    ConvDiff::PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output  = this->write_output;
    pp_data.output_data.output_folder = this->output_directory + "vtu/";
    pp_data.output_data.output_name = this->output_name + "_scalar_" + std::to_string(scalar_index);
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = output_interval_time;
    pp_data.output_data.degree               = degree;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(new ConvDiff::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};


} // namespace Cavity
} // namespace FTI
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_CAVITY_H_ */
