/*
 * periodic_box.h
 *
 *  Created on: Aug 18, 2016
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_BOX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_BOX_H_

#include "../../grid_tools/periodic_box.h"

namespace IncNS
{
namespace PeriodicBox
{
enum class MeshType
{
  Cartesian,
  Curvilinear
};

void
string_to_enum(MeshType & enum_type, std::string const & string_type)
{
  // clang-format off
  if     (string_type == "Cartesian")   enum_type = MeshType::Cartesian;
  else if(string_type == "Curvilinear") enum_type = MeshType::Curvilinear;
  else AssertThrow(false, ExcMessage("Not implemented."));
  // clang-format on
}

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application() : ApplicationBase<dim, Number>("")
  {
  }

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    parse_input(input_file, prm, true, true);

    string_to_enum(mesh_type, mesh_type_string);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("MeshType",  mesh_type_string, "Type of mesh (Cartesian versus curvilinear).", Patterns::Selection("Cartesian|Curvilinear"));
    prm.leave_subsection();
    // clang-format on
  }

  std::string mesh_type_string = "Cartesian";
  MeshType    mesh_type        = MeshType::Cartesian;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::NavierStokes;
    param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    param.right_hand_side             = false;

    // PHYSICAL QUANTITIES
    param.start_time = 0.0;
    param.end_time   = 1.0;
    param.viscosity  = 1.0;

    // TEMPORAL DISCRETIZATION
    param.solver_type = SolverType::Unsteady;
    param.temporal_discretization =
      TemporalDiscretization::BDFDualSplittingScheme; // BDFPressureCorrection;
                                                      // //BDFCoupledSolution;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.cfl                           = 1.0;
    param.max_velocity                  = 1.0;

    // NUMERICAL PARAMETERS
    param.quad_rule_linearization = QuadratureRuleLinearization::Standard; // Overintegration32k;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p = DegreePressure::MixedOrder; // use EqualOrder so that we can also start with
                                                 // k=1 for the velocity!

    // mapping
    param.mapping = MappingType::Affine; // Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // special case: pure DBC's (only periodic BCs -> pure_dirichlet_bc = true)
    param.pure_dirichlet_bc = true;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.apply_penalty_terms_in_postprocessing_step = true;

    // TURBULENCE
    param.use_turbulence_model = false;
    param.turbulence_model     = TurbulenceEddyViscosityModel::Sigma;
    // Smagorinsky: 0.165
    // Vreman: 0.28
    // WALE: 0.50
    // Sigma: 1.35
    param.turbulence_model_constant = 1.35;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::None;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.preconditioner_projection = PreconditionerProjection::None;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.preconditioner_viscous = PreconditionerViscous::None;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // linear solver
    param.solver_momentum         = SolverMomentum::GMRES;
    param.preconditioner_momentum = MomentumPreconditioner::None;

    // COUPLED NAVIER-STOKES SOLVER

    // linear solver
    param.solver_coupled = SolverCoupled::GMRES;

    // preconditioning linear solver
    param.preconditioner_coupled = PreconditionerCoupled::None;

    param.preconditioner_velocity_block = MomentumPreconditioner::None;
    param.preconditioner_pressure_block = SchurComplementPreconditioner::None;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    double const left = -1.0, right = 1.0;
    double const deformation = 0.1;

    bool curvilinear_mesh = false;
    if(mesh_type == MeshType::Cartesian)
    {
      // do nothing
    }
    else if(mesh_type == MeshType::Curvilinear)
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
                        this->n_subdivisions_1d_hypercube,
                        left,
                        right,
                        curvilinear_mesh,
                        deformation);
  }

  void set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim>> /*boundary_descriptor_velocity*/,
    std::shared_ptr<BoundaryDescriptorP<dim>> /*boundary_descriptor_pressure*/)
  {
    // test case with pure periodic BC
    // boundary descriptors remain empty for velocity and pressure
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    (void)degree;

    PostProcessorData<dim> pp_data;

    // no postprocessing

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace PeriodicBox
} // namespace IncNS

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_BOX_H_ */
