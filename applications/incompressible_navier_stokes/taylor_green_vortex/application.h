/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TAYLOR_GREEN_VORTEX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TAYLOR_GREEN_VORTEX_H_

#include <exadg/grid/mesh_movement_functions.h>
#include <exadg/grid/periodic_box.h>

namespace ExaDG
{
namespace IncNS
{
enum class MeshType
{
  Cartesian,
  Curvilinear
};

template<int dim>
class InitialSolutionVelocity : public dealii::Function<dim>
{
public:
  InitialSolutionVelocity(double const V_0, double const L)
    : dealii::Function<dim>(dim, 0.0), V_0(V_0), L(L)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double result = 0.0;

    if(component == 0)
      result = V_0 * std::sin(p[0] / L) * std::cos(p[1] / L) * std::cos(p[2] / L);
    else if(component == 1)
      result = -V_0 * std::cos(p[0] / L) * std::sin(p[1] / L) * std::cos(p[2] / L);
    else if(component == 2)
      result = 0.0;

    return result;
  }

private:
  double const V_0, L;
};

template<int dim>
class InitialSolutionPressure : public dealii::Function<dim>
{
public:
  InitialSolutionPressure(double const V_0, double const L, double const p_0)
    : dealii::Function<dim>(1 /*n_components*/, 0.0), V_0(V_0), L(L), p_0(p_0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const final
  {
    double const result = p_0 + V_0 * V_0 / 16.0 *
                                  (std::cos(2.0 * p[0] / L) + std::cos(2.0 * p[1] / L)) *
                                  (std::cos(2.0 * p[2] / L) + 2.0);

    return result;
  }

private:
  double const V_0, L, p_0;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    prm.enter_subsection("Application");
    {
      prm.add_parameter("MeshType", mesh_type, "Type of mesh (Cartesian versus Curvilinear).");
      prm.add_parameter("NCoarseCells1D",
                        n_subdivisions_1d_hypercube,
                        "Number of cells per direction on coarse grid.",
                        dealii::Patterns::Integer(1, 5));
      prm.add_parameter("ExploitSymmetry",
                        exploit_symmetry,
                        "Exploit symmetry and reduce DoFs by a factor of 8?");
      prm.add_parameter("MovingMesh", ALE, "Moving mesh?");
      prm.add_parameter("Inviscid", inviscid, "Is this an inviscid simulation?");
      prm.add_parameter("ReynoldsNumber", Re, "Reynolds number (ignored if Inviscid = true)");
      prm.add_parameter("WriteRestart", write_restart, "Should restart files be written?");
      prm.add_parameter("ReadRestart", read_restart, "Is this a restarted simulation?");
    }
    prm.leave_subsection();
  }

private:
  void
  parse_parameters() final
  {
    ApplicationBase<dim, Number>::parse_parameters();

    // viscosity needs to be recomputed since the parameters inviscid, Re are
    // read from the input file
    viscosity = inviscid ? 0.0 : V_0 * L / Re;
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type = ProblemType::Unsteady;
    if(inviscid)
      this->param.equation_type = EquationType::Euler;
    else
      this->param.equation_type = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    if(ALE)
      this->param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    this->param.right_hand_side = false;

    // ALE
    this->param.ale_formulation                     = ALE;
    this->param.mesh_movement_type                  = MeshMovementType::Function;
    this->param.neumann_with_variable_normal_vector = false;

    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = end_time;
    this->param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    this->param.solver_type                   = SolverType::Unsteady;
    this->param.temporal_discretization       = TemporalDiscretization::BDFDualSplittingScheme;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    this->param.order_time_integrator         = 2;
    this->param.start_with_low_order          = not read_restart;
    this->param.adaptive_time_stepping        = true;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping_limiting_factor = 3.0;
    this->param.max_velocity                           = max_velocity;
    this->param.cfl                                    = 0.4;
    this->param.cfl_exponent_fe_degree_velocity        = 1.5;
    this->param.time_step_size                         = 1.0e-3;

    // restart
    this->param.restarted_simulation             = read_restart;
    this->param.restart_data.write_restart       = write_restart;
    this->param.restart_data.interval_time       = 1.0;
    this->param.restart_data.interval_wall_time  = 1.e6;
    this->param.restart_data.interval_time_steps = 1e8;
    this->param.restart_data.filename =
      this->output_parameters.directory + this->output_parameters.filename + "restart";

    // output of solver information
    this->param.solver_info_data.interval_time = characteristic_time;

    // NUMERICAL PARAMETERS
    this->param.implement_block_diagonal_preconditioner_matrix_free = false;
    this->param.use_cell_based_face_loops                           = false;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.spatial_discretization  = SpatialDiscretization::L2;
    this->param.mapping_degree          = this->param.degree_u;
    this->param.degree_p                = DegreePressure::MixedOrder;

    // mapping
    if(mesh_type == MeshType::Cartesian)
      this->param.mapping_degree = 1;
    else
      this->param.mapping_degree = this->param.degree_u;

    if(this->param.ale_formulation)
      this->param.mapping_degree = this->param.degree_u;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity pressure coupling terms
    this->param.gradp_formulation = FormulationPressureGradientTerm::Weak;
    this->param.divu_formulation  = FormulationVelocityDivergenceTerm::Weak;

    // div-div and continuity penalty
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    this->param.apply_penalty_terms_in_postprocessing_step = true;
    this->param.continuity_penalty_use_boundary_data       = true;

    // TURBULENCE
    this->param.turbulence_model_data.is_active        = false;
    this->param.turbulence_model_data.turbulence_model = TurbulenceEddyViscosityModel::Sigma;
    // Smagorinsky: 0.165
    // Vreman: 0.28
    // WALE: 0.50
    // Sigma: 1.35
    this->param.turbulence_model_data.constant = 1.35;

    // PROJECTION METHODS

    // pressure Poisson equation
    this->param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 100);
    this->param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    this->param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.solver_data_projection    = SolverData(1000, ABS_TOL, REL_TOL);
    this->param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    this->param.order_extrapolation_pressure_nbc =
      this->param.order_time_integrator <= 2 ? this->param.order_time_integrator : 2;

    // viscous step
    this->param.solver_viscous         = SolverViscous::CG;
    this->param.solver_data_viscous    = SolverData(1000, ABS_TOL, REL_TOL);
    this->param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    this->param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    this->param.solver_momentum = SolverMomentum::GMRES;
    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      this->param.solver_data_momentum = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      this->param.solver_data_momentum = SolverData(1e4, ABS_TOL, REL_TOL, 100);

    this->param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    this->param.update_preconditioner_momentum = true;

    // formulation
    this->param.order_pressure_extrapolation = this->param.order_time_integrator - 1;
    this->param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER
    this->param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    this->param.newton_solver_data_coupled = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    this->param.solver_coupled = SolverCoupled::GMRES;
    if(this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      this->param.solver_data_coupled = SolverData(1e3, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      this->param.solver_data_coupled = SolverData(1e3, ABS_TOL, REL_TOL, 100);

    // preconditioning linear solver
    this->param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    this->param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    this->param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    (void)mapping;
    (void)multigrid_mappings;

    auto const lambda_create_triangulation = [&](dealii::Triangulation<dim, dim> & tria,
                                                 std::vector<dealii::GridTools::PeriodicFacePair<
                                                   typename dealii::Triangulation<
                                                     dim>::cell_iterator>> & periodic_face_pairs,
                                                 unsigned int const          global_refinements,
                                                 std::vector<unsigned int> const &
                                                   vector_local_refinements) {
      (void)vector_local_refinements;

      double const deformation = 0.5;

      if(ALE)
      {
        AssertThrow(mesh_type == MeshType::Cartesian,
                    dealii::ExcMessage(
                      "Taylor-Green vortex: Parameter mesh_type is invalid for ALE."));
      }

      if(mesh_type == MeshType::Curvilinear)
      {
        AssertThrow(
          this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
          dealii::ExcMessage(
            "Manifolds might not be applied correctly for TriangulationType::FullyDistributed. "
            "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));
      }


      if(exploit_symmetry == false) // periodic box
      {
        AssertThrow(
          this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
          dealii::ExcMessage(
            "Periodic faces might not be applied correctly for TriangulationType::FullyDistributed. "
            "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));

        create_periodic_box(tria,
                            global_refinements,
                            periodic_face_pairs,
                            n_subdivisions_1d_hypercube,
                            left,
                            right,
                            mesh_type == MeshType::Curvilinear,
                            deformation);
      }
      else // symmetric box
      {
        dealii::GridGenerator::subdivided_hyper_cube(tria, n_subdivisions_1d_hypercube, 0.0, right);

        if(mesh_type == MeshType::Curvilinear)
        {
          unsigned int const frequency = 2;

          apply_deformed_cube_manifold(tria, 0.0, right, deformation, frequency);
        }

        tria.refine_global(global_refinements);
      }
    };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);
  }

  std::shared_ptr<dealii::Function<dim>>
  create_mesh_movement_function() final
  {
    std::shared_ptr<dealii::Function<dim>> mesh_motion;

    MeshMovementData<dim> data;
    data.temporal                       = MeshMovementAdvanceInTime::Sin;
    data.shape                          = MeshMovementShape::Sin;
    data.dimensions[0]                  = std::abs(right - left);
    data.dimensions[1]                  = std::abs(right - left);
    data.dimensions[2]                  = std::abs(right - left);
    data.amplitude                      = right / 6.0; // use a value <= right/4.0
    data.period                         = 20.0 * characteristic_time;
    data.t_start                        = 0.0;
    data.t_end                          = end_time;
    data.spatial_number_of_oscillations = 1.0;
    mesh_motion.reset(new CubeMeshMovementFunctions<dim>(data));

    return mesh_motion;
  }

  void
  set_boundary_descriptor() final
  {
    if(exploit_symmetry)
    {
      typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
        pair;

      this->boundary_descriptor->velocity->symmetry_bc.insert(
        pair(0, new dealii::Functions::ZeroFunction<dim>(dim))); // function will not be used
      this->boundary_descriptor->pressure->neumann_bc.insert(0);
    }
    else
    {
      // test case with pure periodic BC
      // boundary descriptors remain empty for velocity and pressure
    }
  }


  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new InitialSolutionVelocity<dim>(V_0, L));
    this->field_functions->initial_solution_pressure.reset(
      new InitialSolutionPressure<dim>(V_0, L, p_0));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    std::string name = this->output_parameters.filename + "_l" +
                       std::to_string(this->param.grid.n_refine_global) + "_k" +
                       std::to_string(this->param.degree_u);

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 20.0;
    pp_data.output_data.directory                 = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                  = name;
    pp_data.output_data.write_vorticity           = true;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_velocity_magnitude  = true;
    pp_data.output_data.write_vorticity_magnitude = true;
    pp_data.output_data.write_q_criterion         = true;
    pp_data.output_data.write_processor_id        = true;
    pp_data.output_data.write_higher_order        = false;
    pp_data.output_data.degree                    = this->param.degree_u;

    // calculate div and mass error
    pp_data.mass_data.time_control_data.is_active                = false;
    pp_data.mass_data.time_control_data.start_time               = 0.0;
    pp_data.mass_data.time_control_data.trigger_every_time_steps = 1e2;
    pp_data.mass_data.directory              = this->output_parameters.directory;
    pp_data.mass_data.filename               = name;
    pp_data.mass_data.reference_length_scale = 1.0;

    // kinetic energy
    pp_data.kinetic_energy_data.time_control_data.is_active                = true;
    pp_data.kinetic_energy_data.time_control_data.trigger_every_time_steps = 1;
    pp_data.kinetic_energy_data.time_control_data.start_time               = start_time;

    pp_data.kinetic_energy_data.evaluate_individual_terms = false;
    pp_data.kinetic_energy_data.viscosity                 = viscosity;
    pp_data.kinetic_energy_data.directory                 = this->output_parameters.directory;
    pp_data.kinetic_energy_data.filename                  = name;
    pp_data.kinetic_energy_data.clear_file                = not read_restart;

    // kinetic energy spectrum
    bool const do_fftw_during_simulation                                    = true;
    pp_data.kinetic_energy_spectrum_data.time_control_data.is_active        = true;
    pp_data.kinetic_energy_spectrum_data.time_control_data.trigger_interval = 0.5;
    pp_data.kinetic_energy_spectrum_data.time_control_data.start_time       = start_time;
    pp_data.kinetic_energy_spectrum_data.do_fftw                 = do_fftw_during_simulation;
    pp_data.kinetic_energy_spectrum_data.write_raw_data_to_files = not do_fftw_during_simulation;
    pp_data.kinetic_energy_spectrum_data.directory = this->output_parameters.directory;
    pp_data.kinetic_energy_spectrum_data.filename  = name + "_energy_spectrum";
    pp_data.kinetic_energy_spectrum_data.degree    = this->param.degree_u;
    pp_data.kinetic_energy_spectrum_data.evaluation_points_per_cell = (this->param.degree_u + 1);
    pp_data.kinetic_energy_spectrum_data.exploit_symmetry           = exploit_symmetry;
    pp_data.kinetic_energy_spectrum_data.n_cells_1d_coarse_grid     = n_subdivisions_1d_hypercube;
    pp_data.kinetic_energy_spectrum_data.refine_level            = this->param.grid.n_refine_global;
    pp_data.kinetic_energy_spectrum_data.length_symmetric_domain = right;
    pp_data.kinetic_energy_spectrum_data.clear_file              = not read_restart;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  // mesh type
  MeshType mesh_type = MeshType::Cartesian;

  unsigned int n_subdivisions_1d_hypercube = 1;

  // inviscid limit
  bool inviscid = false;

  // Reynolds number
  double Re = 1600.0;

  // reduce dofs by exploiting symmetry
  bool exploit_symmetry = false;

  // moving mesh
  bool ALE = false;

  // restart
  bool write_restart = false;
  bool read_restart  = false;

  double const V_0                 = 1.0;
  double const L                   = 1.0;
  double const p_0                 = 0.0;
  double const max_velocity        = V_0;
  double const characteristic_time = L / V_0;
  double const start_time          = 0.0;
  double const end_time            = 20.0 * characteristic_time;
  double const left = -dealii::numbers::PI * L, right = dealii::numbers::PI * L;

  // viscosity
  double viscosity = inviscid ? 0.0 : V_0 * L / Re;

  // solver tolerances
  double const ABS_TOL = 1.e-12;
  double const REL_TOL = 1.e-6;

  double const ABS_TOL_LINEAR = 1.e-12;
  double const REL_TOL_LINEAR = 1.e-2;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TAYLOR_GREEN_VORTEX_H_ */
