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
using namespace dealii;

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

template<int dim>
class InitialSolutionVelocity : public Function<dim>
{
public:
  InitialSolutionVelocity(double const V_0, double const L)
    : Function<dim>(dim, 0.0), V_0(V_0), L(L)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
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
class InitialSolutionPressure : public Function<dim>
{
public:
  InitialSolutionPressure(double const V_0, double const L, double const p_0)
    : Function<dim>(1 /*n_components*/, 0.0), V_0(V_0), L(L), p_0(p_0)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const /*component*/) const
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
  typedef typename ApplicationBase<dim, Number>::PeriodicFaces PeriodicFaces;

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);

    string_to_enum(mesh_type, mesh_type_string);

    // viscosity needs to be recomputed since the parameters inviscid, Re are
    // read from the input file
    viscosity = inviscid ? 0.0 : V_0 * L / Re;
  }

  void
  add_parameters(ParameterHandler & prm) final
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("MeshType",        mesh_type_string,                  "Type of mesh (Cartesian versus curvilinear).", Patterns::Selection("Cartesian|Curvilinear"));
      prm.add_parameter("NCoarseCells1D",  this->n_subdivisions_1d_hypercube, "Number of cells per direction on coarse grid.", Patterns::Integer(1,5));
      prm.add_parameter("ExploitSymmetry", exploit_symmetry,                  "Exploit symmetry and reduce DoFs by a factor of 8?");
      prm.add_parameter("MovingMesh",      ALE,                               "Moving mesh?");
      prm.add_parameter("Inviscid",        inviscid,                          "Is this an inviscid simulation?");
      prm.add_parameter("ReynoldsNumber",  Re,                                "Reynolds number (ignored if Inviscid = true)");
      prm.add_parameter("WriteRestart",    write_restart,                     "Should restart files be written?");
      prm.add_parameter("ReadRestart",     read_restart,                      "Is this a restarted simulation?");
    prm.leave_subsection();
    // clang-format on
  }

  // mesh type
  std::string mesh_type_string = "Cartesian";
  MeshType    mesh_type        = MeshType::Cartesian;

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
  double const left = -numbers::PI * L, right = numbers::PI * L;

  // viscosity
  double viscosity = inviscid ? 0.0 : V_0 * L / Re;

  unsigned int refine_level = 0;

  // solver tolerances
  double const ABS_TOL = 1.e-12;
  double const REL_TOL = 1.e-6;

  double const ABS_TOL_LINEAR = 1.e-12;
  double const REL_TOL_LINEAR = 1.e-2;

  void
  set_input_parameters(InputParameters & param) final
  {
    // MATHEMATICAL MODEL
    param.problem_type = ProblemType::Unsteady;
    if(inviscid)
      param.equation_type = EquationType::Euler;
    else
      param.equation_type = EquationType::NavierStokes;
    param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    if(ALE)
      param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;
    param.right_hand_side = false;

    // ALE
    param.ale_formulation                     = ALE;
    param.neumann_with_variable_normal_vector = false;

    // PHYSICAL QUANTITIES
    param.start_time = start_time;
    param.end_time   = end_time;
    param.viscosity  = viscosity;


    // TEMPORAL DISCRETIZATION
    param.solver_type                            = SolverType::Unsteady;
    param.temporal_discretization                = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term           = TreatmentOfConvectiveTerm::Explicit;
    param.order_time_integrator                  = 2;
    param.start_with_low_order                   = !read_restart;
    param.adaptive_time_stepping                 = true;
    param.calculation_of_time_step_size          = TimeStepCalculation::CFL;
    param.adaptive_time_stepping_limiting_factor = 3.0;
    param.max_velocity                           = max_velocity;
    param.cfl                                    = 0.4;
    param.cfl_exponent_fe_degree_velocity        = 1.5;
    param.time_step_size                         = 1.0e-3;

    // restart
    param.restarted_simulation             = read_restart;
    param.restart_data.write_restart       = write_restart;
    param.restart_data.interval_time       = 1.0;
    param.restart_data.interval_wall_time  = 1.e6;
    param.restart_data.interval_time_steps = 1e8;
    param.restart_data.filename            = this->output_directory + this->output_name + "restart";

    // output of solver information
    param.solver_info_data.interval_time = characteristic_time;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TriangulationType::Distributed;
    param.degree_p           = DegreePressure::MixedOrder;

    // mapping
    if(mesh_type == MeshType::Cartesian)
      param.mapping = MappingType::Affine;
    else
      param.mapping = MappingType::Isoparametric;

    if(param.ale_formulation)
      param.mapping = MappingType::Isoparametric;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // velocity pressure coupling terms
    param.gradp_formulation = FormulationPressureGradientTerm::Weak;
    param.divu_formulation  = FormulationVelocityDivergenceTerm::Weak;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.continuity_penalty_components              = ContinuityPenaltyComponents::Normal;
    param.apply_penalty_terms_in_postprocessing_step = true;
    param.continuity_penalty_use_boundary_data       = true;

    // TURBULENCE
    param.use_turbulence_model = false;
    param.turbulence_model     = TurbulenceEddyViscosityModel::Sigma;
    // Smagorinsky: 0.165
    // Vreman: 0.28
    // WALE: 0.50
    // Sigma: 1.35
    param.turbulence_model_constant = 1.35;

    // PROJECTION METHODS

    // formulation
    // this test case one only has periodic or symmetry boundaries so that this parameter is not
    // used. Deactivate in order to reduce memory requirements
    param.store_previous_boundary_values = false;

    // pressure Poisson equation
    param.solver_data_pressure_poisson         = SolverData(1000, ABS_TOL, REL_TOL, 100);
    param.preconditioner_pressure_poisson      = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type = MultigridType::cphMG;

    // projection step
    param.solver_projection         = SolverProjection::CG;
    param.solver_data_projection    = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_projection = PreconditionerProjection::InverseMassMatrix;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    param.solver_momentum = SolverMomentum::GMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_momentum = SolverData(1e4, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      param.solver_data_momentum = SolverData(1e4, ABS_TOL, REL_TOL, 100);

    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = true;

    // formulation
    param.order_pressure_extrapolation = param.order_time_integrator - 1;
    param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER
    param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, ABS_TOL, REL_TOL);

    // linear solver
    param.solver_coupled = SolverCoupled::GMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_coupled = SolverData(1e3, ABS_TOL_LINEAR, REL_TOL_LINEAR, 100);
    else
      param.solver_data_coupled = SolverData(1e3, ABS_TOL, REL_TOL, 100);

    // preconditioning linear solver
    param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block = SchurComplementPreconditioner::CahouetChabard;
  }

  void
  create_grid(std::shared_ptr<Triangulation<dim>> triangulation,
              PeriodicFaces &                     periodic_faces,
              unsigned int const                  n_refine_space,
              std::shared_ptr<Mapping<dim>> &     mapping,
              unsigned int const                  mapping_degree) final
  {
    this->refine_level = n_refine_space;

    double const deformation = 0.5;

    if(ALE)
    {
      AssertThrow(mesh_type == MeshType::Cartesian,
                  ExcMessage("Taylor-Green vortex: Parameter MESH_TYPE is invalid for ALE."));
    }

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

    if(exploit_symmetry == false) // periodic box
    {
      create_periodic_box(triangulation,
                          n_refine_space,
                          periodic_faces,
                          this->n_subdivisions_1d_hypercube,
                          left,
                          right,
                          curvilinear_mesh,
                          deformation);
    }
    else // symmetric box
    {
      GridGenerator::subdivided_hyper_cube(*triangulation,
                                           this->n_subdivisions_1d_hypercube,
                                           0.0,
                                           right);

      if(curvilinear_mesh)
      {
        unsigned int const               frequency = 2;
        static DeformedCubeManifold<dim> manifold(0.0, right, deformation, frequency);
        triangulation->set_all_manifold_ids(1);
        triangulation->set_manifold(1, manifold);

        std::vector<bool> vertex_touched(triangulation->n_vertices(), false);

        for(auto cell : triangulation->active_cell_iterators())
        {
          for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
          {
            if(vertex_touched[cell->vertex_index(v)] == false)
            {
              Point<dim> & vertex                   = cell->vertex(v);
              Point<dim>   new_point                = manifold.push_forward(vertex);
              vertex                                = new_point;
              vertex_touched[cell->vertex_index(v)] = true;
            }
          }
        }
      }

      // perform global refinements
      triangulation->refine_global(n_refine_space);
    }

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
  }

  std::shared_ptr<Function<dim>>
  set_mesh_movement_function() final
  {
    std::shared_ptr<Function<dim>> mesh_motion;

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
  set_boundary_conditions(
    std::shared_ptr<BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<BoundaryDescriptorP<dim>> boundary_descriptor_pressure) final
  {
    if(exploit_symmetry)
    {
      typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

      boundary_descriptor_velocity->symmetry_bc.insert(
        pair(0, new Functions::ZeroFunction<dim>(dim))); // function will not be used
      boundary_descriptor_pressure->neumann_bc.insert(
        pair(0, new Functions::ZeroFunction<dim>(dim))); // dg_u/dt=0 for dual splitting
    }
    else
    {
      // test case with pure periodic BC
      // boundary descriptors remain empty for velocity and pressure
    }
  }


  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions) final
  {
    field_functions->initial_solution_velocity.reset(new InitialSolutionVelocity<dim>(V_0, L));
    field_functions->initial_solution_pressure.reset(new InitialSolutionPressure<dim>(V_0, L, p_0));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm) final
  {
    PostProcessorData<dim> pp_data;

    std::string name =
      this->output_name + "_l" + std::to_string(this->refine_level) + "_k" + std::to_string(degree);

    // write output for visualization of results
    pp_data.output_data.write_output              = this->write_output;
    pp_data.output_data.directory                 = this->output_directory + "vtu/";
    pp_data.output_data.filename                  = name;
    pp_data.output_data.start_time                = start_time;
    pp_data.output_data.interval_time             = (end_time - start_time) / 20;
    pp_data.output_data.write_vorticity           = true;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_velocity_magnitude  = true;
    pp_data.output_data.write_vorticity_magnitude = true;
    pp_data.output_data.write_q_criterion         = true;
    pp_data.output_data.write_processor_id        = true;
    pp_data.output_data.write_higher_order        = false;
    pp_data.output_data.degree                    = degree;

    // calculate div and mass error
    pp_data.mass_data.calculate               = false;
    pp_data.mass_data.start_time              = 0.0;
    pp_data.mass_data.sample_every_time_steps = 1e2;
    pp_data.mass_data.directory               = this->output_directory;
    pp_data.mass_data.filename                = name;
    pp_data.mass_data.reference_length_scale  = 1.0;

    // kinetic energy
    pp_data.kinetic_energy_data.calculate                  = true;
    pp_data.kinetic_energy_data.evaluate_individual_terms  = false;
    pp_data.kinetic_energy_data.calculate_every_time_steps = 1;
    pp_data.kinetic_energy_data.viscosity                  = viscosity;
    pp_data.kinetic_energy_data.directory                  = this->output_directory;
    pp_data.kinetic_energy_data.filename                   = name;
    pp_data.kinetic_energy_data.clear_file                 = !read_restart;

    // kinetic energy spectrum
    bool const do_fftw_during_simulation                               = true;
    pp_data.kinetic_energy_spectrum_data.calculate                     = true;
    pp_data.kinetic_energy_spectrum_data.do_fftw                       = do_fftw_during_simulation;
    pp_data.kinetic_energy_spectrum_data.write_raw_data_to_files       = !do_fftw_during_simulation;
    pp_data.kinetic_energy_spectrum_data.calculate_every_time_interval = 0.5;
    pp_data.kinetic_energy_spectrum_data.directory                     = this->output_directory;
    pp_data.kinetic_energy_spectrum_data.filename                      = name + "_energy_spectrum";
    pp_data.kinetic_energy_spectrum_data.degree                        = degree;
    pp_data.kinetic_energy_spectrum_data.evaluation_points_per_cell    = (degree + 1);
    pp_data.kinetic_energy_spectrum_data.exploit_symmetry              = exploit_symmetry;
    pp_data.kinetic_energy_spectrum_data.n_cells_1d_coarse_grid = this->n_subdivisions_1d_hypercube;
    pp_data.kinetic_energy_spectrum_data.refine_level           = this->refine_level;
    pp_data.kinetic_energy_spectrum_data.length_symmetric_domain = right;
    pp_data.kinetic_energy_spectrum_data.clear_file              = !read_restart;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace IncNS

template<int dim, typename Number>
std::shared_ptr<IncNS::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<IncNS::Application<dim, Number>>(input_file);
}

} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TAYLOR_GREEN_VORTEX_H_ */
