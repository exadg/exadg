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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_BOX_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_BOX_H_

// ExaDG
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
      prm.add_parameter("MeshType", mesh_type, "Type of mesh (Cartesian versus curvilinear).");
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::NavierStokes;
    this->param.formulation_viscous_term    = FormulationViscousTerm::LaplaceFormulation;
    this->param.formulation_convective_term = FormulationConvectiveTerm::DivergenceFormulation;
    this->param.right_hand_side             = false;

    // PHYSICAL QUANTITIES
    this->param.start_time = 0.0;
    this->param.end_time   = 1.0;
    this->param.viscosity  = 1.0;

    // TEMPORAL DISCRETIZATION
    this->param.solver_type = SolverType::Unsteady;
    this->param.temporal_discretization =
      TemporalDiscretization::BDFDualSplittingScheme; // BDFPressureCorrection;
                                                      // //BDFCoupledSolution;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.cfl                           = 1.0;
    this->param.max_velocity                  = 1.0;

    // NUMERICAL PARAMETERS
    this->param.quad_rule_linearization =
      QuadratureRuleLinearization::Standard; // Overintegration32k;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = 1; // this->param.degree_u;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;

    // use EqualOrder so that we can also start with k=1 for the velocity!
    this->param.degree_p = DegreePressure::MixedOrder;

    // convective term
    if(this->param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      this->param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    this->param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // div-div and continuity penalty
    this->param.use_divergence_penalty                     = true;
    this->param.divergence_penalty_factor                  = 1.0e0;
    this->param.use_continuity_penalty                     = true;
    this->param.continuity_penalty_factor                  = this->param.divergence_penalty_factor;
    this->param.apply_penalty_terms_in_postprocessing_step = true;

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
    this->param.solver_pressure_poisson         = SolverPressurePoisson::CG;
    this->param.preconditioner_pressure_poisson = PreconditionerPressurePoisson::None;

    // projection step
    this->param.solver_projection         = SolverProjection::CG;
    this->param.preconditioner_projection = PreconditionerProjection::None;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    if(this->param.temporal_discretization == TemporalDiscretization::BDFDualSplittingScheme)
    {
      this->param.solver_momentum         = SolverMomentum::CG;
      this->param.preconditioner_momentum = MomentumPreconditioner::None;
    }

    // PRESSURE-CORRECTION SCHEME

    // momentum step
    if(this->param.temporal_discretization == TemporalDiscretization::BDFPressureCorrection)
    {
      // linear solver
      this->param.solver_momentum         = SolverMomentum::GMRES;
      this->param.preconditioner_momentum = MomentumPreconditioner::None;
    }

    // COUPLED NAVIER-STOKES SOLVER

    // linear solver
    this->param.solver_coupled = SolverCoupled::GMRES;

    // preconditioning linear solver
    this->param.preconditioner_coupled = PreconditionerCoupled::None;

    this->param.preconditioner_velocity_block = MomentumPreconditioner::None;
    this->param.preconditioner_pressure_block = SchurComplementPreconditioner::None;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    auto const lambda_create_triangulation = [&](dealii::Triangulation<dim, dim> & tria,
                                                 std::vector<dealii::GridTools::PeriodicFacePair<
                                                   typename dealii::Triangulation<
                                                     dim>::cell_iterator>> & periodic_face_pairs,
                                                 unsigned int const          global_refinements,
                                                 std::vector<unsigned int> const &
                                                   vector_local_refinements) {
      (void)vector_local_refinements;

      double const left = -1.0, right = 1.0;
      double const deformation = 0.1;

      AssertThrow(
        this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
        dealii::ExcMessage(
          "Periodic faces might not be applied correctly for TriangulationType::FullyDistributed. "
          "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));

      if(mesh_type == MeshType::Curvilinear)
      {
        AssertThrow(
          this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
          dealii::ExcMessage(
            "Manifolds might not be applied correctly for TriangulationType::FullyDistributed. "
            "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));
      }

      create_periodic_box(tria,
                          global_refinements,
                          periodic_face_pairs,
                          this->n_subdivisions_1d_hypercube,
                          left,
                          right,
                          mesh_type == MeshType::Curvilinear,
                          deformation);
    };

    GridUtilities::create_triangulation_with_multigrid<dim>(grid,
                                                            this->mpi_comm,
                                                            this->param.grid,
                                                            this->param.involves_h_multigrid(),
                                                            lambda_create_triangulation,
                                                            {} /* no local refinements */);

    // mappings
    GridUtilities::create_mapping_with_multigrid(mapping,
                                                 multigrid_mappings,
                                                 this->param.grid.element_type,
                                                 this->param.mapping_degree,
                                                 this->param.mapping_degree_coarse_grids,
                                                 this->param.involves_h_multigrid());
  }

  void
  set_boundary_descriptor() final
  {
    // test case with purely periodic boundary conditions
    // boundary descriptors remain empty for velocity and pressure
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->initial_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->analytical_solution_pressure.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // no postprocessing

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  MeshType mesh_type = MeshType::Cartesian;
};

} // namespace IncNS

} // namespace ExaDG

#include <exadg/incompressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_BOX_H_ */
