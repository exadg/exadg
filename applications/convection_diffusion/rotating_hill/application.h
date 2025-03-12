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

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_

// ExaDG
#include <exadg/grid/deformed_cube_manifold.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim>
class Solution : public dealii::Function<dim>
{
public:
  Solution(unsigned int const n_components = 1, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const final
  {
    double t = this->get_time();

    double radius   = 0.5;
    double omega    = 2.0 * dealii::numbers::PI;
    double center_x = -radius * std::sin(omega * t);
    double center_y = +radius * std::cos(omega * t);
    double result   = std::exp(-50 * pow(p[0] - center_x, 2.0) - 50 * pow(p[1] - center_y, 2.0));

    return result;
  }
};

template<int dim>
class VelocityField : public dealii::Function<dim>
{
public:
  VelocityField(unsigned int const n_components = dim, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & point, unsigned int const component = 0) const final
  {
    double value = 0.0;

    if(component == 0)
      value = -point[1] * 2.0 * dealii::numbers::PI;
    else if(component == 1)
      value = point[0] * 2.0 * dealii::numbers::PI;

    return value;
  }
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
      prm.add_parameter("EnableAdaptivity",
                        enable_adaptivity,
                        "Enable adaptive mesh refinement.",
                        dealii::Patterns::Bool());
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type              = ProblemType::Unsteady;
    this->param.equation_type             = EquationType::Convection;
    this->param.analytical_velocity_field = true;
    this->param.right_hand_side           = false;

    // PHYSICAL QUANTITIES
    this->param.start_time  = start_time;
    this->param.end_time    = end_time;
    this->param.diffusivity = 0.0;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization      = TemporalDiscretization::BDF;
    this->param.order_time_integrator        = 2; // instabilities for BDF 3 and 4
    this->param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
    this->param.start_with_low_order         = false;

    //    this->param.temporal_discretization      = TemporalDiscretization::ExplRK;
    //    this->param.time_integrator_rk           = TimeIntegratorRK::ExplRK3Stage7Reg2;

    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping        = false;
    this->param.time_step_size                = 1.e-2;
    this->param.cfl                           = 0.25;
    this->param.exponent_fe_degree_convection = 1.5;

    // restart
    this->param.restart_data.write_restart = false;
    this->param.restart_data.filename      = "output_conv_diff/rotating_hill";
    this->param.restart_data.interval_time = 0.4;


    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = this->param.degree;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;

    this->param.enable_adaptivity = enable_adaptivity;

    this->param.grid.create_coarse_triangulations = enable_adaptivity;

    this->param.amr_data.trigger_every_n_time_steps        = 30;
    this->param.amr_data.maximum_refinement_level          = 4;
    this->param.amr_data.minimum_refinement_level          = 0;
    this->param.amr_data.preserve_boundary_cells           = false;
    this->param.amr_data.fraction_of_cells_to_be_refined   = 0.025;
    this->param.amr_data.fraction_of_cells_to_be_coarsened = 0.4;

    // convective term
    this->param.numerical_flux_convective_operator =
      NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    this->param.IP_factor = 1.0;

    // SOLVER
    this->param.solver      = Solver::GMRES;
    this->param.solver_data = SolverData(1e3, 1.e-20, 1.e-8, 100);
    this->param.preconditioner =
      Preconditioner::Multigrid; // None; //InverseMassMatrix; //PointJacobi; // BlockJacobi;
                                 // //Multigrid;
    this->param.update_preconditioner = true;

    // BlockJacobi (these parameters are also relevant if used as a smoother in multigrid)
    this->param.implement_block_diagonal_preconditioner_matrix_free = false; // true;
    this->param.solver_block_diagonal                               = Elementwise::Solver::GMRES;
    this->param.preconditioner_block_diagonal = Elementwise::Preconditioner::InverseMassMatrix;
    this->param.solver_data_block_diagonal    = SolverData(1000, 1.e-12, 1.e-2, 1000);

    // Multigrid
    this->param.mg_operator_type    = MultigridOperatorType::ReactionConvection;
    this->param.multigrid_data.type = MultigridType::hMG;

    // MG smoother
    this->param.multigrid_data.smoother_data.smoother       = MultigridSmoother::Jacobi;
    this->param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::PointJacobi;
    this->param.multigrid_data.smoother_data.iterations     = 5;
    this->param.multigrid_data.smoother_data.relaxation_factor = 0.8;

    // MG coarse grid solver
    this->param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 20;

    // NUMERICAL PARAMETERS
    this->param.use_cell_based_face_loops               = false;
    this->param.store_analytical_velocity_in_dof_vector = false;
  }

  void
  create_grid(Grid<dim> &                                       grid,
              std::shared_ptr<dealii::Mapping<dim>> &           mapping,
              std::shared_ptr<MultigridMappings<dim, Number>> & multigrid_mappings) final
  {
    (void)mapping;
    (void)multigrid_mappings;

    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> &                        tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & periodic_face_pairs,
          unsigned int const                                       global_refinements,
          std::vector<unsigned int> const &                        vector_local_refinements) {
        (void)periodic_face_pairs;
        (void)vector_local_refinements;

        // hypercube volume is [left,right]^dim
        dealii::GridGenerator::hyper_cube(tria, left, right);

        // For AMR, we want to consider a mapping as well.
        if(this->param.enable_adaptivity)
        {
          unsigned int const frequency   = 1;
          double const       deformation = (right - left) * 0.1;
          apply_deformed_cube_manifold(tria, left, right, deformation, frequency);
        }

        tria.refine_global(global_refinements);
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
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // problem with pure Dirichlet boundary conditions
    this->boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>()));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new Solution<dim>());
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->velocity.reset(new VelocityField<dim>());
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 20.0;
    pp_data.output_data.directory = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename  = this->output_parameters.filename;
    pp_data.output_data.degree    = this->param.degree;

    pp_data.error_data.time_control_data.is_active        = true;
    pp_data.error_data.time_control_data.start_time       = start_time;
    pp_data.error_data.time_control_data.trigger_interval = (end_time - start_time) / 20.0;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>(1));
    pp_data.error_data.calculate_relative_errors = true;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  double const start_time = 0.0;
  double const end_time   = 1.0;

  double const left  = -1.0;
  double const right = +1.0;

  bool enable_adaptivity = false;
};

} // namespace ConvDiff

} // namespace ExaDG

#include <exadg/convection_diffusion/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_ */
