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

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_H_

// prescribe value of solution at left and right boundary
// Neumann boundaries at upper and lower boundary
// use constant advection velocity from left to right -> boundary layer

namespace ExaDG
{
namespace ConvDiff
{
template<int dim>
class Solution : public dealii::Function<dim>
{
public:
  Solution(double const diffusivity) : dealii::Function<dim>(1, 0.0), diffusivity(diffusivity)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const final
  {
    double phi_l = 1.0, phi_r = 0.0;
    double U = 1.0, L = 2.0;
    double Pe = U * L / diffusivity;

    double result = phi_l + (phi_r - phi_l) * (std::exp(Pe * p[0] / L) - std::exp(-Pe / 2.0)) /
                              (std::exp(Pe / 2.0) - std::exp(-Pe / 2.0));

    return result;
  }

private:
  double const diffusivity;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type              = ProblemType::Steady;
    this->param.equation_type             = EquationType::ConvectionDiffusion;
    this->param.right_hand_side           = false;
    this->param.analytical_velocity_field = true;

    // PHYSICAL QUANTITIES
    this->param.start_time  = start_time;
    this->param.end_time    = end_time;
    this->param.diffusivity = diffusivity;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization       = TemporalDiscretization::BDF;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit;
    this->param.order_time_integrator         = 2;
    this->param.start_with_low_order          = true;
    this->param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    this->param.time_step_size                = 1.0e-1;
    this->param.cfl                           = 0.2;
    this->param.diffusion_number              = 0.01;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type     = TriangulationType::Distributed;
    this->param.mapping_degree              = 1;
    this->param.mapping_degree_coarse_grids = this->param.mapping_degree;

    // convective term
    this->param.numerical_flux_convective_operator =
      NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    this->param.IP_factor = 1.0;

    // SOLVER
    this->param.use_cell_based_face_loops = true;
    this->param.solver                    = Solver::GMRES;
    this->param.solver_data               = SolverData(1e4, 1.e-20, 1.e-8, 100);
    this->param.preconditioner            = Preconditioner::Multigrid; // PointJacobi;
    this->param.mg_operator_type          = MultigridOperatorType::ReactionConvectionDiffusion;
    this->param.multigrid_data.type       = MultigridType::phMG;
    // MG smoother
    this->param.multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi; // Chebyshev;
    // MG smoother data
    this->param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
    this->param.multigrid_data.smoother_data.iterations     = 5;

    // MG coarse grid solver
    this->param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    this->param.update_preconditioner = false;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 20;

    // NUMERICAL PARAMETERS
    this->param.use_combined_operator = true;
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

        // set boundary indicator
        for(auto cell : tria)
        {
          for(auto const & f : cell.face_indices())
          {
            if((std::fabs(cell.face(f)->center()(1) - left) < 1e-12) or
               (std::fabs(cell.face(f)->center()(1) - right) < 1e-12) or
               ((dim == 3) and ((std::fabs(cell.face(f)->center()(2) - left) < 1e-12) or
                                (std::fabs(cell.face(f)->center()(2) - right) < 1e-12))))
            {
              cell.face(f)->set_boundary_id(1);
            }
          }
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

    this->boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>(diffusivity)));
    this->boundary_descriptor->neumann_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(1));
    std::vector<double> velocity = std::vector<double>(dim, 0.0);
    velocity[0]                  = 1.0;
    this->field_functions->velocity.reset(new dealii::Functions::ConstantFunction<dim>(velocity));
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
    pp_data.error_data.analytical_solution.reset(new Solution<dim>(diffusivity));

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  double const left = -1.0, right = 1.0;

  double const diffusivity = 1.0e-1;

  double const start_time = 0.0;
  double const end_time   = 1.0;
};

} // namespace ConvDiff

} // namespace ExaDG

#include <exadg/convection_diffusion/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_BOUNDARY_LAYER_H_ */
