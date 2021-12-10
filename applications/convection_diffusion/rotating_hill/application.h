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

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution(unsigned int const n_components = 1, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const /*component*/) const
  {
    double t = this->get_time();

    double radius   = 0.5;
    double omega    = 2.0 * numbers::PI;
    double center_x = -radius * std::sin(omega * t);
    double center_y = +radius * std::cos(omega * t);
    double result   = std::exp(-50 * pow(p[0] - center_x, 2.0) - 50 * pow(p[1] - center_y, 2.0));

    return result;
  }
};

template<int dim>
class VelocityField : public Function<dim>
{
public:
  VelocityField(unsigned int const n_components = dim, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & point, unsigned int const component = 0) const
  {
    double value = 0.0;

    if(component == 0)
      value = -point[1] * 2.0 * numbers::PI;
    else if(component == 1)
      value = point[0] * 2.0 * numbers::PI;

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
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  double const start_time = 0.0;
  double const end_time   = 1.0;

  double const left  = -1.0;
  double const right = +1.0;

  void
  set_input_parameters(unsigned int const degree) final
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
    this->param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit; // ExplicitOIF;
    this->param.start_with_low_order         = false;
    this->param.time_integrator_oif =
      TimeIntegratorRK::ExplRK2Stage2; // ExplRK3Stage7Reg2; //ExplRK4Stage8Reg2;

    //    this->param.temporal_discretization      = TemporalDiscretization::ExplRK;
    //    this->param.time_integrator_rk           = TimeIntegratorRK::ExplRK3Stage7Reg2;

    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.adaptive_time_stepping        = false;
    this->param.time_step_size                = 1.e-2;
    this->param.cfl_oif                       = 0.25;
    this->param.cfl                           = this->param.cfl_oif * 1.0;
    this->param.exponent_fe_degree_convection = 1.5;

    // restart
    this->param.restart_data.write_restart = false;
    this->param.restart_data.filename      = "output_conv_diff/rotating_hill";
    this->param.restart_data.interval_time = 0.4;


    // SPATIAL DISCRETIZATION
    this->param.triangulation_type = TriangulationType::Distributed;
    this->param.mapping            = MappingType::Affine;
    this->param.degree             = degree;

    // convective term
    this->param.numerical_flux_convective_operator =
      NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    this->param.IP_factor = 1.0;

    // SOLVER
    this->param.solver      = Solver::GMRES;
    this->param.solver_data = SolverData(1e3, 1.e-20, 1.e-8, 100);
    this->param.preconditioner =
      Preconditioner::Multigrid; // None; //InverseMassMatrix; //PointJacobi;
                                 // //BlockJacobi; //Multigrid;
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
    this->param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
    this->param.multigrid_data.smoother_data.iterations     = 5;
    this->param.multigrid_data.smoother_data.relaxation_factor = 0.8;

    // MG coarse grid solver
    this->param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 20;

    // NUMERICAL PARAMETERS
    this->param.use_cell_based_face_loops               = true;
    this->param.store_analytical_velocity_in_dof_vector = false;
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid(GridData const & grid_data) final
  {
    std::shared_ptr<Grid<dim, Number>> grid =
      std::make_shared<Grid<dim, Number>>(grid_data, this->mpi_comm);

    GridGenerator::hyper_cube(*grid->triangulation, left, right);

    grid->triangulation->refine_global(grid_data.n_refine_global);

    return grid;
  }


  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // problem with pure Dirichlet boundary conditions
    this->boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>()));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new Solution<dim>());
    this->field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->velocity.reset(new VelocityField<dim>());
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output  = this->write_output;
    pp_data.output_data.directory     = this->output_directory + "vtu/";
    pp_data.output_data.filename      = this->output_name;
    pp_data.output_data.start_time    = start_time;
    pp_data.output_data.interval_time = (end_time - start_time) / 20;
    pp_data.output_data.degree        = this->param.degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>(1));
    pp_data.error_data.calculate_relative_errors = true;
    pp_data.error_data.error_calc_start_time     = start_time;
    pp_data.error_data.error_calc_interval_time  = (end_time - start_time) / 20;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace ConvDiff

} // namespace ExaDG

#include <exadg/convection_diffusion/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_ */
