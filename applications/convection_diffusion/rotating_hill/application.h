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
  typedef typename ApplicationBase<dim, Number>::PeriodicFaces PeriodicFaces;

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
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
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.problem_type              = ProblemType::Unsteady;
    param.equation_type             = EquationType::Convection;
    param.analytical_velocity_field = true;
    param.right_hand_side           = false;

    // PHYSICAL QUANTITIES
    param.start_time  = start_time;
    param.end_time    = end_time;
    param.diffusivity = 0.0;

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization      = TemporalDiscretization::BDF;
    param.order_time_integrator        = 2; // instabilities for BDF 3 and 4
    param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit; // ExplicitOIF;
    param.start_with_low_order         = false;
    param.time_integrator_oif =
      TimeIntegratorRK::ExplRK2Stage2; // ExplRK3Stage7Reg2; //ExplRK4Stage8Reg2;

    //    param.temporal_discretization      = TemporalDiscretization::ExplRK;
    //    param.time_integrator_rk           = TimeIntegratorRK::ExplRK3Stage7Reg2;

    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.adaptive_time_stepping        = false;
    param.time_step_size                = 1.e-2;
    param.cfl_oif                       = 0.25;
    param.cfl                           = param.cfl_oif * 1.0;
    param.exponent_fe_degree_convection = 1.5;

    // restart
    param.restart_data.write_restart = false;
    param.restart_data.filename      = "output_conv_diff/rotating_hill";
    param.restart_data.interval_time = 0.4;


    // SPATIAL DISCRETIZATION

    // triangulation
    param.triangulation_type = TriangulationType::Distributed;

    // polynomial degree
    param.mapping = MappingType::Affine;

    // convective term
    param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    param.IP_factor = 1.0;

    // SOLVER
    param.solver         = Solver::GMRES;
    param.solver_data    = SolverData(1e3, 1.e-20, 1.e-8, 100);
    param.preconditioner = Preconditioner::Multigrid; // None; //InverseMassMatrix; //PointJacobi;
                                                      // //BlockJacobi; //Multigrid;
    param.update_preconditioner = true;

    // BlockJacobi (these parameters are also relevant if used as a smoother in multigrid)
    param.implement_block_diagonal_preconditioner_matrix_free = false; // true;
    param.solver_block_diagonal                               = Elementwise::Solver::GMRES;
    param.preconditioner_block_diagonal = Elementwise::Preconditioner::InverseMassMatrix;
    param.solver_data_block_diagonal    = SolverData(1000, 1.e-12, 1.e-2, 1000);

    // Multigrid
    param.mg_operator_type    = MultigridOperatorType::ReactionConvection;
    param.multigrid_data.type = MultigridType::hMG;

    // MG smoother
    param.multigrid_data.smoother_data.smoother          = MultigridSmoother::Jacobi;
    param.multigrid_data.smoother_data.preconditioner    = PreconditionerSmoother::BlockJacobi;
    param.multigrid_data.smoother_data.iterations        = 5;
    param.multigrid_data.smoother_data.relaxation_factor = 0.8;

    // MG coarse grid solver
    param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // output of solver information
    param.solver_info_data.interval_time = (end_time - start_time) / 20;

    // NUMERICAL PARAMETERS
    param.use_cell_based_face_loops               = true;
    param.store_analytical_velocity_in_dof_vector = false;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              PeriodicFaces &                                   periodic_faces,
              unsigned int const                                n_refine_space,
              std::shared_ptr<Mapping<dim>> &                   mapping,
              unsigned int const                                mapping_degree)
  {
    (void)periodic_faces;

    GridGenerator::hyper_cube(*triangulation, left, right);
    triangulation->refine_global(n_refine_space);

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // problem with pure Dirichlet boundary conditions
    boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>()));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new Solution<dim>());
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->velocity.reset(new VelocityField<dim>());
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.output_folder        = this->output_directory + "vtu/";
    pp_data.output_data.output_name          = this->output_name;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 20;
    pp_data.output_data.degree               = degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>(1));
    pp_data.error_data.calculate_relative_errors = true;
    pp_data.error_data.error_calc_start_time     = start_time;
    pp_data.error_data.error_calc_interval_time  = (end_time - start_time) / 20;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace ConvDiff

template<int dim, typename Number>
std::shared_ptr<ConvDiff::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<ConvDiff::Application<dim, Number>>(input_file);
}

} // namespace ExaDG


#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_ROTATING_HILL_H_ */
