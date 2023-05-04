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

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DEFORMING_HILL_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DEFORMING_HILL_H_

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
    // The analytical solution is only known at t = start_time and t = end_time

    double center_x = 0.5;
    double center_y = 0.75;
    double factor   = 50.0;
    double result   = std::exp(-factor * (pow(p[0] - center_x, 2.0) + pow(p[1] - center_y, 2.0)));

    return result;
  }
};

template<int dim>
class VelocityField : public dealii::Function<dim>
{
public:
  VelocityField(double const end_time) : dealii::Function<dim>(dim, 0.0), end_time(end_time)
  {
  }

  double
  value(dealii::Point<dim> const & point, unsigned int const component = 0) const final
  {
    double value = 0.0;
    double t     = this->get_time();

    if(component == 0)
      value = 4.0 * std::sin(dealii::numbers::PI * point[0]) *
              std::sin(dealii::numbers::PI * point[0]) * std::sin(dealii::numbers::PI * point[1]) *
              std::cos(dealii::numbers::PI * point[1]) *
              std::cos(dealii::numbers::PI * t / end_time);
    else if(component == 1)
      value = -4.0 * std::sin(dealii::numbers::PI * point[0]) *
              std::cos(dealii::numbers::PI * point[0]) * std::sin(dealii::numbers::PI * point[1]) *
              std::sin(dealii::numbers::PI * point[1]) *
              std::cos(dealii::numbers::PI * t / end_time);

    return value;
  }

private:
  double const end_time;
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
    this->param.problem_type              = ProblemType::Unsteady;
    this->param.equation_type             = EquationType::Convection;
    this->param.analytical_velocity_field = true;
    this->param.right_hand_side           = false;

    // PHYSICAL QUANTITIES
    this->param.start_time  = start_time;
    this->param.end_time    = end_time;
    this->param.diffusivity = 0.0;

    // TEMPORAL DISCRETIZATION
    //    this->param.temporal_discretization = TemporalDiscretization::BDF;
    //    this->param.treatment_of_convective_term = TreatmentOfConvectiveTerm::Implicit;
    //    this->param.order_time_integrator         = 3;
    //    this->param.start_with_low_order          = true;
    this->param.temporal_discretization       = TemporalDiscretization::ExplRK;
    this->param.time_integrator_rk            = TimeIntegratorRK::ExplRK3Stage7Reg2;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.time_step_size                = 1.0e-4;
    this->param.cfl                           = 0.4;
    this->param.exponent_fe_degree_convection = 1.5;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = 1;

    // convective term
    this->param.numerical_flux_convective_operator =
      NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    this->param.IP_factor = 1.0;

    // SOLVER
    this->param.solver         = Solver::GMRES;
    this->param.solver_data    = SolverData(1e4, 1.e-20, 1.e-6, 100);
    this->param.preconditioner = Preconditioner::InverseMassMatrix;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 20;

    // NUMERICAL PARAMETERS
    this->param.use_combined_operator = true;
  }

  void
  create_grid() final
  {
    dealii::GridGenerator::hyper_cube(*this->grid->triangulation, left, right);

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
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
    this->field_functions->velocity.reset(new VelocityField<dim>(end_time));
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

    // analytical solution only available at t = start_time and t = end_time
    pp_data.error_data.time_control_data.is_active        = true;
    pp_data.error_data.time_control_data.start_time       = start_time;
    pp_data.error_data.time_control_data.trigger_interval = (end_time - start_time);
    pp_data.error_data.analytical_solution.reset(new Solution<dim>(1));

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  double const left = 0.0, right = 1.0;

  double const start_time = 0.0;
  double const end_time   = 1.0; // increase end_time for larger deformations of the hill
};

} // namespace ConvDiff

} // namespace ExaDG

#include <exadg/convection_diffusion/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DEFORMING_HILL_H_ */
