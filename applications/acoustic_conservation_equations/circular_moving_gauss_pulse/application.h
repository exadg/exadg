/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#ifndef APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_CIRCULAR_MOVING_GAUSS_PULSE_H_
#define APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_CIRCULAR_MOVING_GAUSS_PULSE_H_

#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <exadg/grid/grid_utilities.h>

namespace ExaDG
{
namespace Acoustics
{
template<int dim>
class SourceTerm : public dealii::Function<dim>
{
public:
  SourceTerm(double const c, double const xi, double const alpha, double const l)
    : dealii::Function<dim>(1, 0.0), c(c), xi(xi), alpha(alpha), l(l)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const) const final
  {
    double const t        = this->get_time();
    double const sin_xi_t = std::sin(xi * t);
    double const cos_xi_t = std::cos(xi * t);

    double const result = 1.0e-6 * 2.0 * alpha *
                          (2.0 * c * c *
                             (-alpha * (l * sin_xi_t - p[0]) * (l * sin_xi_t - p[0]) -
                              alpha * (l * cos_xi_t - p[1]) * (l * cos_xi_t - p[1]) + 1.0) +
                           l * xi * xi *
                             (2.0 * alpha * l * (p[0] * cos_xi_t - p[1] * sin_xi_t) *
                                (p[0] * cos_xi_t - p[1] * sin_xi_t) -
                              p[0] * sin_xi_t - p[1] * cos_xi_t)) *
                          std::exp(-alpha * ((l * sin_xi_t - p[0]) * (l * sin_xi_t - p[0]) +
                                             (l * cos_xi_t - p[1]) * (l * cos_xi_t - p[1]))) /
                          (c * c);

    return result;
  }

private:
  double const c;
  double const xi;
  double const alpha;
  double const l;
};

template<int dim>
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure(double const xi, double const alpha, double const l)
    : dealii::Function<dim>(1, 0.0), xi(xi), alpha(alpha), l(l)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const) const final
  {
    double const t        = this->get_time();
    double const sin_xi_t = std::sin(xi * t);
    double const cos_xi_t = std::cos(xi * t);

    double const result = 1.0e-6 * 2.0 * xi * alpha * l * (p[0] * cos_xi_t - p[1] * sin_xi_t) *
                          std::exp(-1.0 * alpha *
                                   ((p[0] - l * sin_xi_t) * (p[0] - l * sin_xi_t) +
                                    (p[1] - l * cos_xi_t) * (p[1] - l * cos_xi_t)));
    return result;
  }

private:
  double const xi;
  double const alpha;
  double const l;
};

template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  AnalyticalSolutionVelocity(double const xi, double const alpha, double const l)
    : dealii::Function<dim>(dim, 0.0), xi(xi), alpha(alpha), l(l)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const final
  {
    double const t        = this->get_time();
    double const sin_xi_t = std::sin(xi * t);
    double const cos_xi_t = std::cos(xi * t);

    double result = 1.0e-6 * -2.0 * alpha *
                    std::exp(-1.0 * alpha *
                             ((p[0] - l * sin_xi_t) * (p[0] - l * sin_xi_t) +
                              (p[1] - l * cos_xi_t) * (p[1] - l * cos_xi_t)));
    if(component == 0)
      result *= (l * sin_xi_t - p[0]);
    else if(component == 1)
      result *= (l * cos_xi_t - p[1]);

    return result;
  }

private:
  double const xi;
  double const alpha;
  double const l;
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
      prm.add_parameter("RuntimeInNumberOfPulseRotations",
                        number_of_rotations,
                        "Number of pulse rotations during runtime.",
                        dealii::Patterns::Double(1.0e-12));
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.formulation     = Formulation::SkewSymmetric;
    this->param.right_hand_side = true;

    // PHYSICAL QUANTITIES
    this->param.start_time     = start_time;
    this->param.end_time       = number_of_rotations * compute_rotation_duration();
    this->param.speed_of_sound = speed_of_sound;
    this->param.density        = density;

    // TEMPORAL DISCRETIZATION
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.cfl                           = 0.59;
    this->param.order_time_integrator         = 2;
    this->param.start_with_low_order          = false;

    // output of solver information
    this->param.solver_info_data.interval_time = (this->param.end_time - this->param.start_time);

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = 2;
    this->param.degree_p                = this->param.degree_u;
    this->param.degree_u                = this->param.degree_p;
  }

  void
  create_grid(Grid<dim> & grid, std::shared_ptr<dealii::Mapping<dim>> & mapping) final
  {
    (void)mapping;

    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> & tria,
          std::vector<dealii::GridTools::PeriodicFacePair<
            typename dealii::Triangulation<dim>::cell_iterator>> & /*periodic_face_pairs*/,
          unsigned int const global_refinements,
          std::vector<unsigned int> const & /* vector_local_refinements*/) {
        dealii::GridGenerator::hyper_ball_balanced(tria, {}, radius);

        for(const auto & face : tria.active_face_iterators())
          if(face->at_boundary())
            face->set_boundary_id(1);

        tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation<dim>(
      grid, this->mpi_comm, this->param.grid, lambda_create_triangulation, {});

    GridUtilities::create_mapping(mapping,
                                  this->param.grid.element_type,
                                  this->param.mapping_degree);
  }

  void
  set_boundary_descriptor() final
  {
    this->boundary_descriptor->velocity_dbc.insert(
      std::make_pair(1, std::make_shared<AnalyticalSolutionVelocity<dim>>(xi, alpha, l)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_pressure =
      std::make_shared<AnalyticalSolutionPressure<dim>>(xi, alpha, l);

    this->field_functions->initial_solution_velocity =
      std::make_shared<AnalyticalSolutionVelocity<dim>>(xi, alpha, l);

    this->field_functions->right_hand_side =
      std::make_shared<SourceTerm<dim>>(speed_of_sound, xi, alpha, l);
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active  = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time = start_time;
    pp_data.output_data.time_control_data.trigger_interval =
      (this->param.end_time - start_time) / 20.0;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_pressure     = true;
    pp_data.output_data.write_velocity     = true;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree_u;

    // calculation of pressure error
    pp_data.error_data_p.time_control_data.is_active        = true;
    pp_data.error_data_p.time_control_data.start_time       = start_time;
    pp_data.error_data_p.time_control_data.trigger_interval = (this->param.end_time - start_time);
    pp_data.error_data_p.analytical_solution =
      std::make_shared<AnalyticalSolutionPressure<dim>>(xi, alpha, l);
    pp_data.error_data_p.calculate_relative_errors = true;
    pp_data.error_data_p.name                      = "pressure";

    // ... velocity error
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval = (this->param.end_time - start_time);
    pp_data.error_data_u.analytical_solution =
      std::make_shared<AnalyticalSolutionVelocity<dim>>(xi, alpha, l);
    pp_data.error_data_u.calculate_relative_errors = true;
    pp_data.error_data_u.name                      = "velocity";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  // problem specific parameters like physical dimensions, etc.
  double xi                  = 4.0e5;
  double alpha               = 1.6e5;
  double radius              = 0.04;
  double l                   = 0.02;
  double speed_of_sound      = 1500.0;
  double density             = 1000.0;
  double number_of_rotations = 1.0;

  double
  compute_rotation_duration()
  {
    return (2.0 * dealii::numbers::PI / xi);
  }

  double const start_time = 0.0;
};

} // namespace Acoustics

} // namespace ExaDG

#include <exadg/acoustic_conservation_equations/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_CIRCULAR_MOVING_GAUSS_PULSE_H_ \
        */
