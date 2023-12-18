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

#ifndef APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_VIBRATING_MEMBRANE_H_
#define APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_VIBRATING_MEMBRANE_H_

#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <exadg/grid/grid_utilities.h>

namespace ExaDG
{
namespace Acoustics
{
template<int dim>
class AnalyticalSolutionPressure : public dealii::Function<dim>
{
public:
  AnalyticalSolutionPressure(double const modes, double const speed_of_sound)
    : dealii::Function<dim>(1, 0.0), M(modes), c(speed_of_sound)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const) const final
  {
    double const t  = this->get_time();
    double const pi = dealii::numbers::PI;

    double result = std::cos(M * std::sqrt(dim) * pi * c * t);

    if constexpr(dim == 2)
      result *= std::sin(M * pi * p[0]) * std::sin(M * pi * p[1]);
    else if constexpr(dim == 3)
      result *= std::sin(M * pi * p[0]) * std::sin(M * pi * p[1]) * std::sin(M * pi * p[2]);

    return result;
  }

private:
  double const M, c;
};

template<int dim>
class AnalyticalSolutionVelocity : public dealii::Function<dim>
{
public:
  explicit AnalyticalSolutionVelocity(double const modes,
                                      double const speed_of_sound,
                                      double const density)
    : dealii::Function<dim>(dim, 0.0), M(modes), c(speed_of_sound), rho(density)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const final
  {
    double const t  = this->get_time();
    double const pi = dealii::numbers::PI;

    double result = -std::sin(M * std::sqrt(dim) * pi * c * t) / (std::sqrt(dim) * rho * c);

    if constexpr(dim == 2)
    {
      if(component == 0)
        result *= std::cos(M * pi * p[0]) * std::sin(M * pi * p[1]);
      else if(component == 1)
        result *= std::sin(M * pi * p[0]) * std::cos(M * pi * p[1]);
    }
    else if constexpr(dim == 3)
    {
      if(component == 0)
        result *= std::cos(M * pi * p[0]) * std::sin(M * pi * p[1]) * std::sin(M * pi * p[2]);
      else if(component == 1)
        result *= std::sin(M * pi * p[0]) * std::cos(M * pi * p[1]) * std::sin(M * pi * p[2]);
      else if(component == 2)
        result *= std::sin(M * pi * p[0]) * std::sin(M * pi * p[1]) * std::cos(M * pi * p[2]);
    }

    return rho * result;
  }

private:
  double const M, c, rho;
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
      // MATHEMATICAL MODEL
      prm.add_parameter("Formulation", this->param.formulation, "Formulation.");

      // PHYSICAL QUANTITIES
      prm.add_parameter("SpeedOfSound",
                        this->param.speed_of_sound,
                        "Speed of sound.",
                        dealii::Patterns::Double());

      prm.add_parameter("Density", this->param.density, "Density.", dealii::Patterns::Double());


      // TEMPORAL DISCRETIZATION
      prm.add_parameter("TimeIntegrationScheme",
                        this->param.calculation_of_time_step_size,
                        "How to calculate time step size.");

      prm.add_parameter("UserSpecifiedTimeStepSize",
                        this->param.time_step_size,
                        "UserSpecified Timestep size.",
                        dealii::Patterns::Double());

      prm.add_parameter("CFL", this->param.cfl, "CFL number.", dealii::Patterns::Double());

      prm.add_parameter("OrderTimeIntegrator",
                        this->param.order_time_integrator,
                        "Order of time integration.",
                        dealii::Patterns::Integer(1));

      // APPLICATION SPECIFIC
      prm.add_parameter("RuntimeInNumberOfPeriods",
                        number_of_periods,
                        "Number of temporal oscillations during runtime.",
                        dealii::Patterns::Double(1.0e-12));

      prm.add_parameter("Modes", modes, "Number of Modes.", dealii::Patterns::Double(1.0e-12));
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // PHYSICAL QUANTITIES
    this->param.start_time = start_time;
    this->param.end_time   = number_of_periods * compute_period_duration();

    // TEMPORAL DISCRETIZATION
    this->param.start_with_low_order = false;

    // output of solver information
    this->param.solver_info_data.interval_time = (this->param.end_time - this->param.start_time);

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = 1;
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
        dealii::GridGenerator::hyper_cube(tria, left, right);

        for(const auto & face : tria.active_face_iterators())
          if(face->at_boundary())
            face->set_boundary_id(1);

        tria.refine_global(global_refinements);
      };

    GridUtilities::create_triangulation<dim>(
      grid, this->mpi_comm, this->param.grid, lambda_create_triangulation, {});
  }

  void
  set_boundary_descriptor() final
  {
    this->boundary_descriptor->pressure_dbc.insert(std::make_pair(
      1, std::make_shared<AnalyticalSolutionPressure<dim>>(modes, this->param.speed_of_sound)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_pressure =
      std::make_shared<AnalyticalSolutionPressure<dim>>(modes, this->param.speed_of_sound);

    this->field_functions->initial_solution_velocity =
      std::make_shared<AnalyticalSolutionVelocity<dim>>(modes,
                                                        this->param.speed_of_sound,
                                                        this->param.density);
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
      std::make_shared<AnalyticalSolutionPressure<dim>>(modes, this->param.speed_of_sound);
    pp_data.error_data_p.calculate_relative_errors = false; // at some times the solution is 0
    pp_data.error_data_p.name                      = "pressure";

    // ... velocity error
    pp_data.error_data_u.time_control_data.is_active        = true;
    pp_data.error_data_u.time_control_data.start_time       = start_time;
    pp_data.error_data_u.time_control_data.trigger_interval = (this->param.end_time - start_time);
    pp_data.error_data_u.analytical_solution =
      std::make_shared<AnalyticalSolutionVelocity<dim>>(modes,
                                                        this->param.speed_of_sound,
                                                        this->param.density);
    pp_data.error_data_u.calculate_relative_errors = false; // at some times the solution is 0
    pp_data.error_data_u.name                      = "velocity";

    // sound energy calculation
    pp_data.sound_energy_data.time_control_data.is_active  = false;
    pp_data.sound_energy_data.time_control_data.start_time = this->param.start_time;
    pp_data.sound_energy_data.density                      = this->param.density;
    pp_data.sound_energy_data.speed_of_sound               = this->param.speed_of_sound;
    pp_data.sound_energy_data.time_control_data.trigger_every_time_steps = 1;
    pp_data.sound_energy_data.directory = this->output_parameters.directory + "sound_energy/";

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  // problem specific parameters like physical dimensions, etc.
  double modes             = 2.0;
  double number_of_periods = 1.0;

  double
  compute_period_duration()
  {
    AssertThrow(this->param.speed_of_sound > 0.0, dealii::ExcMessage("speed_of_sound not set."));
    return 2.0 / (modes * std::sqrt(dim) * this->param.speed_of_sound);
  }

  double const left  = 0.0;
  double const right = 1.0;

  double const start_time = 0.0;
};

} // namespace Acoustics

} // namespace ExaDG

#include <exadg/acoustic_conservation_equations/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_VIBRATING_MEMBRANE_H_ */
