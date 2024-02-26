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

#ifndef APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_NORMAL_INCIDENTING_WAVE_H_
#define APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_NORMAL_INCIDENTING_WAVE_H_

#include <deal.II/base/function.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>

#include <exadg/grid/grid_utilities.h>

namespace ExaDG
{
namespace Acoustics
{
template<int dim>
class InitialConditionPressure : public dealii::Function<dim>
{
public:
  InitialConditionPressure() : dealii::Function<dim>(1, 0.0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const) const final
  {
    double exponent = 0.0;
    for(unsigned int d = 0; d < dim; ++d)
      exponent += p[d] * p[d];

    return std::exp(-100.0 * exponent);
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
      prm.add_parameter("SpeedOfSound",
                        speed_of_sound,
                        "Speed of sound.",
                        dealii::Patterns::Double(1.0e-12));

      prm.add_parameter("Admittance",
                        admittance,
                        "Admittance.",
                        dealii::Patterns::Double(0.0, 1.0));

      prm.add_parameter("EndTime", end_time, "End time.", dealii::Patterns::Double(1.0e-12));

      prm.add_parameter("Radius", radius, "Radius of domain.", dealii::Patterns::Double(1.0e-12));

      prm.add_parameter("AnalyticalSolutionAvailable",
                        analytical_solution_is_available,
                        "We know the analytical solution for Y=1 after the wave left domain?",
                        dealii::Patterns::Bool());
    }
    prm.leave_subsection();
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.formulation     = Formulation::SkewSymmetric;
    this->param.right_hand_side = false;

    // PHYSICAL QUANTITIES
    this->param.start_time     = 0.0;
    this->param.end_time       = end_time;
    this->param.speed_of_sound = speed_of_sound;

    // TEMPORAL DISCRETIZATION
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.cfl                           = 0.25;
    this->param.order_time_integrator         = 2;
    this->param.start_with_low_order          = true;

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
    this->boundary_descriptor->admittance_bc.insert(
      std::make_pair(1, std::make_shared<dealii::Functions::ConstantFunction<dim>>(admittance, 1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution_pressure =
      std::make_shared<InitialConditionPressure<dim>>();

    this->field_functions->initial_solution_velocity =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim);

    this->field_functions->right_hand_side =
      std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.time_control_data.is_active  = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time = this->param.start_time;
    pp_data.output_data.time_control_data.trigger_interval =
      (this->param.end_time - this->param.start_time) / 20.0;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_pressure     = true;
    pp_data.output_data.write_velocity     = true;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree_u;

    // To be able to compute an analytical solution for testing purposes we have to make
    // multiple assumptions. The application is intended to be used with different
    // admittances which contradicts the assumptions. We compute the analytical
    // solution only if we run with Y=1 and after the wave left the domain.
    if(analytical_solution_is_available)
    {
      // To be able to compute an error and thus test the behavior of the admittance BC for
      // Y=1 (first-oder ABC). We also have to ensure the that the wave left the domain
      // when computing the error.
      AssertThrow(std::abs(admittance - 1.0) < 1.0e-12,
                  dealii::ExcMessage("The analytical solution only holds for Y=1."));
      AssertThrow(this->param.end_time > 1.5 * radius / speed_of_sound,
                  dealii::ExcMessage("The end time has to be after the wave left the domain."));

      // calculation of pressure error
      pp_data.error_data_p.time_control_data.is_active        = true;
      pp_data.error_data_p.time_control_data.start_time       = this->param.start_time;
      pp_data.error_data_p.time_control_data.trigger_interval = this->param.end_time;
      pp_data.error_data_p.analytical_solution =
        std::make_shared<dealii::Functions::ZeroFunction<dim>>(1);
      pp_data.error_data_p.calculate_relative_errors = false; // analytical solution is zero
      pp_data.error_data_p.name                      = "pressure";

      // calculation of velocity error
      pp_data.error_data_u.time_control_data.is_active        = true;
      pp_data.error_data_u.time_control_data.start_time       = this->param.start_time;
      pp_data.error_data_u.time_control_data.trigger_interval = this->param.end_time;
      pp_data.error_data_u.analytical_solution =
        std::make_shared<dealii::Functions::ZeroFunction<dim>>(dim);
      pp_data.error_data_u.calculate_relative_errors = false; // analytical solution is zero
      pp_data.error_data_u.name                      = "velocity";
    }

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  // problem specific parameters
  double speed_of_sound = 340.0;
  double admittance     = 1.0; // Y = 1.0 -> first order ABC
  double end_time       = 1.0;
  double radius         = 1.0;

  // analytical solution only known for first oder ABC and after
  // the wave left the domain
  bool analytical_solution_is_available = false;
};

} // namespace Acoustics

} // namespace ExaDG

#include <exadg/acoustic_conservation_equations/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_ACOUSTIC_CONSERVATION_EQUATIONS_TEST_CASES_NORMAL_INCIDENTING_WAVE_H_ \
        */
