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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_POSTPROCESSOR_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_POSTPROCESSOR_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics_homogeneous.h>
#include <exadg/incompressible_navier_stokes/postprocessor/mean_velocity_calculator.h>

#include "flow_rate_controller.h"

namespace ExaDG
{
namespace IncNS
{
template<int dim>
struct MyPostProcessorData
{
  PostProcessorData<dim>          pp_data;
  MeanVelocityCalculatorData<dim> mean_velocity_data;
  LinePlotDataStatistics<dim>     line_plot_data;
};

template<int dim, typename Number>
class MyPostProcessor : public PostProcessor<dim, Number>
{
public:
  typedef PostProcessor<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef typename Base::Operator Operator;

  MyPostProcessor(MyPostProcessorData<dim> const & my_pp_data_in,
                  MPI_Comm const &                 mpi_comm,
                  double const                     length_in,
                  FlowRateController &             flow_rate_controller_in)
    : Base(my_pp_data_in.pp_data, mpi_comm),
      my_pp_data(my_pp_data_in),
      length(length_in),
      flow_rate_controller(flow_rate_controller_in)
  {
  }

  void
  setup(Operator const & pde_operator)
  {
    // call setup function of base class
    Base::setup(pde_operator);

    // calculation of mean velocity
    mean_velocity_calculator.reset(
      new MeanVelocityCalculator<dim, Number>(pde_operator.get_matrix_free(),
                                              pde_operator.get_dof_index_velocity(),
                                              pde_operator.get_quad_index_velocity_linear(),
                                              my_pp_data.mean_velocity_data,
                                              this->mpi_comm));

    // evaluation of characteristic quantities along lines
    line_plot_calculator_statistics.reset(
      new LinePlotCalculatorStatisticsHomogeneous<dim, Number>(pde_operator.get_dof_handler_u(),
                                                               pde_operator.get_dof_handler_p(),
                                                               *pde_operator.get_mapping(),
                                                               this->mpi_comm));

    line_plot_calculator_statistics->setup(my_pp_data.line_plot_data);
  }

  void
  do_postprocessing(VectorType const &     velocity,
                    VectorType const &     pressure,
                    double const           time,
                    types::time_step const time_step_number)
  {
    Base::do_postprocessing(velocity, pressure, time, time_step_number);

    if(my_pp_data.mean_velocity_data.calculate == true)
    {
      // calculation of flow rate
      double const flow_rate =
        mean_velocity_calculator->calculate_flow_rate_volume(velocity, time, length);

      // update body force
      flow_rate_controller.update_body_force(flow_rate, time, time_step_number);
    }

    // line plot statistics
    if(line_plot_calculator_statistics->time_control_statistics.time_control.needs_evaluation(
         time, time_step_number))
    {
      line_plot_calculator_statistics->evaluate(velocity, pressure);
    }

    if(line_plot_calculator_statistics->time_control_statistics.write_preliminary_results(
         time, time_step_number))
    {
      line_plot_calculator_statistics->write_output();
    }
  }

private:
  // postprocessor data supplemented with data required for periodic hill
  MyPostProcessorData<dim> my_pp_data;

  // calculate flow rate in precursor domain so that the flow rate can be
  // dynamically adjusted by a flow rate controller.
  std::shared_ptr<MeanVelocityCalculator<dim, Number>> mean_velocity_calculator;

  double const length;

  FlowRateController & flow_rate_controller;

  // line plot statistics with averaging in homogeneous direction
  std::shared_ptr<LinePlotCalculatorStatisticsHomogeneous<dim, Number>>
    line_plot_calculator_statistics;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_PERIODIC_HILL_POSTPROCESSOR_H_ */
