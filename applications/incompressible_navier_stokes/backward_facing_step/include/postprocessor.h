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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_POSTPROCESSOR_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_POSTPROCESSOR_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/inflow_data_calculator.h>
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics_homogeneous.h>
#include <exadg/postprocessor/statistics_manager.h>

// backward facing step application
#include "geometry.h"

namespace ExaDG
{
namespace IncNS
{
template<int dim>
struct PostProcessorDataBFS
{
  PostProcessorData<dim>      pp_data;
  TurbulentChannelData        turb_ch_data;
  InflowData<dim>             inflow_data;
  LinePlotDataStatistics<dim> line_plot_data;
};

template<int dim, typename Number>
class PostProcessorBFS : public PostProcessor<dim, Number>
{
public:
  typedef PostProcessor<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef typename Base::Operator Operator;

  PostProcessorBFS(PostProcessorDataBFS<dim> const & pp_data_bfs_in, MPI_Comm const & mpi_comm)
    : Base(pp_data_bfs_in.pp_data, mpi_comm),
      write_final_output(true),
      write_final_output_lines(true),
      pp_data_bfs(pp_data_bfs_in)
  {
  }

  void
  setup(Operator const & pde_operator) final
  {
    // call setup function of base class
    Base::setup(pde_operator);

    // turbulent channel statistics for precursor simulation
    if(pp_data_bfs.turb_ch_data.time_control_data_statistics.time_control_data.is_active)
    {
      statistics_turb_ch.reset(new StatisticsManager<dim, Number>(pde_operator.get_dof_handler_u(),
                                                                  *pde_operator.get_mapping()));

      statistics_turb_ch->setup(&Geometry::grid_transform_turb_channel, pp_data_bfs.turb_ch_data);
    }

    // inflow data
    if(pp_data_bfs.inflow_data.write_inflow_data)
    {
      inflow_data_calculator.reset(
        new InflowDataCalculator<dim, Number>(pp_data_bfs.inflow_data, this->mpi_comm));
      inflow_data_calculator->setup(pde_operator.get_dof_handler_u(), *pde_operator.get_mapping());
    }

    // evaluation of characteristic quantities along lines
    if(pp_data_bfs.line_plot_data.time_control_data_statistics.time_control_data.is_active)
    {
      line_plot_calculator_statistics.reset(
        new LinePlotCalculatorStatisticsHomogeneous<dim, Number>(pde_operator.get_dof_handler_u(),
                                                                 pde_operator.get_dof_handler_p(),
                                                                 *pde_operator.get_mapping(),
                                                                 this->mpi_comm));

      line_plot_calculator_statistics->setup(pp_data_bfs.line_plot_data);
    }
  }

  void
  do_postprocessing(VectorType const &     velocity,
                    VectorType const &     pressure,
                    double const           time,
                    types::time_step const time_step_number) final
  {
    Base::do_postprocessing(velocity, pressure, time, time_step_number);


    // turbulent channel statistics
    if(statistics_turb_ch)
    {
      if(statistics_turb_ch->time_control_statistics.time_control.needs_evaluation(
           time, time_step_number))
      {
        statistics_turb_ch->evaluate(velocity, Utilities::is_unsteady_timestep(time_step_number));
      }

      if(statistics_turb_ch->time_control_statistics.write_preliminary_results(time,
                                                                               time_step_number))
      {
        statistics_turb_ch->write_output();
      }
    }

    // inflow data
    if(pp_data_bfs.inflow_data.write_inflow_data)
      inflow_data_calculator->calculate(velocity);

    // line plot statistics
    if(line_plot_calculator_statistics)
    {
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
  }

  bool                                               write_final_output;
  bool                                               write_final_output_lines;
  PostProcessorDataBFS<dim>                          pp_data_bfs;
  std::shared_ptr<StatisticsManager<dim, Number>>    statistics_turb_ch;
  std::shared_ptr<InflowDataCalculator<dim, Number>> inflow_data_calculator;
  std::shared_ptr<LinePlotCalculatorStatisticsHomogeneous<dim, Number>>
    line_plot_calculator_statistics;
};

} // namespace IncNS
} // namespace ExaDG


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_POSTPROCESSOR_H_ \
        */
