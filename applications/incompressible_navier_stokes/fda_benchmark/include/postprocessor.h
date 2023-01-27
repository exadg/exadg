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

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_POSTPROCESSOR_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_POSTPROCESSOR_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/postprocessor/inflow_data_calculator.h>
#include <exadg/incompressible_navier_stokes/postprocessor/line_plot_calculation_statistics.h>
#include <exadg/incompressible_navier_stokes/postprocessor/mean_velocity_calculator.h>

#include "inflow_data_storage.h"

namespace ExaDG
{
namespace IncNS
{
template<int dim>
struct PostProcessorDataFDA
{
  PostProcessorData<dim>          pp_data;
  InflowData<dim>                 inflow_data;
  MeanVelocityCalculatorData<dim> mean_velocity_data;
  LinePlotDataStatistics<dim>     line_plot_data;
};

template<int dim, typename Number>
class PostProcessorFDA : public PostProcessor<dim, Number>
{
public:
  typedef PostProcessor<dim, Number> Base;

  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef typename Base::Operator Operator;

  PostProcessorFDA(PostProcessorDataFDA<dim> const & pp_data_in,
                   MPI_Comm const &                  mpi_comm_in,
                   double const                      area_in,
                   FlowRateController &              flow_rate_controller_in,
                   InflowDataStorage<dim> &          inflow_data_storage_in,
                   bool const                        use_precursor_in,
                   bool const                        add_random_perturbations_in)
    : Base(pp_data_in.pp_data, mpi_comm_in),
      mpi_comm(mpi_comm_in),
      pp_data_fda(pp_data_in),
      area(area_in),
      flow_rate_controller(flow_rate_controller_in),
      inflow_data_storage(inflow_data_storage_in),
      use_precursor(use_precursor_in),
      add_random_perturbations(add_random_perturbations_in)
  {
  }

  void
  setup(Operator const & pde_operator)
  {
    // call setup function of base class
    Base::setup(pde_operator);

    if(use_precursor)
    {
      // inflow data
      inflow_data_calculator.reset(
        new InflowDataCalculator<dim, Number>(pp_data_fda.inflow_data, mpi_comm));
      inflow_data_calculator->setup(pde_operator.get_dof_handler_u(), *pde_operator.get_mapping());

      // calculation of mean velocity
      mean_velocity_calculator.reset(
        new MeanVelocityCalculator<dim, Number>(pde_operator.get_matrix_free(),
                                                pde_operator.get_dof_index_velocity(),
                                                pde_operator.get_quad_index_velocity_linear(),
                                                pp_data_fda.mean_velocity_data,
                                                this->mpi_comm));
    }

    // evaluation of results along lines
    if(pp_data_fda.line_plot_data.time_control_data_statistics.time_control_data.is_active)
    {
      line_plot_calculator_statistics.reset(
        new LinePlotCalculatorStatistics<dim, Number>(pde_operator.get_dof_handler_u(),
                                                      pde_operator.get_dof_handler_p(),
                                                      *pde_operator.get_mapping(),
                                                      this->mpi_comm));

      line_plot_calculator_statistics->setup(pp_data_fda.line_plot_data);
    }
  }

  void
  do_postprocessing(VectorType const &     velocity,
                    VectorType const &     pressure,
                    double const           time,
                    types::time_step const time_step_number)
  {
    Base::do_postprocessing(velocity, pressure, time, time_step_number);

    if(use_precursor)
    {
      // inflow data
      inflow_data_calculator->calculate(velocity);

      // random perturbations
      if(add_random_perturbations)
        inflow_data_storage.add_random_perturbations();
    }
    else // laminar inflow profile
    {
      // in case of random perturbations, the velocity field at the inflow boundary
      // has to be recomputed after each time step
      if(add_random_perturbations)
        inflow_data_storage.initialize_velocity_values();
    }

    if(pp_data_fda.mean_velocity_data.calculate == true)
    {
      // calculation of flow rate
      double const flow_rate =
        area * mean_velocity_calculator->calculate_mean_velocity_volume(velocity, time);

      // update body force
      flow_rate_controller.update_body_force(flow_rate, time);
    }

    // evaluation of results along lines
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

private:
  MPI_Comm const mpi_comm;

  // postprocessor data supplemented with data required for FDA benchmark
  PostProcessorDataFDA<dim> pp_data_fda;

  // calculate flow rate in precursor domain so that the flow rate can be
  // dynamically adjusted by a flow rate controller.
  std::shared_ptr<MeanVelocityCalculator<dim, Number>> mean_velocity_calculator;

  double const area;

  FlowRateController & flow_rate_controller;

  // interpolate velocity field to a predefined set of interpolation points
  std::shared_ptr<InflowDataCalculator<dim, Number>> inflow_data_calculator;

  InflowDataStorage<dim> & inflow_data_storage;

  bool const use_precursor;
  bool const add_random_perturbations;

  // evaluation of results along lines
  std::shared_ptr<LinePlotCalculatorStatistics<dim, Number>> line_plot_calculator_statistics;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FDA_POSTPROCESSOR_H_ */
