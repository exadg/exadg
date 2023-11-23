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

#ifndef EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_TIME_INTEGRATION_TIME_INT_ABM_H_
#define EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_TIME_INTEGRATION_TIME_INT_ABM_H_


#include <exadg/acoustic_conservation_equations/spatial_discretization/interface.h>
#include <exadg/time_integration/time_int_abm_base.h>
#include <exadg/time_integration/time_step_calculation.h>

namespace ExaDG
{
namespace Acoustics
{
template<typename Number>
class TimeIntAdamsBashforthMoulton
  : public TimeIntAdamsBashforthMoultonBase<Interface::SpatialOperator<Number>,
                                            dealii::LinearAlgebra::distributed::BlockVector<Number>>
{
public:
  TimeIntAdamsBashforthMoulton(std::shared_ptr<Interface::SpatialOperator<Number>> pde_operator_in,
                               Parameters const &                                  param_in,
                               std::shared_ptr<PostProcessorInterface<Number>>     postprocessor_in,
                               MPI_Comm const &                                    mpi_comm_in,
                               bool const                                          is_test_in)
    : TimeIntAdamsBashforthMoultonBase<Interface::SpatialOperator<Number>,
                                       dealii::LinearAlgebra::distributed::BlockVector<Number>>(
        pde_operator_in,
        param_in.start_time,
        param_in.end_time,
        param_in.max_number_of_time_steps,
        param_in.order_time_integrator,
        param_in.start_with_low_order,
        param_in.adaptive_time_stepping,
        param_in.restart_data,
        mpi_comm_in,
        is_test_in),
      param(param_in),
      postprocessor(postprocessor_in),
      pcout(std::cout, dealii::Utilities::MPI::this_mpi_process(mpi_comm_in) == 0)
  {
  }

  bool
  print_solver_info() const final
  {
    return param.solver_info_data.write(this->global_timer.wall_time(),
                                        this->time - this->start_time,
                                        this->time_step_number);
  }

private:
  double
  calculate_time_step_size() final
  {
    pcout << std::endl << "Calculation of time step size:" << std::endl << std::endl;

    double time_step = std::numeric_limits<double>::max();

    if(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified)
    {
      time_step = calculate_const_time_step(param.time_step_size, param.n_refine_time);

      print_parameter(pcout, "time step size", time_step);
    }
    else
    {
      AssertThrow(
        false, dealii::ExcMessage("Specified type of time step calculation is not implemented."));
    }

    return time_step;
  }

  double
  recalculate_time_step_size() const final
  {
    AssertThrow(not(param.calculation_of_time_step_size == TimeStepCalculation::UserSpecified),
                dealii::ExcMessage(
                  "Adaptive time step is not implemented for user specified time step size."));

    return {};
  }

  void
  postprocessing() const final
  {
    dealii::Timer timer;
    timer.restart();

    postprocessor->do_postprocessing(this->get_solution(),
                                     this->get_time(),
                                     this->time_step_number);

    this->timer_tree->insert({"Timeloop", "Postprocessing"}, timer.wall_time());
  }

  Parameters const param;

  std::shared_ptr<PostProcessorInterface<Number>> postprocessor;

  dealii::ConditionalOStream pcout;
};

} // namespace Acoustics
} // namespace ExaDG

#endif /* EXADG_ACOUSTIC_CONSERVATION_EQUATIONS_TIME_INTEGRATION_TIME_INT_ABM_H_*/
