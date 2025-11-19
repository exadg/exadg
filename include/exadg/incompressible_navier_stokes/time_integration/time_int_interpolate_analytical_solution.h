/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2024 by the ExaDG authors
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

#ifndef EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_INTERPOLATE_ANALYTICAL_SOLUTION_H_
#define EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_INTERPOLATE_ANALYTICAL_SOLUTION_H_

// ExaDG
#include <exadg/incompressible_navier_stokes/time_integration/time_int_bdf.h>
#include <exadg/time_integration/vector_handling.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
namespace IncNS
{
/**
 * A class that sets the analytical solution for the next time instead of computing it.
 * To be able to compute temporal derivatives etc. this class is based on TimeIntBDF.
 */
template<int dim, typename Number>
class TimeIntInterpolateAnalyticalSolution : public TimeIntBDF<dim, Number>
{
  using Base       = TimeIntBDF<dim, Number>;
  using VectorType = typename Base::VectorType;
  using Operator   = SpatialOperatorBase<dim, Number>;

public:
  TimeIntInterpolateAnalyticalSolution(
    std::shared_ptr<Operator>                       operator_in,
    std::shared_ptr<HelpersALE<dim, Number> const>  helpers_ale_in,
    std::shared_ptr<PostProcessorInterface<Number>> postprocessor_in,
    Parameters const &                              param_in,
    MPI_Comm const &                                mpi_comm_in,
    bool const                                      is_test_in)
    : Base(operator_in, helpers_ale_in, postprocessor_in, param_in, mpi_comm_in, is_test_in),
      velocity(this->order),
      pressure(this->order)
  {
  }

  void
  print_iterations() const final
  {
    std::vector<std::string> names = {"Interpolation analytical solution"};
    std::vector<double>      iterations_avg{0.0};
    print_list_of_iterations(this->pcout, names, iterations_avg);
  }

  VectorType const &
  get_velocity() const final
  {
    return velocity[0];
  }


  VectorType const &
  get_pressure() const final
  {
    return pressure[0];
  }

  VectorType const &
  get_velocity_np() const final
  {
    return velocity_np;
  }
  VectorType const &
  get_pressure_np() const final
  {
    return pressure_np;
  }

private:
  VectorType const &
  get_velocity(unsigned int i /* t_{n-i} */) const final
  {
    return velocity[i];
  }


  VectorType const &
  get_pressure(unsigned int i /* t_{n-i} */) const final
  {
    return pressure[i];
  }

  void
  allocate_vectors() final
  {
    Base::allocate_vectors();

    // velocity
    for(unsigned int i = 0; i < velocity.size(); ++i)
      this->operator_base->initialize_vector_velocity(velocity[i]);
    this->operator_base->initialize_vector_velocity(velocity_np);

    // pressure
    for(unsigned int i = 0; i < pressure.size(); ++i)
      this->operator_base->initialize_vector_pressure(pressure[i]);
    this->operator_base->initialize_vector_pressure(pressure_np);
  }


  /**
   * This function simply sets the analytical
   */
  void
  do_timestep_solve() final
  {
    this->operator_base->interpolate_analytical_solution(velocity_np,
                                                         pressure_np,
                                                         this->get_next_time());
  }

  void
  prepare_vectors_for_next_timestep() final
  {
    Base::prepare_vectors_for_next_timestep();

    swap_back_one_step(velocity);
    velocity[0].swap(velocity_np);

    swap_back_one_step(pressure);
    pressure[0].swap(pressure_np);
  }

  void
  solve_steady_problem() final
  {
    AssertThrow(false, dealii::ExcMessage("TimeIntAnalytic not implemented for steady problem."));
  }


  void
  initialize_current_solution() final
  {
    if(this->param.ale_formulation)
      this->helpers_ale->move_grid(this->get_time());

    this->operator_base->interpolate_analytical_solution(velocity[0],
                                                         pressure[0],
                                                         this->get_time());
  }

  void
  initialize_former_multistep_dof_vectors() final
  {
    // note that the loop begins with i=1! (we could also start with i=0 but this is not necessary)
    for(unsigned int i = 1; i < velocity.size(); ++i)
    {
      if(this->param.ale_formulation)
        this->helpers_ale->move_grid(this->get_previous_time(i));

      this->operator_base->interpolate_analytical_solution(velocity[i],
                                                           pressure[i],
                                                           this->get_previous_time(i));
    }
  }

  void
  set_velocity(VectorType const & velocity_in, unsigned int const i /* t_{n-i} */) final
  {
    velocity[i] = velocity_in;
  }

  void
  set_pressure(VectorType const & pressure_in, unsigned int const i /* t_{n-i} */) final
  {
    pressure[i] = pressure_in;
  }

  std::vector<VectorType> velocity;
  std::vector<VectorType> pressure;
  VectorType              velocity_np;
  VectorType              pressure_np;
};

} // namespace IncNS
} // namespace ExaDG

#endif /* EXADG_INCOMPRESSIBLE_NAVIER_STOKES_TIME_INTEGRATION_TIME_INT_INTERPOLATE_ANALYTICAL_SOLUTION_H_ \
        */
