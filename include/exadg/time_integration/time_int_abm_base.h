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

#ifndef INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_ABM_BASE_H_
#define INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_ABM_BASE_H_


#include <exadg/time_integration/ab_constants.h>
#include <exadg/time_integration/am_constants.h>
#include <exadg/time_integration/push_back_vectors.h>
#include <exadg/time_integration/time_int_multistep_base.h>
#include <exadg/utilities/print_solver_results.h>

namespace ExaDG
{
/**
 * This class implements the purely explicit Adams--Bashforth--Moulton predictor corrector method.
 */
template<typename Operator, typename VectorType>
class TimeIntAdamsBashforthMoultonBase : public TimeIntMultistepBase
{
  using Number = typename VectorType::value_type;

public:
  TimeIntAdamsBashforthMoultonBase(std::shared_ptr<Operator> pde_operator_in,
                                   double const              start_time_,
                                   double const              end_time_,
                                   unsigned int const        max_number_of_time_steps_,
                                   unsigned int const        order_,
                                   bool const                start_with_low_order_,
                                   bool const                adaptive_time_stepping_,
                                   RestartData const &       restart_data_,
                                   MPI_Comm const &          mpi_comm_,
                                   bool const                is_test_)
    : TimeIntMultistepBase(start_time_,
                           end_time_,
                           max_number_of_time_steps_,
                           order_,
                           start_with_low_order_,
                           adaptive_time_stepping_,
                           restart_data_,
                           mpi_comm_,
                           is_test_),
      pde_operator(pde_operator_in),
      // order of predictor can be chosen one below order of corrector
      ab(order_ - 1, start_with_low_order_),
      am(order_, start_with_low_order_),
      vec_evaluated_operators(order_ - 1)
  {
    AssertThrow(order_ >= 1,
                dealii::ExcMessage("Oder of ABM time integrator has to be at least 1."));
  }

  void
  print_iterations() const
  {
    // explicit time integration -> no iterations
    print_list_of_iterations(pcout, {"Adams-Bashforth-Moulton"}, {0});
  }

  void
  ale_update()
  {
    AssertThrow(false, dealii::ExcMessage("not yet implemented"));
  }

  VectorType const &
  get_solution() const
  {
    return solution;
  }

private:
  void
  update_time_integrator_constants() final
  {
    ab.update(time_step_number, adaptive_time_stepping, time_steps);
    am.update(time_step_number, adaptive_time_stepping, time_steps);
  }

  void
  allocate_vectors() final
  {
    pde_operator->initialize_dof_vector(solution);
    pde_operator->initialize_dof_vector(prediction);

    pde_operator->initialize_dof_vector(evaluated_operator_np);
    for(auto & evaluated_operator : vec_evaluated_operators)
      pde_operator->initialize_dof_vector(evaluated_operator);
  }

  void
  initialize_current_solution() final
  {
    pde_operator->prescribe_initial_conditions(solution, get_time());
  }

  void
  initialize_former_multistep_dof_vectors() final
  {
    if(start_with_low_order)
    {
      if(vec_evaluated_operators.size() > 0)
        pde_operator->evaluate(vec_evaluated_operators[0], solution, get_time());
    }
    else // start with high order
    {
      // fill evaluated operators
      VectorType temp_sol;
      pde_operator->initialize_dof_vector(temp_sol);
      for(unsigned int i = 0; i < vec_evaluated_operators.size(); ++i)
      {
        pde_operator->prescribe_initial_conditions(temp_sol, get_previous_time(i));
        pde_operator->evaluate(vec_evaluated_operators[i], temp_sol, get_previous_time(i));
      }
    }
  }

  void
  setup_derived() final
  {
  }

  void
  do_timestep_predict()
  {
    dealii::Timer timer;
    timer.restart();

    predrict_solution(prediction, solution, vec_evaluated_operators);

    timer_tree->insert({"Timeloop", "Adams-Bashforth-Moulton"}, timer.wall_time());
  }

  void
  do_timestep_correct()
  {
    dealii::Timer timer;
    timer.restart();

    // evaluate operator given the predicted solution
    pde_operator->evaluate(evaluated_operator_np, prediction, get_next_time());
    // correct solution
    correct_solution(solution, evaluated_operator_np, vec_evaluated_operators);
    // correct operator by evaluating operator with correct solution
    pde_operator->evaluate(evaluated_operator_np, solution, get_next_time());

    // write output
    if(this->print_solver_info() and not(this->is_test))
    {
      pcout << std::endl << "Adams-Bashforth-Moulton:";
      print_wall_time(pcout, timer.wall_time());
    }

    timer_tree->insert({"Timeloop", "Adams-Bashforth-Moulton"}, timer.wall_time());
  }

  void
  do_timestep_solve() final
  {
    do_timestep_predict();
    do_timestep_correct();
  }

  void
  correct_solution(VectorType &                    dst,
                   VectorType const &              op_np,
                   std::vector<VectorType> const & ops) const
  {
    dst.add(static_cast<Number>(get_time_step_size() * am.get_gamma0()), op_np);
    for(unsigned int i = 0; i < this->am.get_order() - 1; ++i)
      dst.add(static_cast<Number>(get_time_step_size() * am.get_alpha(i)), ops[i]);
  }

  void
  predrict_solution(VectorType &                    dst,
                    VectorType const &              src,
                    std::vector<VectorType> const & ops) const
  {
    dst = src;
    for(unsigned int i = 0; i < this->ab.get_order(); ++i)
      dst.add(static_cast<Number>(get_time_step_size() * ab.get_alpha(i)), ops[i]);
  }

  void
  prepare_vectors_for_next_timestep() final
  {
    if(vec_evaluated_operators.size() > 0)
    {
      push_back(vec_evaluated_operators);
      std::swap(vec_evaluated_operators[0], evaluated_operator_np);
    }
  }

  void
  read_restart_vectors(boost::archive::binary_iarchive & ia) final
  {
    ia >> solution;
    ia >> prediction;
  }

  void
  write_restart_vectors(boost::archive::binary_oarchive & oa) const final
  {
    oa << solution;
    oa << prediction;
  }

  void
  solve_steady_problem() final
  {
    AssertThrow(false, dealii::ExcMessage("Steady not implemented."));
  }

  // spatial pde operator
  std::shared_ptr<Operator> pde_operator;

  // Time integration constants
  ABTimeIntegratorConstants ab;
  AMTimeIntegratorConstants am;

  // solution vector
  VectorType solution;
  // temporary vector to store prediction
  VectorType prediction;

  // store evaluated operators from previous time steps
  VectorType              evaluated_operator_np;
  std::vector<VectorType> vec_evaluated_operators;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_TIME_INTEGRATION_TIME_INT_ABM_BASE_H_*/
