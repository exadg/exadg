/*
 * CheckMultigrid.h
 *
 *  Created on: Oct 4, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CHECKMULTIGRID_H_
#define INCLUDE_CHECKMULTIGRID_H_

#include <deal.II/numerics/data_out.h>

template<int dim, typename value_type, typename Operator, typename Preconditioner>
class CheckMultigrid
{
public:
  CheckMultigrid(Operator const                        &underlying_operator_in,
                 std_cxx11::shared_ptr<Preconditioner> preconditioner_in)
    :
    underlying_operator(underlying_operator_in),
    preconditioner(preconditioner_in)
  {}

  /*
   *  Function that verifies the multgrid algorithm,
   *  especially the smoothing (on the finest level) an
   *  the whole multigrid cycle applied to a random solution vector
   *
   *
   *  Richardson iteration: x^{k+1} = x^{k} + M^{-1} * ( b - A*x^{k} )
   *
   *  A: system matrix
   *  M^{-1}: preconditioner (should approximate A^{-1})
   *
   *  rhs vector b: b = 0 (-> x_ex = 0)
   *  initial guess x^{0} = rand()
   *
   *  --> calculate x^{1} = x^{0} - M^{-1}*A*x^{0}
   *  --> If multigrid cycle works well, x^{1} should be small (entries < 0.1)
   */
  void check()
  {
    /*
     *  Whole MG Cycle
     */
    parallel::distributed::Vector<value_type> initial_solution;
    underlying_operator.initialize_dof_vector(initial_solution);
    parallel::distributed::Vector<value_type> solution_after_mg_cylce(initial_solution), tmp(initial_solution);

    for (unsigned int i=0; i<initial_solution.size(); ++i)
      initial_solution(i) = (double)rand()/RAND_MAX;

    underlying_operator.vmult(tmp, initial_solution);
    tmp *= -1.0;
    preconditioner->vmult(solution_after_mg_cylce, tmp);
    solution_after_mg_cylce += initial_solution;

    /*
     *  Smoothing
     */
    typedef float Number;
    parallel::distributed::Vector<Number> initial_solution_float;
    initial_solution_float = initial_solution;
    parallel::distributed::Vector<Number> solution_after_smoothing, tmp_float;
    solution_after_smoothing = initial_solution;
    tmp_float = tmp;

    preconditioner->apply_smoother_on_fine_level(solution_after_smoothing,tmp_float);
    solution_after_smoothing += initial_solution_float;

    /*
     *  Output
     */
    DataOut<dim> data_out;
    unsigned int dof_index = underlying_operator.get_dof_index();
    data_out.attach_dof_handler (underlying_operator.get_data().get_dof_handler(dof_index));

    std::vector<std::string> initial (dim, "initial");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      initial_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (underlying_operator.get_data().get_dof_handler(dof_index),initial_solution, initial, initial_component_interpretation);

    std::vector<std::string> mg_cycle (dim, "mg_cycle");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      mg_cylce_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (underlying_operator.get_data().get_dof_handler(dof_index),solution_after_mg_cylce, mg_cycle, mg_cylce_component_interpretation);

    std::vector<std::string> smoother (dim, "smoother");
    std::vector<DataComponentInterpretation::DataComponentInterpretation>
      smoother_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (underlying_operator.get_data().get_dof_handler(dof_index),solution_after_smoothing, smoother, smoother_component_interpretation);

    data_out.build_patches (underlying_operator.get_data().get_dof_handler(dof_index).get_fe().degree*3);
    std::ostringstream filename;
    filename << "smoothing.vtk";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtk(output);

    /*
     *  Terminate simulation
     */
//    std::abort();
  }

private:
  Operator const &underlying_operator;
  std_cxx11::shared_ptr<Preconditioner> preconditioner;
};


#endif /* INCLUDE_CHECKMULTIGRID_H_ */
