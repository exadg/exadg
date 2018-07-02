/*
 * MultigridPreconditionerScalarConvDiff.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_

#include "solvers_and_preconditioners/multigrid_preconditioner_dg.h"

namespace ConvDiff
{

/*
 *  Multigrid preconditioner for (reaction-)convection-diffusion
 *  operator of the scalar (reaction-)convection-diffusion equation.
 */
template<int dim, typename value_type, typename Operator, typename UnderlyingOperator>
class MultigridPreconditioner : public MyMultigridPreconditionerDG<dim,value_type,Operator,UnderlyingOperator>
{
public:
  MultigridPreconditioner() {}

  virtual ~MultigridPreconditioner(){};

  /*
   *  This function updates the multigrid preconditioner.
   */
  virtual void update(MatrixOperatorBase const * matrix_operator)
  {
    UnderlyingOperator const *underlying_operator =
        dynamic_cast<UnderlyingOperator const *>(matrix_operator);

    AssertThrow(underlying_operator != nullptr,
        ExcMessage("Multigrid preconditioner: UnderlyingOperator and MatrixOperator are not compatible!"));

    update_mg_matrices(underlying_operator->get_evaluation_time(),
                       underlying_operator->get_scaling_factor_time_derivative_term());
    update_smoothers();
    this->update_coarse_solver();
  }

private:
  /*
   *  This function updates mg_matrices
   *  To do this, three functions are called:
   *   - set_evaluation_time
   *   - set_scaling_factor_time_derivative_term
   */
  void update_mg_matrices(double const &evaluation_time,
                          double const &scaling_factor_time_derivative_term)
  {
    set_evaluation_time(evaluation_time);
    set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);
  }

  /*
   *  This function updates the evaluation time.
   *  In order to update mg_matrices[level] this function has to be called.
   *  (This is due to the fact that the velocity field of the convective term
   *  is a function of the time.)
   */
  void set_evaluation_time(double const &evaluation_time)
  {
    for (int level = this->n_global_levels-1; level>=0; --level)
    {
      this->mg_matrices[level]->set_evaluation_time(evaluation_time);
    }
  }

  /*
   *  This function updates scaling_factor_time_derivative_term.
   *  In order to update mg_matrices[level] this function has to be called.
   *  This is necessary if adaptive time stepping is used where
   *  the scaling factor of the derivative term is variable.
   */
  void set_scaling_factor_time_derivative_term(double const &scaling_factor_time_derivative_term)
  {
    for (int level = this->n_global_levels-1; level>=0; --level)
    {
      this->mg_matrices[level]->set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);
    }
  }

  /*
   *  This function updates the smoother for all levels of the multigrid
   *  algorithm.
   *  The prerequisite to call this function is that mg_matrices[level] have
   *  been updated.
   */
  void update_smoothers()
  {
    // Start with level = 1!
    for (unsigned int level = 1; level<this->n_global_levels; ++level)
    {
      this->update_smoother(level);
    }
  }
};

}


#endif /* INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_ */
