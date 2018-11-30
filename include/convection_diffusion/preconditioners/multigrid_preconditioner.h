/*
 * MultigridPreconditionerScalarConvDiff.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_

#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/operators/convection_diffusion_operator.h"

namespace ConvDiff
{
/*
 *  Multigrid preconditioner for scalar (reaction-)convection-diffusion operator.
 */
template<int dim, int degree, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef MultigridOperatorBase<dim, MultigridNumber> MG_OPERATOR_BASE;

  typedef ConvectionDiffusionOperator<dim, degree, Number>          PDEOperator;
  typedef ConvectionDiffusionOperator<dim, degree, MultigridNumber> MultigridOperator;

  MultigridPreconditioner()
    : MultigridPreconditionerBase<dim, Number, MultigridNumber>(
        std::shared_ptr<MG_OPERATOR_BASE>(new MultigridOperator()))
  {
  }

  virtual ~MultigridPreconditioner(){};

  /*
   *  This function updates the multigrid preconditioner.
   */
  virtual void
  update(LinearOperatorBase const * matrix_operator)
  {
    PDEOperator const * pde_operator = dynamic_cast<PDEOperator const *>(matrix_operator);

    AssertThrow(pde_operator != nullptr,
                ExcMessage("PDEOperator and MatrixOperator are not compatible!"));

    update_mg_matrices(pde_operator->get_evaluation_time(),
                       pde_operator->get_scaling_factor_time_derivative_term());
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
  void
  update_mg_matrices(double const & evaluation_time,
                     double const & scaling_factor_time_derivative_term)
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
  void
  set_evaluation_time(double const & evaluation_time)
  {
    for(int level = this->n_global_levels - 1; level >= 0; --level)
    {
      // this->mg_matrices[level] is a std::shared_ptr<MultigridOperatorBase>:
      // so we have to dereference the shared_ptr, get the reference to it and
      // finally we can cast it to pointer of type Operator
      dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])
        ->set_evaluation_time(evaluation_time);
    }
  }

  /*
   *  This function updates scaling_factor_time_derivative_term.
   *  In order to update mg_matrices[level] this function has to be called.
   *  This is necessary if adaptive time stepping is used where
   *  the scaling factor of the derivative term is variable.
   */
  void
  set_scaling_factor_time_derivative_term(double const & scaling_factor_time_derivative_term)
  {
    for(int level = this->n_global_levels - 1; level >= 0; --level)
    {
      // this->mg_matrices[level] is a std::shared_ptr<MultigridOperatorBase>:
      // so we have to dereference the shared_ptr, get the reference to it and
      // finally we can cast it to pointer of type Operator
      dynamic_cast<MultigridOperator *>(&*this->mg_matrices[level])
        ->set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);
    }
  }

  /*
   *  This function updates the smoother for all levels of the multigrid
   *  algorithm.
   *  The prerequisite to call this function is that mg_matrices[level] have
   *  been updated.
   */
  void
  update_smoothers()
  {
    // Start with level = 1!
    for(unsigned int level = 1; level < this->n_global_levels; ++level)
    {
      this->update_smoother(level);
    }
  }
};

} // namespace ConvDiff


#endif /* INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_ */
