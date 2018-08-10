/*
 * MultigridPreconditionerDG.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_

#include "multigrid_preconditioner_adapter_base.h"

template<int dim, typename value_type, typename Operator>
class MyMultigridPreconditionerDG
  : public MyMultigridPreconditionerBase<dim,
                                         value_type,
                                         MultigridOperatorBase<dim, typename Operator::value_type>>
{
public:
  MyMultigridPreconditionerDG()
    : MyMultigridPreconditionerBase<dim, value_type, OPERATOR_BASE>(
        std::shared_ptr<OPERATOR_BASE>(new Operator()))
  {
  }

  typedef MultigridOperatorBase<dim, typename Operator::value_type>     OPERATOR_BASE;
  typedef MyMultigridPreconditionerBase<dim, value_type, OPERATOR_BASE> BASE;
};

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_ */
