/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_

#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/laplace_operator.h"

namespace Poisson
{
/*
 *  Multigrid preconditioner for scalar Laplace operator.
 */
template<int dim, int degree, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  // TODO: remove unnecessary typedefs
  typedef PreconditionableOperator<dim, MultigridNumber> MG_OPERATOR_BASE;

  typedef LaplaceOperator<dim, degree, Number>          PDEOperator;
  typedef LaplaceOperator<dim, degree, MultigridNumber> MultigridOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> BASE;

  typedef typename BASE::VectorType   VectorType;
  typedef typename BASE::VectorTypeMG VectorTypeMG;

  MultigridPreconditioner()
    : MultigridPreconditionerBase<dim, Number, MultigridNumber>(
        std::shared_ptr<MG_OPERATOR_BASE>(new MultigridOperator()))
  {
  }
};

} // namespace Poisson


#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_ */
