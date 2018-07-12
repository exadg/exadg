/*
 * MultigridPreconditionerScalarConvDiff.h
 *
 *  Created on:
 *      Author: 
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_

#include "../../solvers_and_preconditioners/multigrid_preconditioner_dg.h"

namespace Laplace
{

/*
 *  Multigrid preconditioner for (reaction-)convection-diffusion
 *  operator of the scalar (reaction-)convection-diffusion equation.
 */
template<int dim, typename value_type, typename Operator, typename UnderlyingOperator=Operator>
class MultigridPreconditioner : public MyMultigridPreconditionerDG<dim,value_type,Operator,UnderlyingOperator>
{
public:
  MultigridPreconditioner() {}
  
  virtual ~MultigridPreconditioner(){};
  
};

}


#endif /* INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_ */

