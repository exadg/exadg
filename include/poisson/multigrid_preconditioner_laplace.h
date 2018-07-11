/*
 * MultigridPreconditionerLaplace.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_LAPLACE_H_
#define INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_LAPLACE_H_

#include "../solvers_and_preconditioners/multigrid/multigrid_preconditioner_adapter_base.h"

template <int dim, typename value_type, typename Operator, typename OperatorData>
class MyMultigridPreconditionerLaplace
    : public MyMultigridPreconditionerBase<dim, value_type, MatrixOperatorBaseNew<dim, typename Operator::value_type>> {
public:
  MyMultigridPreconditionerLaplace()
      : BASE(std::shared_ptr<MatrixOperatorBaseNew<dim, typename Operator::value_type>>(new Operator())) {}

  typedef MatrixOperatorBaseNew<dim, typename Operator::value_type> OPERATOR_BASE;
  typedef MyMultigridPreconditionerBase<dim, value_type, OPERATOR_BASE> BASE;
  
  
  virtual ~MyMultigridPreconditionerLaplace() {}

  virtual void update(MatrixOperatorBase const * /*matrix_operator*/) {}

  void initialize(const MultigridData &mg_data_in,
                  const DoFHandler<dim> &dof_handler, 
                  const Mapping<dim> &mapping,
                  const OperatorData &operator_data_in,
                  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const &dirichlet_bc) {

      
      
      
    BASE::initialize(mg_data_in, dof_handler, mapping, 
            (void *)operator_data_in, dirichlet_bc);
  }
  
};

#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_LAPLACE_H_ */
