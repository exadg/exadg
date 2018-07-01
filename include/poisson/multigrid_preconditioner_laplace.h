/*
 * MultigridPreconditionerLaplace.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_LAPLACE_H_
#define INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_LAPLACE_H_

#include "../solvers_and_preconditioners/multigrid/multigrid_preconditioner_adapter_base.h"

template <int dim, typename value_type, typename Operator,
          typename OperatorData>
class MyMultigridPreconditionerLaplace
    : public MyMultigridPreconditionerBase<dim, value_type, MatrixOperatorBaseNew<dim, typename Operator::value_type>> {
public:
  MyMultigridPreconditionerLaplace()
      : BASE(std::shared_ptr<MatrixOperatorBaseNew<dim, typename Operator::value_type>>(new Operator())) {}

  const Mapping<dim> *mapping;
  const OperatorData *operator_data_in;
  std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const
      *dirichlet_bc;

  typedef MyMultigridPreconditionerBase<dim, value_type, MatrixOperatorBaseNew<dim, typename Operator::value_type>> BASE;

  virtual ~MyMultigridPreconditionerLaplace() {}

  virtual void update(MatrixOperatorBase const * /*matrix_operator*/) {}

  void
  initialize(const MultigridData &mg_data_in,
             const DoFHandler<dim> &dof_handler, const Mapping<dim> &mapping,
             const OperatorData &operator_data_in,
             std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const
                 &dirichlet_bc) {
    // save mg-setup
    this->mapping = &mapping;
    this->operator_data_in = &operator_data_in;
    this->dirichlet_bc = &dirichlet_bc;

    BASE::initialize(mg_data_in, dof_handler);
  }

public:
  void initialize_mg_constrained_dofs(const DoFHandler<dim> &dof_handler,
                                      MGConstrainedDoFs &constrained_dofs) {
    std::set<types::boundary_id> dirichlet_boundary;
    for (auto& it : *dirichlet_bc)
      dirichlet_boundary.insert(it.first);
    constrained_dofs.initialize(dof_handler);
    constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                    dirichlet_boundary);
  }

  void initialize_mg_matrix(
      const DoFHandler<dim> &dof_handler,
      MatrixOperatorBaseNew<dim, typename Operator::value_type> *matrix,
      int level, int tria_level) {

    matrix->reinit(dof_handler, *mapping, (void *)operator_data_in,
                   level == -1 ? *this->cg_constrained_dofs_local
                               : *this->mg_constrained_dofs_local[level],
                   tria_level);
  }
};

#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_LAPLACE_H_ */
