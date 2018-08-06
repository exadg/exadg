/*
 * MultigridPreconditionerDG.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_

#include "multigrid_preconditioner_adapter_base.h"

template<int dim, typename value_type, typename Operator, typename UnderlyingOperator>
class MyMultigridPreconditionerDG
  : public MyMultigridPreconditionerBase<dim,
                                         value_type,
                                         MatrixOperatorBaseNew<dim, typename Operator::value_type>>
{
public:
  MyMultigridPreconditionerDG()
    : MyMultigridPreconditionerBase<dim, value_type, OPERATOR_BASE>(
        std::shared_ptr<OPERATOR_BASE>(new Operator()))
  {
  }

  typedef MatrixOperatorBaseNew<dim, typename Operator::value_type>     OPERATOR_BASE;
  typedef MyMultigridPreconditionerBase<dim, value_type, OPERATOR_BASE> BASE;
  typedef std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    VectorPeriodicFacePair;

  virtual ~MyMultigridPreconditionerDG(){};

  void
  initialize(const MultigridData &      mg_data_in,
             const DoFHandler<dim> &    dof_handler,
             const Mapping<dim> &       mapping,
             const UnderlyingOperator & underlying_operator,
             const VectorPeriodicFacePair & /*periodic_face_pairs_level0*/)
  {
    // empty map for DG
    auto & dirichlet_bc = underlying_operator.get_operator_data().bc->dirichlet_bc;

    BASE::initialize(
      mg_data_in, dof_handler, mapping, dirichlet_bc, (void *)&underlying_operator.get_operator_data());
  }

  void
  initialize(const MultigridData &                                                mg_data_in,
             const DoFHandler<dim> &                                              dof_handler,
             const Mapping<dim> &                                                 mapping,
             std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc,
             void *                                                               operator_data_in)
  {
    BASE::initialize(mg_data_in, dof_handler, mapping, dirichlet_bc, operator_data_in);
  }
};

#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_ */
