/*
 * MultigridPreconditionerDG.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_


#include "multigrid/multigrid_preconditioner_adapter_base.h"

template<int dim, typename value_type, typename Operator, typename UnderlyingOperator>
class MyMultigridPreconditionerDG : public MyMultigridPreconditionerBase<dim,value_type,MatrixOperatorBaseNew<dim, typename Operator::value_type>>
{
public:
  MyMultigridPreconditionerDG() : 
    MyMultigridPreconditionerBase<dim,value_type,OPERATOR_BASE>(std::shared_ptr<OPERATOR_BASE>(new Operator())){}

  const UnderlyingOperator* underlying_operator;
  
  typedef MatrixOperatorBaseNew<dim, typename Operator::value_type> OPERATOR_BASE;
  typedef MyMultigridPreconditionerBase<dim, value_type, OPERATOR_BASE> BASE;
  typedef std::vector<GridTools::PeriodicFacePair<typename
                    Triangulation<dim>::cell_iterator> > VectorPeriodicFacePair;
  
  const VectorPeriodicFacePair* periodic_face_pairs_level0;
    
  virtual ~MyMultigridPreconditionerDG(){};

  virtual void update(MatrixOperatorBase const * /*matrix_operator*/){}

  void initialize(const MultigridData          &mg_data_in,
                  const DoFHandler<dim>        &dof_handler,
                  const Mapping<dim>           &mapping,
                  const UnderlyingOperator     &underlying_operator,
                  const VectorPeriodicFacePair &periodic_face_pairs_level0)
  {
    // save mg-setup
    this->underlying_operator = &  underlying_operator;
    this->periodic_face_pairs_level0 = &periodic_face_pairs_level0;
    
    BASE::initialize(mg_data_in, dof_handler, mapping, 
            (void *)&underlying_operator.get_operator_data());
  }

private:
  void initialize_mg_constrained_dofs(const DoFHandler<dim> &dof_handler,
                                      MGConstrainedDoFs &constrained_dofs)
  {
    // initialize mg_contstrained_dofs
    constrained_dofs.clear();
    constrained_dofs.initialize(dof_handler);
  }
  
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_ */
