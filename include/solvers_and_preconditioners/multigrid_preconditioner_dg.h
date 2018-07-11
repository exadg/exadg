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

  const Mapping<dim> *mapping;
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
    this->mapping = &mapping;
    this->underlying_operator = &  underlying_operator;
    this->periodic_face_pairs_level0 = &periodic_face_pairs_level0;
      
//    this->mg_data = mg_data_in;
//
//    const parallel::Triangulation<dim> *tria =
//      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());
//
//    this->n_global_levels = tria->n_global_levels();
//
//    initialize_mg_constrained_dofs(dof_handler);
//
//    initialize_mg_matrices(dof_handler, mapping, underlying_operator, periodic_face_pairs_level0);
//
//    this->initialize_smoothers();
//
//    this->initialize_coarse_solver();
//
//    this->initialize_mg_transfer(dof_handler, periodic_face_pairs_level0);
//
//    this->initialize_multigrid_preconditioner(dof_handler);
      BASE::initialize(mg_data_in, dof_handler);
  }

private:
  void initialize_mg_constrained_dofs(const DoFHandler<dim> &dof_handler,
                                      MGConstrainedDoFs &constrained_dofs)
  {
    // initialize mg_contstrained_dofs
    constrained_dofs.clear();
    constrained_dofs.initialize(dof_handler);
  }

  /*
   *  This function initializes mg_matrices on all levels with level >= 0.
   */
  virtual void initialize_mg_matrix(const DoFHandler<dim> &dof_handler,
        OPERATOR_BASE * matrix, int level, int tria_level) {
      
      matrix->reinit(dof_handler, *mapping, (void *)&this->underlying_operator->get_operator_data(),
                   level == -1 ? *this->cg_constrained_dofs_local
                               : *this->mg_constrained_dofs_local[level],
                   tria_level);

  }
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_ */
