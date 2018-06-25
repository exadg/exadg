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
class MyMultigridPreconditionerDG : public MyMultigridPreconditionerBase<dim,value_type,Operator>
{
public:
  MyMultigridPreconditionerDG(Operator& o) : 
    MyMultigridPreconditionerBase<dim,value_type,Operator>(o){}

  virtual ~MyMultigridPreconditionerDG(){};

  virtual void update(MatrixOperatorBase const * /*matrix_operator*/){}

  void initialize(const MultigridData                                     &/*mg_data_in*/,
                  const DoFHandler<dim>                                   &/*dof_handler*/,
                  const Mapping<dim>                                      &/*mapping*/,
                  const UnderlyingOperator                                &/*underlying_operator*/,
                  const std::vector<GridTools::PeriodicFacePair<typename
                    Triangulation<dim>::cell_iterator> >                  &/*periodic_face_pairs_level0*/)
  {
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
  }

private:
  void initialize_mg_constrained_dofs(const DoFHandler<dim> &dof_handler,
                                      MGConstrainedDoFs &/*constrained_dofs*/)
  {
    // initialize mg_contstrained_dofs
    this->mg_constrained_dofs.clear();
    this->mg_constrained_dofs.initialize(dof_handler);
  }

  /*
   *  This function initializes mg_matrices on all levels with level >= 0.
   */
  virtual void initialize_mg_matrices(const DoFHandler<dim>                           &dof_handler,
                                      const Mapping<dim>                              &mapping,
                                      const UnderlyingOperator                        &underlying_operator,
                                      const std::vector<GridTools::PeriodicFacePair<
                                        typename Triangulation<dim>::cell_iterator> > &periodic_face_pairs_level0)
  {
    // resize
    this->mg_matrices.resize(0, this->n_global_levels-1);

    // initialize mg_matrices on all levels
    for (unsigned int level = 0; level<this->n_global_levels; ++level)
    {
      this->mg_matrices[level]->initialize_mg_matrix(level, dof_handler, mapping, underlying_operator, periodic_face_pairs_level0);
    }
  }
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_DG_H_ */
