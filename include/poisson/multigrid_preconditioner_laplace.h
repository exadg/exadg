/*
 * MultigridPreconditionerLaplace.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_LAPLACE_H_
#define INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_LAPLACE_H_


#include "solvers_and_preconditioners/multigrid_preconditioner_adapter_base.h"

template<int dim, typename value_type, typename Operator, typename OperatorData>
class MyMultigridPreconditionerLaplace : public MyMultigridPreconditionerBase<dim,value_type,Operator>
{
public:
  MyMultigridPreconditionerLaplace(){}

  virtual ~MyMultigridPreconditionerLaplace(){}

  virtual void update(MatrixOperatorBase const * /*matrix_operator*/){}

  void initialize(const MultigridData                             &mg_data_in,
                  const DoFHandler<dim>                           &dof_handler,
                  const Mapping<dim>                              &mapping,
                  const OperatorData                              &operator_data_in,
                  std::map<types::boundary_id,
                    std::shared_ptr<Function<dim> > > const &dirichlet_bc)
  {
    this->mg_data = mg_data_in;

    const parallel::Triangulation<dim> *tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation());

    this->n_global_levels = tria->n_global_levels();

    initialize_mg_constrained_dofs(dof_handler,dirichlet_bc);

    initialize_mg_matrices(dof_handler, mapping, operator_data_in);

    this->initialize_smoothers();

    this->initialize_coarse_solver();

    this->initialize_mg_transfer(dof_handler, operator_data_in.periodic_face_pairs_level0);

    this->initialize_multigrid_preconditioner(dof_handler);
  }

private:
  void initialize_mg_constrained_dofs(const DoFHandler<dim>                           &dof_handler,
                                      std::map<types::boundary_id,
                                        std::shared_ptr<Function<dim> > > const &dirichlet_bc)
  {
    // needed for continuous elements
    this->mg_constrained_dofs.clear();
    std::set<types::boundary_id> dirichlet_boundary;
    for(typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::const_iterator
        it = dirichlet_bc.begin(); it != dirichlet_bc.end(); ++it)
      dirichlet_boundary.insert(it->first);
    this->mg_constrained_dofs.initialize(dof_handler);
    this->mg_constrained_dofs.make_zero_boundary_constraints(dof_handler, dirichlet_boundary);
    // needed for continuous elements
  }

  virtual void initialize_mg_matrices(const DoFHandler<dim>   &dof_handler,
                                      const Mapping<dim>      &mapping,
                                      const OperatorData      &operator_data_in)
  {
    // resize
    this->mg_matrices.resize(0, this->n_global_levels-1);

    // initialize mg_matrices on all levels
    for (unsigned int level = 0; level<this->n_global_levels; ++level)
    {
      this->mg_matrices[level].reinit(dof_handler, mapping, operator_data_in, this->mg_constrained_dofs, level);
    }
  }
};


#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_LAPLACE_H_ */
