/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_

#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/laplace_operator.h"

namespace Poisson
{
/*
 *  Multigrid preconditioner for scalar Laplace operator.
 */
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef LaplaceOperator<dim, MultigridNumber> Laplace;

  typedef MultigridOperatorBase<dim, MultigridNumber>      MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, Laplace> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> BASE;
  typedef typename BASE::Map                                        Map;

  void
  initialize(MultigridData const &                mg_data,
             const parallel::Triangulation<dim> * tria,
             const FiniteElement<dim> &           fe,
             Mapping<dim> const &                 mapping,
             LaplaceOperatorData<dim> const &     operator_data_in,
             Map const *                          dirichlet_bc = nullptr,
             std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> *
               periodic_face_pairs = nullptr)
  {
    operator_data            = operator_data_in;
    operator_data.dof_index  = 0;
    operator_data.quad_index = 0;

    BASE::initialize(mg_data,
                     tria,
                     fe,
                     mapping,
                     operator_data.operator_is_singular,
                     dirichlet_bc,
                     periodic_face_pairs);
  }

  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  initialize_matrix_free(unsigned int const level, Mapping<dim> const & mapping)
  {
    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);

    // setup MatrixFree::AdditionalData
    typename MatrixFree<dim, MultigridNumber>::AdditionalData additional_data;
    additional_data.level_mg_handler     = this->global_levels[level].level;
    additional_data.mapping_update_flags = operator_data.get_mapping_update_flags();

    if(this->global_levels[level].is_dg)
    {
      additional_data.mapping_update_flags_inner_faces =
        operator_data.get_mapping_update_flags_inner_faces();
      additional_data.mapping_update_flags_boundary_faces =
        operator_data.get_mapping_update_flags_boundary_faces();
    }

    if(operator_data.do_use_cell_based_loops() && this->global_levels[level].is_dg)
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &this->mg_dofhandler[level]->get_triangulation());
      Categorization::do_cell_based_loops(*tria, additional_data, this->global_levels[level].level);
    }

    QGauss<1> const quad(this->global_levels[level].degree + 1);
    matrix_free->reinit(
      mapping, *this->mg_dofhandler[level], *this->mg_constraints[level], quad, additional_data);

    return matrix_free;
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<Laplace> pde_operator(new Laplace());
    pde_operator->reinit_multigrid(*this->mg_matrixfree[level],
                                   *this->mg_constraints[level],
                                   operator_data);

    // initialize MGOperator which is a wrapper around the PDEOperator
    std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

    return mg_operator;
  }

  LaplaceOperatorData<dim> operator_data;
};

} // namespace Poisson


#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_ */
