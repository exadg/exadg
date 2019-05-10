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
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef PreconditionableOperator<dim, MultigridNumber> MG_OPERATOR_BASE;

  typedef LaplaceOperator<dim, MultigridNumber> MultigridOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> BASE;
  typedef typename BASE::Map                                        Map;

  MultigridPreconditioner()
    : MultigridPreconditionerBase<dim, Number, MultigridNumber>(
        std::shared_ptr<MG_OPERATOR_BASE>(new MultigridOperator()))
  {
  }

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
    auto operator_data       = operator_data_in;
    operator_data.dof_index  = 0;
    operator_data.quad_index = 0;

    BASE::initialize(mg_data, tria, fe, mapping, operator_data, dirichlet_bc, periodic_face_pairs);
  }
};

} // namespace Poisson


#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_ */
