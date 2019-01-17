/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_

#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "compatible_laplace_operator.h"

namespace IncNS
{
/*
 *  Multigrid preconditioner for compatible Laplace operator.
 */
template<int dim, int degree_u, int degree_p, typename Number, typename MultigridNumber>
class CompatibleLaplaceMultigridPreconditioner
  : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  // TODO: remove unnecessary typedefs
  typedef PreconditionableOperator<dim, MultigridNumber> MG_OPERATOR_BASE;

  typedef CompatibleLaplaceOperator<dim, degree_u, degree_p, Number>          PDEOperator;
  typedef CompatibleLaplaceOperator<dim, degree_u, degree_p, MultigridNumber> MultigridOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> BASE;
  typedef typename BASE::Map                                        Map;

  typedef typename BASE::VectorType   VectorType;
  typedef typename BASE::VectorTypeMG VectorTypeMG;

  CompatibleLaplaceMultigridPreconditioner()
    : MultigridPreconditionerBase<dim, Number, MultigridNumber>(
        std::shared_ptr<MG_OPERATOR_BASE>(new MultigridOperator()))
  {
  }

  void
  initialize(MultigridData const &                      mg_data,
             const parallel::Triangulation<dim> *       tria,
             const FiniteElement<dim> &                 fe,
             Mapping<dim> const &                       mapping,
             CompatibleLaplaceOperatorData<dim> const & operator_data_in,
             Map const *                                dirichlet_bc = nullptr,
             std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> *
               periodic_face_pairs = nullptr)
  {
    auto operator_data               = operator_data_in;
    operator_data.dof_index_velocity = 0;
    operator_data.dof_index_pressure = 1;

    BASE::initialize(mg_data, tria, fe, mapping, operator_data, dirichlet_bc, periodic_face_pairs);
  }
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_ \
        */
