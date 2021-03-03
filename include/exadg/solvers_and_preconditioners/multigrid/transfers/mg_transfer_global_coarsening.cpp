/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfers/mg_transfer_global_coarsening.h>


namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number, typename VectorType>
void
MGTransferGlobalCoarsening<dim, Number, VectorType>::restrict_and_add(unsigned int const level,
                                                                      VectorType &       dst,
                                                                      VectorType const & src) const
{
  mg_transfer_global_coarsening->restrict_and_add(level, dst, src);
}

template<int dim, typename Number, typename VectorType>
void
MGTransferGlobalCoarsening<dim, Number, VectorType>::prolongate(unsigned int const level,
                                                                VectorType &       dst,
                                                                VectorType const & src) const
{
  mg_transfer_global_coarsening->prolongate(level, dst, src);
}

template<int dim, typename Number, typename VectorType>
void
MGTransferGlobalCoarsening<dim, Number, VectorType>::reinit(
  MGLevelObject<std::shared_ptr<MatrixFree<dim, Number>>> &   mg_matrixfree,
  MGLevelObject<std::shared_ptr<AffineConstraints<Number>>> & mg_constraints,
  unsigned int const                                          dof_handler_index)
{
  std::vector<MGLevelInfo>            global_levels;
  std::vector<MGDoFHandlerIdentifier> p_levels;

  unsigned int const min_level = mg_matrixfree.min_level();
  AssertThrow(min_level == 0, ExcMessage("Currently, we expect min_level==0!"));

  unsigned int const max_level = mg_matrixfree.max_level();

  // construct global_levels
  for(unsigned int global_level = min_level; global_level <= max_level; global_level++)
  {
    auto const &       matrixfree = mg_matrixfree[global_level];
    auto const &       fe         = matrixfree->get_dof_handler(dof_handler_index).get_fe();
    bool const         is_dg      = fe.dofs_per_vertex == 0;
    unsigned int const level = matrixfree->get_dof_handler().get_triangulation().n_global_levels();
    unsigned int const degree =
      (int)round(std::pow(fe.n_dofs_per_cell() / fe.n_components(), 1.0 / dim)) - 1;

    global_levels.push_back(MGLevelInfo(level, degree, is_dg));
  }

  // construct and p_levels
  for(auto i : global_levels)
    p_levels.push_back(i.dof_handler_id());

  sort(p_levels.begin(), p_levels.end());
  p_levels.erase(unique(p_levels.begin(), p_levels.end()), p_levels.end());
  std::reverse(std::begin(p_levels), std::end(p_levels));

  // create transfer-operator instances
  transfers.resize(0, global_levels.size() - 1);

  // fill mg_transfer with the correct transfers
  for(unsigned int i = 1; i < global_levels.size(); i++)
  {
    auto coarse_level = global_levels[i - 1];
    auto fine_level   = global_levels[i];

    if(coarse_level.h_level() != fine_level.h_level()) // h-transfer
    {
      transfers[i].reinit_geometric_transfer(mg_matrixfree[i]->get_dof_handler(dof_handler_index),
                                             mg_matrixfree[i - 1]->get_dof_handler(
                                               dof_handler_index),
                                             *mg_constraints[i],
                                             *mg_constraints[i - 1]);
    }
    else if(coarse_level.degree() != fine_level.degree() || // p-transfer
            coarse_level.is_dg() != fine_level.is_dg())     // c-transfer
    {
      transfers[i].reinit_polynomial_transfer(mg_matrixfree[i]->get_dof_handler(dof_handler_index),
                                              mg_matrixfree[i - 1]->get_dof_handler(
                                                dof_handler_index),
                                              *mg_constraints[i],
                                              *mg_constraints[i - 1]);
    }
    else
    {
      AssertThrow(false, ExcMessage("Cannot create MGTransfer!"));
    }
  }

  mg_transfer_global_coarsening =
    std::make_unique<dealii::MGTransferGlobalCoarsening<dim, VectorType>>(transfers);
}

template<int dim, typename Number, typename VectorType>
void
MGTransferGlobalCoarsening<dim, Number, VectorType>::interpolate(unsigned int const level,
                                                                 VectorType &       dst,
                                                                 VectorType const & src) const
{
  (void)level;
  (void)dst;
  (void)src;
  AssertThrow(false, ExcNotImplemented());
}

typedef dealii::LinearAlgebra::distributed::Vector<float>  VectorTypeFloat;
typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

template class MGTransferGlobalCoarsening<2, float, VectorTypeFloat>;

template class MGTransferGlobalCoarsening<3, float, VectorTypeFloat>;

template class MGTransferGlobalCoarsening<2, double, VectorTypeDouble>;

template class MGTransferGlobalCoarsening<3, double, VectorTypeDouble>;

} // namespace ExaDG
