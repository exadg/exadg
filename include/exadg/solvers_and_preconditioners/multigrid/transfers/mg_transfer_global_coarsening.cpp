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
template<int dim, typename Number, typename VectorType>
void
MGTransferGlobalCoarsening<dim, Number, VectorType>::reinit(
  dealii::MGLevelObject<std::shared_ptr<dealii::MatrixFree<dim, Number>>> & mg_matrixfree,
  unsigned int const                                                        dof_handler_index,
  bool const                                                                with_global_refinement)
{
  std::vector<MGLevelInfo> global_levels;

  unsigned int const min_level = mg_matrixfree.min_level();
  AssertThrow(min_level == 0, dealii::ExcMessage("Currently, we expect min_level==0!"));

  unsigned int const max_level = mg_matrixfree.max_level();

  // construct global_levels
  for(unsigned int global_level = min_level; global_level <= max_level; global_level++)
  {
    auto const &       matrixfree = mg_matrixfree[global_level];
    auto const &       fe         = matrixfree->get_dof_handler(dof_handler_index).get_fe();
    bool const         is_dg      = fe.dofs_per_vertex == 0;
    unsigned int const level =
      with_global_refinement ? matrixfree->get_mg_level() :
                               matrixfree->get_dof_handler().get_triangulation().n_global_levels();
    unsigned int const degree = fe.degree;

    global_levels.push_back(MGLevelInfo(level, degree, is_dg));
  }

  // create transfer-operator instances
  transfers.resize(0, global_levels.size() - 1);

  // fill mg_transfer with the correct transfers
  for(unsigned int i = 1; i < global_levels.size(); i++)
  {
    auto const coarse_level = global_levels[i - 1];
    auto const fine_level   = global_levels[i];

    if(coarse_level.h_level() != fine_level.h_level()) // h-transfer
    {
      transfers[i].reinit_geometric_transfer(
        mg_matrixfree[i]->get_dof_handler(dof_handler_index),
        mg_matrixfree[i - 1]->get_dof_handler(dof_handler_index),
        mg_matrixfree[i]->get_affine_constraints(dof_handler_index),
        mg_matrixfree[i - 1]->get_affine_constraints(dof_handler_index),
        with_global_refinement ? fine_level.h_level() : dealii::numbers::invalid_unsigned_int,
        with_global_refinement ? coarse_level.h_level() : dealii::numbers::invalid_unsigned_int);
    }
    else if(coarse_level.degree() != fine_level.degree() or // p-transfer
            coarse_level.is_dg() != fine_level.is_dg())     // c-transfer
    {
      transfers[i].reinit_polynomial_transfer(
        mg_matrixfree[i]->get_dof_handler(dof_handler_index),
        mg_matrixfree[i - 1]->get_dof_handler(dof_handler_index),
        mg_matrixfree[i]->get_affine_constraints(dof_handler_index),
        mg_matrixfree[i - 1]->get_affine_constraints(dof_handler_index),
        with_global_refinement ? fine_level.h_level() : dealii::numbers::invalid_unsigned_int,
        with_global_refinement ? coarse_level.h_level() : dealii::numbers::invalid_unsigned_int);
    }
  }

  mg_transfer_global_coarsening =
    std::make_unique<dealii::MGTransferGlobalCoarsening<dim, VectorType>>(
      transfers, [&](const auto l, auto & vec) { mg_matrixfree[l]->initialize_dof_vector(vec); });
}

template<int dim, typename Number, typename VectorType>
void
MGTransferGlobalCoarsening<dim, Number, VectorType>::interpolate(unsigned int const level,
                                                                 VectorType &       dst,
                                                                 VectorType const & src) const
{
  transfers[level].interpolate(dst, src);
}

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
MGTransferGlobalCoarsening<dim, Number, VectorType>::prolongate_and_add(
  unsigned int const level,
  VectorType &       dst,
  VectorType const & src) const
{
  mg_transfer_global_coarsening->prolongate_and_add(level, dst, src);
}

typedef dealii::LinearAlgebra::distributed::Vector<float>  VectorTypeFloat;
typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

template class MGTransferGlobalCoarsening<2, float, VectorTypeFloat>;

template class MGTransferGlobalCoarsening<3, float, VectorTypeFloat>;

template class MGTransferGlobalCoarsening<2, double, VectorTypeDouble>;

template class MGTransferGlobalCoarsening<3, double, VectorTypeDouble>;

} // namespace ExaDG
