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

#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer_mg_level_object.h>

#include <exadg/solvers_and_preconditioners/multigrid/levels_hybrid_multigrid.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer_c.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer_h.h>
#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer_p.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number, typename VectorType>
void
MGTransfer_MGLevelObject<dim, Number, VectorType>::reinit(
  const Mapping<dim> &                                        mapping,
  MGLevelObject<std::shared_ptr<MatrixFree<dim, Number>>> &   mg_matrixfree,
  MGLevelObject<std::shared_ptr<AffineConstraints<Number>>> & mg_constraints,
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &         mg_constrained_dofs,
  unsigned int const                                          dof_handler_index)
{
  std::vector<MGLevelInfo>            global_levels;
  std::vector<MGDoFHandlerIdentifier> p_levels;

  unsigned int const min_level = mg_matrixfree.min_level();
  AssertThrow(min_level == 0, ExcMessage("Currently, we expect min_level==0!"));

  unsigned int const max_level = mg_matrixfree.max_level();
  int const          n_components =
    mg_matrixfree[max_level]->get_dof_handler(dof_handler_index).get_fe().n_components();

  // construct global_levels
  for(unsigned int global_level = min_level; global_level <= max_level; global_level++)
  {
    const auto &       matrixfree = mg_matrixfree[global_level];
    const auto &       fe         = matrixfree->get_dof_handler(dof_handler_index).get_fe();
    const bool         is_dg      = fe.dofs_per_vertex == 0;
    unsigned int const level      = matrixfree->get_mg_level();
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
  mg_level_object.resize(0, global_levels.size() - 1);

  std::map<MGDoFHandlerIdentifier, std::shared_ptr<MGTransferH<dim, Number>>> mg_tranfers_temp;
  std::map<MGDoFHandlerIdentifier, std::map<unsigned int, unsigned int>>
    map_global_level_to_h_levels;

  // initialize maps so that we do not have to check existence later on
  for(auto deg : p_levels)
    map_global_level_to_h_levels[deg] = {};

  // fill the maps
  for(unsigned int i = 0; i < global_levels.size(); i++)
  {
    auto level = global_levels[i];

    map_global_level_to_h_levels[level.dof_handler_id()][i] = level.h_level();
  }

  // create h-transfer operators between levels
  for(auto deg : p_levels)
  {
    if(map_global_level_to_h_levels[deg].size() > 1)
    {
      // create actual h-transfer-operator
      unsigned int global_level = map_global_level_to_h_levels[deg].begin()->first;
      std::shared_ptr<MGTransferH<dim, Number>> transfer(new MGTransferH<dim, Number>(
        map_global_level_to_h_levels[deg],
        mg_matrixfree[global_level]->get_dof_handler(dof_handler_index)));

      // dof-handlers and constrains are saved for global levels
      // so we have to convert degree to any global level which has this degree
      // (these share the same dof-handlers and constraints)
      transfer->initialize_constraints(*mg_constrained_dofs[global_level]);
      transfer->build(mg_matrixfree[global_level]->get_dof_handler(dof_handler_index));
      mg_tranfers_temp[deg] = transfer;
    } // else: there is only one global level (and one h-level) on this p-level
  }

  // fill mg_transfer with the correct transfers
  for(unsigned int i = 1; i < global_levels.size(); i++)
  {
    auto coarse_level = global_levels[i - 1];
    auto fine_level   = global_levels[i];

    std::shared_ptr<MGTransfer<VectorType>> temp;

    if(coarse_level.h_level() != fine_level.h_level()) // h-transfer
    {
      temp =
        mg_tranfers_temp[coarse_level.dof_handler_id()]; // get the previously h-transfer operator
    }
    else if(coarse_level.degree() != fine_level.degree()) // p-transfer
    {
      if(n_components == 1)
      {
        temp.reset(new MGTransferP<dim, Number, VectorType, 1>(&*mg_matrixfree[i],
                                                               &*mg_matrixfree[i - 1],
                                                               fine_level.degree(),
                                                               coarse_level.degree(),
                                                               dof_handler_index));
      }
      else if(n_components == dim)
      {
        temp.reset(new MGTransferP<dim, Number, VectorType, dim>(&*mg_matrixfree[i],
                                                                 &*mg_matrixfree[i - 1],
                                                                 fine_level.degree(),
                                                                 coarse_level.degree(),
                                                                 dof_handler_index));
      }
      else
      {
        AssertThrow(false, ExcMessage("Cannot create MGTransferP!"));
      }
    }
    else if(coarse_level.is_dg() != fine_level.is_dg()) // c-transfer
    {
      if(n_components == 1)
      {
        temp.reset(new MGTransferC<dim, Number, VectorType, 1>(mapping,
                                                               *mg_matrixfree[i],
                                                               *mg_matrixfree[i - 1],
                                                               *mg_constraints[i],
                                                               *mg_constraints[i - 1],
                                                               fine_level.h_level(),
                                                               coarse_level.degree(),
                                                               dof_handler_index));
      }
      else if(n_components == dim)
      {
        temp.reset(new MGTransferC<dim, Number, VectorType, dim>(mapping,
                                                                 *mg_matrixfree[i],
                                                                 *mg_matrixfree[i - 1],
                                                                 *mg_constraints[i],
                                                                 *mg_constraints[i - 1],
                                                                 fine_level.h_level(),
                                                                 coarse_level.degree(),
                                                                 dof_handler_index));
      }
      else
      {
        AssertThrow(false, ExcMessage("Cannot create MGTransferP!"));
      }
    }
    mg_level_object[i] = temp;
  }
}

template<int dim, typename Number, typename VectorType>
void
MGTransfer_MGLevelObject<dim, Number, VectorType>::interpolate(unsigned int const level,
                                                               VectorType &       dst,
                                                               const VectorType & src) const
{
  this->mg_level_object[level]->interpolate(level, dst, src);
}

template<int dim, typename Number, typename VectorType>
void
MGTransfer_MGLevelObject<dim, Number, VectorType>::restrict_and_add(unsigned int const level,
                                                                    VectorType &       dst,
                                                                    const VectorType & src) const
{
  this->mg_level_object[level]->restrict_and_add(level, dst, src);
}

template<int dim, typename Number, typename VectorType>
void
MGTransfer_MGLevelObject<dim, Number, VectorType>::prolongate(unsigned int const level,
                                                              VectorType &       dst,
                                                              const VectorType & src) const
{
  this->mg_level_object[level]->prolongate(level, dst, src);
}

typedef dealii::LinearAlgebra::distributed::Vector<float>  VectorTypeFloat;
typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

template class MGTransfer_MGLevelObject<2, float, VectorTypeFloat>;

template class MGTransfer_MGLevelObject<3, float, VectorTypeFloat>;

template class MGTransfer_MGLevelObject<2, double, VectorTypeDouble>;

template class MGTransfer_MGLevelObject<3, double, VectorTypeDouble>;

} // namespace ExaDG
