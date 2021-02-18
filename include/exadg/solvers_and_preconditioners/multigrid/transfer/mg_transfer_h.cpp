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
#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer_h.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
MGTransferH<dim, Number>::MGTransferH(
  std::map<unsigned int, unsigned int> level_to_triangulation_level_map,
  DoFHandler<dim> const &              dof_handler)
  : underlying_operator(0),
    level_to_triangulation_level_map(level_to_triangulation_level_map),
    dof_handler(dof_handler)
{
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::set_operator(
  const MGLevelObject<std::shared_ptr<MultigridOperatorBase<dim, Number>>> & operator_in)
{
  underlying_operator = &operator_in;
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::prolongate(unsigned int const to_level,
                                     VectorType &       dst,
                                     VectorType const & src) const
{
  MGTransferMatrixFree<dim, Number>::prolongate(level_to_triangulation_level_map[to_level],
                                                dst,
                                                src);
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::restrict_and_add(unsigned int const from_level,
                                           VectorType &       dst,
                                           VectorType const & src) const
{
  MGTransferMatrixFree<dim, Number>::restrict_and_add(level_to_triangulation_level_map[from_level],
                                                      dst,
                                                      src);
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::interpolate(unsigned int const level_in,
                                      VectorType &       dst,
                                      VectorType const & src) const
{
  auto & fe    = dof_handler.get_fe();
  auto   level = level_to_triangulation_level_map[level_in];

  LinearAlgebra::distributed::Vector<Number> src_ghosted;
  IndexSet                                   relevant_dofs;
  DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
  src_ghosted.reinit(dof_handler.locally_owned_mg_dofs(level),
                     relevant_dofs,
                     src.get_mpi_communicator());
  src_ghosted.copy_locally_owned_data_from(src);
  src_ghosted.update_ghost_values();

  std::vector<Number>                     dof_values_coarse(fe.dofs_per_cell);
  Vector<Number>                          dof_values_fine(fe.dofs_per_cell);
  Vector<Number>                          tmp(fe.dofs_per_cell);
  std::vector<types::global_dof_index>    dof_indices(fe.dofs_per_cell);
  typename DoFHandler<dim>::cell_iterator cell = dof_handler.begin(level - 1);
  typename DoFHandler<dim>::cell_iterator endc = dof_handler.end(level - 1);
  for(; cell != endc; ++cell)
    if(cell->is_locally_owned_on_level())
    {
      Assert(cell->has_children(), ExcNotImplemented());
      std::fill(dof_values_coarse.begin(), dof_values_coarse.end(), 0.);
      for(unsigned int child = 0; child < cell->n_children(); ++child)
      {
        cell->child(child)->get_mg_dof_indices(dof_indices);

        for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          dof_values_fine(i) = src_ghosted(dof_indices[i]);
        fe.get_restriction_matrix(child, cell->refinement_case()).vmult(tmp, dof_values_fine);
        for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
          if(fe.restriction_is_additive(i))
            dof_values_coarse[i] += tmp[i];
          else if(tmp(i) != 0.)
            dof_values_coarse[i] = tmp[i];
      }
      cell->get_mg_dof_indices(dof_indices);
      for(unsigned int i = 0; i < fe.dofs_per_cell; ++i)
        dst(dof_indices[i]) = dof_values_coarse[i];
    }

  dst.zero_out_ghosts();
}

template<int dim, typename Number>
void
MGTransferH<dim, Number>::copy_to_mg(const DoFHandler<dim, dim> & mg_dof,
                                     MGLevelObject<VectorType> &  dst,
                                     VectorType const &           src) const
{
  AssertThrow(underlying_operator != 0, ExcNotInitialized());

  for(unsigned int level = dst.min_level(); level <= dst.max_level(); ++level)
    (*underlying_operator)[level]->initialize_dof_vector(dst[level]);

  MGLevelGlobalTransfer<VectorType>::copy_to_mg(mg_dof, dst, src);
}

template class MGTransferH<2, float>;
template class MGTransferH<3, float>;

template class MGTransferH<2, double>;
template class MGTransferH<3, double>;

} // namespace ExaDG
