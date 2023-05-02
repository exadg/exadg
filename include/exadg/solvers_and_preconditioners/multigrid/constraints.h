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

#ifndef OPERATOR_CONSTRAINTS_H
#define OPERATOR_CONSTRAINTS_H

#include <deal.II/multigrid/mg_constrained_dofs.h>

namespace ExaDG
{
namespace ConstraintUtil
{
namespace // anonymous namespace
{
template<int dim, typename Number>
void
add_periodicity_constraints(unsigned int const                                    level,
                            unsigned int const                                    target_level,
                            typename dealii::DoFHandler<dim>::face_iterator const face1,
                            typename dealii::DoFHandler<dim>::face_iterator const face2,
                            dealii::AffineConstraints<Number> &                   constraints)
{
  if(level == 0)
  {
    // level of interest has been reached
    unsigned int const dofs_per_face = face1->get_fe(0).dofs_per_face;

    std::vector<dealii::types::global_dof_index> dofs_1(dofs_per_face);
    std::vector<dealii::types::global_dof_index> dofs_2(dofs_per_face);

    face1->get_mg_dof_indices(target_level, dofs_1, 0);
    face2->get_mg_dof_indices(target_level, dofs_2, 0);

    for(unsigned int i = 0; i < dofs_per_face; ++i)
    {
      if(constraints.can_store_line(dofs_2[i]) and constraints.can_store_line(dofs_1[i]) and
         not constraints.is_constrained(dofs_2[i]))
      {
        // constraint dof and ...
        constraints.add_line(dofs_2[i]);
        // specify type of constraint: equality (dof_2[i]=dof_1[j]*1.0)
        constraints.add_entry(dofs_2[i], dofs_1[i], 1.);
      }
    }
  }
  else if(face1->has_children() and face2->has_children())
  {
    // recursively visit all subfaces
    for(unsigned int c = 0; c < face1->n_children(); ++c)
    {
      add_periodicity_constraints<dim>(
        level - 1, target_level, face1->child(c), face2->child(c), constraints);
    }
  }
}


template<int dim, typename Number>
void
add_periodicity_constraints(
  dealii::DoFHandler<dim> const &                                dof_handler,
  unsigned int const                                             level,
  std::vector<typename dealii::GridTools::PeriodicFacePair<
    typename dealii::Triangulation<dim>::cell_iterator>> const & periodic_face_pairs_level0,
  dealii::AffineConstraints<Number> &                            affine_constraints_own)
{
  // loop over all periodic face pairs of level 0
  for(auto & it : periodic_face_pairs_level0)
  {
    // get reference to the cells on level 0 sharing the periodic face
    typename dealii::DoFHandler<dim>::cell_iterator cell1(&dof_handler.get_triangulation(),
                                                          0,
                                                          it.cell[1]->index(),
                                                          &dof_handler);
    typename dealii::DoFHandler<dim>::cell_iterator cell0(&dof_handler.get_triangulation(),
                                                          0,
                                                          it.cell[0]->index(),
                                                          &dof_handler);

    // get reference to periodic faces on level and add recursively their
    // subfaces on the given level
    add_periodicity_constraints<dim, Number>(level,
                                             level,
                                             cell1->face(it.face_idx[1]),
                                             cell0->face(it.face_idx[0]),
                                             affine_constraints_own);
  }
}

template<int dim, typename Number>
void
add_constraints(bool                                is_dg,
                bool                                operator_is_singular,
                dealii::DoFHandler<dim> const &     dof_handler,
                dealii::AffineConstraints<Number> & affine_constraints_own,
                dealii::MGConstrainedDoFs const &   mg_constrained_dofs,
                std::vector<dealii::GridTools::PeriodicFacePair<
                  typename dealii::Triangulation<dim>::cell_iterator>> const & periodic_face_pairs,
                unsigned int const                                             level)
{
  if(is_dg)
  {
    // for DG: nothing to do
    affine_constraints_own.close();
    return;
  }
  // 0) clear old content (to be on the safe side)
  affine_constraints_own.clear();

  // ... and set local dofs
  dealii::IndexSet relevant_dofs;
  if(level != dealii::numbers::invalid_unsigned_int)
    dealii::DoFTools::extract_locally_relevant_level_dofs(dof_handler, level, relevant_dofs);
  else
    dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, relevant_dofs);
  affine_constraints_own.reinit(relevant_dofs);

  // 1) add periodic BCs
  add_periodicity_constraints<dim, Number>(dof_handler,
                                           level,
                                           periodic_face_pairs,
                                           affine_constraints_own);

  // 2) add Dirichlet BCs
  if(mg_constrained_dofs.have_boundary_indices())
    affine_constraints_own.add_lines(mg_constrained_dofs.get_boundary_indices(level));

  // constrain zeroth DoF in continuous case (the mean value constraint will
  // be applied in the DG case). In case we have interface matrices, there are
  // Dirichlet constraints on parts of the boundary and no such transformation
  // is required.
  if(operator_is_singular and affine_constraints_own.can_store_line(0))
  {
    // if dof 0 is constrained, it must be a periodic dof, so we take the
    // value on the other side
    dealii::types::global_dof_index line_index = 0;
    while(true)
    {
      auto const * lines = affine_constraints_own.get_constraint_entries(line_index);

      if(lines == 0)
      {
        affine_constraints_own.add_line(line_index);
        // add the constraint back to the dealii::MGConstrainedDoFs field. This
        // is potentially dangerous but we know what we are doing... ;-)
        if(mg_constrained_dofs.have_boundary_indices() &&
           level != dealii::numbers::invalid_unsigned_int)
        {
          if(mg_constrained_dofs.get_boundary_indices(level).size() != dof_handler.n_dofs(level))
            const_cast<dealii::IndexSet &>(mg_constrained_dofs.get_boundary_indices(level))
              .set_size(dof_handler.n_dofs(level));
          const_cast<dealii::IndexSet &>(mg_constrained_dofs.get_boundary_indices(level))
            .add_index(line_index);
        }

        break;
      }
      else
      {
        Assert(lines->size() == 1 and std::abs((*lines)[0].second - 1.) < 1e-15,
               dealii::ExcMessage("Periodic index expected, bailing out"));

        line_index = (*lines)[0].first;
      }
    }
  }

  affine_constraints_own.close();
}


} // namespace
} // namespace ConstraintUtil
} // namespace ExaDG

#endif
