#ifndef OPERATOR_CONSTRAINTS_H
#define OPERATOR_CONSTRAINTS_H

namespace ConstraintUtil
{
namespace // anonymous namespace
{
template<int dim>
void
add_periodicity_constraints(unsigned int const                            level,
                            unsigned int const                            target_level,
                            typename DoFHandler<dim>::face_iterator const face1,
                            typename DoFHandler<dim>::face_iterator const face2,
                            AffineConstraints<double> &                   constraints)
{
  if(level == 0)
  {
    // level of interest has been reached
    unsigned int const dofs_per_face = face1->get_fe(0).dofs_per_face;

    std::vector<types::global_dof_index> dofs_1(dofs_per_face);
    std::vector<types::global_dof_index> dofs_2(dofs_per_face);

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    face1->get_mg_dof_indices(target_level, dofs_1, 0);
    face2->get_mg_dof_indices(target_level, dofs_2, 0);
#pragma GCC diagnostic pop

    for(unsigned int i = 0; i < dofs_per_face; ++i)
    {
      if(constraints.can_store_line(dofs_2[i]) && constraints.can_store_line(dofs_1[i]) &&
         !constraints.is_constrained(dofs_2[i]))
      {
        // constraint dof and ...
        constraints.add_line(dofs_2[i]);
        // specify type of constraint: equality (dof_2[i]=dof_1[j]*1.0)
        constraints.add_entry(dofs_2[i], dofs_1[i], 1.);
      }
    }
  }
  else if(face1->has_children() && face2->has_children())
  {
    // recursively visit all subfaces
    for(unsigned int c = 0; c < face1->n_children(); ++c)
    {
      add_periodicity_constraints<dim>(
        level - 1, target_level, face1->child(c), face2->child(c), constraints);
    }
  }
}


template<int dim>
void
add_periodicity_constraints(
  DoFHandler<dim> const & dof_handler,
  unsigned int const      level,
  std::vector<typename GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                              periodic_face_pairs_level0,
  AffineConstraints<double> & constraint_own)
{
  // loop over all periodic face pairs of level 0
  for(auto & it : periodic_face_pairs_level0)
  {
    // get reference to the cells on level 0 sharing the periodic face
    typename DoFHandler<dim>::cell_iterator cell1(&dof_handler.get_triangulation(),
                                                  0,
                                                  it.cell[1]->index(),
                                                  &dof_handler);
    typename DoFHandler<dim>::cell_iterator cell0(&dof_handler.get_triangulation(),
                                                  0,
                                                  it.cell[0]->index(),
                                                  &dof_handler);

    // get reference to periodic faces on level and add recursively their
    // subfaces on the given level
    add_periodicity_constraints<dim>(
      level, level, cell1->face(it.face_idx[1]), cell0->face(it.face_idx[0]), constraint_own);
  }
}

template<int dim>
void
add_constraints(
  bool                        is_dg,
  bool                        operator_is_singular,
  DoFHandler<dim> const &     dof_handler,
  AffineConstraints<double> & constraint_own,
  MGConstrainedDoFs const &   mg_constrained_dofs,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                     periodic_face_pairs,
  unsigned int const level)
{
  if(is_dg)
  {
    // for DG: nothing to do
    constraint_own.close();
    return;
  }
  // 0) clear old content (to be on the safe side)
  constraint_own.clear();

  // 1) add periodic BCs
  add_periodicity_constraints<dim>(dof_handler, level, periodic_face_pairs, constraint_own);

  // 2) add Dirichlet BCs
  constraint_own.add_lines(mg_constrained_dofs.get_boundary_indices(level));

  // constrain zeroth DoF in continuous case (the mean value constraint will
  // be applied in the DG case). In case we have interface matrices, there are
  // Dirichlet constraints on parts of the boundary and no such transformation
  // is required.
  if(operator_is_singular && constraint_own.can_store_line(0))
  {
    // if dof 0 is constrained, it must be a periodic dof, so we take the
    // value on the other side
    types::global_dof_index line_index = 0;
    while(true)
    {
      std::vector<std::pair<types::global_dof_index, double>> const * lines =
        constraint_own.get_constraint_entries(line_index);

      if(lines == 0)
      {
        constraint_own.add_line(line_index);
        // add the constraint back to the MGConstrainedDoFs field. This
        // is potentially dangerous but we know what we are doing... ;-)
        if(level != numbers::invalid_unsigned_int)
        {
          const_cast<IndexSet &>(mg_constrained_dofs.get_boundary_indices(level))
            .add_index(line_index);
        }

        break;
      }
      else
      {
        Assert(lines->size() == 1 && std::abs((*lines)[0].second - 1.) < 1e-15,
               ExcMessage("Periodic index expected, bailing out"));

        line_index = (*lines)[0].first;
      }
    }
  }
  constraint_own.close();
}


} // namespace
} // namespace ConstraintUtil

#endif
