/*
 * restrict_vector_to_coarser_level.h
 *
 *  Created on: Nov 30, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_UTIL_RESTRICT_VECTOR_TO_COARSER_LEVEL_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_UTIL_RESTRICT_VECTOR_TO_COARSER_LEVEL_H_


template<int dim, typename MultigridNumber, typename VectorType>
void
restrict_to_coarser_level(VectorType &            vector_coarse_level,
                          VectorType const &      vector_fine_level,
                          DoFHandler<dim> const & dof_handler_velocity,
                          unsigned int const      level)
{
  unsigned int dofs_per_cell = dof_handler_velocity.get_fe().dofs_per_cell;

  IndexSet relevant_dofs;
  DoFTools::extract_locally_relevant_level_dofs(dof_handler_velocity, level + 1, relevant_dofs);

  VectorType ghosted_vector(dof_handler_velocity.locally_owned_mg_dofs(level + 1),
                            relevant_dofs,
                            MPI_COMM_WORLD);

  ghosted_vector = vector_fine_level;
  ghosted_vector.update_ghost_values();

  Vector<MultigridNumber>              dof_values_fine(dofs_per_cell);
  Vector<MultigridNumber>              tmp(dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices(dofs_per_cell);
  std::vector<MultigridNumber>         dof_values_coarse(dofs_per_cell);

  typename DoFHandler<dim>::cell_iterator cell = dof_handler_velocity.begin(level);
  typename DoFHandler<dim>::cell_iterator endc = dof_handler_velocity.end(level);
  for(; cell != endc; ++cell)
  {
    if(cell->is_locally_owned_on_level())
    {
      Assert(cell->has_children(), ExcNotImplemented());
      std::fill(dof_values_coarse.begin(), dof_values_coarse.end(), 0.);

      for(unsigned int child = 0; child < cell->n_children(); ++child)
      {
        cell->child(child)->get_mg_dof_indices(dof_indices);
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          dof_values_fine(i) = ghosted_vector(dof_indices[i]);

        dof_handler_velocity.get_fe()
          .get_restriction_matrix(child, cell->refinement_case())
          .vmult(tmp, dof_values_fine);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
        {
          if(dof_handler_velocity.get_fe().restriction_is_additive(i)) // discontinuous case
            dof_values_coarse[i] += tmp[i];
          else if(tmp[i] != 0.) // continuous case
            dof_values_coarse[i] = tmp[i];
        }
      }
      cell->get_mg_dof_indices(dof_indices);
      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        vector_coarse_level(dof_indices[i]) = dof_values_coarse[i];
    }
  }
  vector_coarse_level.compress(VectorOperation::insert); // continuous case
}


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_UTIL_RESTRICT_VECTOR_TO_COARSER_LEVEL_H_ */
