/*
 * mg_transfer_mf_h.h
 *
 *  Created on: Nov 29, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_TRANSFER_MG_TRANSFER_MF_H_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_TRANSFER_MG_TRANSFER_MF_H_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/multigrid/mg_transfer_matrix_free.h>

#include "../../operators/operator_preconditionable.h"

#include "mg_transfer_mf.h"

using namespace dealii;

// Specialized matrix-free implementation that overloads the copy_to_mg
// function for proper initialization of the vectors in matrix-vector products.
template<int dim, typename Number>
class MGTransferMFH : public MGTransferMatrixFree<dim, Number>,
                      public MGTransferMF<LinearAlgebra::distributed::Vector<Number>>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MGTransferMFH(std::map<unsigned int, unsigned int> level_to_triangulation_level_map,
                const DoFHandler<dim> &              dof_handler)
    : underlying_operator(0),
      level_to_triangulation_level_map(level_to_triangulation_level_map),
      dof_handler(dof_handler)
  {
  }

  virtual ~MGTransferMFH()
  {
  }

  void
  set_operator(
    const MGLevelObject<std::shared_ptr<PreconditionableOperator<dim, Number>>> & operator_in)
  {
    underlying_operator = &operator_in;
  }

  virtual void
  prolongate(unsigned int const to_level, VectorType & dst, VectorType const & src) const
  {
    MGTransferMatrixFree<dim, Number>::prolongate(level_to_triangulation_level_map[to_level],
                                                  dst,
                                                  src);
  }

  virtual void
  restrict_and_add(unsigned int const from_level, VectorType & dst, VectorType const & src) const
  {
    MGTransferMatrixFree<dim, Number>::restrict_and_add(
      level_to_triangulation_level_map[from_level], dst, src);
  }

  virtual void
  interpolate(unsigned int const level_in, VectorType & dst, VectorType const & src) const
  {
    auto & fe    = dof_handler.get_fe();
    auto   level = level_to_triangulation_level_map[level_in];

    auto & src_ghosted = this->ghosted_level_vector[level - 0];
    auto & dst_ghosted = this->ghosted_level_vector[level - 1];

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
          dst_ghosted(dof_indices[i]) = dof_values_coarse[i];
      }

    dst.copy_locally_owned_data_from(dst_ghosted);
  }

  /**
   * Overload copy_to_mg from MGTransferMatrixFree
   */
  template<class InVector, int spacedim>
  void
  copy_to_mg(const DoFHandler<dim, spacedim> & mg_dof,
             MGLevelObject<VectorType> &       dst,
             const InVector &                  src) const
  {
    AssertThrow(underlying_operator != 0, ExcNotInitialized());

    for(unsigned int level = dst.min_level(); level <= dst.max_level(); ++level)
      (*underlying_operator)[level]->initialize_dof_vector(dst[level]);

    MGLevelGlobalTransfer<VectorType>::copy_to_mg(mg_dof, dst, src);
  }

private:
  const MGLevelObject<std::shared_ptr<PreconditionableOperator<dim, Number>>> * underlying_operator;

  // this map converts the multigrid level as used in the V-cycle to an actual
  // level in the triangulation (this is necessary since both numbers might not
  // equal e.g. in the case of hp-MG multiple (p-)levels
  // are on the zeroth triangulation level)
  mutable std::map<unsigned int, unsigned int> level_to_triangulation_level_map;

  const DoFHandler<dim> & dof_handler;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_TRANSFER_MG_TRANSFER_MF_H_H_ */
