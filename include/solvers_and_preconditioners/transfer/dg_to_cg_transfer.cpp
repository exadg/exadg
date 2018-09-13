#include "dg_to_cg_transfer.h"

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/matrix_free/fe_evaluation.h>

template<int dim, typename Number>
CGToDGTransfer<dim, Number>::CGToDGTransfer(const MF &         data_dg,
                                            const MF &         data_cg,
                                            const unsigned int level,
                                            const unsigned int /*fe_degree*/)
  : data_dg(data_dg), data_cg(data_cg), level(level)
{
  // get reference to dof_handlers
  const DoFHandler<dim> & dh1 = data_cg.get_dof_handler();
  const DoFHandler<dim> & dh2 = data_dg.get_dof_handler();

  // get global index (TODO)
  std::vector<types::global_dof_index> dof_indices1(dh1.get_fe().dofs_per_cell);
  std::vector<types::global_dof_index> dof_indices2(dh2.get_fe().dofs_per_cell);

  // check if multigrid
  const bool is_mg = !(level == numbers::invalid_unsigned_int);

  // fill in the indices
  if(is_mg)
  {
    // description: as in the non-mg-case
    auto start1 = dh1.begin_mg(level);
    auto start2 = dh2.begin_mg(level);
    auto end    = dh1.end_mg(level);

    for(auto cell1 = start1, cell2 = start2; cell1 < end; cell1++, cell2++)
      if(cell1->is_locally_owned_on_level())
      {
        cell1->get_mg_dof_indices(dof_indices1);
        for(auto i : dof_indices1)
          dof_indices_cg.push_back(i);
        cell2->get_mg_dof_indices(dof_indices2);
        dof_indices_dg.push_back(dof_indices2[0]);
      }
  }
  else
  {
    // get iterators
    auto start1 = dh1.begin();
    auto start2 = dh2.begin();
    auto end    = dh1.end();

    // loop over all local cells
    for(auto cell1 = start1, cell2 = start2; cell1 < end; cell1++, cell2++)
      if(cell1->is_locally_owned())
      {
        // get indices for CG and ...
        cell1->get_dof_indices(dof_indices1);
        // ... save them
        for(auto i : dof_indices1)
          dof_indices_cg.push_back(i);
        // get indices for DG and ...
        cell2->get_dof_indices(dof_indices2);
        // ... save the first entry (rest not needed: consecutive elements)
        dof_indices_dg.push_back(dof_indices2[0]);
      }
  }
}

template<int dim, typename Number>
CGToDGTransfer<dim, Number>::~CGToDGTransfer()
{
}

template<int dim, typename Number>
void
CGToDGTransfer<dim, Number>::toCG(VectorType & dst, const VectorType & src) const
{
  transfer(dst, src, data_cg, data_dg);
  dst.compress(VectorOperation::add);
}

template<int dim, typename Number>
void
CGToDGTransfer<dim, Number>::toDG(VectorType & dst, const VectorType & src) const
{
  src.update_ghost_values();
  transfer(dst, src, data_dg, data_cg);
}

template<int dim, typename Number>
void
CGToDGTransfer<dim, Number>::transfer(VectorType &       dst,
                                      const VectorType & src,
                                      const MF &         data_dst,
                                      const MF &         data_src) const
{
  // get reference to dof_handlers
  const DoFHandler<dim> & dh1 = data_src.get_dof_handler();

  // get numbering of shape functions
  auto & num_src = data_src.get_shape_info().lexicographic_numbering;
  auto & num_dst = data_dst.get_shape_info().lexicographic_numbering;

  // compute dofs per cell
  const unsigned int delta = dh1.get_fe().dofs_per_cell;

  // do transfer:
  if(dh1.get_fe().dofs_per_vertex == 0)
    // DG -> CG
    for(unsigned int i = 0, k = 0; i < dof_indices_cg.size(); i += delta, k++)
      for(unsigned int j = 0; j < delta; j++)
        dst[dof_indices_cg[i + num_dst[j]]] += src[dof_indices_dg[k] + num_src[j]];
  else
    // CG -> DG
    for(unsigned int i = 0, k = 0; i < dof_indices_cg.size(); i += delta, k++)
      for(unsigned int j = 0; j < delta; j++)
        dst[dof_indices_dg[k] + num_dst[j]] += src[dof_indices_cg[i + num_src[j]]];
}

#include "dg_to_cg_transfer.hpp"
