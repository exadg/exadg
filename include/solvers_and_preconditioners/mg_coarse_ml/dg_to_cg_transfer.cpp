#include "dg_to_cg_transfer.h"

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/matrix_free/fe_evaluation.h>

template <int dim, typename Number>
CGToDGTransfer<dim, Number>::CGToDGTransfer(const MF &data_dg, const MF &data_cg,
                                            const unsigned int level,
                                            const unsigned int fe_degree)
    : data_dg(data_dg), data_cg(data_cg), level(level),
      temp_src(std::pow(fe_degree + 1, dim)),
      temp_dst(std::pow(fe_degree + 1, dim)) {}

template <int dim, typename Number>
CGToDGTransfer<dim, Number>::~CGToDGTransfer() {}

template <int dim, typename Number>
void CGToDGTransfer<dim, Number>::toCG(VNumber &dst, const VNumber &src) const {

  transfer(dst, src, data_cg, data_dg);
  dst.compress(VectorOperation::add);
}

template <int dim, typename Number>
void CGToDGTransfer<dim, Number>::toDG(VNumber &dst, const VNumber &src) const {

  src.update_ghost_values();
  transfer(dst, src, data_dg, data_cg);
}

template <int dim, typename Number>
void CGToDGTransfer<dim, Number>::transfer(VNumber &dst, const VNumber &src,
                                           const MF &data_dst,
                                           const MF &data_src) const {

  // get reference to dof_handlers
  const DoFHandler<dim> &dh1 = data_src.get_dof_handler();
  const DoFHandler<dim> &dh2 = data_dst.get_dof_handler();

  // get numbering of shape functions
  auto &num_src = data_src.get_shape_info().lexicographic_numbering;
  auto &num_dst = data_dst.get_shape_info().lexicographic_numbering;

  // check if multi grid
  const bool is_mg = !(level == numbers::invalid_unsigned_int);

  if (is_mg) {

    // get global index (TODO)
    std::vector<types::global_dof_index> dof_indices1(
        dh1.get_fe().dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices2(
        dh2.get_fe().dofs_per_cell);

    // get iterator
    auto start1 = dh1.begin_mg(level);
    auto start2 = dh2.begin_mg(level);
    auto end = dh1.end_mg(level);

    // loop over all local cells
    for (auto cell1 = start1, cell2 = start2; cell1 < end; cell1++, cell2++)
      if (cell1->is_locally_owned_on_level()) {

        // gather values (TODO: any alternatives?)
        cell1->get_mg_dof_indices(dof_indices1);
        for (unsigned int i = 0; i < dof_indices1.size(); i++)
          temp_src[i] = src[dof_indices1[i]];

        // bring dof_values into the right order
        // (needed: numbering of shape functions of fe_q and fe_dgq different)
        for (unsigned int j = 0; j < temp_src.size(); j++)
          temp_dst[num_dst[j]] = temp_src[num_src[j]];

        // scatter values (TODO: any alternatives?)
        cell2->get_mg_dof_indices(dof_indices2);
        for (unsigned int i = 0; i < dof_indices2.size(); i++)
          dst[dof_indices2[i]] += temp_dst[i];
      }
  } else {
    // get iterator
    auto start1 = dh1.begin();
    auto start2 = dh2.begin();
    auto end = dh1.end();

    // loop over all local cells
    for (auto cell1 = start1, cell2 = start2; cell1 < end; cell1++, cell2++)
      if (cell1->is_locally_owned()) {
        // gather values
        cell1->get_dof_values(src, temp_src);

        // bring dof_values into the right order
        // (needed: numbering of shape functions of fe_q and fe_dgq different)
        for (unsigned int j = 0; j < temp_src.size(); j++)
          temp_dst[num_dst[j]] = temp_src[num_src[j]];
        
        // scatter values
        cell2->distribute_local_to_global(temp_dst, dst);
      }
  }
}

#include "dg_to_cg_transfer.hpp"
