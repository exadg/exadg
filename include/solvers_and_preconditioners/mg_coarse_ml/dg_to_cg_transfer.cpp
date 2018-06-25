#include "dg_to_cg_transfer.h"

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/matrix_free/fe_evaluation.h>

template <int dim, typename Number>
CGToDGTransfer<dim, Number>::CGToDGTransfer(const MF &data_1, const MF &data_2,
                                            const int level,
                                            const int fe_degree)
    : data_1(data_1), data_2(data_2), level(level),
      temp(std::pow(fe_degree + 1, dim)) {}

template <int dim, typename Number>
CGToDGTransfer<dim, Number>::~CGToDGTransfer() {}

template <int dim, typename Number>
void CGToDGTransfer<dim, Number>::toCG(VNumber &dst, const VNumber &src) const {

  transfer(dst, src, data_2, data_1);
  dst.compress(VectorOperation::add);
}

template <int dim, typename Number>
void CGToDGTransfer<dim, Number>::toDG(VNumber &dst, const VNumber &src) const {

  src.update_ghost_values();
  transfer(dst, src, data_1, data_2);
}

template <int dim, typename Number>
void CGToDGTransfer<dim, Number>::transfer(VNumber &dst, const VNumber &src,
                                           const MF &data_dst,
                                           const MF &data_src) const {

  const DoFHandler<dim> &dh1 = data_src.get_dof_handler();
  const DoFHandler<dim> &dh2 = data_dst.get_dof_handler();
  for (auto cell1 = dh1.begin_mg(level), cell2 = dh2.begin_mg(level);
       cell1 < dh1.end_mg(level); cell1++, cell2++)
    if (cell1->is_locally_owned()) {
      cell1->get_dof_values(src, temp);
      cell2->distribute_local_to_global(temp, dst);
    }
}

#include "dg_to_cg_transfer.hpp"
