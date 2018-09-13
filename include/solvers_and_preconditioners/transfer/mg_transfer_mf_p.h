#ifndef MG_TRANSFER_MF_P
#define MG_TRANSFER_MF_P

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_base.h>

using namespace dealii;

template<int dim, int fe_degree_1, int fe_degree_2, typename Number, typename VectorType>
class MGTransferMatrixFreeP : virtual public MGTransferBase<VectorType>
{
public:
  typedef Number value_type;

  MGTransferMatrixFreeP();

  MGTransferMatrixFreeP(const DoFHandler<dim> & dof_handler_1,
                        const DoFHandler<dim> & dof_handler_2,
                        const unsigned int      level);

  void
  reinit(const DoFHandler<dim> & dof_handler_1,
         const DoFHandler<dim> & dof_handler_2,
         const unsigned int      level);

  void
  initialize_dof_vector(VectorType & vec_1, VectorType & vec_2);

  ~MGTransferMatrixFreeP();

  virtual void
  restrict_and_add(const unsigned int /*level*/, VectorType & dst, const VectorType & src) const;

  virtual void
  prolongate(const unsigned int /*level*/, VectorType & dst, const VectorType & src) const;

private:
  MatrixFree<dim, value_type>            data_1;
  MatrixFree<dim, value_type>            data_2;
  AlignedVector<VectorizedArray<Number>> shape_values_rest;
  AlignedVector<VectorizedArray<Number>> shape_values_prol;

  void
  restrict_and_add_local(const MatrixFree<dim, value_type> & /*data*/,
                         VectorType &                                  dst,
                         const VectorType &                            src,
                         const std::pair<unsigned int, unsigned int> & cell_range) const;

  void
  prolongate_local(const MatrixFree<dim, value_type> & /*data*/,
                   VectorType &                                  dst,
                   const VectorType &                            src,
                   const std::pair<unsigned int, unsigned int> & cell_range) const;

  void
  convert_to_eo(AlignedVector<VectorizedArray<Number>> & shape_values,
                AlignedVector<VectorizedArray<Number>> & shape_values_eo,
                unsigned int                             fe_degree,
                unsigned int                             n_q_points_1d);

  void
  fill_shape_values(AlignedVector<VectorizedArray<Number>> & shape_values,
                    unsigned int                             fe_degree_src,
                    unsigned int                             fe_degree_dst);
};

#endif