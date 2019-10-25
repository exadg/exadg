#ifndef MG_TRANSFER_MF_P
#define MG_TRANSFER_MF_P

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_tools.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_base.h>

#include "mg_transfer_mf.h"

using namespace dealii;

template<int dim, typename Number, typename VectorType, int components = 1>
class MGTransferMFP : virtual public MGTransferMF<VectorType>
{
public:
  typedef Number value_type;

  MGTransferMFP();

  MGTransferMFP(const MatrixFree<dim, value_type> * data_1_cm,
                const MatrixFree<dim, value_type> * data_2_cm,
                int                                 degree_1,
                int                                 degree_2,
                int                                 dof_handler_index = 0);

  void
  reinit(const MatrixFree<dim, value_type> * data_1_cm,
         const MatrixFree<dim, value_type> * data_2_cm,
         int                                 degree_1,
         int                                 degree_2,
         int                                 dof_handler_index = 0);

  virtual ~MGTransferMFP();

  virtual void
  interpolate(const unsigned int level, VectorType & dst, const VectorType & src) const;

  virtual void
  restrict_and_add(const unsigned int /*level*/, VectorType & dst, const VectorType & src) const;

  virtual void
  prolongate(const unsigned int /*level*/, VectorType & dst, const VectorType & src) const;

private:
  const MatrixFree<dim, value_type> *    data_1_cm;
  const MatrixFree<dim, value_type> *    data_2_cm;
  AlignedVector<VectorizedArray<Number>> prolongation_matrix_1d;
  AlignedVector<VectorizedArray<Number>> interpolation_matrix_1d;

  template<int fe_degree_1, int fe_degree_2>
  void
  do_interpolate(VectorType & dst, const VectorType & src) const;

  template<int fe_degree_1, int fe_degree_2>
  void
  do_restrict_and_add(VectorType & dst, const VectorType & src) const;

  template<int fe_degree_1, int fe_degree_2>
  void
  do_prolongate(VectorType & dst, const VectorType & src) const;

  int          degree_1;
  int          degree_2;
  int          dof_handler_index;
  unsigned int quad_index;
  
  AlignedVector<VectorizedArray<Number>> weights;

  bool is_dg;
};

#endif
