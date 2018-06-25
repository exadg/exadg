#ifndef MG_COARSE_ML_WRAPPER_CG
#define MG_COARSE_ML_WRAPPER_CG

#include "mg_coarse_ml_wrapper.h"

#include "dg_to_cg_transfer.h"

template <int DIM, typename Number>
class MGCoarseMLCG : public MGCoarseMLWrapper<DIM, Number> {
public:
  MGCoarseMLCG(const int level,
               const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix_dg,
               const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix_q,
               TrilinosWrappers::SparseMatrix &system_matrix);

  const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix_dg;

  virtual void init_system();

  virtual void vmult_pre(LinearAlgebra::distributed::Vector<Number> &dst,
                         const LinearAlgebra::distributed::Vector<Number> &src);

  virtual void
  vmult_post(LinearAlgebra::distributed::Vector<Number> &dst,
             const LinearAlgebra::distributed::Vector<Number> &src);

private:
  CGToDGTransfer<DIM, Number> transfer;
};

#endif