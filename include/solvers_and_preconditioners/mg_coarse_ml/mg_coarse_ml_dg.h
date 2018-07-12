#ifndef MG_COARSE_ML_WRAPPER_DG
#define MG_COARSE_ML_WRAPPER_DG

#include "mg_coarse_ml_wrapper.h"

template <int DIM, typename Number>
class MGCoarseMLDG : public MGCoarseMLWrapper<DIM, Number> {
public:
  MGCoarseMLDG(const int level,
               const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix,
               const MatrixOperatorBase & /*coarse_matrix*/,
               TrilinosWrappers::SparseMatrix &system_matrix);

  virtual void vmult_pre(LinearAlgebra::distributed::Vector<Number> &dst,
                         const LinearAlgebra::distributed::Vector<Number> &src);

  virtual void
  vmult_post(LinearAlgebra::distributed::Vector<Number> &dst,
             const LinearAlgebra::distributed::Vector<Number> &src);
};

#endif