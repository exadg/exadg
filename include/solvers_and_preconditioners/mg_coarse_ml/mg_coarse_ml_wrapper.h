#ifndef MG_COARSE_ML_WRAPPER
#define MG_COARSE_ML_WRAPPER

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include "../../operators/matrix_operator_base_new.h"

using namespace dealii;

#ifdef DEAL_II_WITH_TRILINOS

template <int DIM, typename Number> class MGCoarseMLWrapper {

public:
  MGCoarseMLWrapper(const int level,
                    const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix,
                    TrilinosWrappers::SparseMatrix &system_matrix)
      : coarse_matrix(coarse_matrix), system_matrix(system_matrix),
        level(level),
        degree(coarse_matrix.get_data().get_dof_handler().get_fe().degree) {}

  virtual void init_system(){
      coarse_matrix.init_system_matrix(system_matrix);
      coarse_matrix.calculate_system_matrix(system_matrix);
  }

  virtual void
  vmult_pre(LinearAlgebra::distributed::Vector<Number> &dst,
            const LinearAlgebra::distributed::Vector<Number> &src) = 0;

  virtual void
  vmult_post(LinearAlgebra::distributed::Vector<Number> &dst,
             const LinearAlgebra::distributed::Vector<Number> &src) = 0;

  virtual void
  init_vectors(LinearAlgebra::distributed::Vector<Number> &vec_1,
               LinearAlgebra::distributed::Vector<Number> &vec_2) const {
    this->coarse_matrix.initialize_dof_vector(vec_1);
    this->coarse_matrix.initialize_dof_vector(vec_2);
  }

protected:
  const MatrixOperatorBaseNew<DIM, Number> &coarse_matrix;
  TrilinosWrappers::SparseMatrix &system_matrix;
  const int level;
  const int degree;
};

#endif

#endif