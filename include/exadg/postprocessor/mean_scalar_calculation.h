//
// Created by bergbauer on 12/10/20.
//

#ifndef INCLUDE_EXADG_POSTPROCESSOR_MEAN_SCALAR_CALCULATION_H
#define INCLUDE_EXADG_POSTPROCESSOR_MEAN_SCALAR_CALCULATION_H

#include <exadg/matrix_free/integrators.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number>
class MeanScalarCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef FaceIntegrator<dim, 1, Number> FaceIntegratorScalar;

  typedef VectorizedArray<Number> scalar;

  MeanScalarCalculator(MatrixFree<dim, Number> const & matrix_free_in,
                       unsigned int                    dof_index_in,
                       unsigned int                    quad_index_in,
                       MPI_Comm const &                mpi_comm_in);

  Number
  calculate_mean_scalar(const VectorType &                     solution,
                        std::map<types::boundary_id, Number> & mean_scalar);

private:
  MatrixFree<dim, Number> const & matrix_free;
  unsigned int                    dof_index, quad_index;

  MPI_Comm const & mpi_comm;
};

} // namespace ExaDG



#endif // INCLUDE_EXADG_POSTPROCESSOR_MEAN_SCALAR_CALCULATION_H
