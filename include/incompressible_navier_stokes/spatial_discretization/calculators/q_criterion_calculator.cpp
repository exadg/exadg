/*
 * q_criterion_calculator.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "q_criterion_calculator.h"

namespace IncNS
{
template<int dim, typename Number>
QCriterionCalculator<dim, Number>::QCriterionCalculator()
  : matrix_free(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0)
{
}

template<int dim, typename Number>
void
QCriterionCalculator<dim, Number>::initialize(MatrixFree<dim, Number> const & matrix_free_in,
                                              unsigned int const              dof_index_u_in,
                                              unsigned int const              dof_index_u_scalar_in,
                                              unsigned int const              quad_index_in)
{
  matrix_free        = &matrix_free_in;
  dof_index_u        = dof_index_u_in;
  dof_index_u_scalar = dof_index_u_scalar_in;
  quad_index         = quad_index_in;
}

template<int dim, typename Number>
void
QCriterionCalculator<dim, Number>::compute(VectorType & dst, VectorType const & src) const
{
  dst = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number>
void
QCriterionCalculator<dim, Number>::cell_loop(MatrixFree<dim, Number> const & matrix_free,
                                             VectorType &                    dst,
                                             VectorType const &              src,
                                             Range const &                   cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_u, quad_index);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_u_scalar, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_vector.reinit(cell);
    integrator_vector.gather_evaluate(src, false, true);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_scalar.n_q_points; q++)
    {
      tensor gradu = integrator_vector.get_gradient(q);
      tensor Om, S;
      for(unsigned int i = 0; i < dim; i++)
      {
        for(unsigned int j = 0; j < dim; j++)
        {
          Om[i][j] = 0.5 * (gradu[i][j] - gradu[j][i]);
          S[i][j]  = 0.5 * (gradu[i][j] + gradu[j][i]);
        }
      }

      scalar const Q = 0.5 * (Om.norm_square() - S.norm_square());
      integrator_scalar.submit_value(Q, q);
    }
    integrator_scalar.integrate_scatter(true, false, dst);
  }
}

template class QCriterionCalculator<2, float>;
template class QCriterionCalculator<2, double>;

template class QCriterionCalculator<3, float>;
template class QCriterionCalculator<3, double>;

} // namespace IncNS
