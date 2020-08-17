/*
 * vorticity_calculator.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "vorticity_calculator.h"

namespace ExaDG
{
namespace IncNS
{
using namespace dealii;

template<int dim, typename Number>
VorticityCalculator<dim, Number>::VorticityCalculator()
  : matrix_free(nullptr), dof_index(0), quad_index(0)
{
}

template<int dim, typename Number>
void
VorticityCalculator<dim, Number>::initialize(MatrixFree<dim, Number> const & matrix_free_in,
                                             unsigned int const              dof_index_in,
                                             unsigned int const              quad_index_in)
{
  matrix_free = &matrix_free_in;
  dof_index   = dof_index_in;
  quad_index  = quad_index_in;
}

template<int dim, typename Number>
void
VorticityCalculator<dim, Number>::compute_vorticity(VectorType & dst, VectorType const & src) const
{
  dst = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number>
void
VorticityCalculator<dim, Number>::cell_loop(MatrixFree<dim, Number> const & matrix_free,
                                            VectorType &                    dst,
                                            VectorType const &              src,
                                            Range const &                   cell_range) const
{
  Integrator integrator(matrix_free, dof_index, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.gather_evaluate(src, false, true, false);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // omega is a scalar quantity in 2D and a vector with dim components in 3D
      Tensor<1, number_vorticity_components, VectorizedArray<Number>> omega =
        integrator.get_curl(q);

      // omega_vector is a vector with dim components
      // for dim=3: omega_vector[i] = omega[i], i=1,...,dim
      // for dim=2: omega_vector[0] = omega,
      //            omega_vector[1] = 0
      vector omega_vector;
      for(unsigned int d = 0; d < number_vorticity_components; ++d)
        omega_vector[d] = omega[d];

      integrator.submit_value(omega_vector, q);
    }

    integrator.integrate_scatter(true, false, dst);
  }
}

template class VorticityCalculator<2, float>;
template class VorticityCalculator<2, double>;

template class VorticityCalculator<3, float>;
template class VorticityCalculator<3, double>;

} // namespace IncNS
} // namespace ExaDG
