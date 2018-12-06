/*
 * vorticity_calculator.cpp
 *
 *  Created on: Dec 6, 2018
 *      Author: fehn
 */

#include "vorticity_calculator.h"

namespace IncNS
{
template<int dim, int degree, typename Number>
VorticityCalculator<dim, degree, Number>::VorticityCalculator()
  : data(nullptr), dof_index(0), quad_index(0)
{
}

template<int dim, int degree, typename Number>
void
VorticityCalculator<dim, degree, Number>::initialize(MatrixFree<dim, Number> const & data_in,
                                                     unsigned int const              dof_index_in,
                                                     unsigned int const              quad_index_in)
{
  this->data = &data_in;
  dof_index  = dof_index_in;
  quad_index = quad_index_in;
}

template<int dim, int degree, typename Number>
void
VorticityCalculator<dim, degree, Number>::compute_vorticity(VectorType &       dst,
                                                            VectorType const & src) const
{
  dst = 0;

  data->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, int degree, typename Number>
void
VorticityCalculator<dim, degree, Number>::cell_loop(MatrixFree<dim, Number> const & data,
                                                    VectorType &                    dst,
                                                    VectorType const &              src,
                                                    Range const & cell_range) const
{
  FEEval fe_eval_velocity(data, dof_index, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    fe_eval_velocity.reinit(cell);
    fe_eval_velocity.gather_evaluate(src, false, true, false);

    for(unsigned int q = 0; q < fe_eval_velocity.n_q_points; ++q)
    {
      // omega is a scalar quantity in 2D and a vector with dim components in 3D
      Tensor<1, number_vorticity_components, VectorizedArray<Number>> omega =
        fe_eval_velocity.get_curl(q);

      // omega_vector is a vector with dim components
      // for dim=3: omega_vector[i] = omega[i], i=1,...,dim
      // for dim=2: omega_vector[0] = omega,
      //            omega_vector[1] = 0
      vector omega_vector;
      for(unsigned int d = 0; d < number_vorticity_components; ++d)
        omega_vector[d] = omega[d];

      fe_eval_velocity.submit_value(omega_vector, q);
    }

    fe_eval_velocity.integrate_scatter(true, false, dst);
  }
}

} // namespace IncNS

#include "vorticity_calculator.hpp"
