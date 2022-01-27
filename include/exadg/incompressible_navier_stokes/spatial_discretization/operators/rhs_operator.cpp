/*
 * rhs_operator.cpp
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/rhs_operator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
RHSOperator<dim, Number>::RHSOperator() : matrix_free(nullptr), time(0.0), temperature(nullptr)
{
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                     RHSOperatorData<dim> const &            data_in)
{
  this->matrix_free = &matrix_free_in;
  this->data        = data_in;

  kernel.reinit(data.kernel_data);
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::evaluate(VectorType & dst, Number const evaluation_time) const
{
  time = evaluation_time;

  VectorType src;
  matrix_free->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::evaluate_add(VectorType & dst, Number const evaluation_time) const
{
  time = evaluation_time;

  VectorType src;
  matrix_free->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::set_temperature(VectorType const & T)
{
  this->temperature = &T;
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::do_cell_integral(Integrator &       integrator,
                                           IntegratorScalar & integrator_temperature) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_value(kernel.get_volume_flux(integrator, integrator_temperature, q, time), q);
  }
}

template<int dim, typename Number>
void
RHSOperator<dim, Number>::cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
                                    VectorType &                            dst,
                                    VectorType const &                      src,
                                    Range const &                           cell_range) const
{
  (void)src;

  Integrator integrator(matrix_free, data.dof_index, data.quad_index);

  IntegratorScalar integrator_temperature(matrix_free, data.dof_index_scalar, data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    if(data.kernel_data.boussinesq_term)
    {
      integrator_temperature.reinit(cell);
      integrator_temperature.gather_evaluate(*temperature, true, false);
    }

    do_cell_integral(integrator, integrator_temperature);

    integrator.integrate_scatter(true, false, dst);
  }
}

template class RHSOperator<2, float>;
template class RHSOperator<2, double>;

template class RHSOperator<3, float>;
template class RHSOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
