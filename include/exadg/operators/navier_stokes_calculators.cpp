/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#include <exadg/operators/navier_stokes_calculators.h>

namespace ExaDG
{
template<int dim, typename Number>
DivergenceCalculator<dim, Number>::DivergenceCalculator()
  : matrix_free(nullptr), dof_index_vector(1), dof_index_scalar(2), quad_index(0)
{
}

template<int dim, typename Number>
void
DivergenceCalculator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_vector_in,
  unsigned int const                      dof_index_scalar_in,
  unsigned int const                      quad_index_in)
{
  matrix_free      = &matrix_free_in;
  dof_index_vector = dof_index_vector_in;
  dof_index_scalar = dof_index_scalar_in;
  quad_index       = quad_index_in;
}

template<int dim, typename Number>
void
DivergenceCalculator<dim, Number>::compute_divergence(VectorType &       dst,
                                                      VectorType const & src) const
{
  dst = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number>
void
DivergenceCalculator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_vector, quad_index, 0);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_scalar, quad_index, 0);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_vector.reinit(cell);
    integrator_vector.read_dof_values(src);
    integrator_vector.evaluate(dealii::EvaluationFlags::gradients);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_vector.n_q_points; q++)
    {
      scalar divergence = integrator_vector.get_divergence(q);
      integrator_scalar.submit_value(divergence, q);
    }

    integrator_scalar.integrate(dealii::EvaluationFlags::values);
    integrator_scalar.set_dof_values(dst);
  }
}

template<int dim, typename Number>
ShearRateCalculator<dim, Number>::ShearRateCalculator()
  : matrix_free(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0)
{
}

template<int dim, typename Number>
void
ShearRateCalculator<dim, Number>::initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                             unsigned int const                      dof_index_u_in,
                                             unsigned int const dof_index_u_scalar_in,
                                             unsigned int const quad_index_in)
{
  matrix_free        = &matrix_free_in;
  dof_index_u        = dof_index_u_in;
  dof_index_u_scalar = dof_index_u_scalar_in;
  quad_index         = quad_index_in;
}

template<int dim, typename Number>
void
ShearRateCalculator<dim, Number>::compute_shear_rate(VectorType & dst, VectorType const & src) const
{
  dst = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number>
void
ShearRateCalculator<dim, Number>::cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
                                            VectorType &                            dst,
                                            VectorType const &                      src,
                                            Range const & cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_u, quad_index);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_u_scalar, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_vector.reinit(cell);
    integrator_vector.gather_evaluate(src, dealii::EvaluationFlags::gradients);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_scalar.n_q_points; q++)
    {
      symmetrictensor sym_grad_u = integrator_vector.get_symmetric_gradient(q);

      // Shear rate definition according to Galdi et al., 2008
      // ("Hemodynamical Flows: Modeling, Analysis and Simulation").
      // sqrt(2*trace(sym_grad_u^2)) = sqrt(2*sym_grad_u : sym_grad_u)
      scalar shear_rate = std::sqrt(2.0 * scalar_product(sym_grad_u, sym_grad_u));

      integrator_scalar.submit_value(shear_rate, q);
    }

    integrator_scalar.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
VorticityCalculator<dim, Number>::VorticityCalculator()
  : matrix_free(nullptr), dof_index(0), quad_index(0)
{
}

template<int dim, typename Number>
void
VorticityCalculator<dim, Number>::initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                             unsigned int const                      dof_index_in,
                                             unsigned int const                      quad_index_in)
{
  matrix_free = &matrix_free_in;
  dof_index   = dof_index_in;
  quad_index  = quad_index_in;
}

template<int dim, typename Number>
void
VorticityCalculator<dim, Number>::compute_vorticity(VectorType & dst, VectorType const & src) const
{
  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number>
void
VorticityCalculator<dim, Number>::cell_loop(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  VectorType &                                  dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  CellIntegratorVector integrator(matrix_free, dof_index, quad_index, 0);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);
    integrator.evaluate(dealii::EvaluationFlags::gradients);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      dealii::Tensor<1, number_vorticity_components, dealii::VectorizedArray<Number>> omega =
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

    integrator.integrate(dealii::EvaluationFlags::values);
    integrator.set_dof_values(dst);
  }
}

template<int dim, typename Number>
MagnitudeCalculator<dim, Number>::MagnitudeCalculator()
  : matrix_free(nullptr), dof_index_u(0), dof_index_u_scalar(0), quad_index(0)
{
}

template<int dim, typename Number>
void
MagnitudeCalculator<dim, Number>::initialize(dealii::MatrixFree<dim, Number> const & matrix_free_in,
                                             unsigned int const                      dof_index_u_in,
                                             unsigned int const dof_index_u_scalar_in,
                                             unsigned int const quad_index_in)
{
  matrix_free        = &matrix_free_in;
  dof_index_u        = dof_index_u_in;
  dof_index_u_scalar = dof_index_u_scalar_in;
  quad_index         = quad_index_in;
}

template<int dim, typename Number>
void
MagnitudeCalculator<dim, Number>::compute(VectorType & dst, VectorType const & src) const
{
  dst = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number>
void
MagnitudeCalculator<dim, Number>::cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
                                            VectorType &                            dst,
                                            VectorType const &                      src,
                                            Range const & cell_range) const
{
  IntegratorVector integrator_vector(matrix_free, dof_index_u, quad_index);
  IntegratorScalar integrator_scalar(matrix_free, dof_index_u_scalar, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_vector.reinit(cell);
    integrator_vector.gather_evaluate(src, dealii::EvaluationFlags::values);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_scalar.n_q_points; q++)
    {
      scalar magnitude = integrator_vector.get_value(q).norm();
      integrator_scalar.submit_value(magnitude, q);
    }
    integrator_scalar.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number>
QCriterionCalculator<dim, Number>::QCriterionCalculator()
  : matrix_free(nullptr),
    dof_index_u(0),
    dof_index_u_scalar(0),
    quad_index(0),
    compressible_flow(false)
{
}

template<int dim, typename Number>
void
QCriterionCalculator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_u_in,
  unsigned int const                      dof_index_u_scalar_in,
  unsigned int const                      quad_index_in,
  bool const                              compressible_flow_in)
{
  matrix_free        = &matrix_free_in;
  dof_index_u        = dof_index_u_in;
  dof_index_u_scalar = dof_index_u_scalar_in;
  quad_index         = quad_index_in;
  compressible_flow  = compressible_flow_in;
}

template<int dim, typename Number>
void
QCriterionCalculator<dim, Number>::compute(VectorType & dst, VectorType const & src) const
{
  dst = 0;

  matrix_free->cell_loop(&This::cell_loop, this, dst, src, compressible_flow);
}

template<int dim, typename Number>
void
QCriterionCalculator<dim, Number>::cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
                                             VectorType &                            dst,
                                             VectorType const &                      src,
                                             Range const & cell_range) const
{
  CellIntegratorVector integrator_vector(matrix_free, dof_index_u, quad_index);
  CellIntegratorScalar integrator_scalar(matrix_free, dof_index_u_scalar, quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator_vector.reinit(cell);
    integrator_vector.gather_evaluate(src, dealii::EvaluationFlags::gradients);

    integrator_scalar.reinit(cell);

    for(unsigned int q = 0; q < integrator_scalar.n_q_points; q++)
    {
      tensor grad_u = integrator_vector.get_gradient(q);
      tensor Om, S;
      for(unsigned int i = 0; i < dim; i++)
      {
        for(unsigned int j = 0; j < dim; j++)
        {
          Om[i][j] = 0.5 * (grad_u[i][j] - grad_u[j][i]);
          S[i][j]  = 0.5 * (grad_u[i][j] + grad_u[j][i]);
        }
      }

      // Q criterion for compressible flow is based on the deviatoric
      // part of S (Kolar, V.: Compressibility Effect in Vortex Identification,
      // AIAA, 2009. https://doi.org/10.2514/1.40131)
      if(compressible_flow)
      {
        scalar const one_third_trace_grad_u = trace(grad_u) / 3.0;
        for(unsigned int i = 0; i < dim; i++)
        {
          S[i][i] -= one_third_trace_grad_u;
        }
      }

      scalar const Q = 0.5 * (Om.norm_square() - S.norm_square());
      integrator_scalar.submit_value(Q, q);
    }
    integrator_scalar.integrate_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template class DivergenceCalculator<2, float>;
template class DivergenceCalculator<2, double>;

template class DivergenceCalculator<3, float>;
template class DivergenceCalculator<3, double>;

template class ShearRateCalculator<2, float>;
template class ShearRateCalculator<2, double>;

template class ShearRateCalculator<3, float>;
template class ShearRateCalculator<3, double>;

template class VorticityCalculator<2, float>;
template class VorticityCalculator<2, double>;

template class VorticityCalculator<3, float>;
template class VorticityCalculator<3, double>;

template class MagnitudeCalculator<2, float>;
template class MagnitudeCalculator<2, double>;

template class MagnitudeCalculator<3, float>;
template class MagnitudeCalculator<3, double>;

template class QCriterionCalculator<2, float>;
template class QCriterionCalculator<2, double>;

template class QCriterionCalculator<3, float>;
template class QCriterionCalculator<3, double>;

} // namespace ExaDG
