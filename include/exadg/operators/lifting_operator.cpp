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

#include <exadg/operators/lifting_operator.h>

namespace ExaDG
{
template<int dim, typename Number, int n_components>
LiftingOperator<dim, Number, n_components>::LiftingOperator() : matrix_free(nullptr), time(0.0)
{
}

template<int dim, typename Number, int n_components>
void
LiftingOperator<dim, Number, n_components>::initialize(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  LiftingOperatorData<dim> const &            data_in)
{
  this->matrix_free = &matrix_free_in;
  this->data        = data_in;

  kernel.reinit(data.kernel_data);
}

template<int dim, typename Number, int n_components>
void
LiftingOperator<dim, Number, n_components>::evaluate(VectorType & dst,
                                                 double const evaluation_time) const
{
  dst = 0;
  evaluate_add(dst, evaluation_time);
}

template<int dim, typename Number, int n_components>
void
LiftingOperator<dim, Number, n_components>::evaluate_add(VectorType & dst,
                                                     double const evaluation_time) const
{
  this->time = evaluation_time;

  VectorType src;
  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number, int n_components>
void
LiftingOperator<dim, Number, n_components>::evalute_lifting_operator(VectorType const & solution,
                                                                     VectorType &       lifting_term,
                                                                     double const       evaluation_term) const
{
  this->time = evaluation_time;

  matrix_free->loop(&This::cell_loop,
                    &This::face_loop_lifting,
                    &This::boundary_face_loop_lifting,
                    this,
                    lifting_term,
                    solution);

}

template<int dim, typename Number, int n_components>
void
LiftingOperator<dim, Number, n_components>::apply_inverse_mass_matrix(VectorType & dst,
                                                     double const evaluation_time) const
{
  this->time = evaluation_time;

  VectorType src;
  matrix_free->cell_loop(&This::cell_loop, this, dst, src);
}

template<int dim, typename Number, int n_components>
void
LiftingOperator<dim, Number, n_components>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_value(kernel.get_volume_flux(integrator, q, time), q);
  }

  integrator.integrate(dealii::EvaluationFlags::values);
}

template<int dim, typename Number, int n_components>
void
LiftingOperator<dim, Number< n_components>::face_loop_lifting(dealii::MatrixFree<dim, Number> const & matrix_free,
                                                              VectorType & dst,
                                                              VectorType const & src,
                                                              Range const & face_range) const
{
  IntegratorFaceScalar integrator_m(matrix_free, true, data.dof_index_scalar, data.quad_index);
  IntegratorFaceScalar integrator_p(matrix_free, false, data.dof_index_scalar, data.quad_index);

  IntegratorFaceVector integrator integrator_lift(matrix_free, true, data.dof_index_vector, data.quad_index);

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    integrator_m.reinit(face);
    integrator_p.reinit(face);

    integrator_m.gather_evaluate(src, dealii::EvaluationFlags::values);
    integrator_p.gather_evaluate(src, dealii::EvaluationFlags::values);

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      scalar val_m = integrator_m.get_value(q);
      scalar val_p = integrator_p.get_value(q);
      vector normal = integrator_m.get_normal_vector(q);

      vector flux = kernel.get_face_integral(val_m, val_p, normal);

      integrator_lift.submit_value(flux, q);
    }
    integrator_lift.integrator_scatter(dealii::EvaluationFlags::values, dst);
  }
}

template<int dim, typename Number, int n_components>
void
LiftingOperator<dim, Number, n_components>::cell_loop(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           cell_range) const
{
  (void)src;

  IntegratorCell integrator(matrix_free, data.dof_index, data.quad_index);

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    do_cell_integral(integrator);

    integrator.distribute_local_to_global(dst);
  }
}

template class LiftingOperator<2, float, 1>;
template class LiftingOperator<2, double, 1>;
template class LiftingOperator<2, float, 2>;
template class LiftingOperator<2, double, 2>;

template class LiftingOperator<3, float, 1>;
template class LiftingOperator<3, double, 1>;
template class LiftingOperator<3, float, 3>;
template class LiftingOperator<3, double, 3>;

} // namespace ExaDG
