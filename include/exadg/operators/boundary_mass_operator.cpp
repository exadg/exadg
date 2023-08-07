/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#include <exadg/operators/boundary_mass_operator.h>

namespace ExaDG
{
template<int dim, typename Number, int n_components>
BoundaryMassOperator<dim, Number, n_components>::BoundaryMassOperator() : scaling_factor(1.0)
{
}

template<int dim, typename Number, int n_components>
bool
BoundaryMassOperator<dim, Number, n_components>::non_empty() const
{
  return this->ids_normal_coefficients.size() > 0;
}

template<int dim, typename Number, int n_components>
IntegratorFlags
BoundaryMassOperator<dim, Number, n_components>::get_integrator_flags() const
{
  return kernel.get_integrator_flags();
}

template<int dim, typename Number, int n_components>
MappingFlags
BoundaryMassOperator<dim, Number, n_components>::get_mapping_flags()
{
  return BoundaryMassKernel<dim, Number>::get_mapping_flags();
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::initialize(
  dealii::MatrixFree<dim, Number> const &       matrix_free_in,
  dealii::AffineConstraints<Number> const &     affine_constraints,
  BoundaryMassOperatorData<dim, Number> const & data)
{
  Base::reinit(matrix_free_in, affine_constraints, data);

  this->integrator_flags        = this->get_integrator_flags();
  this->ids_normal_coefficients = data.ids_normal_coefficients;
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::set_scaling_factor(Number const & factor) const
{
  scaling_factor = factor;
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::set_ids_normal_coefficients(
  std::map<dealii::types::boundary_id, std::pair<bool, Number>> const & ids_normal_coefficients_in)
  const
{
  this->ids_normal_coefficients = ids_normal_coefficients_in;
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::evaluate_add(VectorType &       dst,
                                                              VectorType const & src) const
{
  this->matrix_free->loop(&This::cell_loop_empty,
                          &This::face_loop_empty,
                          &This::boundary_face_loop_full_operator,
                          this,
                          dst,
                          src,
                          false /* zero_dst_vector */);
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::cell_loop_empty(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::face_loop_empty(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::boundary_face_loop_full_operator(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorFace integrator_m =
    IntegratorFace(matrix_free, true, this->get_dof_index(), this->get_quad_index());

  for(unsigned int face = range.first; face < range.second; face++)
  {
    dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

    // integrate over selected boundaries and *do not* gather_evaluate/integrate_scatter on others
    if(auto it{this->ids_normal_coefficients.find(boundary_id)};
       it != this->ids_normal_coefficients.end())
    {
      Number scaled_coefficient = it->second.second * scaling_factor;
      bool   normal_projection  = it->second.first;

      integrator_m.reinit(face);
      integrator_m.gather_evaluate(src, this->integrator_flags.face_evaluate);

      this->do_boundary_segment_integral(integrator_m, scaled_coefficient, normal_projection);

      integrator_m.integrate_scatter(this->integrator_flags.face_integrate, dst);
    }
  }
}

template<int dim, typename Number, int n_components>
void
BoundaryMassOperator<dim, Number, n_components>::do_boundary_segment_integral(
  IntegratorFace & integrator_m,
  Number const &   scaled_coefficient,
  bool const       normal_projection) const
{
  if(normal_projection)
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      integrator_m.submit_value(
        kernel.get_boundary_mass_normal_value(scaled_coefficient,
                                              integrator_m.get_normal_vector(q),
                                              integrator_m.get_value(q)),
        q);
    }
  }
  else
  {
    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      integrator_m.submit_value(kernel.get_boundary_mass_value(scaled_coefficient,
                                                               integrator_m.get_value(q)),
                                q);
    }
  }
}

template class BoundaryMassOperator<2, float, 2>;
template class BoundaryMassOperator<2, double, 2>;

template class BoundaryMassOperator<3, float, 3>;
template class BoundaryMassOperator<3, double, 3>;
} // namespace ExaDG
