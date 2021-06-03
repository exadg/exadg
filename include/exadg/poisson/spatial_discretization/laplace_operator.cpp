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

// deal.II
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/poisson/spatial_discretization/weak_boundary_conditions.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::initialize(
  MatrixFree<dim, Number> const &        matrix_free,
  AffineConstraints<Number> const &      affine_constraints,
  LaplaceOperatorData<rank, dim> const & data)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  kernel.reinit(matrix_free, data.kernel_data, data.dof_index);

  this->integrator_flags = kernel.get_integrator_flags(this->is_dg);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::calculate_penalty_parameter(
  MatrixFree<dim, Number> const & matrix_free,
  unsigned int const              dof_index)
{
  kernel.calculate_penalty_parameter(matrix_free, dof_index);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::update_penalty_parameter()
{
  calculate_penalty_parameter(this->get_matrix_free(), this->get_data().dof_index);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::rhs_add_dirichlet_bc_from_dof_vector(
  VectorType &       dst,
  VectorType const & src) const
{
  VectorType tmp;
  tmp.reinit(dst, false /* init with 0 */);

  this->matrix_free->loop(&This::cell_loop_empty,
                          &This::face_loop_empty,
                          &This::boundary_face_loop_inhom_operator_dirichlet_bc_from_dof_vector,
                          this,
                          tmp,
                          src,
                          false /*zero_dst_vector = false*/);

  // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
  dst.add(-1.0, tmp);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  kernel.reinit_face(*this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  kernel.reinit_boundary_face(*this->integrator_m);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::reinit_face_cell_based(
  unsigned int const       cell,
  unsigned int const       face,
  types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  kernel.reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    integrator.submit_gradient(integrator.get_gradient(q), q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_face_integral(IntegratorFace & integrator_m,
                                                             IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value value_m = integrator_m.get_value(q);
    value value_p = integrator_p.get_value(q);

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    value normal_gradient_m = integrator_m.get_normal_derivative(q);
    value normal_gradient_p = integrator_p.get_normal_derivative(q);

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_p.submit_normal_derivative(gradient_flux, q);

    integrator_m.submit_value(-value_flux, q);
    integrator_p.submit_value(value_flux, q); // + sign since n⁺ = -n⁻
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_face_int_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set exterior value to zero
    value value_m = integrator_m.get_value(q);
    value value_p = value();

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    // set exterior value to zero
    value normal_gradient_m = integrator_m.get_normal_derivative(q);
    value normal_gradient_p = value();

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_face_ext_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    // set value_m to zero
    value value_p = integrator_p.get_value(q);
    value value_m = value();

    value gradient_flux = kernel.calculate_gradient_flux(value_p, value_m);

    // minus sign to get the correct normal vector n⁺ = -n⁻
    value normal_gradient_p = -integrator_p.get_normal_derivative(q);
    value normal_gradient_m = value();

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_p, normal_gradient_m, value_p, value_m);

    integrator_p.submit_normal_derivative(-gradient_flux, q); // opposite sign since n⁺ = -n⁻
    integrator_p.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_boundary_integral(
  IntegratorFace &           integrator_m,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value value_m =
      calculate_interior_value<dim, Number, n_components, rank>(q, integrator_m, operator_type);
    value value_p = calculate_exterior_value<dim, Number, n_components, rank>(value_m,
                                                                              q,
                                                                              integrator_m,
                                                                              operator_type,
                                                                              boundary_type,
                                                                              boundary_id,
                                                                              operator_data.bc,
                                                                              this->time);

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    value normal_gradient_m =
      calculate_interior_normal_gradient<dim, Number, n_components, rank>(q,
                                                                          integrator_m,
                                                                          operator_type);
    value normal_gradient_p =
      calculate_exterior_normal_gradient<dim, Number, n_components, rank>(normal_gradient_m,
                                                                          q,
                                                                          integrator_m,
                                                                          operator_type,
                                                                          boundary_type,
                                                                          boundary_id,
                                                                          operator_data.bc,
                                                                          this->time);

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::cell_loop_empty(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::face_loop_empty(
  MatrixFree<dim, Number> const & matrix_free,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;

  // do nothing
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::
  boundary_face_loop_inhom_operator_dirichlet_bc_from_dof_vector(
    MatrixFree<dim, Number> const & matrix_free,
    VectorType &                    dst,
    VectorType const &              src,
    Range const &                   range) const
{
  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(face);

    // deviating from the standard function boundary_face_loop_inhom_operator()
    // because the boundary condition comes from the vector src
    this->integrator_m->gather_evaluate(src,
                                        this->integrator_flags.face_evaluate.value,
                                        this->integrator_flags.face_evaluate.gradient);

    do_boundary_integral_dirichlet_bc_from_dof_vector(*this->integrator_m,
                                                      OperatorType::inhomogeneous,
                                                      matrix_free.get_boundary_id(face));

    this->integrator_m->integrate_scatter(this->integrator_flags.face_integrate.value,
                                          this->integrator_flags.face_integrate.gradient,
                                          dst);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_boundary_integral_dirichlet_bc_from_dof_vector(
  IntegratorFace &           integrator_m,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value value_m =
      calculate_interior_value<dim, Number, n_components, rank>(q, integrator_m, operator_type);

    // deviating from the standard boundary_face_loop_inhom_operator() function,
    // because the boundary condition comes from the vector src
    Assert(operator_type == OperatorType::inhomogeneous,
           ExcMessage("This function is only implemented for OperatorType::inhomogeneous."));

    value value_p = value();
    if(boundary_type == BoundaryType::Dirichlet)
    {
      value_p = 2.0 * integrator_m.get_value(q);
    }
    else if(boundary_type == BoundaryType::Neumann)
    {
      // do nothing
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    value gradient_flux = kernel.calculate_gradient_flux(value_m, value_p);

    value normal_gradient_m =
      calculate_interior_normal_gradient<dim, Number, n_components, rank>(q,
                                                                          integrator_m,
                                                                          operator_type);
    value normal_gradient_p =
      calculate_exterior_normal_gradient<dim, Number, n_components, rank>(normal_gradient_m,
                                                                          q,
                                                                          integrator_m,
                                                                          operator_type,
                                                                          boundary_type,
                                                                          boundary_id,
                                                                          operator_data.bc,
                                                                          this->time);

    value value_flux =
      kernel.calculate_value_flux(normal_gradient_m, normal_gradient_p, value_m, value_p);

    integrator_m.submit_normal_derivative(gradient_flux, q);
    integrator_m.submit_value(-value_flux, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::do_boundary_integral_continuous(
  IntegratorFace &           integrator_m,
  types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    value neumann_value = calculate_neumann_value<dim, Number, n_components, rank>(
      q, integrator_m, boundary_type, boundary_id, operator_data.bc, this->time);

    integrator_m.submit_value(-neumann_value, q);
  }
}

template<int dim, typename Number, int n_components>
void
LaplaceOperator<dim, Number, n_components>::set_constrained_values(VectorType & dst,
                                                                   double const time) const
{
  // standard Dirichlet boundary conditions
  std::map<types::global_dof_index, double> boundary_values;
  for(auto dbc : operator_data.bc->dirichlet_bc)
  {
    dbc.second->set_time(time);

    ComponentMask mask     = ComponentMask();
    auto          dbc_mask = operator_data.bc->dirichlet_bc_component_mask.find(dbc.first);
    if(dbc_mask != operator_data.bc->dirichlet_bc_component_mask.end())
      mask = dbc_mask->second;

    VectorTools::interpolate_boundary_values(*this->matrix_free->get_mapping_info().mapping,
                                             this->matrix_free->get_dof_handler(
                                               operator_data.dof_index),
                                             dbc.first,
                                             *dbc.second,
                                             boundary_values,
                                             mask);
  }

  // set Dirichlet values in solution vector
  for(auto m : boundary_values)
    if(dst.get_partitioner()->in_local_range(m.first))
      dst[m.first] = m.second;

  dst.update_ghost_values();

  // Dirichlet mortar type boundary conditions
  if(not(operator_data.bc->dirichlet_mortar_bc.empty()))
  {
    unsigned int const dof_index  = operator_data.dof_index;
    unsigned int const quad_index = operator_data.quad_index_gauss_lobatto;

    IntegratorFace integrator(*this->matrix_free, true, dof_index, quad_index);

    for(unsigned int face = this->matrix_free->n_inner_face_batches();
        face <
        this->matrix_free->n_inner_face_batches() + this->matrix_free->n_boundary_face_batches();
        ++face)
    {
      types::boundary_id const boundary_id = this->matrix_free->get_boundary_id(face);

      BoundaryType const boundary_type = operator_data.bc->get_boundary_type(boundary_id);

      if(boundary_type == BoundaryType::DirichletMortar)
      {
        integrator.reinit(face);
        integrator.read_dof_values(dst);

        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          unsigned int const local_face_number =
            this->matrix_free->get_face_info(face).interior_face_no;

          unsigned int const index = this->matrix_free->get_shape_info(dof_index, quad_index)
                                       .face_to_cell_index_nodal[local_face_number][q];

          Tensor<rank, dim, VectorizedArray<Number>> g;

          if(boundary_type == BoundaryType::DirichletMortar)
          {
            auto bc = operator_data.bc->dirichlet_mortar_bc.find(boundary_id)->second;

            g = FunctionEvaluator<rank, dim, Number>::value(bc, face, q, quad_index);
          }
          else
          {
            AssertThrow(false, ExcMessage("Not implemented."));
          }

          integrator.submit_dof_value(g, index);
        }

        integrator.set_dof_values_plain(dst);
      }
      else
      {
        AssertThrow(boundary_type == BoundaryType::Dirichlet ||
                      boundary_type == BoundaryType::Neumann,
                    ExcMessage("BoundaryType not implemented."));
      }
    }
  }
}

template class LaplaceOperator<2, float, 1>;
template class LaplaceOperator<2, double, 1>;
template class LaplaceOperator<2, float, 2>;
template class LaplaceOperator<2, double, 2>;

template class LaplaceOperator<3, float, 1>;
template class LaplaceOperator<3, double, 1>;
template class LaplaceOperator<3, float, 3>;
template class LaplaceOperator<3, double, 3>;

} // namespace Poisson
} // namespace ExaDG
