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

#ifndef LAPLACE_OPERATOR_H
#define LAPLACE_OPERATOR_H

#include <exadg/operators/interior_penalty_parameter.h>
#include <exadg/operators/operator_base.h>
#include <exadg/operators/operator_type.h>
#include <exadg/poisson/user_interface/boundary_descriptor.h>

namespace ExaDG
{
namespace Poisson
{
namespace Operators
{
struct LaplaceKernelData
{
  LaplaceKernelData() : IP_factor(1.0)
  {
  }

  double IP_factor;
};

template<int dim, typename Number, int n_components = 1>
class LaplaceKernel
{
private:
  typedef dealii::LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef dealii::VectorizedArray<Number> scalar;

  typedef FaceIntegrator<dim, n_components, Number> IntegratorFace;

  bool
  using_simplex(IntegratorFace & integrator) const
  {
    return integrator.get_matrix_free()
      .get_dof_handler()
      .get_triangulation()
      .all_reference_cells_are_simplex();
  }

public:
  LaplaceKernel() : degree(1), tau(dealii::make_vectorized_array<Number>(0.0))
  {
  }

  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free,
         LaplaceKernelData const &               data_in,
         unsigned int const                      dof_index)
  {
    data = data_in;

    dealii::FiniteElement<dim> const & fe = matrix_free.get_dof_handler(dof_index).get_fe();
    degree                                = fe.degree;

    calculate_penalty_parameter(matrix_free, dof_index);
  }

  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index)
  {
    IP::calculate_penalty_parameter<dim, Number>(array_penalty_parameter, matrix_free, dof_index);
  }

  IntegratorFlags
  get_integrator_flags(bool const is_dg) const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = dealii::EvaluationFlags::gradients;
    flags.cell_integrate = dealii::EvaluationFlags::gradients;

    if(is_dg)
    {
      flags.face_evaluate  = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
      flags.face_integrate = dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients;
    }
    else
    {
      // evaluation of Neumann BCs for continuous elements
      flags.face_evaluate  = dealii::EvaluationFlags::nothing;
      flags.face_integrate = dealii::EvaluationFlags::values;
    }

    return flags;
  }

  static MappingFlags
  get_mapping_flags(bool const compute_interior_face_integrals,
                    bool const compute_boundary_face_integrals)
  {
    MappingFlags flags;

    flags.cells = dealii::update_gradients | dealii::update_JxW_values;

    if(compute_interior_face_integrals)
    {
      flags.inner_faces =
        dealii::update_gradients | dealii::update_JxW_values | dealii::update_normal_vectors;
    }

    if(compute_boundary_face_integrals)
    {
      flags.boundary_faces = dealii::update_gradients | dealii::update_JxW_values |
                             dealii::update_normal_vectors | dealii::update_quadrature_points;
    }

    return flags;
  }

  void
  reinit_face(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const
  {
    tau = std::max(integrator_m.read_cell_data(array_penalty_parameter),
                   integrator_p.read_cell_data(array_penalty_parameter)) *
          IP::get_penalty_factor<dim, Number>(degree, using_simplex(integrator_m), data.IP_factor);
  }

  void
  reinit_boundary_face(IntegratorFace & integrator_m) const
  {
    tau = integrator_m.read_cell_data(array_penalty_parameter) *
          IP::get_penalty_factor<dim, Number>(degree, using_simplex(integrator_m), data.IP_factor);
  }

  void
  reinit_face_cell_based(dealii::types::boundary_id const boundary_id,
                         IntegratorFace &                 integrator_m,
                         IntegratorFace &                 integrator_p) const
  {
    if(boundary_id == dealii::numbers::internal_face_boundary_id) // internal face
    {
      tau =
        std::max(integrator_m.read_cell_data(array_penalty_parameter),
                 integrator_p.read_cell_data(array_penalty_parameter)) *
        IP::get_penalty_factor<dim, Number>(degree, using_simplex(integrator_m), data.IP_factor);
    }
    else // boundary face
    {
      tau =
        integrator_m.read_cell_data(array_penalty_parameter) *
        IP::get_penalty_factor<dim, Number>(degree, using_simplex(integrator_m), data.IP_factor);
    }
  }

  template<typename T>
  inline DEAL_II_ALWAYS_INLINE //
    T
    calculate_gradient_flux(T const & value_m, T const & value_p) const
  {
    return -0.5 * (value_m - value_p);
  }

  template<typename T>
  inline DEAL_II_ALWAYS_INLINE //
    T
    calculate_value_flux(T const & normal_gradient_m,
                         T const & normal_gradient_p,
                         T const & value_m,
                         T const & value_p) const
  {
    return 0.5 * (normal_gradient_m + normal_gradient_p) - tau * (value_m - value_p);
  }

private:
  LaplaceKernelData data;

  unsigned int degree;

  dealii::AlignedVector<scalar> array_penalty_parameter;

  mutable scalar tau;
};

} // namespace Operators

template<int rank, int dim>
struct LaplaceOperatorData : public OperatorBaseData
{
  LaplaceOperatorData() : OperatorBaseData(), quad_index_gauss_lobatto(0)
  {
  }

  Operators::LaplaceKernelData kernel_data;

  // continuous FE:
  // for DirichletCached boundary conditions, another quadrature rule
  // is needed to set the constrained DoFs.
  unsigned int quad_index_gauss_lobatto;

  std::shared_ptr<BoundaryDescriptor<rank, dim> const> bc;
};

template<int dim, typename Number, int n_components>
class LaplaceOperator : public OperatorBase<dim, Number, n_components>
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  typedef OperatorBase<dim, Number, n_components>    Base;
  typedef LaplaceOperator<dim, Number, n_components> This;

  typedef typename Base::IntegratorCell IntegratorCell;
  typedef typename Base::IntegratorFace IntegratorFace;

  typedef typename Base::Range Range;

  typedef dealii::Tensor<rank, dim, dealii::VectorizedArray<Number>> value;

  typedef typename Base::VectorType VectorType;

public:
  typedef Number value_type;

  void
  initialize(dealii::MatrixFree<dim, Number> const &   matrix_free,
             dealii::AffineConstraints<Number> const & affine_constraints,
             LaplaceOperatorData<rank, dim> const &    data);

  LaplaceOperatorData<rank, dim> const &
  get_data() const
  {
    return operator_data;
  }

  void
  calculate_penalty_parameter(dealii::MatrixFree<dim, Number> const & matrix_free,
                              unsigned int const                      dof_index);

  void
  update_penalty_parameter();

  // Some more functionality on top of what is provided by the base class.
  // This function evaluates the inhomogeneous boundary face integrals in DG where the
  // Dirichlet boundary condition is extracted from a dof vector instead of a dealii::Function<dim>.
  void
  rhs_add_dirichlet_bc_from_dof_vector(VectorType & dst, VectorType const & src) const;

  // continuous FE: This function sets the constrained Dirichlet boundary values.
  void
  set_constrained_values(VectorType & solution, double const time) const final;

private:
  void
  reinit_face(unsigned int const face) const final;

  void
  reinit_boundary_face(unsigned int const face) const final;

  void
  reinit_face_cell_based(unsigned int const               cell,
                         unsigned int const               face,
                         dealii::types::boundary_id const boundary_id) const final;

  void
  do_cell_integral(IntegratorCell & integrator) const final;

  void
  do_face_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_face_int_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_face_ext_integral(IntegratorFace & integrator_m, IntegratorFace & integrator_p) const final;

  void
  do_boundary_integral(IntegratorFace &                   integrator_m,
                       OperatorType const &               operator_type,
                       dealii::types::boundary_id const & boundary_id) const final;

  void
  cell_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           range) const;

  void
  face_loop_empty(dealii::MatrixFree<dim, Number> const & matrix_free,
                  VectorType &                            dst,
                  VectorType const &                      src,
                  Range const &                           range) const;

  // DG
  void
  boundary_face_loop_inhom_operator_dirichlet_bc_from_dof_vector(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    VectorType &                            dst,
    VectorType const &                      src,
    Range const &                           range) const;

  // DG
  void
  do_boundary_integral_dirichlet_bc_from_dof_vector(
    IntegratorFace &                   integrator_m,
    OperatorType const &               operator_type,
    dealii::types::boundary_id const & boundary_id) const;

  // continuous FE: calculates Neumann boundary integral
  void
  do_boundary_integral_continuous(IntegratorFace &                   integrator_m,
                                  dealii::types::boundary_id const & boundary_id) const final;

  LaplaceOperatorData<rank, dim> operator_data;

  Operators::LaplaceKernel<dim, Number, n_components> kernel;
};

} // namespace Poisson
} // namespace ExaDG

#endif
