/*
 * convection_diffusion_operator_merged.cpp
 *
 *  Created on: Jun 6, 2019
 *      Author: fehn
 */

#include "convection_diffusion_operator_merged.h"

#include "verify_boundary_conditions.h"
#include "weak_boundary_conditions.h"

namespace ConvDiff
{
template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::reinit(
  MatrixFree<dim, Number> const &                    matrix_free,
  AffineConstraints<double> const &                  constraint_matrix,
  ConvectionDiffusionOperatorMergedData<dim> const & operator_data) const
{
  Base::reinit(matrix_free, constraint_matrix, operator_data);

  if(this->operator_data.unsteady_problem)
    mass_kernel.reinit(operator_data.scaling_factor_mass_matrix);

  if(this->operator_data.convective_problem)
    convective_kernel.reinit(matrix_free,
                             operator_data.convective_kernel_data,
                             operator_data.quad_index,
                             this->is_mg);

  if(this->operator_data.diffusive_problem)
    diffusive_kernel.reinit(matrix_free,
                            operator_data.diffusive_kernel_data,
                            operator_data.dof_index);

  if(this->operator_data.unsteady_problem)
    this->integrator_flags = this->integrator_flags || mass_kernel.get_integrator_flags();
  if(this->operator_data.convective_problem)
    this->integrator_flags = this->integrator_flags || convective_kernel.get_integrator_flags();
  if(this->operator_data.diffusive_problem)
    this->integrator_flags = this->integrator_flags || diffusive_kernel.get_integrator_flags();
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::set_velocity_copy(
  VectorType const & velocity_in) const
{
  convective_kernel.set_velocity_copy(velocity_in);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::set_velocity_ptr(
  VectorType const & velocity_in) const
{
  convective_kernel.set_velocity_ptr(velocity_in);
}

template<int dim, typename Number>
Number
ConvectionDiffusionOperatorMerged<dim, Number>::get_scaling_factor_mass_matrix() const
{
  return mass_kernel.get_scaling_factor();
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::set_scaling_factor_mass_matrix(
  Number const & number) const
{
  mass_kernel.set_scaling_factor(number);
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
ConvectionDiffusionOperatorMerged<dim, Number>::get_velocity() const
{
  return convective_kernel.get_velocity();
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  if(this->operator_data.convective_problem)
    convective_kernel.reinit_cell(cell);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  if(this->operator_data.convective_problem)
    convective_kernel.reinit_face(face);
  if(this->operator_data.diffusive_problem)
    diffusive_kernel.reinit_face(*this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  if(this->operator_data.convective_problem)
    convective_kernel.reinit_boundary_face(face);
  if(this->operator_data.diffusive_problem)
    diffusive_kernel.reinit_boundary_face(*this->integrator_m);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::reinit_face_cell_based(
  unsigned int const       cell,
  unsigned int const       face,
  types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  if(this->operator_data.convective_problem)
    convective_kernel.reinit_face_cell_based(cell, face, boundary_id);
  if(this->operator_data.diffusive_problem)
    diffusive_kernel.reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    scalar value = make_vectorized_array<Number>(0.0);
    if(this->operator_data.unsteady_problem || this->operator_data.convective_problem)
      value = integrator.get_value(q);

    vector flux;
    if(this->operator_data.convective_problem)
      flux += convective_kernel.get_volume_flux(value, integrator, q, this->eval_time);
    if(this->operator_data.diffusive_problem)
      flux += diffusive_kernel.get_volume_flux(integrator, q);

    if(this->operator_data.unsteady_problem)
      integrator.submit_value(mass_kernel.get_volume_flux(value), q);

    integrator.submit_gradient(flux, q);
  }
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::do_face_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = integrator_m.get_value(q);
    scalar value_p = integrator_p.get_value(q);

    scalar value_flux = make_vectorized_array<Number>(0.0);

    if(this->operator_data.convective_problem)
      value_flux +=
        convective_kernel.calculate_flux(q, integrator_m, value_m, value_p, this->eval_time, true);

    if(this->operator_data.diffusive_problem)
    {
      scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
      scalar normal_gradient_p = integrator_p.get_normal_derivative(q);

      value_flux += -diffusive_kernel.calculate_value_flux(normal_gradient_m,
                                                           normal_gradient_p,
                                                           value_m,
                                                           value_p);
    }

    integrator_m.submit_value(value_flux, q);
    integrator_p.submit_value(-value_flux, q);

    if(this->operator_data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel.calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
      integrator_p.submit_normal_derivative(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::do_face_int_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set value_p to zero
    scalar value_p = make_vectorized_array<Number>(0.0);
    scalar value_m = integrator_m.get_value(q);

    scalar value_flux = make_vectorized_array<Number>(0.0);

    if(this->operator_data.convective_problem)
      value_flux +=
        convective_kernel.calculate_flux(q, integrator_m, value_m, value_p, this->eval_time, true);

    if(this->operator_data.diffusive_problem)
    {
      // set exterior value to zero
      scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
      scalar normal_gradient_p = make_vectorized_array<Number>(0.0);

      value_flux += -diffusive_kernel.calculate_value_flux(normal_gradient_m,
                                                           normal_gradient_p,
                                                           value_m,
                                                           value_p);
    }

    integrator_m.submit_value(value_flux, q);

    if(this->operator_data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel.calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
    }
  }
}

// TODO can be removed later once matrix-free evaluation allows accessing neighboring data for
// cell-based face loops
template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::do_face_int_integral_cell_based(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    // set value_p to zero
    scalar value_p = make_vectorized_array<Number>(0.0);
    scalar value_m = integrator_m.get_value(q);

    scalar value_flux = make_vectorized_array<Number>(0.0);

    if(this->operator_data.convective_problem)
    {
      // TODO
      // The matrix-free implementation in deal.II does currently not allow to access neighboring
      // data in case of cell-based face loops. We therefore have to use integrator_velocity_m twice
      // to avoid the problem of accessing data of the neighboring element. Note that this variant
      // calculates the diagonal and block-diagonal only approximately. The theoretically correct
      // version using integrator_velocity_p is currently not implemented in deal.II.
      bool exterior_velocity_available =
        false; // TODO -> set to true once functionality is available
      value_flux += convective_kernel.calculate_flux(
        q, integrator_m, value_m, value_p, this->eval_time, exterior_velocity_available);
    }

    if(this->operator_data.diffusive_problem)
    {
      // set exterior value to zero
      scalar normal_gradient_m = integrator_m.get_normal_derivative(q);
      scalar normal_gradient_p = make_vectorized_array<Number>(0.0);

      value_flux += -diffusive_kernel.calculate_value_flux(normal_gradient_m,
                                                           normal_gradient_p,
                                                           value_m,
                                                           value_p);
    }

    integrator_m.submit_value(value_flux, q);

    if(this->operator_data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel.calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::do_face_ext_integral(
  IntegratorFace & integrator_m,
  IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    // set value_m to zero
    scalar value_m = make_vectorized_array<Number>(0.0);
    scalar value_p = integrator_p.get_value(q);

    scalar value_flux = make_vectorized_array<Number>(0.0);

    if(this->operator_data.convective_problem)
    {
      // minus sign for convective flux since n⁺ = -n⁻
      value_flux -=
        convective_kernel.calculate_flux(q, integrator_p, value_m, value_p, this->eval_time, true);
    }

    if(this->operator_data.diffusive_problem)
    {
      // set gradient_m to zero
      scalar normal_gradient_m = make_vectorized_array<Number>(0.0);
      // minus sign to get the correct normal vector n⁺ = -n⁻
      scalar normal_gradient_p = -integrator_p.get_normal_derivative(q);

      value_flux += -diffusive_kernel.calculate_value_flux(normal_gradient_p,
                                                           normal_gradient_m,
                                                           value_p,
                                                           value_m);
    }

    integrator_p.submit_value(value_flux, q);

    if(this->operator_data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel.calculate_gradient_flux(value_p, value_m);
      // opposite sign since n⁺ = -n⁻
      integrator_p.submit_normal_derivative(-gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::do_boundary_integral(
  IntegratorFace &           integrator_m,
  OperatorType const &       operator_type,
  types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    scalar value_m = calculate_interior_value(q, integrator_m, operator_type);
    scalar value_p = calculate_exterior_value(value_m,
                                              q,
                                              integrator_m,
                                              operator_type,
                                              boundary_type,
                                              boundary_id,
                                              this->operator_data.bc,
                                              this->eval_time);

    scalar value_flux = make_vectorized_array<Number>(0.0);

    if(this->operator_data.convective_problem)
    {
      // In case of numerical velocity field:
      // Simply use velocity_p = velocity_m on boundary faces -> exterior_velocity_available =
      // false.
      value_flux +=
        convective_kernel.calculate_flux(q, integrator_m, value_m, value_p, this->eval_time, false);
    }

    if(this->operator_data.diffusive_problem)
    {
      scalar normal_gradient_m = calculate_interior_normal_gradient(q, integrator_m, operator_type);
      scalar normal_gradient_p = calculate_exterior_normal_gradient(normal_gradient_m,
                                                                    q,
                                                                    integrator_m,
                                                                    operator_type,
                                                                    boundary_type,
                                                                    boundary_id,
                                                                    this->operator_data.bc,
                                                                    this->eval_time);

      value_flux += -diffusive_kernel.calculate_value_flux(normal_gradient_m,
                                                           normal_gradient_p,
                                                           value_m,
                                                           value_p);
    }

    integrator_m.submit_value(value_flux, q);

    if(this->operator_data.diffusive_problem)
    {
      scalar gradient_flux = diffusive_kernel.calculate_gradient_flux(value_m, value_p);
      integrator_m.submit_normal_derivative(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::do_verify_boundary_conditions(
  types::boundary_id const                           boundary_id,
  ConvectionDiffusionOperatorMergedData<dim> const & operator_data,
  std::set<types::boundary_id> const &               periodic_boundary_ids) const
{
  do_verify_boundary_conditions(boundary_id, operator_data, periodic_boundary_ids);
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim, Number>::apply_inverse_block_diagonal(
  VectorType &       dst,
  VectorType const & src) const
{
  // matrix-free
  if(this->operator_data.implement_block_diagonal_preconditioner_matrix_free)
  {
    // Solve elementwise block Jacobi problems iteratively using an elementwise solver vectorized
    // over several elements.
    bool update_preconditioner = false;
    elementwise_solver->solve(dst, src, update_preconditioner);
  }
  else // matrix-based
  {
    // Simply apply inverse of block matrices (using the LU factorization that has been computed
    // before).
    Base::apply_inverse_block_diagonal_matrix_based(dst, src);
  }
}

template<int dim, typename Number>
void
ConvectionDiffusionOperatorMerged<dim,
                                  Number>::initialize_block_diagonal_preconditioner_matrix_free()
  const
{
  elementwise_operator.reset(new ELEMENTWISE_OPERATOR(*this));

  if(this->operator_data.preconditioner_block_jacobi == PreconditionerBlockDiagonal::None)
  {
    typedef Elementwise::PreconditionerIdentity<VectorizedArray<Number>> IDENTITY;
    elementwise_preconditioner.reset(new IDENTITY(elementwise_operator->get_problem_size()));
  }
  else if(this->operator_data.preconditioner_block_jacobi ==
          PreconditionerBlockDiagonal::InverseMassMatrix)
  {
    typedef Elementwise::InverseMassMatrixPreconditioner<dim, 1 /*scalar equation*/, Number>
      INVERSE_MASS;

    elementwise_preconditioner.reset(
      new INVERSE_MASS(this->get_matrix_free(), this->get_dof_index(), this->get_quad_index()));
  }
  else
  {
    AssertThrow(false, ExcMessage("Not implemented."));
  }

  Elementwise::IterativeSolverData iterative_solver_data;
  iterative_solver_data.solver_type = Elementwise::SolverType::GMRES;
  iterative_solver_data.solver_data = this->operator_data.block_jacobi_solver_data;

  elementwise_solver.reset(new ELEMENTWISE_SOLVER(
    *std::dynamic_pointer_cast<ELEMENTWISE_OPERATOR>(elementwise_operator),
    *std::dynamic_pointer_cast<ELEMENTWISE_PRECONDITIONER>(elementwise_preconditioner),
    iterative_solver_data));
}

template class ConvectionDiffusionOperatorMerged<2, float>;
template class ConvectionDiffusionOperatorMerged<2, double>;

template class ConvectionDiffusionOperatorMerged<3, float>;
template class ConvectionDiffusionOperatorMerged<3, double>;

} // namespace ConvDiff
