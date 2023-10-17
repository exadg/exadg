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

#include <exadg/structure/spatial_discretization/operators/boundary_conditions.h>
#include <exadg/structure/spatial_discretization/operators/continuum_mechanics.h>
#include <exadg/structure/spatial_discretization/operators/nonlinear_operator.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  OperatorData<dim> const &                 data)
{
  Base::initialize(matrix_free, affine_constraints, data);

  integrator_lin = std::make_shared<IntegratorCell>(*this->matrix_free,
                                                    this->operator_data.dof_index_inhomogeneous,
                                                    this->operator_data.quad_index);

  // it should not make a difference here whether we use dof_index or dof_index_inhomogeneous
  this->matrix_free->initialize_dof_vector(displacement_lin, this->operator_data.dof_index);
  displacement_lin.update_ghost_values();

  if(this->operator_data.spatial_integration)
  {
    // Deep copy of matrix_free to use different mappings.
    this->matrix_free_spatial.copy_from(*this->matrix_free);

    // Setup spatial mapping based on linearization vector and undeformed mapping,
    // where parameter check enforces mapping_degree == degree.
    this->mapping_spatial =
      std::make_shared<MappingDoFVector<dim, Number>>(this->operator_data.mapping_degree);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::evaluate_nonlinear(VectorType & dst, VectorType const & src) const
{
  AssertThrow(this->operator_data.pull_back_traction == false,
              dealii::ExcMessage("Neumann data expected in respective "
                                 "configuration for comparison."));

  if(this->operator_data.spatial_integration and (not this->operator_data.force_material_residual))
  {
    this->matrix_free_spatial.loop(&This::cell_loop_nonlinear,
                                   &This::face_loop_nonlinear,
                                   &This::boundary_face_loop_nonlinear,
                                   this,
                                   dst,
                                   src,
                                   true);
  }
  else
  {
    this->matrix_free->loop(&This::cell_loop_nonlinear,
                            &This::face_loop_nonlinear,
                            &This::boundary_face_loop_nonlinear,
                            this,
                            dst,
                            src,
                            true);
  }
}

template<int dim, typename Number>
bool
NonLinearOperator<dim, Number>::valid_deformation(VectorType const & displacement) const
{
  Number dst = 0.0;

  // dst has to remain zero for a valid deformation state
  this->matrix_free->cell_loop(&This::cell_loop_valid_deformation,
                               this,
                               dst,
                               displacement,
                               false /* no zeroing of dst vector */);

  // sum over all MPI processes
  Number valid = 0.0;
  valid        = dealii::Utilities::MPI::sum(
    dst, this->matrix_free->get_dof_handler(this->operator_data.dof_index).get_communicator());

  return (valid == 0.0);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::set_mapping_undeformed(
  std::shared_ptr<dealii::Mapping<dim> const> mapping) const
{
  this->mapping_undeformed = mapping;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::set_solution_linearization(VectorType const & vector,
                                                           bool const         update_mapping) const
{
  // Check for valid deformation state.
  bool const valid_deformation_field = valid_deformation(vector);
  if(not valid_deformation_field)
  {
	std::cout << "the linearization vector does not correspond to an invertible mapping on lvl "
			  << this->get_level() << "  ## \n";
  }

  if(valid_deformation_field || this->operator_data.check_type != 1)
  {
    displacement_lin = vector;
    displacement_lin.update_ghost_values();

    // update cached linearization data
    this->set_cell_linearization_data(displacement_lin);

    // update mapping to spatial configuration
    if(this->operator_data.spatial_integration and update_mapping)
    {
      this->mapping_spatial->initialize_mapping_q_cache(this->mapping_undeformed,
                                                        displacement_lin,
                                                        this->matrix_free->get_dof_handler());
      this->matrix_free_spatial.update_mapping(*mapping_spatial);
    }
  }
}

template<int dim, typename Number>
typename NonLinearOperator<dim, Number>::VectorType const &
NonLinearOperator<dim, Number>::get_solution_linearization() const
{
  return displacement_lin;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::apply(VectorType & dst, VectorType const & src) const
{
  AssertThrow(not this->is_dg, dealii::ExcMessage("NonLinearOperator::apply supports CG only"));

  if(this->operator_data.spatial_integration)
  {
    // Compute matrix-vector product. Constrained degrees of freedom in the src-vector will not be
    // used. The function read_dof_values() (or gather_evaluate()) uses the homogeneous boundary
    // data passed to MatrixFree via AffineConstraints with the standard "dof_index".
    this->matrix_free_spatial.cell_loop(&This::cell_loop, this, dst, src, true);

    // Constrained degree of freedom are not removed from the system of equations.
    // Instead, we set the diagonal entries of the matrix to 1 for these constrained
    // degrees of freedom. This means that we simply copy the constrained values to the
    // dst vector.
    for(unsigned int const constrained_index :
        this->matrix_free_spatial.get_constrained_dofs(this->operator_data.dof_index))
    {
      dst.local_element(constrained_index) = src.local_element(constrained_index);
    }
  }
  else
  {
    OperatorBase<dim, Number, dim /* n_components */>::apply(dst, src);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::apply_add(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;
  AssertThrow(false, dealii::ExcMessage("NonLinearOperator::apply_add not implemented."));
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::rhs(VectorType & dst) const
{
  (void)dst;
  AssertThrow(false, dealii::ExcMessage("NonLinearOperator::rhs not implemented."));
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::rhs_add(VectorType & dst) const
{
  (void)dst;
  AssertThrow(false, dealii::ExcMessage("NonLinearOperator::rhs_add not implemented."));
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::evaluate(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;
  AssertThrow(false, dealii::ExcMessage("NonLinearOperator::evaluate not implemented."));
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::evaluate_add(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;
  AssertThrow(false, dealii::ExcMessage("NonLinearOperator::evaluate_add not implemented."));
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::add_diagonal(VectorType & diagonal) const
{
  AssertThrow(not this->is_dg,
              dealii::ExcMessage("NonLinearOperator::add_diagonal supports CG only"));

  if(this->operator_data.spatial_integration)
  {
    dealii::MatrixFreeTools::
      compute_diagonal<dim, -1, 0, dim /* n_components */, Number, dealii::VectorizedArray<Number>>(
        this->matrix_free_spatial,
        diagonal,
        [&](auto & integrator) -> void {
          // TODO: this is currently done for every column, but would only be necessary
          // once per cell
          this->reinit_cell_derived(integrator, integrator.get_current_cell_index());

          integrator.evaluate(this->integrator_flags.cell_evaluate);

          this->do_cell_integral(integrator);

          integrator.integrate(this->integrator_flags.cell_integrate);
        },
        this->operator_data.dof_index,
        this->operator_data.quad_index);
  }
  else
  {
    OperatorBase<dim, Number, dim /* n_components */>::add_diagonal(diagonal);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::reinit_cell_nonlinear(IntegratorCell &   integrator,
                                                      unsigned int const cell) const
{
  integrator.reinit(cell);

  this->material_handler.reinit(*this->matrix_free, cell);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::cell_loop_nonlinear(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorCell integrator_inhom(matrix_free,
                                  this->operator_data.dof_index_inhomogeneous,
                                  this->operator_data.quad_index);

  IntegratorCell integrator(matrix_free,
                            this->operator_data.dof_index,
                            this->operator_data.quad_index);

  auto const unsteady_flag = this->operator_data.unsteady ? dealii::EvaluationFlags::values :
                                                            dealii::EvaluationFlags::nothing;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    reinit_cell_nonlinear(integrator_inhom, cell);
    integrator.reinit(cell);

    if(this->operator_data.spatial_integration and
       (not this->operator_data.force_material_residual))
    {
      integrator_lin->reinit(cell);
      integrator_lin->read_dof_values(displacement_lin);
      integrator_lin->evaluate(dealii::EvaluationFlags::gradients);
    }

    integrator_inhom.gather_evaluate(src, unsteady_flag | dealii::EvaluationFlags::gradients);

    do_cell_integral_nonlinear(integrator_inhom);

    // make sure that we do not write into Dirichlet degrees of freedom
    integrator_inhom.integrate(unsteady_flag | dealii::EvaluationFlags::gradients,
                               integrator.begin_dof_values());
    integrator.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::set_cell_linearization_data(
  VectorType const & linearization_vector) const
{
  if(this->operator_data.cache_level > 0)
  {
    VectorType dummy;
    this->matrix_free->cell_loop(&This::cell_loop_set_linearization_data,
                                 this,
                                 dummy,
                                 linearization_vector);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::cell_loop_set_linearization_data(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;

  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    integrator_lin->reinit(cell);
    this->material_handler.reinit(*this->matrix_free, cell);

    integrator_lin->read_dof_values(src);
    integrator_lin->evaluate(dealii::EvaluationFlags::gradients);

    material->do_set_cell_linearization_data(integrator_lin, cell);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::cell_loop_valid_deformation(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  Number &                                dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorCell integrator(matrix_free,
                            this->operator_data.dof_index_inhomogeneous,
                            this->operator_data.quad_index);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    reinit_cell_nonlinear(integrator, cell);

    integrator.gather_evaluate(src, dealii::EvaluationFlags::gradients);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // material displacement gradient
      tensor const Grad_d = integrator.get_gradient(q);

      // material deformation gradient
      tensor const F = get_F(Grad_d);
      scalar const J = determinant(F);
      for(unsigned int v = 0; v < J.size(); ++v)
      {
        // if deformation is invalid, add a positive value to dst
        if(J[v] <= 0.0)
        {
          dst += 1.0;
        }
      }
    }
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::face_loop_nonlinear(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  (void)matrix_free;
  (void)dst;
  (void)src;
  (void)range;
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::boundary_face_loop_nonlinear(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range) const
{
  IntegratorFace integrator_m_inhom(matrix_free,
                                    true,
                                    this->operator_data.dof_index_inhomogeneous,
                                    this->operator_data.quad_index);

  IntegratorFace integrator_m = IntegratorFace(matrix_free,
                                               true,
                                               this->operator_data.dof_index,
                                               this->operator_data.quad_index);

  // apply Neumann BCs
  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(integrator_m_inhom, face);
    integrator_m.reinit(face);

    // In case of a pull-back of the traction vector, we need to evaluate
    // the displacement gradient to obtain the surface area ratio da/dA.
    // We write the integrator flags explicitly in this case since they
    // depend on the parameter pull_back_traction.
    if(this->operator_data.pull_back_traction)
    {
      integrator_m_inhom.gather_evaluate(src, dealii::EvaluationFlags::gradients);
    }

    do_boundary_integral_continuous(integrator_m_inhom, matrix_free.get_boundary_id(face));

    // make sure that we do not write into Dirichlet degrees of freedom
    integrator_m_inhom.integrate(this->integrator_flags.face_integrate,
                                 integrator_m.begin_dof_values());
    integrator_m.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_cell_integral_nonlinear(IntegratorCell & integrator) const
{
  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  if(this->operator_data.spatial_integration and (not this->operator_data.force_material_residual))
  {
    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // material gradient of the linearization vector
      tensor Grad_d_lin;
      if(this->operator_data.cache_level < 2)
      {
        Grad_d_lin = integrator_lin->get_gradient(q);
      }
      else
      {
        // Grad_d_lin : dummy tensor sufficient for function call.
      }

      scalar one_over_J;
      if(this->operator_data.cache_level == 0)
      {
        scalar J;
        tensor F;
        get_modified_F_J(F, J, Grad_d_lin, this->operator_data.check_type, true /* compute_J */);
        one_over_J = 1.0 / J;
      }
      else
      {
        one_over_J = material->one_over_J(integrator.get_current_cell_index(), q);
      }

      // Kirchhoff stresses
      tensor const tau =
        material->kirchhoff_stress(Grad_d_lin, integrator.get_current_cell_index(), q);

      // integral over spatial domain
      integrator.submit_gradient(tau * one_over_J, q);

      if(this->operator_data.unsteady)
      {
        integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                  integrator.get_value(q) * one_over_J,
                                q);
      }
    }
  }
  else
  {
    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // material gradient of the linearization vector and deformation gradient
      tensor Grad_d, F;
      if(this->operator_data.cache_level < 2)
      {
        Grad_d = integrator.get_gradient(q);
        scalar J;
        get_modified_F_J(F, J, Grad_d, this->operator_data.check_type, false /* compute_J */);
      }
      else
      {
        // Grad_d : dummy tensor sufficient for function call.
        F = material->deformation_gradient(integrator.get_current_cell_index(), q);
      }

      // 2nd Piola-Kirchhoff stresses
      tensor const S =
        material->second_piola_kirchhoff_stress(Grad_d, integrator.get_current_cell_index(), q);

      // 1st Piola-Kirchhoff stresses P = F * S
      tensor const P = F * S;

      // Grad_v : P
      integrator.submit_gradient(P, q);

      if(this->operator_data.unsteady)
      {
        integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                  integrator.get_value(q),
                                q);
      }
    }
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_boundary_integral_continuous(
  IntegratorFace &                   integrator,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryType boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    auto traction = calculate_neumann_value<dim, Number>(
      q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->time);

    if(this->operator_data.pull_back_traction)
    {
      tensor const F = get_F(integrator.get_gradient(q));
      vector const N = integrator.get_normal_vector(q);
      // da/dA * n = det F F^{-T} * N := n_star
      // -> da/dA = n_star.norm()
      vector const n_star = determinant(F) * transpose(invert(F)) * N;
      // t_0 = da/dA * t
      traction *= n_star.norm();
    }

    integrator.submit_value(-traction, q);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::reinit_cell_derived(IntegratorCell &   integrator,
                                                    unsigned int const cell) const
{
  Base::reinit_cell_derived(integrator, cell);

  integrator_lin->reinit(cell);

  integrator_lin->read_dof_values(displacement_lin);
  integrator_lin->evaluate(dealii::EvaluationFlags::gradients);
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  if(this->operator_data.spatial_integration)
  {
    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // spatial gradient of the displacement increment
      tensor const grad_delta = integrator.get_gradient(q);

      // material gradient of the linearization vector and displacement gradient
      scalar J_lin;
      tensor Grad_d_lin, F_lin;
      if(this->operator_data.cache_level < 2)
      {
        Grad_d_lin = integrator_lin->get_gradient(q);
        get_modified_F_J(
          F_lin, J_lin, Grad_d_lin, this->operator_data.check_type, false /* compute_J */);
      }
      else
      {
        // Grad_d_lin : dummy tensor sufficient for function call.
        F_lin = material->deformation_gradient(integrator.get_current_cell_index(), q);
      }

      scalar one_over_J;
      if(this->operator_data.cache_level == 0)
      {
        J_lin      = determinant(F_lin);
        one_over_J = 1.0 / J_lin;
      }
      else
      {
        one_over_J = material->one_over_J(integrator.get_current_cell_index(), q);
      }

      // Kirchhoff stresses
      tensor const tau_lin =
        material->kirchhoff_stress(Grad_d_lin, integrator.get_current_cell_index(), q);

      // material part of the directional derivative
      tensor delta_tau = material->contract_with_J_times_C(
        0.5 * (grad_delta + transpose(grad_delta)), F_lin, integrator.get_current_cell_index(), q);

      // integral over spatial domain
      integrator.submit_gradient((delta_tau + grad_delta * tau_lin) * one_over_J, q);

      if(this->operator_data.unsteady)
      {
        integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                  integrator.get_value(q) * one_over_J,
                                q);
      }
    }
  }
  else
  {
    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // material gradient of the displacement increment
      tensor const Grad_delta = integrator.get_gradient(q);

      // material gradient of the linearization vector and displacement gradient
      tensor Grad_d_lin, F_lin;
      if(this->operator_data.cache_level < 2)
      {
        Grad_d_lin = integrator_lin->get_gradient(q);
        scalar J_lin;
        get_modified_F_J(
          F_lin, J_lin, Grad_d_lin, this->operator_data.check_type, false /* compute_J */);
      }
      else
      {
        // Grad_d_lin : dummy tensor sufficient for function call.
        F_lin = material->deformation_gradient(integrator.get_current_cell_index(), q);
      }

      // 2nd Piola-Kirchhoff stresses
      tensor const S_lin =
        material->second_piola_kirchhoff_stress(Grad_d_lin, integrator.get_current_cell_index(), q);

      // directional derivative of 1st Piola-Kirchhoff stresses P

      // 1. elastic and initial displacement stiffness contributions
      tensor delta_P = F_lin * material->second_piola_kirchhoff_stress_displacement_derivative(
                                 Grad_delta, F_lin, integrator.get_current_cell_index(), q);

      // 2. geometric (or initial stress) stiffness contribution
      delta_P += Grad_delta * S_lin;

      // Grad_v : delta_P
      integrator.submit_gradient(delta_P, q);

      if(this->operator_data.unsteady)
      {
        integrator.submit_value(this->scaling_factor_mass * this->operator_data.density *
                                  integrator.get_value(q),
                                q);
      }
    }
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::cell_loop(dealii::MatrixFree<dim, Number> const & matrix_free,
                                          VectorType &                            dst,
                                          VectorType const &                      src,
                                          Range const &                           range) const
{
  if(this->operator_data.spatial_integration)
  {
    IntegratorCell integrator = IntegratorCell(this->matrix_free_spatial,
                                               this->operator_data.dof_index,
                                               this->operator_data.quad_index);

    for(auto cell = range.first; cell < range.second; ++cell)
    {
      integrator.reinit(cell);
      this->reinit_cell_derived(integrator, cell);

      integrator.gather_evaluate(src, this->integrator_flags.cell_evaluate);

      this->do_cell_integral(integrator);

      integrator.integrate_scatter(this->integrator_flags.cell_integrate, dst);
    }
  }
  else
  {
    OperatorBase<dim, Number, dim /* n_components */>::cell_loop(matrix_free, dst, src, range);
  }
}

#ifdef DEAL_II_WITH_TRILINOS
template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::calculate_system_matrix(
  dealii::TrilinosWrappers::SparseMatrix & system_matrix) const
{
  if(this->operator_data.spatial_integration)
  {
    this->matrix_free_spatial.cell_loop(
      &OperatorBase<dim, Number, dim /* n_components */>::cell_loop_calculate_system_matrix,
      this,
      system_matrix,
      system_matrix);

    // communicate overlapping matrix parts
    system_matrix.compress(dealii::VectorOperation::add);
  }
  else
  {
    OperatorBase<dim, Number, dim /* n_components */>::internal_calculate_system_matrix(
      system_matrix);
  }
}
#endif

#ifdef DEAL_II_WITH_PETSC
template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::calculate_system_matrix(
  dealii::PETScWrappers::MPI::SparseMatrix & system_matrix) const
{
  if(this->operator_data.spatial_integration)
  {
    this->matrix_free_spatial.cell_loop(
      &OperatorBase<dim, Number, dim /* n_components */>::cell_loop_calculate_system_matrix,
      this,
      system_matrix,
      system_matrix);

    // communicate overlapping matrix parts
    system_matrix.compress(dealii::VectorOperation::add);
  }
  else
  {
    OperatorBase<dim, Number, dim /* n_components */>::internal_calculate_system_matrix(
      system_matrix);
  }
}
#endif

template class NonLinearOperator<2, float>;
template class NonLinearOperator<2, double>;

template class NonLinearOperator<3, float>;
template class NonLinearOperator<3, double>;

} // namespace Structure
} // namespace ExaDG
