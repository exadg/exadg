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

  // integrator_lin always refers to evaluation in the reference configuration.
  integrator_lin = std::make_shared<IntegratorCell>(*this->matrix_free,
                                                    this->operator_data.dof_index_inhomogeneous,
                                                    this->operator_data.quad_index);

  // It should not make a difference here whether we use dof_index or dof_index_inhomogeneous.
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
NonLinearOperator<dim, Number>::set_solution_linearization(
  VectorType const & vector,
  bool const         update_cell_data,
  bool const         update_mapping,
  bool const         update_matrix_if_necessary) const
{
  // Check for valid deformation state.
  bool const valid_deformation_field = valid_deformation(vector);
  if(not valid_deformation_field)
  {
    std::cout << "the linearization vector does not correspond to an invertible mapping on lvl "
              << this->get_level() << "  ## \n";
  }

  if(valid_deformation_field or this->operator_data.check_type != 1)
  {
    displacement_lin = vector;
    displacement_lin.update_ghost_values();

    // update cached linearization data
    if(update_cell_data)
    {
      this->set_cell_linearization_data(displacement_lin);
    }

    // update mapping to spatial configuration
    if(this->operator_data.spatial_integration and update_mapping)
    {
      this->mapping_spatial->initialize_mapping_from_dof_vector(
        this->mapping_undeformed, displacement_lin, this->matrix_free->get_dof_handler());
      this->matrix_free_spatial.update_mapping(*mapping_spatial->get_mapping());
    }

    if(update_matrix_if_necessary)
    {
      this->assemble_matrix_if_necessary();
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
NonLinearOperator<dim, Number>::apply_before_after(
  VectorType &                                                        dst,
  VectorType const &                                                  src,
  std::function<void(const unsigned int, const unsigned int)> const & before,
  std::function<void(const unsigned int, const unsigned int)> const & after) const
{
  AssertThrow(not this->is_dg, dealii::ExcMessage("NonLinearOperator::apply supports CG only"));

  if(this->operator_data.spatial_integration)
  {
    // Compute matrix-vector product. Constrained degrees of freedom in the src-vector will not be
    // used. The function read_dof_values() (or gather_evaluate()) uses the homogeneous boundary
    // data passed to MatrixFree via AffineConstraints with the standard "dof_index".
    this->matrix_free_spatial.cell_loop(&This::cell_loop, this, dst, src, before, after);

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
    OperatorBase<dim, Number, dim /* n_components */>::apply_before_after(dst, src, before, after);
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
  // Note that `cache_level` does not play a role here, as the residual
  // is evaluated with `src` in any case.
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

    // Evaluating the stress terms requires interpolation in the reference
    // configuration. Since we call cell_loop_nonlinear() on the most recent
    // iterate, this is only needed for the spatial integration approach.
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
      tensor const F = compute_F(Grad_d);
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

  // apply Neumann or Robin BCs
  for(unsigned int face = range.first; face < range.second; face++)
  {
    this->reinit_boundary_face(integrator_m_inhom, face);
    integrator_m.reinit(face);

    // In case of a pull-back of the traction vector, we need to evaluate the displacement gradient
    // to obtain the surface area ratio da/dA. We write the integrator flags explicitly in this case
    // since they depend on the parameter pull_back_traction. On Robin boundaries, we need the
    // solution values.
    BoundaryType const boundary_type =
      this->operator_data.bc->get_boundary_type(matrix_free.get_boundary_id(face));
    bool const values_or_gradients_required =
      boundary_type == BoundaryType::Neumann or boundary_type == BoundaryType::NeumannCached or
      boundary_type == BoundaryType::RobinSpringDashpotPressure or
      this->operator_data.pull_back_traction;
    if(values_or_gradients_required)
    {
      integrator_m_inhom.gather_evaluate(src,
                                         dealii::EvaluationFlags::gradients |
                                           dealii::EvaluationFlags::values);
    }

    do_boundary_integral_continuous(integrator_m_inhom,
                                    OperatorType::full,
                                    matrix_free.get_boundary_id(face));

    // make sure that we do not write into Dirichlet degrees of freedom
    integrator_m_inhom.integrate(this->integrator_flags.face_integrate,
                                 integrator_m.begin_dof_values());
    integrator_m.distribute_local_to_global(dst);
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_boundary_integral_continuous(
  IntegratorFace &                   integrator,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  BoundaryType const boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

#ifdef DEBUG
  if(this->operator_data.spatial_integration)
  {
    // Assert nonzero Robin boundary terms for spatial integration.
    auto const it = this->operator_data.bc->robin_k_c_p_param.find(boundary_id);
    if(it != this->operator_data.bc->robin_k_c_p_param.end())
    {
      double const coefficient_displacement = it->second.second[0];
      double const coefficient_velocity     = it->second.second[1];

      if(std::abs(coefficient_displacement) > 1e-20 or std::abs(coefficient_velocity) > 1e-20)
      {
        AssertThrow(boundary_type != BoundaryType::RobinSpringDashpotPressure,
                    dealii::ExcMessage(
                      "Linearization of Robin terms incomplete for spatial integration."));
      }
    }
  }
#endif

  bool const spatial_residual_evaluation =
    this->operator_data.spatial_integration and not this->operator_data.force_material_residual;

  vector traction;

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    traction = 0.0;

    // integrate standard (stored) traction or exterior pressure on Robin boundaries
    if(boundary_type == BoundaryType::Neumann or boundary_type == BoundaryType::NeumannCached or
       boundary_type == BoundaryType::RobinSpringDashpotPressure)
    {
      if(operator_type == OperatorType::inhomogeneous or operator_type == OperatorType::full)
      {
        traction -= calculate_neumann_value<dim, Number>(
          q, integrator, boundary_type, boundary_id, this->operator_data.bc, this->time);

        if(this->operator_data.pull_back_traction or this->operator_data.spatial_integration)
        {
          tensor F = compute_F(integrator.get_gradient(q));
          vector N = integrator.get_normal_vector(q);
          // da/dA * n = det F F^{-T} * N := n_star
          // -> da/dA = n_star.norm()
          vector n_star = determinant(F) * transpose(invert(F)) * N;
          if(this->operator_data.pull_back_traction)
          {
            // t_0 = da/dA * t
            traction *= n_star.norm();
            AssertThrow(not spatial_residual_evaluation,
                        dealii::ExcMessage("Invalid combination."));
          }
          else if(spatial_residual_evaluation)
          {
            // t = dA/da * t_0
            traction /= n_star.norm();
            AssertThrow(not this->operator_data.pull_back_traction,
                        dealii::ExcMessage("Invalid combination."));
          }
        }
      }
    }

    // check boundary ID in robin_k_c_p_param to add boundary mass integrals from Robin boundaries
    // on BoundaryType::NeumannCached or BoundaryType::RobinSpringDashpotPressure
    if(boundary_type == BoundaryType::NeumannCached or
       boundary_type == BoundaryType::RobinSpringDashpotPressure)
    {
      if(operator_type == OperatorType::homogeneous or operator_type == OperatorType::full)
      {
        auto const it = this->operator_data.bc->robin_k_c_p_param.find(boundary_id);

        if(it != this->operator_data.bc->robin_k_c_p_param.end())
        {
          bool const   normal_projection_displacement = it->second.first[0];
          double const coefficient_displacement       = it->second.second[0];

          if(normal_projection_displacement)
          {
            vector const N = integrator.get_normal_vector(q);
            traction += N * (coefficient_displacement * (N * integrator.get_value(q)));
          }
          else
          {
            traction += coefficient_displacement * integrator.get_value(q);
          }

          if(this->operator_data.unsteady)
          {
            bool const   normal_projection_velocity = it->second.first[1];
            double const coefficient_velocity       = it->second.second[1];

            if(normal_projection_velocity)
            {
              vector const N = integrator.get_normal_vector(q);
              traction += N * (coefficient_velocity * this->scaling_factor_mass_boundary *
                               (N * integrator.get_value(q)));
            }
            else
            {
              traction +=
                coefficient_velocity * this->scaling_factor_mass_boundary * integrator.get_value(q);
            }
          }
        }
      }
    }

    integrator.submit_value(traction, q);
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
NonLinearOperator<dim, Number>::do_cell_integral_nonlinear(IntegratorCell & integrator) const
{
  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  unsigned int constexpr check_type = 0;
  AssertThrow(this->operator_data.check_type == 0,
              dealii::ExcMessage("A `check_type` > 0 is currently not implemented here."));

  if(this->operator_data.spatial_integration and (not this->operator_data.force_material_residual))
  {
    // Evaluate the residual operator in the spatial configuration.
    if(this->operator_data.stable_formulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        tensor const Grad_d_lin = integrator_lin->get_gradient(q);
        scalar const Jm1_lin =
          compute_modified_Jm1<dim, Number, check_type, true /* stable_formulation */>(Grad_d_lin);
        scalar const           one_over_J_lin = 1.0 / (Jm1_lin + 1.0);
        symmetric_tensor const tau_lin =
          material->kirchhoff_stress_eval(Grad_d_lin, integrator.get_current_cell_index(), q);

        integrator.submit_gradient(tau_lin * one_over_J_lin, q);

        if(this->operator_data.unsteady)
        {
          integrator.submit_value((this->scaling_factor_mass * this->operator_data.density *
                                   one_over_J_lin) *
                                    integrator.get_value(q),
                                  q);
        }
      }
    }
    else
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        tensor const Grad_d_lin = integrator_lin->get_gradient(q);
        scalar const Jm1_lin =
          compute_modified_Jm1<dim, Number, check_type, true /* stable_formulation */>(Grad_d_lin);
        scalar const           one_over_J_lin = 1.0 / (Jm1_lin + 1.0);
        symmetric_tensor const tau_lin =
          material->kirchhoff_stress_eval(Grad_d_lin, integrator.get_current_cell_index(), q);

        integrator.submit_gradient(tau_lin * one_over_J_lin, q);

        if(this->operator_data.unsteady)
        {
          integrator.submit_value(((this->scaling_factor_mass * this->operator_data.density) *
                                   one_over_J_lin) *
                                    integrator.get_value(q),
                                  q);
        }
      }
    }
  }
  else
  {
    // Evaluate the residual operator in the material configuration.
    // `integrator_lin` is not initialized on this cell, since we call
    // this cell loop only on the most recent iterate.
    if(this->operator_data.stable_formulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        tensor const Grad_d_lin = integrator.get_gradient(q);
        tensor const F_lin =
          compute_modified_F<dim, Number, check_type, true /* stable_formulation */>(Grad_d_lin);
        symmetric_tensor const S_lin =
          material->second_piola_kirchhoff_stress_eval(Grad_d_lin,
                                                       integrator.get_current_cell_index(),
                                                       q);

        integrator.submit_gradient(F_lin * S_lin, q);

        if(this->operator_data.unsteady)
        {
          integrator.submit_value((this->scaling_factor_mass * this->operator_data.density) *
                                    integrator.get_value(q),
                                  q);
        }
      }
    }
    else
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        tensor const Grad_d_lin = integrator.get_gradient(q);
        tensor const F_lin =
          compute_modified_F<dim, Number, check_type, false /* stable_formulation */>(Grad_d_lin);
        symmetric_tensor const S_lin =
          material->second_piola_kirchhoff_stress_eval(Grad_d_lin,
                                                       integrator.get_current_cell_index(),
                                                       q);

        integrator.submit_gradient(F_lin * S_lin, q);

        if(this->operator_data.unsteady)
        {
          integrator.submit_value((this->scaling_factor_mass * this->operator_data.density) *
                                    integrator.get_value(q),
                                  q);
        }
      }
    }
  }
}

template<int dim, typename Number>
void
NonLinearOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

  unsigned int constexpr check_type = 0;
  AssertThrow(this->operator_data.check_type == 0,
              dealii::ExcMessage("A `check_type` > 0 is currently not implemented here."));

  AssertThrow(this->operator_data.cache_level < 3,
              dealii::ExcMessage("We expect a `cache_level` in [0,1,2]."));

  if(this->operator_data.spatial_integration)
  {
    // Evaluate the linearized operator in the spatial configuration.
    if(this->operator_data.cache_level == 0)
    {
      if(this->operator_data.stable_formulation)
      {
        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          tensor const           Grad_d_lin = integrator_lin->get_gradient(q);
          tensor const           grad_delta = integrator.get_gradient(q);
          symmetric_tensor const delta_tau  = material->contract_with_J_times_C(
            symmetrize(grad_delta), Grad_d_lin, integrator.get_current_cell_index(), q);
          scalar const Jm1_lin =
            compute_modified_Jm1<dim, Number, check_type, true /* stable_formulation */>(
              Grad_d_lin);
          scalar const           one_over_J_lin = 1.0 / (Jm1_lin + 1.0);
          symmetric_tensor const tau_lin =
            material->kirchhoff_stress(Grad_d_lin, integrator.get_current_cell_index(), q);

          integrator.submit_gradient((delta_tau + grad_delta * tau_lin) * one_over_J_lin, q);

          if(this->operator_data.unsteady)
          {
            integrator.submit_value(((this->scaling_factor_mass * this->operator_data.density) *
                                     one_over_J_lin) *
                                      integrator.get_value(q),
                                    q);
          }
        }
      }
      else
      {
        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          tensor const           Grad_d_lin = integrator_lin->get_gradient(q);
          tensor const           grad_delta = integrator.get_gradient(q);
          symmetric_tensor const delta_tau  = material->contract_with_J_times_C(
            symmetrize(grad_delta), Grad_d_lin, integrator.get_current_cell_index(), q);
          scalar const Jm1_lin =
            compute_modified_Jm1<dim, Number, check_type, false /* stable_formulation */>(
              Grad_d_lin);
          scalar const           one_over_J_lin = 1.0 / (Jm1_lin + 1.0);
          symmetric_tensor const tau_lin =
            material->kirchhoff_stress(Grad_d_lin, integrator.get_current_cell_index(), q);

          integrator.submit_gradient((delta_tau + grad_delta * tau_lin) * one_over_J_lin, q);

          if(this->operator_data.unsteady)
          {
            integrator.submit_value(((this->scaling_factor_mass * this->operator_data.density) *
                                     one_over_J_lin) *
                                      integrator.get_value(q),
                                    q);
          }
        }
      }
    }
    else if(this->operator_data.cache_level == 1)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        tensor const           Grad_d_lin = integrator_lin->get_gradient(q);
        tensor const           grad_delta = integrator.get_gradient(q);
        symmetric_tensor const delta_tau  = material->contract_with_J_times_C(
          symmetrize(grad_delta), Grad_d_lin, integrator.get_current_cell_index(), q);
        scalar const one_over_J_lin = material->one_over_J(integrator.get_current_cell_index(), q);
        symmetric_tensor const tau_lin =
          material->kirchhoff_stress(Grad_d_lin, integrator.get_current_cell_index(), q);

        integrator.submit_gradient((delta_tau + grad_delta * tau_lin) * one_over_J_lin, q);

        if(this->operator_data.unsteady)
        {
          integrator.submit_value(((this->scaling_factor_mass * this->operator_data.density) *
                                   one_over_J_lin) *
                                    integrator.get_value(q),
                                  q);
        }
      }
    }
    else
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        tensor const           grad_delta = integrator.get_gradient(q);
        symmetric_tensor const delta_tau =
          material->contract_with_J_times_C(symmetrize(grad_delta),
                                            integrator.get_current_cell_index(),
                                            q);
        scalar const one_over_J_lin = material->one_over_J(integrator.get_current_cell_index(), q);
        symmetric_tensor const tau_lin =
          material->kirchhoff_stress(integrator.get_current_cell_index(), q);
        integrator.submit_gradient((delta_tau + grad_delta * tau_lin) * one_over_J_lin, q);

        if(this->operator_data.unsteady)
        {
          integrator.submit_value(((this->scaling_factor_mass * this->operator_data.density) *
                                   one_over_J_lin) *
                                    integrator.get_value(q),
                                  q);
        }
      }
    }
  }
  else
  {
    // Evaluate the linearized operator in the material configuration.
    if(this->operator_data.cache_level < 2)
    {
      if(this->operator_data.stable_formulation)
      {
        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          tensor const Grad_d_lin = integrator_lin->get_gradient(q);
          tensor const Grad_delta = integrator.get_gradient(q);
          tensor const F_lin =
            compute_modified_F<dim, Number, check_type, true /* stable_formulation */>(Grad_d_lin);
          symmetric_tensor const S_lin =
            material->second_piola_kirchhoff_stress(Grad_d_lin,
                                                    integrator.get_current_cell_index(),
                                                    q);
          tensor const delta_P =
            Grad_delta * S_lin +
            F_lin * material->second_piola_kirchhoff_stress_displacement_derivative(
                      Grad_delta, Grad_d_lin, integrator.get_current_cell_index(), q);

          integrator.submit_gradient(delta_P, q);

          if(this->operator_data.unsteady)
          {
            integrator.submit_value((this->scaling_factor_mass * this->operator_data.density) *
                                      integrator.get_value(q),
                                    q);
          }
        }
      }
      else
      {
        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          tensor const Grad_d_lin = integrator_lin->get_gradient(q);
          tensor const Grad_delta = integrator.get_gradient(q);
          tensor const F_lin =
            compute_modified_F<dim, Number, check_type, false /* stable_formulation */>(Grad_d_lin);
          symmetric_tensor const S_lin =
            material->second_piola_kirchhoff_stress(Grad_d_lin,
                                                    integrator.get_current_cell_index(),
                                                    q);
          tensor const delta_P =
            Grad_delta * S_lin +
            F_lin * material->second_piola_kirchhoff_stress_displacement_derivative(
                      Grad_delta, Grad_d_lin, integrator.get_current_cell_index(), q);

          integrator.submit_gradient(delta_P, q);

          if(this->operator_data.unsteady)
          {
            integrator.submit_value((this->scaling_factor_mass * this->operator_data.density) *
                                      integrator.get_value(q),
                                    q);
          }
        }
      }
    }
    else
    {
      if(this->operator_data.stable_formulation)
      {
        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          // TODO: compare these variants.
          // tensor const Grad_d_lin = integrator_lin->get_gradient(q);
          tensor const Grad_d_lin =
            material->gradient_displacement(integrator.get_current_cell_index(), q);

          tensor const Grad_delta = integrator.get_gradient(q);
          tensor const F_lin =
            compute_modified_F<dim, Number, check_type, true /*stable_formulation*/>(Grad_d_lin);
          symmetric_tensor const S_lin =
            material->second_piola_kirchhoff_stress(integrator.get_current_cell_index(), q);
          tensor const delta_P =
            Grad_delta * S_lin +
            F_lin * material->second_piola_kirchhoff_stress_displacement_derivative(
                      Grad_delta, Grad_d_lin, integrator.get_current_cell_index(), q);

          integrator.submit_gradient(delta_P, q);

          if(this->operator_data.unsteady)
          {
            integrator.submit_value((this->scaling_factor_mass * this->operator_data.density) *
                                      integrator.get_value(q),
                                    q);
          }
        }
      }
      else
      {
        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          // TODO: compare these variants.
          // tensor const Grad_d_lin = integrator_lin->get_gradient(q);
          tensor const Grad_d_lin =
            material->gradient_displacement(integrator.get_current_cell_index(), q);

          tensor const Grad_delta = integrator.get_gradient(q);
          tensor const F_lin =
            compute_modified_F<dim, Number, check_type, false /*stable_formulation*/>(Grad_d_lin);
          symmetric_tensor const S_lin =
            material->second_piola_kirchhoff_stress(integrator.get_current_cell_index(), q);
          tensor const delta_P =
            Grad_delta * S_lin +
            F_lin * material->second_piola_kirchhoff_stress_displacement_derivative(
                      Grad_delta, Grad_d_lin, integrator.get_current_cell_index(), q);

          integrator.submit_gradient(delta_P, q);

          if(this->operator_data.unsteady)
          {
            integrator.submit_value((this->scaling_factor_mass * this->operator_data.density) *
                                      integrator.get_value(q),
                                    q);
          }
        }
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
    this->matrix_free_spatial.cell_loop(&This::cell_loop_calculate_system_matrix,
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
    this->matrix_free_spatial.cell_loop(&This::cell_loop_calculate_system_matrix,
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
