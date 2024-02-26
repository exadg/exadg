#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
MomentumOperator<dim, Number>::MomentumOperator() : scaling_factor_mass(1.0)
{
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &   matrix_free,
  dealii::AffineConstraints<Number> const & affine_constraints,
  MomentumOperatorData<dim> const &         data)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  // create new objects and initialize kernels
  if(operator_data.unsteady_problem)
  {
    this->mass_kernel = std::make_shared<MassKernel<dim, Number>>();
  }

  if(operator_data.convective_problem)
  {
    this->convective_kernel = std::make_shared<Operators::ConvectiveKernel<dim, Number>>();
    this->convective_kernel->reinit(matrix_free,
                                    operator_data.convective_kernel_data,
                                    operator_data.dof_index,
                                    operator_data.quad_index,
                                    true);
  }

  if(operator_data.viscous_problem)
  {
    this->viscous_kernel = std::make_shared<Operators::ViscousKernel<dim, Number>>();
    this->viscous_kernel->reinit(matrix_free,
                                 operator_data.viscous_kernel_data,
                                 operator_data.dof_index,
                                 operator_data.quad_index);
  }

  if(operator_data.unsteady_problem)
    this->integrator_flags = this->integrator_flags | this->mass_kernel->get_integrator_flags();
  if(operator_data.convective_problem)
    this->integrator_flags =
      this->integrator_flags | this->convective_kernel->get_integrator_flags();
  if(operator_data.viscous_problem)
    this->integrator_flags = this->integrator_flags | this->viscous_kernel->get_integrator_flags();
}

template<int dim, typename Number>
MomentumOperatorData<dim> const &
MomentumOperator<dim, Number>::get_data() const
{
  return operator_data;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::initialize(
  dealii::MatrixFree<dim, Number> const &                   matrix_free,
  dealii::AffineConstraints<Number> const &                 affine_constraints,
  MomentumOperatorData<dim> const &                         data,
  std::shared_ptr<Operators::ViscousKernel<dim, Number>>    viscous_kernel,
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel)
{
  operator_data = data;

  Base::reinit(matrix_free, affine_constraints, data);

  // mass kernel: create new object and initialize kernel
  if(operator_data.unsteady_problem)
  {
    this->mass_kernel = std::make_shared<MassKernel<dim, Number>>();
  }

  // simply set pointers for convective and viscous kernels
  this->convective_kernel = convective_kernel;
  this->viscous_kernel    = viscous_kernel;

  if(operator_data.unsteady_problem)
    this->integrator_flags = this->integrator_flags | this->mass_kernel->get_integrator_flags();
  if(operator_data.convective_problem)
    this->integrator_flags =
      this->integrator_flags | this->convective_kernel->get_integrator_flags();
  if(operator_data.viscous_problem)
    this->integrator_flags = this->integrator_flags | this->viscous_kernel->get_integrator_flags();
}

template<int dim, typename Number>
Operators::ConvectiveKernelData
MomentumOperator<dim, Number>::get_convective_kernel_data() const
{
  if(operator_data.convective_problem)
    return convective_kernel->get_data();
  else
    return Operators::ConvectiveKernelData();
}

template<int dim, typename Number>
Operators::ViscousKernelData
MomentumOperator<dim, Number>::get_viscous_kernel_data() const
{
  if(operator_data.viscous_problem)
    return viscous_kernel->get_data();
  else
    return Operators::ViscousKernelData();
}

template<int dim, typename Number>
dealii::LinearAlgebra::distributed::Vector<Number> const &
MomentumOperator<dim, Number>::get_velocity() const
{
  return convective_kernel->get_velocity();
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_solution_linearization(VectorType const & velocity)
{
  this->set_velocity_ptr(velocity);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::update_after_grid_motion()
{
  if(operator_data.viscous_problem)
    viscous_kernel->calculate_penalty_parameter(this->get_matrix_free(),
                                                this->get_data().dof_index);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_velocity_copy(VectorType const & velocity) const
{
  if(operator_data.convective_problem)
    convective_kernel->set_velocity_copy(velocity);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_velocity_ptr(VectorType const & velocity) const
{
  if(operator_data.convective_problem)
    convective_kernel->set_velocity_ptr(velocity);
}

template<int dim, typename Number>
Number
MomentumOperator<dim, Number>::get_scaling_factor_mass_operator() const
{
  return this->scaling_factor_mass;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_scaling_factor_mass_operator(Number const & number)
{
  this->scaling_factor_mass = number;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::rhs(VectorType & dst) const
{
  (void)dst;

  AssertThrow(
    false, dealii::ExcMessage("The function rhs() does not make sense for the momentum operator."));
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::rhs_add(VectorType & dst) const
{
  (void)dst;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function rhs_add() does not make sense for the momentum operator."));
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::evaluate(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function evaluate() does not make sense for the momentum operator."));
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::evaluate_add(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;

  AssertThrow(false,
              dealii::ExcMessage(
                "The function evaluate_add() does not make sense for the momentum operator."));
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_cell_derived(IntegratorCell &   integrator,
                                                   unsigned int const cell) const
{
  (void)integrator;

  if(operator_data.convective_problem)
    convective_kernel->reinit_cell(cell);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_face_derived(IntegratorFace &   integrator_m,
                                                   IntegratorFace &   integrator_p,
                                                   unsigned int const face) const
{
  if(operator_data.convective_problem)
    convective_kernel->reinit_face(face);

  if(operator_data.viscous_problem)
    viscous_kernel->reinit_face(integrator_m, integrator_p, operator_data.dof_index);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_boundary_face_derived(IntegratorFace &   integrator_m,
                                                            unsigned int const face) const
{
  if(operator_data.convective_problem)
    convective_kernel->reinit_boundary_face(face);

  if(operator_data.viscous_problem)
    viscous_kernel->reinit_boundary_face(integrator_m, operator_data.dof_index);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_face_cell_based_derived(
  IntegratorFace &                 integrator_m,
  IntegratorFace &                 integrator_p,
  unsigned int const               cell,
  unsigned int const               face,
  dealii::types::boundary_id const boundary_id) const
{
  if(operator_data.convective_problem)
    convective_kernel->reinit_face_cell_based(cell, face, boundary_id);

  if(operator_data.viscous_problem)
    viscous_kernel->reinit_face_cell_based(boundary_id,
                                           integrator_m,
                                           integrator_p,
                                           operator_data.dof_index);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector value_flux;
    tensor gradient_flux;

    vector value;
    if(this->integrator_flags.cell_evaluate & dealii::EvaluationFlags::values)
      value = integrator.get_value(q);

    tensor gradient;
    if(this->integrator_flags.cell_evaluate & dealii::EvaluationFlags::gradients)
      gradient = integrator.get_gradient(q);

    if(operator_data.unsteady_problem)
    {
      value_flux += mass_kernel->get_volume_flux(scaling_factor_mass, value);
    }

    if(operator_data.convective_problem)
    {
      if(operator_data.convective_kernel_data.formulation ==
         FormulationConvectiveTerm::DivergenceFormulation)
      {
        gradient_flux +=
          convective_kernel->get_volume_flux_linearized_divergence_formulation(value, q);
      }
      else if(operator_data.convective_kernel_data.formulation ==
              FormulationConvectiveTerm::ConvectiveFormulation)
      {
        value_flux +=
          convective_kernel->get_volume_flux_linearized_convective_formulation(value, gradient, q);
      }
      else
      {
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
      }
    }

    if(operator_data.viscous_problem)
    {
      scalar viscosity = viscous_kernel->get_viscosity_cell(integrator.get_current_cell_index(), q);
      gradient_flux += viscous_kernel->get_volume_flux(gradient, viscosity);
    }

    if(this->integrator_flags.cell_integrate & dealii::EvaluationFlags::values)
      integrator.submit_value(value_flux, q);

    if(this->integrator_flags.cell_integrate & dealii::EvaluationFlags::gradients)
      integrator.submit_gradient(gradient_flux, q);
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_face_integral(IntegratorFace & integrator_m,
                                                IntegratorFace & integrator_p) const
{
  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector value_m  = integrator_m.get_value(q);
    vector value_p  = integrator_p.get_value(q);
    vector normal_m = integrator_m.get_normal_vector(q);

    vector value_flux_m, value_flux_p;
    tensor gradient_flux;

    if(operator_data.convective_problem)
    {
      vector u_m = convective_kernel->get_velocity_m(q);
      vector u_p = convective_kernel->get_velocity_p(q);

      std::tuple<vector, vector> flux =
        convective_kernel->calculate_flux_linearized_interior_and_neighbor(
          u_m, u_p, value_m, value_p, normal_m, q);

      value_flux_m += std::get<0>(flux);
      value_flux_p += std::get<1>(flux);
    }

    if(operator_data.viscous_problem)
    {
      scalar average_viscosity =
        viscous_kernel->get_viscosity_interior_face(integrator_m.get_current_cell_index(), q);

      gradient_flux =
        viscous_kernel->calculate_gradient_flux(value_m, value_p, normal_m, average_viscosity);

      vector normal_gradient_m = viscous_kernel->calculate_normal_gradient(q, integrator_m);
      vector normal_gradient_p = viscous_kernel->calculate_normal_gradient(q, integrator_p);

      vector value_flux = viscous_kernel->calculate_value_flux(
        normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, average_viscosity);

      value_flux_m += -value_flux;
      value_flux_p += value_flux; // + sign since n⁺ = -n⁻
    }

    integrator_m.submit_value(value_flux_m, q);
    integrator_p.submit_value(value_flux_p, q);

    if(operator_data.viscous_problem)
    {
      integrator_m.submit_gradient(gradient_flux, q);
      integrator_p.submit_gradient(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_face_int_integral(IntegratorFace & integrator_m,
                                                    IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector value_m = integrator_m.get_value(q);
    vector value_p; // set exterior value to zero
    vector normal_m = integrator_m.get_normal_vector(q);

    vector value_flux_m;
    tensor gradient_flux;

    if(operator_data.convective_problem)
    {
      vector u_m = convective_kernel->get_velocity_m(q);
      vector u_p = convective_kernel->get_velocity_p(q);

      value_flux_m += convective_kernel->calculate_flux_linearized_interior(
        u_m, u_p, value_m, value_p, normal_m, q);
    }

    if(operator_data.viscous_problem)
    {
      scalar average_viscosity =
        viscous_kernel->get_viscosity_interior_face(integrator_m.get_current_cell_index(), q);

      gradient_flux +=
        viscous_kernel->calculate_gradient_flux(value_m, value_p, normal_m, average_viscosity);

      vector normal_gradient_m = viscous_kernel->calculate_normal_gradient(q, integrator_m);
      vector normal_gradient_p; // set exterior gradient to zero

      vector value_flux = viscous_kernel->calculate_value_flux(
        normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, average_viscosity);

      value_flux_m += -value_flux;
    }

    integrator_m.submit_value(value_flux_m, q);

    if(operator_data.viscous_problem)
    {
      integrator_m.submit_gradient(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_face_int_integral_cell_based(IntegratorFace & integrator_m,
                                                               IntegratorFace & integrator_p) const
{
  (void)integrator_p;

  for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
  {
    vector value_m = integrator_m.get_value(q);
    vector value_p; // set exterior value to zero
    vector normal_m = integrator_m.get_normal_vector(q);

    vector value_flux_m;
    tensor gradient_flux;

    if(operator_data.convective_problem)
    {
      vector u_m = convective_kernel->get_velocity_m(q);
      // TODO
      // Accessing exterior data is currently not available in deal.II/matrixfree.
      // Hence, we simply use the interior value, but note that the diagonal and block-diagonal
      // are not calculated exactly.
      vector u_p = u_m;

      value_flux_m += convective_kernel->calculate_flux_linearized_interior(
        u_m, u_p, value_m, value_p, normal_m, q);
    }

    if(operator_data.viscous_problem)
    {
      scalar average_viscosity =
        viscous_kernel->get_viscosity_interior_face(integrator_m.get_current_cell_index(), q);

      gradient_flux +=
        viscous_kernel->calculate_gradient_flux(value_m, value_p, normal_m, average_viscosity);

      vector normal_gradient_m = viscous_kernel->calculate_normal_gradient(q, integrator_m);
      vector normal_gradient_p; // set exterior gradient to zero

      vector value_flux = viscous_kernel->calculate_value_flux(
        normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, average_viscosity);

      value_flux_m += -value_flux;
    }

    integrator_m.submit_value(value_flux_m, q);

    if(operator_data.viscous_problem)
    {
      integrator_m.submit_gradient(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_face_ext_integral(IntegratorFace & integrator_m,
                                                    IntegratorFace & integrator_p) const
{
  (void)integrator_m;

  for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
  {
    vector value_m; // set exterior values to zero
    vector value_p = integrator_p.get_value(q);
    // multiply by -1.0 to get the correct normal vector !
    vector normal_p = -integrator_p.get_normal_vector(q);

    vector value_flux_p;
    tensor gradient_flux;

    if(operator_data.convective_problem)
    {
      vector u_m = convective_kernel->get_velocity_m(q);
      vector u_p = convective_kernel->get_velocity_p(q);

      value_flux_p += convective_kernel->calculate_flux_linearized_interior(
        u_p, u_m, value_p, value_m, normal_p, q);
    }

    if(operator_data.viscous_problem)
    {
      scalar average_viscosity =
        viscous_kernel->get_viscosity_interior_face(integrator_p.get_current_cell_index(), q);

      gradient_flux +=
        viscous_kernel->calculate_gradient_flux(value_p, value_m, normal_p, average_viscosity);

      // set exterior gradient to zero
      vector normal_gradient_m;
      // multiply by -1.0 since normal vector n⁺ = -n⁻ !
      vector normal_gradient_p = -viscous_kernel->calculate_normal_gradient(q, integrator_p);

      vector value_flux = viscous_kernel->calculate_value_flux(
        normal_gradient_p, normal_gradient_m, value_p, value_m, normal_p, average_viscosity);

      value_flux_p += -value_flux;
    }

    integrator_p.submit_value(value_flux_p, q);

    if(operator_data.viscous_problem)
    {
      integrator_p.submit_gradient(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_boundary_integral(
  IntegratorFace &                   integrator,
  OperatorType const &               operator_type,
  dealii::types::boundary_id const & boundary_id) const
{
  // make sure that this function is only accessed for OperatorType::homogeneous
  AssertThrow(
    operator_type == OperatorType::homogeneous,
    dealii::ExcMessage(
      "For the linearized momentum operator, only OperatorType::homogeneous makes sense."));

  BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector value_m = integrator.get_value(q);
    vector value_p;
    vector normal_m = integrator.get_normal_vector(q);

    vector value_flux_m;
    tensor gradient_flux;

    if(operator_data.convective_problem)
    {
      // value_p is calculated differently for the convective term and the viscous term
      value_p = convective_kernel->calculate_exterior_value_linearized(value_m,
                                                                       q,
                                                                       integrator,
                                                                       boundary_type);

      vector u_m = convective_kernel->get_velocity_m(q);
      vector u_p =
        calculate_exterior_value_nonlinear(u_m,
                                           q,
                                           integrator,
                                           boundary_type,
                                           operator_data.convective_kernel_data.type_dirichlet_bc,
                                           boundary_id,
                                           operator_data.bc,
                                           this->time);

      value_flux_m += convective_kernel->calculate_flux_linearized_boundary(
        u_m, u_p, value_m, value_p, normal_m, boundary_type, q);
    }

    if(operator_data.viscous_problem)
    {
      // value_p is calculated differently for the convective term and the viscous term
      value_p = calculate_exterior_value(value_m,
                                         q,
                                         integrator,
                                         operator_type,
                                         boundary_type,
                                         boundary_id,
                                         operator_data.bc,
                                         this->time);

      scalar viscosity =
        viscous_kernel->get_viscosity_boundary_face(integrator.get_current_cell_index(), q);
      gradient_flux +=
        viscous_kernel->calculate_gradient_flux(value_m, value_p, normal_m, viscosity);

      vector normal_gradient_m =
        viscous_kernel->calculate_interior_normal_gradient(q, integrator, operator_type);

      vector normal_gradient_p;

      normal_gradient_p =
        calculate_exterior_normal_gradient(normal_gradient_m,
                                           q,
                                           integrator,
                                           operator_type,
                                           boundary_type,
                                           boundary_id,
                                           operator_data.bc,
                                           this->time,
                                           viscous_kernel->get_data().variable_normal_vector);

      vector value_flux = viscous_kernel->calculate_value_flux(
        normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, viscosity);

      value_flux_m += -value_flux;
    }

    integrator.submit_value(value_flux_m, q);

    if(operator_data.viscous_problem)
    {
      integrator.submit_gradient(gradient_flux, q);
    }
  }
}

template class MomentumOperator<2, float>;
template class MomentumOperator<2, double>;

template class MomentumOperator<3, float>;
template class MomentumOperator<3, double>;

} // namespace IncNS
} // namespace ExaDG
