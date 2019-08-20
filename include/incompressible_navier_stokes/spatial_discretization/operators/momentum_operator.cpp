#include "momentum_operator.h"

namespace IncNS
{
template<int dim, typename Number>
MomentumOperator<dim, Number>::MomentumOperator() : scaling_factor_mass_matrix(1.0)
{
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit(MatrixFree<dim, Number> const &   matrix_free,
                                      AffineConstraints<double> const & constraint_matrix,
                                      MomentumOperatorData<dim> const & data)
{
  (void)matrix_free;
  (void)constraint_matrix;
  (void)data;

  AssertThrow(false, ExcMessage("This reinit function is not implemented for MomentumOperator."));
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit(
  MatrixFree<dim, Number> const &         matrix_free,
  AffineConstraints<double> const &       constraint_matrix,
  MomentumOperatorData<dim> const &       data,
  Operators::ConvectiveKernelData const & convective_kernel_data,
  Operators::ViscousKernelData const &    viscous_kernel_data)
{
  Base::reinit(matrix_free, constraint_matrix, data);

  // create new objects and initialize kernels
  if(this->data.unsteady_problem)
  {
    this->mass_kernel.reset(new MassMatrixKernel<dim, Number>());
    this->scaling_factor_mass_matrix = this->data.scaling_factor_mass_matrix;
  }

  if(this->data.convective_problem)
  {
    this->convective_kernel.reset(new Operators::ConvectiveKernel<dim, Number>());
    this->convective_kernel->reinit(matrix_free,
                                    convective_kernel_data,
                                    this->data.dof_index,
                                    this->data.quad_index,
                                    this->is_mg);
  }

  if(this->data.viscous_problem)
  {
    this->viscous_kernel.reset(new Operators::ViscousKernel<dim, Number>());
    this->viscous_kernel->reinit(matrix_free, viscous_kernel_data, this->data.dof_index);
  }

  if(this->data.unsteady_problem)
    this->integrator_flags = this->integrator_flags || this->mass_kernel->get_integrator_flags();
  if(this->data.convective_problem)
    this->integrator_flags =
      this->integrator_flags || this->convective_kernel->get_integrator_flags();
  if(this->data.viscous_problem)
    this->integrator_flags = this->integrator_flags || this->viscous_kernel->get_integrator_flags();
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit(
  MatrixFree<dim, Number> const &                           matrix_free,
  AffineConstraints<double> const &                         constraint_matrix,
  MomentumOperatorData<dim> const &                         data,
  std::shared_ptr<Operators::ViscousKernel<dim, Number>>    viscous_kernel,
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel)
{
  Base::reinit(matrix_free, constraint_matrix, data);

  // mass kernel: create new object and initialize kernel
  if(this->data.unsteady_problem)
  {
    this->mass_kernel.reset(new MassMatrixKernel<dim, Number>());
    this->scaling_factor_mass_matrix = this->data.scaling_factor_mass_matrix;
  }

  // simply set pointers for convective and viscous kernels
  this->convective_kernel = convective_kernel;
  this->viscous_kernel    = viscous_kernel;

  if(this->data.unsteady_problem)
    this->integrator_flags = this->integrator_flags || this->mass_kernel->get_integrator_flags();
  if(this->data.convective_problem)
    this->integrator_flags =
      this->integrator_flags || this->convective_kernel->get_integrator_flags();
  if(this->data.viscous_problem)
    this->integrator_flags = this->integrator_flags || this->viscous_kernel->get_integrator_flags();
}

template<int dim, typename Number>
Operators::ConvectiveKernelData
MomentumOperator<dim, Number>::get_convective_kernel_data() const
{
  if(this->data.convective_problem)
    return convective_kernel->get_data();
  else
    return Operators::ConvectiveKernelData();
}

template<int dim, typename Number>
Operators::ViscousKernelData
MomentumOperator<dim, Number>::get_viscous_kernel_data() const
{
  if(this->data.viscous_problem)
    return viscous_kernel->get_data();
  else
    return Operators::ViscousKernelData();
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number> const &
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
MomentumOperator<dim, Number>::set_velocity_copy(VectorType const & velocity) const
{
  convective_kernel->set_velocity_copy(velocity);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_velocity_ptr(VectorType const & velocity) const
{
  convective_kernel->set_velocity_ptr(velocity);
}

template<int dim, typename Number>
Number
MomentumOperator<dim, Number>::get_scaling_factor_mass_matrix() const
{
  return this->scaling_factor_mass_matrix;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::set_scaling_factor_mass_matrix(Number const & number)
{
  this->scaling_factor_mass_matrix = number;
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::rhs(VectorType & dst) const
{
  (void)dst;

  AssertThrow(false,
              ExcMessage("The function rhs() does not make sense for the momentum operator."));
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::rhs_add(VectorType & dst) const
{
  (void)dst;

  AssertThrow(false,
              ExcMessage("The function rhs_add() does not make sense for the momentum operator."));
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::evaluate(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;

  AssertThrow(false,
              ExcMessage("The function evaluate() does not make sense for the momentum operator."));
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::evaluate_add(VectorType & dst, VectorType const & src) const
{
  (void)dst;
  (void)src;

  AssertThrow(false,
              ExcMessage(
                "The function evaluate_add() does not make sense for the momentum operator."));
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_cell(unsigned int const cell) const
{
  Base::reinit_cell(cell);

  if(this->data.convective_problem)
    convective_kernel->reinit_cell(cell);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_face(unsigned int const face) const
{
  Base::reinit_face(face);

  if(this->data.convective_problem)
    convective_kernel->reinit_face(face);

  if(this->data.viscous_problem)
    viscous_kernel->reinit_face(*this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_boundary_face(unsigned int const face) const
{
  Base::reinit_boundary_face(face);

  if(this->data.convective_problem)
    convective_kernel->reinit_boundary_face(face);

  if(this->data.viscous_problem)
    viscous_kernel->reinit_boundary_face(*this->integrator_m);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::reinit_face_cell_based(unsigned int const       cell,
                                                      unsigned int const       face,
                                                      types::boundary_id const boundary_id) const
{
  Base::reinit_face_cell_based(cell, face, boundary_id);

  if(this->data.convective_problem)
    convective_kernel->reinit_face_cell_based(cell, face, boundary_id);

  if(this->data.viscous_problem)
    viscous_kernel->reinit_face_cell_based(boundary_id, *this->integrator_m, *this->integrator_p);
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_cell_integral(IntegratorCell & integrator) const
{
  bool const get_value = this->data.unsteady_problem || this->data.convective_problem;

  bool const get_gradient =
    this->data.viscous_problem ||
    (this->data.convective_problem &&
     this->data.formulation_convective_term == FormulationConvectiveTerm::ConvectiveFormulation);

  bool const submit_value =
    this->data.unsteady_problem ||
    (this->data.convective_problem &&
     this->data.formulation_convective_term == FormulationConvectiveTerm::ConvectiveFormulation);

  bool const submit_gradient =
    this->data.viscous_problem ||
    (this->data.convective_problem &&
     this->data.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector value_flux;
    tensor gradient_flux;

    vector value;
    if(get_value)
      value = integrator.get_value(q);

    tensor gradient;
    if(get_gradient)
      gradient = integrator.get_gradient(q);

    if(this->data.unsteady_problem)
    {
      value_flux += mass_kernel->get_volume_flux(scaling_factor_mass_matrix, value);
    }

    if(this->data.convective_problem)
    {
      vector u = convective_kernel->get_velocity_cell(q);

      if(this->data.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      {
        gradient_flux +=
          convective_kernel->get_volume_flux_linearized_divergence_formulation(u, value);
      }
      else if(this->data.formulation_convective_term ==
              FormulationConvectiveTerm::ConvectiveFormulation)
      {
        tensor grad_u = convective_kernel->get_velocity_gradient_cell(q);

        value_flux += convective_kernel->get_volume_flux_linearized_convective_formulation(
          u, value, grad_u, gradient);
      }
      else
      {
        AssertThrow(false, ExcMessage("Not implemented."));
      }
    }

    if(this->data.viscous_problem)
    {
      scalar viscosity = viscous_kernel->get_viscosity_cell(integrator.get_cell_index(), q);
      gradient_flux += viscous_kernel->get_volume_flux(gradient, viscosity);
    }

    if(submit_value)
      integrator.submit_value(value_flux, q);

    if(submit_gradient)
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

    if(this->data.convective_problem)
    {
      vector u_m = convective_kernel->get_velocity_m(q);
      vector u_p = convective_kernel->get_velocity_p(q);

      std::tuple<vector, vector> flux =
        convective_kernel->calculate_flux_linearized_interior_and_neighbor(
          u_m, u_p, value_m, value_p, normal_m);

      value_flux_m += std::get<0>(flux);
      value_flux_p += std::get<1>(flux);
    }

    if(this->data.viscous_problem)
    {
      scalar average_viscosity =
        viscous_kernel->get_viscosity_interior_face(integrator_m.get_face_index(), q);

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

    if(this->data.viscous_problem)
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

    if(this->data.convective_problem)
    {
      vector u_m = convective_kernel->get_velocity_m(q);
      vector u_p = convective_kernel->get_velocity_p(q);

      value_flux_m +=
        convective_kernel->calculate_flux_linearized_interior(u_m, u_p, value_m, value_p, normal_m);
    }

    if(this->data.viscous_problem)
    {
      scalar average_viscosity =
        viscous_kernel->get_viscosity_interior_face(integrator_m.get_face_index(), q);

      gradient_flux +=
        viscous_kernel->calculate_gradient_flux(value_m, value_p, normal_m, average_viscosity);

      vector normal_gradient_m = viscous_kernel->calculate_normal_gradient(q, integrator_m);
      vector normal_gradient_p; // set exterior gradient to zero

      vector value_flux = viscous_kernel->calculate_value_flux(
        normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, average_viscosity);

      value_flux_m += -value_flux;
    }

    integrator_m.submit_value(value_flux_m, q);

    if(this->data.viscous_problem)
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

    if(this->data.convective_problem)
    {
      vector u_m = convective_kernel->get_velocity_m(q);
      // TODO
      // Accessing exterior data is currently not available in deal.II/matrixfree.
      // Hence, we simply use the interior value, but note that the diagonal and block-diagonal
      // are not calculated exactly.
      vector u_p = u_m;

      value_flux_m +=
        convective_kernel->calculate_flux_linearized_interior(u_m, u_p, value_m, value_p, normal_m);
    }

    if(this->data.viscous_problem)
    {
      scalar average_viscosity =
        viscous_kernel->get_viscosity_interior_face(integrator_m.get_face_index(), q);

      gradient_flux +=
        viscous_kernel->calculate_gradient_flux(value_m, value_p, normal_m, average_viscosity);

      vector normal_gradient_m = viscous_kernel->calculate_normal_gradient(q, integrator_m);
      vector normal_gradient_p; // set exterior gradient to zero

      vector value_flux = viscous_kernel->calculate_value_flux(
        normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, average_viscosity);

      value_flux_m += -value_flux;
    }

    integrator_m.submit_value(value_flux_m, q);

    if(this->data.viscous_problem)
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
    // multiply by -1.0 to get the correct normal vector !!!
    vector normal_p = -integrator_p.get_normal_vector(q);

    vector value_flux_p;
    tensor gradient_flux;

    if(this->data.convective_problem)
    {
      vector u_m = convective_kernel->get_velocity_m(q);
      vector u_p = convective_kernel->get_velocity_p(q);

      value_flux_p +=
        convective_kernel->calculate_flux_linearized_interior(u_p, u_m, value_p, value_m, normal_p);
    }

    if(this->data.viscous_problem)
    {
      scalar average_viscosity =
        viscous_kernel->get_viscosity_interior_face(integrator_p.get_face_index(), q);

      gradient_flux +=
        viscous_kernel->calculate_gradient_flux(value_p, value_m, normal_p, average_viscosity);

      // set exterior gradient to zero
      vector normal_gradient_m;
      // multiply by -1.0 since normal vector n⁺ = -n⁻ !!!
      vector normal_gradient_p = -viscous_kernel->calculate_normal_gradient(q, integrator_p);

      vector value_flux = viscous_kernel->calculate_value_flux(
        normal_gradient_p, normal_gradient_m, value_p, value_m, normal_p, average_viscosity);

      value_flux_p += -value_flux;
    }

    integrator_p.submit_value(value_flux_p, q);

    if(this->data.viscous_problem)
    {
      integrator_p.submit_gradient(gradient_flux, q);
    }
  }
}

template<int dim, typename Number>
void
MomentumOperator<dim, Number>::do_boundary_integral(IntegratorFace &           integrator,
                                                    OperatorType const &       operator_type,
                                                    types::boundary_id const & boundary_id) const
{
  // make sure that this function is only accessed for OperatorType::homogeneous
  AssertThrow(
    operator_type == OperatorType::homogeneous,
    ExcMessage(
      "For the linearized momentum operator, only OperatorType::homogeneous makes sense."));

  BoundaryTypeU boundary_type = this->data.bc->get_boundary_type(boundary_id);

  for(unsigned int q = 0; q < integrator.n_q_points; ++q)
  {
    vector value_m = integrator.get_value(q);
    vector value_p;
    vector normal_m = integrator.get_normal_vector(q);

    vector value_flux_m;
    tensor gradient_flux;

    if(this->data.convective_problem)
    {
      // value_p is calculated differently for the convective term and the viscous term
      value_p = convective_kernel->calculate_exterior_value_linearized(value_m,
                                                                       q,
                                                                       integrator,
                                                                       boundary_type);

      vector u_m = convective_kernel->get_velocity_m(q);
      vector u_p = convective_kernel->calculate_exterior_value_nonlinear(
        u_m, q, integrator, boundary_type, boundary_id, this->data.bc, this->time);

      value_flux_m += convective_kernel->calculate_flux_linearized_boundary(
        u_m, u_p, value_m, value_p, normal_m, boundary_type);
    }

    if(this->data.viscous_problem)
    {
      // value_p is calculated differently for the convective term and the viscous term
      value_p = calculate_exterior_value(value_m,
                                         q,
                                         integrator,
                                         operator_type,
                                         boundary_type,
                                         boundary_id,
                                         this->data.bc,
                                         this->time);

      scalar viscosity =
        viscous_kernel->get_viscosity_boundary_face(integrator.get_face_index(), q);
      gradient_flux +=
        viscous_kernel->calculate_gradient_flux(value_m, value_p, normal_m, viscosity);

      vector normal_gradient_m =
        viscous_kernel->calculate_interior_normal_gradient(q, integrator, operator_type);

      vector normal_gradient_p;

      normal_gradient_p = calculate_exterior_normal_gradient(normal_gradient_m,
                                                             q,
                                                             integrator,
                                                             operator_type,
                                                             boundary_type,
                                                             boundary_id,
                                                             this->data.bc,
                                                             this->time,
                                                             viscous_kernel->get_data().variable_normal_vector);

      vector value_flux = viscous_kernel->calculate_value_flux(
        normal_gradient_m, normal_gradient_p, value_m, value_p, normal_m, viscosity);

      value_flux_m += -value_flux;
    }

    integrator.submit_value(value_flux_m, q);

    if(this->data.viscous_problem)
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
