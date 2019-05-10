/*
 * convective_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

#include "../../../functionalities/evaluate_functions.h"
#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/input_parameters.h"

using namespace dealii;

namespace IncNS
{
template<int dim>
struct ConvectiveOperatorData
{
  ConvectiveOperatorData()
    : formulation(FormulationConvectiveTerm::DivergenceFormulation),
      dof_index(0),
      quad_index(0),
      upwind_factor(1.0),
      use_outflow_bc(false),
      type_dirichlet_bc(TypeDirichletBCs::Mirror),
      use_cell_based_loops(false)
  {
  }

  FormulationConvectiveTerm formulation;

  unsigned int dof_index;

  unsigned int quad_index;

  double upwind_factor;

  bool use_outflow_bc;

  TypeDirichletBCs type_dirichlet_bc;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;

  // use cell based loops
  bool use_cell_based_loops;
};



template<int dim, typename Number>
class ConvectiveOperator
{
public:
  typedef ConvectiveOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> CellIntegratorU;
  typedef FaceIntegrator<dim, dim, Number> FaceIntegratorU;

  ConvectiveOperator()
    : matrix_free(nullptr),
      eval_time(0.0),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD))
  {
  }

  ConvectiveOperatorData<dim> const &
  get_operator_data() const
  {
    return operator_data;
  }

  void
  set_solution_linearization(VectorType const & src) const
  {
    velocity_linearization = src;

    velocity_linearization.update_ghost_values();
  }

  VectorType const &
  get_solution_linearization() const
  {
    return velocity_linearization;
  }

  void
  initialize(MatrixFree<dim, Number> const &     matrix_free_in,
             ConvectiveOperatorData<dim> const & operator_data_in)
  {
    matrix_free   = &matrix_free_in;
    operator_data = operator_data_in;

    matrix_free->initialize_dof_vector(velocity_linearization, operator_data.dof_index);

    // Block Jacobi elementwise
    integrator.reset(
      new CellIntegratorU(*matrix_free, operator_data.dof_index, operator_data.quad_index));
    integrator_m.reset(
      new FaceIntegratorU(*matrix_free, true, operator_data.dof_index, operator_data.quad_index));
    integrator_p.reset(
      new FaceIntegratorU(*matrix_free, false, operator_data.dof_index, operator_data.quad_index));
    integrator_linearization.reset(
      new CellIntegratorU(*matrix_free, operator_data.dof_index, operator_data.quad_index));
    integrator_linearization_m.reset(
      new FaceIntegratorU(*matrix_free, true, operator_data.dof_index, operator_data.quad_index));
    integrator_linearization_p.reset(
      new FaceIntegratorU(*matrix_free, false, operator_data.dof_index, operator_data.quad_index));
  }

  /*
   * Evaluate nonlinear convective operator.
   */
  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop_nonlinear_operator,
                      &This::face_loop_nonlinear_operator,
                      &This::boundary_face_loop_nonlinear_operator,
                      this,
                      dst,
                      src,
                      true /*zero_dst_vector = true*/,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  void
  evaluate_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop_nonlinear_operator,
                      &This::face_loop_nonlinear_operator,
                      &This::boundary_face_loop_nonlinear_operator,
                      this,
                      dst,
                      src,
                      false /*zero_dst_vector = false*/,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  // OIF splitting
  /*
   * Evaluate operator (linear transport with a divergence-free velocity, required for OIF
   * splitting).
   */
  void
  evaluate_linear_transport(VectorType &       dst,
                            VectorType const & src,
                            Number const       evaluation_time,
                            VectorType const & velocity_transport) const
  {
    set_solution_linearization(velocity_transport);

    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop_linear_transport,
                      &This::face_loop_linear_transport,
                      &This::boundary_face_loop_linear_transport,
                      this,
                      dst,
                      src,
                      true /*zero_dst_vector = true*/,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  /*
   * Apply linearized convective operator.
   */
  void
  apply(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop_linearized_operator,
                      &This::face_loop_linearized_operator,
                      &This::boundary_face_loop_linearized_operator,
                      this,
                      dst,
                      src,
                      true /*zero_dst_vector = true*/,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  void
  apply_add(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop_linearized_operator,
                      &This::face_loop_linearized_operator,
                      &This::boundary_face_loop_linearized_operator,
                      this,
                      dst,
                      src,
                      false /*zero_dst_vector = false*/,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  /*
   * Calculate diagonal of linearized convective operator.
   */
  void
  calculate_diagonal(VectorType & diagonal, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;

    matrix_free->loop(&This::cell_loop_diagonal,
                      &This::face_loop_diagonal,
                      &This::boundary_face_loop_diagonal,
                      this,
                      diagonal,
                      src,
                      true /*zero_dst_vector = true*/,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  void
  add_diagonal(VectorType & diagonal, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;

    matrix_free->loop(&This::cell_loop_diagonal,
                      &This::face_loop_diagonal,
                      &This::boundary_face_loop_diagonal,
                      this,
                      diagonal,
                      src,
                      false /*zero_dst_vector = false*/,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  /*
   * Assemble block diagonal matrices for linearized convective operator.
   */
  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<Number>> & matrices,
                              Number const                            evaluation_time) const
  {
    this->eval_time = evaluation_time;

    VectorType src;

    if(operator_data.use_cell_based_loops)
    {
      matrix_free->cell_loop(&This::cell_based_loop_calculate_block_diagonal, this, matrices, src);
    }
    else
    {
      AssertThrow(
        n_mpi_processes == 1,
        ExcMessage(
          "Block diagonal calculation with separate loops over cells and faces only works in serial. "
          "Use cell based loops for parallel computations."));

      matrix_free->loop(&This::cell_loop_calculate_block_diagonal,
                        &This::face_loop_calculate_block_diagonal,
                        &This::boundary_face_loop_calculate_block_diagonal,
                        this,
                        matrices,
                        src);
    }
  }

  /*
   *  Apply block-diagonal operator as a global operation (only needed in order to test the
   * implementation for assembling the block-diagonal).
   */
  void
  apply_block_diagonal(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop_linearized_operator,
                      &This::face_loop_apply_block_diagonal_linearized_operator,
                      &This::boundary_face_loop_linearized_operator,
                      this,
                      dst,
                      src,
                      true /*zero_dst_vector = true*/,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  /*
   *  Apply-add block-diagonal operator as a global operation (only needed in order to test the
   * implementation for assembling the block-diagonal).
   */
  void
  apply_block_diagonal_add(VectorType &       dst,
                           VectorType const & src,
                           Number const       evaluation_time) const
  {
    this->eval_time = evaluation_time;

    matrix_free->loop(&This::cell_loop_linearized_operator,
                      &This::face_loop_apply_block_diagonal_linearized_operator,
                      &This::boundary_face_loop_linearized_operator,
                      this,
                      dst,
                      src,
                      false /*zero_dst_vector = false*/,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values,
                      MatrixFree<dim, Number>::DataAccessOnFaces::values);
  }

  void
  apply_add_block_diagonal_elementwise(unsigned int const   cell,
                                       scalar * const       dst,
                                       scalar const * const src,
                                       unsigned int const   problem_size = 1) const
  {
    (void)problem_size;

    unsigned int dofs_per_cell = integrator->dofs_per_cell;

    integrator_linearization->reinit(cell);

    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
      integrator_linearization->gather_evaluate(velocity_linearization, true, false, false);
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
      integrator_linearization->gather_evaluate(velocity_linearization, true, true, false);
    else
      AssertThrow(false, ExcMessage("Not implemented."));

    integrator->reinit(cell);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      integrator->begin_dof_values()[i] = src[i];

    do_cell_integral_linearized_operator(*integrator, *integrator_linearization);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dst[i] += integrator->begin_dof_values()[i];

    // loop over all faces
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      integrator_linearization_m->reinit(cell, face);
      integrator_linearization_m->gather_evaluate(velocity_linearization, true, false);

      // TODO
      //        AssertThrow(false, ExcMessage("We have to evaluate the linearized velocity field on
      //        the neighbor. This functionality is not implemented in deal.II/matrix_free."));
      //        integrator_linearization_p->reinit(cell, face);
      //        integrator_linearization_p->gather_evaluate(velocity_linearization, true, false);

      integrator_m->reinit(cell, face);
      integrator_p->reinit(cell, face);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        integrator_m->begin_dof_values()[i] = src[i];

      // do not need to read dof values for integrator_p (already initialized with 0)

      integrator_m->evaluate(true, false);

      auto bids        = matrix_free->get_faces_by_cells_boundary_id(cell, face);
      auto boundary_id = bids[0];

      if(boundary_id == numbers::internal_face_boundary_id) // internal face
      {
        // TODO
        //            do_face_int_integral_linearized_operator(*integrator_m, *integrator_p,
        //            *integrator_linearization_m, *integrator_linearization_p);

        // plug in integrator_linearization_m twice to avoid the above problem with accessing dofs
        // of the neighboring element
        do_face_int_integral_linearized_operator(*integrator_m,
                                                 *integrator_p,
                                                 *integrator_linearization_m,
                                                 *integrator_linearization_m);
      }
      else // boundary face
      {
        do_boundary_integral_linearized_operator(*integrator_m,
                                                 *integrator_linearization_m,
                                                 boundary_id);
      }

      integrator_m->integrate(true, false);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst[i] += integrator_m->begin_dof_values()[i];
    }
  }

private:
  // nonlinear operator
  template<typename Integrator>
  void
  do_cell_integral_nonlinear_operator(Integrator & integrator) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      integrator.evaluate(true, false, false);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        vector u = integrator.get_value(q);
        tensor F = outer_product(u, u);
        // minus sign due to integration by parts
        integrator.submit_gradient(-F, q);
      }
      integrator.integrate(false, true);
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      integrator.evaluate(true, true, false);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // convective formulation: (u * grad) u = grad(u) * u
        vector u          = integrator.get_value(q);
        tensor gradient_u = integrator.get_gradient(q);
        vector F          = gradient_u * u;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        integrator.submit_value(F, q);
      }
      integrator.integrate(true, false);
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      integrator.evaluate(true, true, false);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        vector u          = integrator.get_value(q);
        tensor F          = outer_product(u, u);
        scalar divergence = integrator.get_divergence(q);
        vector div_term   = -0.5 * divergence * u;
        // minus sign due to integration by parts
        integrator.submit_gradient(-F, q);
        integrator.submit_value(div_term, q);
      }
      integrator.integrate(true, true);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename Integrator>
  void
  do_face_integral_nonlinear_operator(Integrator & integrator_m, Integrator & integrator_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM     = integrator_m.get_value(q);
        vector uP     = integrator_p.get_value(q);
        vector normal = integrator_m.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normal);

        integrator_m.submit_value(flux, q);
        integrator_p.submit_value(-flux, q); // minus sign since n⁺ = - n⁻
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM     = integrator_m.get_value(q);
        vector uP     = integrator_p.get_value(q);
        vector normal = integrator_m.get_normal_vector(q);

        vector flux_times_normal       = calculate_upwind_flux(uM, uP, normal);
        scalar average_normal_velocity = 0.5 * (uM + uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator_m.submit_value(flux_times_normal - average_normal_velocity * uM, q);
        // opposite signs since n⁺ = - n⁻
        integrator_p.submit_value(-flux_times_normal + average_normal_velocity * uP, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM     = integrator_m.get_value(q);
        vector uP     = integrator_p.get_value(q);
        vector jump   = uM - uP;
        vector normal = integrator_m.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normal);

        // corrections to obtain an energy preserving flux (which is not conservative!)
        vector flux_m = flux + 0.25 * jump * normal * uP;
        vector flux_p = -flux + 0.25 * jump * normal * uM;

        integrator_m.submit_value(flux_m, q);
        integrator_p.submit_value(flux_p, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename Integrator>
  void
  do_boundary_integral_nonlinear_operator(Integrator &               integrator,
                                          types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator.get_value(q);
        vector uP = calculate_exterior_value(uM, q, integrator, boundary_type, boundary_id);

        vector normalM = integrator.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normalM);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux, uM * normalM);

        integrator.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator.get_value(q);
        vector uP = calculate_exterior_value(uM, q, integrator, boundary_type, boundary_id);

        vector normal = integrator.get_normal_vector(q);

        vector flux_times_normal = calculate_upwind_flux(uM, uP, normal);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux_times_normal, uM * normal);

        scalar average_normal_velocity = 0.5 * (uM + uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator.submit_value(flux_times_normal - average_normal_velocity * uM, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM      = integrator.get_value(q);
        vector uP      = calculate_exterior_value(uM, q, integrator, boundary_type, boundary_id);
        vector normalM = integrator.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normalM);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux, uM * normalM);

        // corrections to obtain an energy preserving flux (which is not conservative!)
        flux = flux + 0.25 * (uM - uP) * normalM * uP;
        integrator.submit_value(flux, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // OIF splitting
  template<typename Integrator>
  void
  do_cell_integral_linear_transport(Integrator & integrator,
                                    Integrator & integrator_transport) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      integrator.evaluate(true, false, false);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // nonlinear convective flux F = uw
        vector u = integrator.get_value(q);
        vector w = integrator_transport.get_value(q);
        tensor F = outer_product(u, w);
        // minus sign due to integration by parts
        integrator.submit_gradient(-F, q);
      }
      integrator.integrate(false, true);
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      integrator.evaluate(false, true, false);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // convective formulation: grad(u) * w
        vector w      = integrator_transport.get_value(q);
        tensor grad_u = integrator.get_gradient(q);
        vector F      = grad_u * w;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        integrator.submit_value(F, q);
      }
      integrator.integrate(true, false);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // splitting
  template<typename Integrator>
  void
  do_face_integral_linear_transport(Integrator & integrator_m,
                                    Integrator & integrator_p,
                                    Integrator & integrator_transport_m,
                                    Integrator & integrator_transport_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_m.get_value(q);
        vector uP = integrator_p.get_value(q);

        vector wM = integrator_transport_m.get_value(q);
        vector wP = integrator_transport_p.get_value(q);

        vector normal = integrator_m.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux_linear_transport(uM, uP, wM, wP, normal);

        integrator_m.submit_value(flux, q);
        integrator_p.submit_value(-flux, q); // minus sign since n⁺ = - n⁻
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_m.get_value(q);
        vector uP = integrator_p.get_value(q);

        vector wM = integrator_transport_m.get_value(q);
        vector wP = integrator_transport_p.get_value(q);

        vector normal = integrator_m.get_normal_vector(q);

        vector flux_times_normal = calculate_upwind_flux_linear_transport(uM, uP, wM, wP, normal);

        scalar average_normal_velocity = 0.5 * (wM + wP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator_m.submit_value(flux_times_normal - average_normal_velocity * uM, q);
        // opposite signs since n⁺ = - n⁻
        integrator_p.submit_value(-flux_times_normal + average_normal_velocity * uP, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // OIF splitting
  template<typename Integrator>
  void
  do_boundary_integral_linear_transport(Integrator &               integrator,
                                        Integrator &               integrator_transport,
                                        types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator.get_value(q);
        vector uP = calculate_exterior_value(uM, q, integrator, boundary_type, boundary_id);

        vector wM = integrator_transport.get_value(q);

        vector normal = integrator.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux_linear_transport(uM, uP, wM, wM, normal);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux, wM * normal);

        integrator.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator.get_value(q);
        vector uP = calculate_exterior_value(uM, q, integrator, boundary_type, boundary_id);

        // concerning the transport velocity w, use the same value for interior and
        // exterior states, i.e., do not prescribe boundary conditions
        vector w = integrator_transport.get_value(q);

        vector normal = integrator.get_normal_vector(q);

        vector flux_times_normal = calculate_upwind_flux_linear_transport(uM, uP, w, w, normal);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux_times_normal, w * normal);

        scalar average_normal_velocity = w * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator.submit_value(flux_times_normal - average_normal_velocity * uM, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // linearized operator
  template<typename Integrator>
  void
  do_cell_integral_linearized_operator(Integrator & integrator,
                                       Integrator & integrator_linearization) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      integrator.evaluate(true, false, false);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector delta_u = integrator.get_value(q);
        vector u       = integrator_linearization.get_value(q);
        tensor F       = outer_product(u, delta_u);
        integrator.submit_gradient(-(F + transpose(F)),
                                   q); // minus sign due to integration by parts
      }
      integrator.integrate(false, true);
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      integrator.evaluate(true, true, false);
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        // convective term: grad(u) * u
        vector u            = integrator_linearization.get_value(q);
        tensor grad_u       = integrator_linearization.get_gradient(q);
        vector delta_u      = integrator.get_value(q);
        tensor grad_delta_u = integrator.get_gradient(q);

        vector F = grad_u * delta_u + grad_delta_u * u;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        integrator.submit_value(F, q);
      }
      integrator.integrate(true, false);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename Integrator>
  void
  do_face_integral_linearized_operator(Integrator & integrator_m,
                                       Integrator & integrator_p,
                                       Integrator & integrator_linearization_m,
                                       Integrator & integrator_linearization_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_linearization_m.get_value(q);
        vector uP = integrator_linearization_p.get_value(q);

        vector delta_uM = integrator_m.get_value(q);
        vector delta_uP = integrator_p.get_value(q);

        vector normal = integrator_m.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        integrator_m.submit_value(flux, q);
        integrator_p.submit_value(-flux, q); // minus sign since n⁺ = -n⁻
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_linearization_m.get_value(q);
        vector uP = integrator_linearization_p.get_value(q);

        vector delta_uM = integrator_m.get_value(q);
        vector delta_uP = integrator_p.get_value(q);

        vector normal = integrator_m.get_normal_vector(q);

        vector flux_times_normal =
          calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator_m.submit_value(flux_times_normal - average_normal_velocity * delta_uM -
                                    delta_average_normal_velocity * uM,
                                  q);
        // opposite signs since n⁺ = - n⁻
        integrator_p.submit_value(-flux_times_normal + average_normal_velocity * delta_uP +
                                    delta_average_normal_velocity * uP,
                                  q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename Integrator>
  void
  do_face_int_integral_linearized_operator(Integrator & integrator_m,
                                           Integrator & /* integrator_p */,
                                           Integrator & integrator_linearization_m,
                                           Integrator & integrator_linearization_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_linearization_m.get_value(q);
        vector uP = integrator_linearization_p.get_value(q);

        vector delta_uM = integrator_m.get_value(q);
        vector delta_uP; // set exterior value to zero

        vector normal_m = integrator_m.get_normal_vector(q);

        vector flux =
          calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal_m);

        integrator_m.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
      {
        vector uM = integrator_linearization_m.get_value(q);
        vector uP = integrator_linearization_p.get_value(q);

        vector delta_uM = integrator_m.get_value(q);
        vector delta_uP; // set exterior value to zero

        vector normal = integrator_m.get_normal_vector(q);

        vector flux_times_normal =
          calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator_m.submit_value(flux_times_normal - average_normal_velocity * delta_uM -
                                    delta_average_normal_velocity * uM,
                                  q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename Integrator>
  void
  do_face_ext_integral_linearized_operator(Integrator & /* integrator_m */,
                                           Integrator & integrator_p,
                                           Integrator & integrator_linearization_m,
                                           Integrator & integrator_linearization_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
      {
        vector uM = integrator_linearization_m.get_value(q);
        vector uP = integrator_linearization_p.get_value(q);

        vector delta_uM; // set exterior value to zero
        vector delta_uP = integrator_p.get_value(q);

        vector normal_p = -integrator_p.get_normal_vector(q);

        vector flux =
          calculate_lax_friedrichs_flux_linearized(uP, uM, delta_uP, delta_uM, normal_p);

        integrator_p.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator_p.n_q_points; ++q)
      {
        vector uM = integrator_linearization_m.get_value(q);
        vector uP = integrator_linearization_p.get_value(q);

        vector delta_uM; // set exterior value to zero
        vector delta_uP = integrator_p.get_value(q);

        vector normal_p = -integrator_p.get_normal_vector(q);

        vector flux_times_normal =
          calculate_upwind_flux_linearized(uP, uM, delta_uP, delta_uM, normal_p);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal_p;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal_p;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        // opposite signs since n⁺ = - n⁻
        integrator_p.submit_value(flux_times_normal - average_normal_velocity * delta_uP -
                                    delta_average_normal_velocity * uP,
                                  q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename Integrator>
  void
  do_boundary_integral_linearized_operator(Integrator &               integrator,
                                           Integrator &               integrator_linearization,
                                           types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator_linearization.get_value(q);
        vector uP =
          calculate_exterior_value(uM, q, integrator_linearization, boundary_type, boundary_id);

        vector delta_uM = integrator.get_value(q);
        vector delta_uP =
          calculate_exterior_value_linearized(delta_uM, q, integrator, boundary_type);

        vector normal = integrator.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux, uM * normal);

        integrator.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        vector uM = integrator_linearization.get_value(q);
        vector uP =
          calculate_exterior_value(uM, q, integrator_linearization, boundary_type, boundary_id);

        vector delta_uM = integrator.get_value(q);
        vector delta_uP =
          calculate_exterior_value_linearized(delta_uM, q, integrator, boundary_type);

        vector normal = integrator.get_normal_vector(q);

        vector flux_times_normal =
          calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux_times_normal, uM * normal);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        integrator.submit_value(flux_times_normal - average_normal_velocity * delta_uM -
                                  delta_average_normal_velocity * uM,
                                q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  /*
   *  This function calculates the exterior velocity on boundary faces
   *  according to:
   *
   *  Dirichlet boundary: u⁺ = -u⁻ + 2g
   *  Neumann boundary:   u⁺ = u⁻
   *  symmetry boundary:  u⁺ = u⁻ -(u⁻*n)n - (u⁻*n)n = u⁻ - 2 (u⁻*n)n
   */
  template<typename Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_value(vector const &           uM,
                             unsigned int const       q,
                             Integrator &             integrator,
                             BoundaryTypeU const &    boundary_type,
                             types::boundary_id const boundary_id) const
  {
    vector uP;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
      it                          = operator_data.bc->dirichlet_bc.find(boundary_id);
      Point<dim, scalar> q_points = integrator.quadrature_point(q);

      vector g = evaluate_vectorial_function(it->second, q_points, eval_time);

      if(operator_data.type_dirichlet_bc == TypeDirichletBCs::Mirror)
      {
        uP = -uM + make_vectorized_array<Number>(2.0) * g;
      }
      else if(operator_data.type_dirichlet_bc == TypeDirichletBCs::Direct)
      {
        uP = g;
      }
      else
      {
        AssertThrow(
          false,
          ExcMessage(
            "Type of imposition of Dirichlet BC's for convective term is not implemented."));
      }
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      uP = uM;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      vector normalM = integrator.get_normal_vector(q);

      uP = uM - 2. * (uM * normalM) * normalM;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return uP;
  }

  /*
   *  Calculate Lax-Friedrichs flux for linearized operator on boundary faces.
   *
   *  Homogeneous linearized operator:
   *  Dirichlet boundary: delta_u⁺ = - delta_u⁻
   *  Neumann boundary:   delta_u⁺ = + delta_u⁻
   *  symmetry boundary:  delta_u⁺ = delta_u⁻ - 2 (delta_u⁻*n)n
   */
  template<typename Integrator>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_value_linearized(vector &              delta_uM,
                                        unsigned int const    q,
                                        Integrator &          integrator,
                                        BoundaryTypeU const & boundary_type) const
  {
    // element e⁺
    vector delta_uP;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      if(operator_data.type_dirichlet_bc == TypeDirichletBCs::Mirror)
      {
        delta_uP = -delta_uM;
      }
      else if(operator_data.type_dirichlet_bc == TypeDirichletBCs::Direct)
      {
        // delta_uP = 0
        // do nothing, delta_uP is already initialized with zero
      }
      else
      {
        AssertThrow(
          false,
          ExcMessage(
            "Type of imposition of Dirichlet BC's for convective term is not implemented."));
      }
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      delta_uP = delta_uM;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      vector normalM = integrator.get_normal_vector(q);
      delta_uP       = delta_uM - 2. * (delta_uM * normalM) * normalM;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return delta_uP;
  }

  /*
   *  Lax-Friedrichs flux (divergence formulation)
   *  Calculation of lambda according to Shahbazi et al.:
   *  lambda = max ( max |lambda(flux_jacobian_M)| , max |lambda(flux_jacobian_P)| )
   *         = max ( | 2*(uM)^T*normal | , | 2*(uP)^T*normal | )
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_lambda(scalar const & uM_n, scalar const & uP_n) const
  {
    return this->operator_data.upwind_factor * 2.0 * std::max(std::abs(uM_n), std::abs(uP_n));
  }

  /*
   *  Calculate Lax-Friedrichs flux for nonlinear operator (divergence formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux(vector const & uM,
                                  vector const & uP,
                                  vector const & normalM) const
  {
    scalar uM_n = uM * normalM;
    scalar uP_n = uP * normalM;

    vector average_normal_flux = make_vectorized_array<Number>(0.5) * (uM * uM_n + uP * uP_n);

    vector jump_value = uM - uP;

    scalar lambda = calculate_lambda(uM_n, uP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   *  Calculate Lax-Friedrichs flux for nonlinear operator (linear transport).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux_linear_transport(vector const & uM,
                                                   vector const & uP,
                                                   vector const & wM,
                                                   vector const & wP,
                                                   vector const & normalM) const
  {
    scalar wM_n = wM * normalM;
    scalar wP_n = wP * normalM;

    vector average_normal_flux = make_vectorized_array<Number>(0.5) * (uM * wM_n + uP * wP_n);

    vector jump_value = uM - uP;

    scalar lambda = calculate_lambda(wM_n, wP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   *  Calculate Lax-Friedrichs flux for linearized operator (divergence formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_lax_friedrichs_flux_linearized(vector const & uM,
                                             vector const & uP,
                                             vector const & delta_uM,
                                             vector const & delta_uP,
                                             vector const & normalM) const
  {
    scalar uM_n = uM * normalM;
    scalar uP_n = uP * normalM;

    scalar delta_uM_n = delta_uM * normalM;
    scalar delta_uP_n = delta_uP * normalM;

    vector average_normal_flux =
      make_vectorized_array<Number>(0.5) *
      (uM * delta_uM_n + delta_uM * uM_n + uP * delta_uP_n + delta_uP * uP_n);

    vector jump_value = delta_uM - delta_uP;

    scalar lambda = calculate_lambda(uM_n, uP_n);

    return (average_normal_flux + 0.5 * lambda * jump_value);
  }

  /*
   *  Calculate upwind flux for nonlinear operator (convective formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_upwind_flux(vector const & uM, vector const & uP, vector const & normalM) const
  {
    vector average_velocity = 0.5 * (uM + uP);

    scalar average_normal_velocity = average_velocity * normalM;

    vector jump_value = uM - uP;

    return (average_normal_velocity * average_velocity + this->operator_data.upwind_factor * 0.5 *
                                                           std::abs(average_normal_velocity) *
                                                           jump_value);
  }

  /*
   *  Calculate upwind flux for convective operator (linear transport, OIF splitting).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_upwind_flux_linear_transport(vector const & uM,
                                           vector const & uP,
                                           vector const & wM,
                                           vector const & wP,
                                           vector const & normalM) const
  {
    vector average_velocity = 0.5 * (uM + uP);

    vector jump_value = uM - uP;

    scalar average_normal_velocity = 0.5 * (wM + wP) * normalM;

    return (average_normal_velocity * average_velocity + this->operator_data.upwind_factor * 0.5 *
                                                           std::abs(average_normal_velocity) *
                                                           jump_value);
  }

  /*
   * outflow BC according to Gravemeier et al. (2012)
   */
  inline DEAL_II_ALWAYS_INLINE //
    void
    apply_outflow_bc(vector & flux, scalar const & uM_n) const
  {
    // we need a factor indicating whether we have inflow or outflow
    // on the Neumann part of the boundary.
    // outflow: factor =  1.0 (do nothing, neutral element of multiplication)
    // inflow:  factor = 0.0 (set convective flux to zero)
    scalar outflow_indicator = make_vectorized_array<Number>(1.0);

    for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
    {
      if(uM_n[v] < 0.0) // backflow at outflow boundary
        outflow_indicator[v] = 0.0;
    }

    // set flux to zero in case of backflow
    flux = outflow_indicator * flux;
  }

  /*
   *  Calculate upwind flux for linearized operator (convective formulation).
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_upwind_flux_linearized(vector const & uM,
                                     vector const & uP,
                                     vector const & delta_uM,
                                     vector const & delta_uP,
                                     vector const & normalM) const
  {
    vector average_velocity       = 0.5 * (uM + uP);
    vector delta_average_velocity = 0.5 * (delta_uM + delta_uP);

    scalar average_normal_velocity       = average_velocity * normalM;
    scalar delta_average_normal_velocity = delta_average_velocity * normalM;

    vector jump_value = delta_uM - delta_uP;

    return (average_normal_velocity * delta_average_velocity +
            delta_average_normal_velocity * average_velocity +
            this->operator_data.upwind_factor * 0.5 * std::abs(average_normal_velocity) *
              jump_value);
  }

  /*
   *  Evaluation of nonlinear convective operator.
   */
  void
  cell_loop_nonlinear_operator(MatrixFree<dim, Number> const & data,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   cell_range) const
  {
    CellIntegratorU integrator(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src);

      do_cell_integral_nonlinear_operator(integrator);

      integrator.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_nonlinear_operator(MatrixFree<dim, Number> const & data,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   face_range) const
  {
    FaceIntegratorU integrator_m(data, true, operator_data.dof_index, operator_data.quad_index);
    FaceIntegratorU integrator_p(data, false, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      integrator_m.gather_evaluate(src, true, false);
      integrator_p.gather_evaluate(src, true, false);

      do_face_integral_nonlinear_operator(integrator_m, integrator_p);

      integrator_m.integrate_scatter(true, false, dst);
      integrator_p.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_nonlinear_operator(MatrixFree<dim, Number> const & data,
                                        VectorType &                    dst,
                                        VectorType const &              src,
                                        Range const &                   face_range) const
  {
    FaceIntegratorU integrator(data, true, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator.reinit(face);

      integrator.gather_evaluate(src, true, false);

      do_boundary_integral_nonlinear_operator(integrator, data.get_boundary_id(face));

      integrator.integrate_scatter(true, false, dst);
    }
  }

  /*
   *  OIF splitting: evaluation convective operator (linear transport).
   */
  void
  cell_loop_linear_transport(MatrixFree<dim, Number> const & data,
                             VectorType &                    dst,
                             VectorType const &              src,
                             Range const &                   cell_range) const
  {
    CellIntegratorU integrator(data, operator_data.dof_index, operator_data.quad_index);

    CellIntegratorU integrator_transport(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);
      integrator.read_dof_values(src);

      integrator_transport.reinit(cell);
      integrator_transport.gather_evaluate(velocity_linearization, true, false, false);

      do_cell_integral_linear_transport(integrator, integrator_transport);

      integrator.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_linear_transport(MatrixFree<dim, Number> const & data,
                             VectorType &                    dst,
                             VectorType const &              src,
                             Range const &                   face_range) const
  {
    FaceIntegratorU integrator_m(data, true, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_p(data, false, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_transport_m(data,
                                           true,
                                           operator_data.dof_index,
                                           operator_data.quad_index);

    FaceIntegratorU integrator_transport_p(data,
                                           false,
                                           operator_data.dof_index,
                                           operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      integrator_m.gather_evaluate(src, true, false);
      integrator_p.gather_evaluate(src, true, false);

      integrator_transport_m.reinit(face);
      integrator_transport_p.reinit(face);

      integrator_transport_m.gather_evaluate(velocity_linearization, true, false);
      integrator_transport_p.gather_evaluate(velocity_linearization, true, false);

      do_face_integral_linear_transport(integrator_m,
                                        integrator_p,
                                        integrator_transport_m,
                                        integrator_transport_p);

      integrator_m.integrate_scatter(true, false, dst);
      integrator_p.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_linear_transport(MatrixFree<dim, Number> const & data,
                                      VectorType &                    dst,
                                      VectorType const &              src,
                                      Range const &                   face_range) const
  {
    FaceIntegratorU integrator(data, true, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_transport(data,
                                         true,
                                         operator_data.dof_index,
                                         operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator.reinit(face);
      integrator.gather_evaluate(src, true, false);

      integrator_transport.reinit(face);
      integrator_transport.gather_evaluate(velocity_linearization, true, false);

      do_boundary_integral_linear_transport(integrator,
                                            integrator_transport,
                                            data.get_boundary_id(face));

      integrator.integrate_scatter(true, false, dst);
    }
  }

  /*
   *  Evaluate linearized convective operator.
   */
  void
  cell_loop_linearized_operator(MatrixFree<dim, Number> const & data,
                                VectorType &                    dst,
                                VectorType const &              src,
                                Range const &                   cell_range) const
  {
    CellIntegratorU integrator(data, operator_data.dof_index, operator_data.quad_index);

    CellIntegratorU integrator_linearization(data,
                                             operator_data.dof_index,
                                             operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator_linearization.reinit(cell);

      if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
        integrator_linearization.gather_evaluate(velocity_linearization, true, false, false);
      else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
        integrator_linearization.gather_evaluate(velocity_linearization, true, true, false);
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      integrator.reinit(cell);
      integrator.read_dof_values(src);

      do_cell_integral_linearized_operator(integrator, integrator_linearization);

      integrator.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_linearized_operator(MatrixFree<dim, Number> const & data,
                                VectorType &                    dst,
                                VectorType const &              src,
                                Range const &                   face_range) const
  {
    FaceIntegratorU integrator_m(data, true, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_p(data, false, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_linearization_m(data,
                                               true,
                                               operator_data.dof_index,
                                               operator_data.quad_index);

    FaceIntegratorU integrator_linearization_p(data,
                                               false,
                                               operator_data.dof_index,
                                               operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_m.reinit(face);
      integrator_p.reinit(face);

      integrator_linearization_m.reinit(face);
      integrator_linearization_p.reinit(face);

      integrator_m.gather_evaluate(src, true, false);
      integrator_p.gather_evaluate(src, true, false);

      integrator_linearization_m.gather_evaluate(velocity_linearization, true, false);
      integrator_linearization_p.gather_evaluate(velocity_linearization, true, false);

      do_face_integral_linearized_operator(integrator_m,
                                           integrator_p,
                                           integrator_linearization_m,
                                           integrator_linearization_p);

      integrator_m.integrate_scatter(true, false, dst);
      integrator_p.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_linearized_operator(MatrixFree<dim, Number> const & data,
                                         VectorType &                    dst,
                                         VectorType const &              src,
                                         Range const &                   face_range) const
  {
    FaceIntegratorU integrator(data, true, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_linearization(data,
                                             true,
                                             operator_data.dof_index,
                                             operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator.reinit(face);
      integrator_linearization.reinit(face);

      integrator.gather_evaluate(src, true, false);
      integrator_linearization.gather_evaluate(velocity_linearization, true, false);

      do_boundary_integral_linearized_operator(integrator,
                                               integrator_linearization,
                                               data.get_boundary_id(face));

      integrator.integrate_scatter(true, false, dst);
    }
  }


  /*
   *  Calculation of diagonal of linearized convective operator.
   */
  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & cell_range) const
  {
    CellIntegratorU integrator(data, operator_data.dof_index, operator_data.quad_index);

    CellIntegratorU integrator_linearization(data,
                                             operator_data.dof_index,
                                             operator_data.quad_index);

    unsigned int const    dofs_per_cell = integrator.dofs_per_cell;
    AlignedVector<scalar> local_diagonal_vector(dofs_per_cell);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator_linearization.reinit(cell);

      if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
        integrator_linearization.gather_evaluate(velocity_linearization, true, false, false);
      else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
        integrator_linearization.gather_evaluate(velocity_linearization, true, true, false);
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      integrator.reinit(cell);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        do_cell_integral_linearized_operator(integrator, integrator_linearization);

        local_diagonal_vector[j] = integrator.begin_dof_values()[j];
      }

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator.begin_dof_values()[j] = local_diagonal_vector[j];

      integrator.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & face_range) const
  {
    FaceIntegratorU integrator_m(data, true, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_p(data, false, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_linearization_m(data,
                                               true,
                                               operator_data.dof_index,
                                               operator_data.quad_index);

    FaceIntegratorU integrator_linearization_p(data,
                                               false,
                                               operator_data.dof_index,
                                               operator_data.quad_index);

    unsigned int const    dofs_per_cell = integrator_m.dofs_per_cell;
    AlignedVector<scalar> local_diagonal_vector(dofs_per_cell);

    // Perform face integrals for element e⁻
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_linearization_m.reinit(face);
      integrator_linearization_m.gather_evaluate(velocity_linearization, true, false);

      integrator_linearization_p.reinit(face);
      integrator_linearization_p.gather_evaluate(velocity_linearization, true, false);

      integrator_m.reinit(face);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_m.evaluate(true, false);

        do_face_int_integral_linearized_operator(integrator_m,
                                                 integrator_p,
                                                 integrator_linearization_m,
                                                 integrator_linearization_p);

        integrator_m.integrate(true, false);

        local_diagonal_vector[j] = integrator_m.begin_dof_values()[j];
      }

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator_m.begin_dof_values()[j] = local_diagonal_vector[j];

      integrator_m.distribute_local_to_global(dst);
    }

    unsigned int const    dofs_per_cell_neighbor = integrator_p.dofs_per_cell;
    AlignedVector<scalar> local_diagonal_vector_neighbor(dofs_per_cell_neighbor);

    // Perform face integrals for element e⁺
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_linearization_m.reinit(face);
      integrator_linearization_m.gather_evaluate(velocity_linearization, true, false);

      integrator_linearization_p.reinit(face);
      integrator_linearization_p.gather_evaluate(velocity_linearization, true, false);

      integrator_p.reinit(face);

      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell_neighbor; ++i)
          integrator_p.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_p.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_p.evaluate(true, false);

        do_face_ext_integral_linearized_operator(integrator_m,
                                                 integrator_p,
                                                 integrator_linearization_m,
                                                 integrator_linearization_p);

        integrator_p.integrate(true, false);

        local_diagonal_vector_neighbor[j] = integrator_p.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
        integrator_p.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];

      integrator_p.distribute_local_to_global(dst);
    }
  }

  void
  boundary_face_loop_diagonal(MatrixFree<dim, Number> const & data,
                              VectorType &                    dst,
                              VectorType const &,
                              Range const & face_range) const
  {
    FaceIntegratorU integrator(data, true, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_linearization(data,
                                             true,
                                             operator_data.dof_index,
                                             operator_data.quad_index);

    unsigned int const    dofs_per_cell = integrator.dofs_per_cell;
    AlignedVector<scalar> local_diagonal_vector(dofs_per_cell);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator.reinit(face);

      integrator_linearization.reinit(face);
      integrator_linearization.gather_evaluate(velocity_linearization, true, false);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(true, false);

        do_boundary_integral_linearized_operator(integrator,
                                                 integrator_linearization,
                                                 data.get_boundary_id(face));

        integrator.integrate(true, false);

        local_diagonal_vector[j] = integrator.begin_dof_values()[j];
      }

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator.begin_dof_values()[j] = local_diagonal_vector[j];

      integrator.distribute_local_to_global(dst);
    }
  }

  /*
   *  Apply block-diagonal operator (only needed for testing): re-implement face_loop (cell_loop and
   * boundary_face_loop are identical to linearized homogeneous operator.
   */
  void
  face_loop_apply_block_diagonal_linearized_operator(MatrixFree<dim, Number> const & data,
                                                     VectorType &                    dst,
                                                     VectorType const &              src,
                                                     Range const & face_range) const
  {
    FaceIntegratorU integrator_m(data, true, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_p(data, false, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_linearization_m(data,
                                               true,
                                               operator_data.dof_index,
                                               operator_data.quad_index);

    FaceIntegratorU integrator_linearization_p(data,
                                               false,
                                               operator_data.dof_index,
                                               operator_data.quad_index);

    // Perform face integral for element e⁻
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_linearization_m.reinit(face);
      integrator_linearization_m.gather_evaluate(velocity_linearization, true, false);

      integrator_linearization_p.reinit(face);
      integrator_linearization_p.gather_evaluate(velocity_linearization, true, false);

      integrator_m.reinit(face);
      integrator_m.gather_evaluate(src, true, false);

      do_face_int_integral_linearized_operator(integrator_m,
                                               integrator_p,
                                               integrator_linearization_m,
                                               integrator_linearization_p);

      integrator_m.integrate_scatter(true, false, dst);
    }

    // Perform face integral for element e⁺
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_linearization_m.reinit(face);
      integrator_linearization_m.gather_evaluate(velocity_linearization, true, false);

      integrator_linearization_p.reinit(face);
      integrator_linearization_p.gather_evaluate(velocity_linearization, true, false);

      integrator_p.reinit(face);
      integrator_p.gather_evaluate(src, true, false);

      do_face_ext_integral_linearized_operator(integrator_m,
                                               integrator_p,
                                               integrator_linearization_m,
                                               integrator_linearization_p);

      integrator_p.integrate_scatter(true, false, dst);
    }
  }

  /*
   * Calculate/assemble block-diagonal matrices via matrix-free operator evaluation.
   */
  void
  cell_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & cell_range) const
  {
    CellIntegratorU integrator(data, operator_data.dof_index, operator_data.quad_index);

    CellIntegratorU integrator_linearization(data,
                                             operator_data.dof_index,
                                             operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator_linearization.reinit(cell);

      if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
        integrator_linearization.gather_evaluate(velocity_linearization, true, false, false);
      else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
        integrator_linearization.gather_evaluate(velocity_linearization, true, true, false);
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      integrator.reinit(cell);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        do_cell_integral_linearized_operator(integrator, integrator_linearization);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              integrator.begin_dof_values()[i][v];
      }
    }
  }

  void
  face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & face_range) const
  {
    FaceIntegratorU integrator_m(data, true, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_p(data, false, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_linearization_m(data,
                                               true,
                                               operator_data.dof_index,
                                               operator_data.quad_index);

    FaceIntegratorU integrator_linearization_p(data,
                                               false,
                                               operator_data.dof_index,
                                               operator_data.quad_index);

    // Perform face integrals for element e⁻.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_linearization_m.reinit(face);
      integrator_linearization_m.gather_evaluate(velocity_linearization, true, false);

      integrator_linearization_p.reinit(face);
      integrator_linearization_p.gather_evaluate(velocity_linearization, true, false);

      integrator_m.reinit(face);

      unsigned int dofs_per_cell = integrator_m.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_m.evaluate(true, false);

        do_face_int_integral_linearized_operator(integrator_m,
                                                 integrator_p,
                                                 integrator_linearization_m,
                                                 integrator_linearization_p);

        integrator_m.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += integrator_m.begin_dof_values()[i][v];
        }
      }
    }

    // Perform face integrals for element e⁺.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator_linearization_m.reinit(face);
      integrator_linearization_m.gather_evaluate(velocity_linearization, true, false);

      integrator_linearization_p.reinit(face);
      integrator_linearization_p.gather_evaluate(velocity_linearization, true, false);

      integrator_p.reinit(face);

      unsigned int dofs_per_cell = integrator_p.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator_p.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator_p.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator_p.evaluate(true, false);

        do_face_ext_integral_linearized_operator(integrator_m,
                                                 integrator_p,
                                                 integrator_linearization_m,
                                                 integrator_linearization_p);

        integrator_p.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_exterior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += integrator_p.begin_dof_values()[i][v];
        }
      }
    }
  }

  void
  boundary_face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                              std::vector<LAPACKFullMatrix<Number>> & matrices,
                                              VectorType const &,
                                              Range const & face_range) const
  {
    FaceIntegratorU integrator(data, true, operator_data.dof_index, operator_data.quad_index);

    FaceIntegratorU integrator_linearization(data,
                                             true,
                                             operator_data.dof_index,
                                             operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator.reinit(face);

      integrator_linearization.reinit(face);
      integrator_linearization.gather_evaluate(velocity_linearization, true, false);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(true, false);

        do_boundary_integral_linearized_operator(integrator,
                                                 integrator_linearization,
                                                 data.get_boundary_id(face));

        integrator.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += integrator.begin_dof_values()[i][v];
        }
      }
    }
  }

  void
  cell_based_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                           std::vector<LAPACKFullMatrix<Number>> & matrices,
                                           VectorType const &,
                                           Range const & cell_range) const
  {
    CellIntegratorU integrator(data, operator_data.dof_index, operator_data.quad_index);
    FaceIntegratorU integrator_m(data, true, operator_data.dof_index, operator_data.quad_index);
    FaceIntegratorU integrator_p(data, false, operator_data.dof_index, operator_data.quad_index);

    CellIntegratorU integrator_linearization(data,
                                             operator_data.dof_index,
                                             operator_data.quad_index);
    FaceIntegratorU integrator_linearization_m(data,
                                               true,
                                               operator_data.dof_index,
                                               operator_data.quad_index);
    FaceIntegratorU integrator_linearization_p(data,
                                               false,
                                               operator_data.dof_index,
                                               operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // cell integral
      unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);

      integrator_linearization.reinit(cell);

      if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
        integrator_linearization.gather_evaluate(velocity_linearization, true, false, false);
      else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
        integrator_linearization.gather_evaluate(velocity_linearization, true, true, false);
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      integrator.reinit(cell);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        do_cell_integral_linearized_operator(integrator, integrator_linearization);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              integrator.begin_dof_values()[i][v];
      }

      // loop over all faces
      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        integrator_linearization_m.reinit(cell, face);
        integrator_linearization_m.gather_evaluate(velocity_linearization, true, false);

        // TODO
        //        AssertThrow(false, ExcMessage("We have to evaluate the linearized velocity field
        //        on the neighbor. This functionality is not implemented in deal.II/matrix_free."));
        //        integrator_linearization_p.reinit(cell, face);
        //        integrator_linearization_p.gather_evaluate(velocity_linearization, true, false);

        integrator_m.reinit(cell, face);
        integrator_p.reinit(cell, face);

        auto bids        = data.get_faces_by_cells_boundary_id(cell, face);
        auto boundary_id = bids[0];

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            integrator_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
          integrator_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

          integrator_m.evaluate(true, false);

          if(boundary_id == numbers::internal_face_boundary_id) // internal face
          {
            // TODO
            //            do_face_int_integral_linearized_operator(integrator_m,
            //            integrator_p,integrator_linearization_m, integrator_linearization_p);

            // plug in integrator_linearization_m twice to avoid the above problem with accessing
            // dofs of the neighboring element
            do_face_int_integral_linearized_operator(integrator_m,
                                                     integrator_p,
                                                     integrator_linearization_m,
                                                     integrator_linearization_m);
          }
          else // boundary face
          {
            do_boundary_integral_linearized_operator(integrator_m,
                                                     integrator_linearization_m,
                                                     boundary_id);
          }

          integrator_m.integrate(true, false);

          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
                integrator_m.begin_dof_values()[i][v];
        }
      }
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

  ConvectiveOperatorData<dim> operator_data;

  mutable Number eval_time;

  mutable VectorType velocity_linearization;

  unsigned int n_mpi_processes;

  std::shared_ptr<CellIntegratorU> integrator;
  std::shared_ptr<FaceIntegratorU> integrator_m;
  std::shared_ptr<FaceIntegratorU> integrator_p;

  std::shared_ptr<CellIntegratorU> integrator_linearization;
  std::shared_ptr<FaceIntegratorU> integrator_linearization_m;
  std::shared_ptr<FaceIntegratorU> integrator_linearization_p;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_ \
        */
