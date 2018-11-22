/*
 * convective_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation.h>

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



template<int dim, int degree, typename Number>
class ConvectiveOperator
{
public:
  typedef ConvectiveOperator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef std::pair<unsigned int, unsigned int> Range;

  static const unsigned int n_q_points_overint = degree + (degree + 2) / 2;

  typedef FEEvaluation<dim, degree, n_q_points_overint, dim, Number>     FEEvalCellOverint;
  typedef FEFaceEvaluation<dim, degree, n_q_points_overint, dim, Number> FEEvalFaceOverint;

  ConvectiveOperator()
    : data(nullptr),
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
  initialize(MatrixFree<dim, Number> const &     data_in,
             ConvectiveOperatorData<dim> const & operator_data_in)
  {
    data          = &data_in;
    operator_data = operator_data_in;

    data->initialize_dof_vector(velocity_linearization, operator_data.dof_index);

    // Block Jacobi elementwise
    fe_eval.reset(new FEEvalCellOverint(*data, operator_data.dof_index, operator_data.quad_index));
    fe_eval_m.reset(
      new FEEvalFaceOverint(*data, true, operator_data.dof_index, operator_data.quad_index));
    fe_eval_p.reset(
      new FEEvalFaceOverint(*data, false, operator_data.dof_index, operator_data.quad_index));
    fe_eval_linearization.reset(
      new FEEvalCellOverint(*data, operator_data.dof_index, operator_data.quad_index));
    fe_eval_linearization_m.reset(
      new FEEvalFaceOverint(*data, true, operator_data.dof_index, operator_data.quad_index));
    fe_eval_linearization_p.reset(
      new FEEvalFaceOverint(*data, false, operator_data.dof_index, operator_data.quad_index));
  }

  /*
   * Evaluate nonlinear convective operator.
   */
  void
  evaluate(VectorType & dst, VectorType const & src, Number const evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop_nonlinear_operator,
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

    data->loop(&This::cell_loop_nonlinear_operator,
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

    data->loop(&This::cell_loop_linear_transport,
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

    data->loop(&This::cell_loop_linearized_operator,
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

    data->loop(&This::cell_loop_linearized_operator,
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

    data->loop(&This::cell_loop_diagonal,
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

    data->loop(&This::cell_loop_diagonal,
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
      data->cell_loop(&This::cell_based_loop_calculate_block_diagonal, this, matrices, src);
    }
    else
    {
      AssertThrow(
        n_mpi_processes == 1,
        ExcMessage(
          "Block diagonal calculation with separate loops over cells and faces only works in serial. "
          "Use cell based loops for parallel computations."));

      data->loop(&This::cell_loop_calculate_block_diagonal,
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

    data->loop(&This::cell_loop_linearized_operator,
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

    data->loop(&This::cell_loop_linearized_operator,
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

    unsigned int dofs_per_cell = fe_eval->dofs_per_cell;

    fe_eval_linearization->reinit(cell);

    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
      fe_eval_linearization->gather_evaluate(velocity_linearization, true, false, false);
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
      fe_eval_linearization->gather_evaluate(velocity_linearization, true, true, false);
    else
      AssertThrow(false, ExcMessage("Not implemented."));

    fe_eval->reinit(cell);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      fe_eval->begin_dof_values()[i] = src[i];

    do_cell_integral_linearized_operator(*fe_eval, *fe_eval_linearization);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dst[i] += fe_eval->begin_dof_values()[i];

    // loop over all faces
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      fe_eval_linearization_m->reinit(cell, face);
      fe_eval_linearization_m->gather_evaluate(velocity_linearization, true, false);

      // TODO
      //        AssertThrow(false, ExcMessage("We have to evaluate the linearized velocity field on
      //        the neighbor. This functionality is not implemented in deal.II/matrix_free."));
      //        fe_eval_linearization_p->reinit(cell, face);
      //        fe_eval_linearization_p->gather_evaluate(velocity_linearization, true, false);

      fe_eval_m->reinit(cell, face);
      fe_eval_p->reinit(cell, face);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        fe_eval_m->begin_dof_values()[i] = src[i];

      // do not need to read dof values for fe_eval_p (already initialized with 0)

      fe_eval_m->evaluate(true, false);

      auto bids        = data->get_faces_by_cells_boundary_id(cell, face);
      auto boundary_id = bids[0];

      if(boundary_id == numbers::internal_face_boundary_id) // internal face
      {
        // TODO
        //            do_face_int_integral_linearized_operator(*fe_eval_m, *fe_eval_p,
        //            *fe_eval_linearization_m, *fe_eval_linearization_p);

        // plug in fe_eval_linearization_m twice to avoid the above problem with accessing dofs of
        // the neighboring element
        do_face_int_integral_linearized_operator(*fe_eval_m,
                                                 *fe_eval_p,
                                                 *fe_eval_linearization_m,
                                                 *fe_eval_linearization_m);
      }
      else // boundary face
      {
        do_boundary_integral_linearized_operator(*fe_eval_m, *fe_eval_linearization_m, boundary_id);
      }

      fe_eval_m->integrate(true, false);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst[i] += fe_eval_m->begin_dof_values()[i];
    }
  }

private:
  // nonlinear operator
  template<typename FEEvaluation>
  void
  do_cell_integral_nonlinear_operator(FEEvaluation & fe_eval) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      fe_eval.evaluate(true, false, false);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        vector u = fe_eval.get_value(q);
        tensor F = outer_product(u, u);
        // minus sign due to integration by parts
        fe_eval.submit_gradient(-F, q);
      }
      fe_eval.integrate(false, true);
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      fe_eval.evaluate(true, true, false);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // convective formulation: (u * grad) u = grad(u) * u
        vector u          = fe_eval.get_value(q);
        tensor gradient_u = fe_eval.get_gradient(q);
        vector F          = gradient_u * u;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        fe_eval.submit_value(F, q);
      }
      fe_eval.integrate(true, false);
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      fe_eval.evaluate(true, true, false);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // nonlinear convective flux F(u) = uu
        vector u          = fe_eval.get_value(q);
        tensor F          = outer_product(u, u);
        scalar divergence = fe_eval.get_divergence(q);
        vector div_term   = -0.5 * divergence * u;
        // minus sign due to integration by parts
        fe_eval.submit_gradient(-F, q);
        fe_eval.submit_value(div_term, q);
      }
      fe_eval.integrate(true, true);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename FEEvaluation>
  void
  do_face_integral_nonlinear_operator(FEEvaluation & fe_eval_m, FEEvaluation & fe_eval_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM     = fe_eval_m.get_value(q);
        vector uP     = fe_eval_p.get_value(q);
        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normal);

        fe_eval_m.submit_value(flux, q);
        fe_eval_p.submit_value(-flux, q); // minus sign since n⁺ = - n⁻
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM     = fe_eval_m.get_value(q);
        vector uP     = fe_eval_p.get_value(q);
        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux_times_normal       = calculate_upwind_flux(uM, uP, normal);
        scalar average_normal_velocity = 0.5 * (uM + uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fe_eval_m.submit_value(flux_times_normal - average_normal_velocity * uM, q);
        // opposite signs since n⁺ = - n⁻
        fe_eval_p.submit_value(-flux_times_normal + average_normal_velocity * uP, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM     = fe_eval_m.get_value(q);
        vector uP     = fe_eval_p.get_value(q);
        vector jump   = uM - uP;
        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normal);

        // corrections to obtain an energy preserving flux (which is not conservative!)
        vector flux_m = flux + 0.25 * jump * normal * uP;
        vector flux_p = -flux + 0.25 * jump * normal * uM;

        fe_eval_m.submit_value(flux_m, q);
        fe_eval_p.submit_value(flux_p, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename FEEvaluation>
  void
  do_boundary_integral_nonlinear_operator(FEEvaluation &             fe_eval,
                                          types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector uM = fe_eval.get_value(q);
        vector uP = calculate_exterior_value(uM, q, fe_eval, boundary_type, boundary_id);

        vector normalM = fe_eval.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normalM);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux, uM * normalM);

        fe_eval.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector uM = fe_eval.get_value(q);
        vector uP = calculate_exterior_value(uM, q, fe_eval, boundary_type, boundary_id);

        vector normal = fe_eval.get_normal_vector(q);

        vector flux_times_normal = calculate_upwind_flux(uM, uP, normal);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux_times_normal, uM * normal);

        scalar average_normal_velocity = 0.5 * (uM + uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fe_eval.submit_value(flux_times_normal - average_normal_velocity * uM, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::EnergyPreservingFormulation)
    {
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector uM      = fe_eval.get_value(q);
        vector uP      = calculate_exterior_value(uM, q, fe_eval, boundary_type, boundary_id);
        vector normalM = fe_eval.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux(uM, uP, normalM);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux, uM * normalM);

        // corrections to obtain an energy preserving flux (which is not conservative!)
        flux = flux + 0.25 * (uM - uP) * normalM * uP;
        fe_eval.submit_value(flux, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // OIF splitting
  template<typename FEEvaluation>
  void
  do_cell_integral_linear_transport(FEEvaluation & fe_eval, FEEvaluation & fe_eval_transport) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      fe_eval.evaluate(true, false, false);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // nonlinear convective flux F = uw
        vector u = fe_eval.get_value(q);
        vector w = fe_eval_transport.get_value(q);
        tensor F = outer_product(u, w);
        // minus sign due to integration by parts
        fe_eval.submit_gradient(-F, q);
      }
      fe_eval.integrate(false, true);
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      fe_eval.evaluate(false, true, false);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // convective formulation: grad(u) * w
        vector w      = fe_eval_transport.get_value(q);
        tensor grad_u = fe_eval.get_gradient(q);
        vector F      = grad_u * w;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        fe_eval.submit_value(F, q);
      }
      fe_eval.integrate(true, false);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // splitting
  template<typename FEEvaluation>
  void
  do_face_integral_linear_transport(FEEvaluation & fe_eval_m,
                                    FEEvaluation & fe_eval_p,
                                    FEEvaluation & fe_eval_transport_m,
                                    FEEvaluation & fe_eval_transport_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM = fe_eval_m.get_value(q);
        vector uP = fe_eval_p.get_value(q);

        vector wM = fe_eval_transport_m.get_value(q);
        vector wP = fe_eval_transport_p.get_value(q);

        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux_linear_transport(uM, uP, wM, wP, normal);

        fe_eval_m.submit_value(flux, q);
        fe_eval_p.submit_value(-flux, q); // minus sign since n⁺ = - n⁻
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM = fe_eval_m.get_value(q);
        vector uP = fe_eval_p.get_value(q);

        vector wM = fe_eval_transport_m.get_value(q);
        vector wP = fe_eval_transport_p.get_value(q);

        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux_times_normal = calculate_upwind_flux_linear_transport(uM, uP, wM, wP, normal);

        scalar average_normal_velocity = 0.5 * (wM + wP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fe_eval_m.submit_value(flux_times_normal - average_normal_velocity * uM, q);
        // opposite signs since n⁺ = - n⁻
        fe_eval_p.submit_value(-flux_times_normal + average_normal_velocity * uP, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // OIF splitting
  template<typename FEEvaluation>
  void
  do_boundary_integral_linear_transport(FEEvaluation &             fe_eval,
                                        FEEvaluation &             fe_eval_transport,
                                        types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector uM = fe_eval.get_value(q);
        vector uP = calculate_exterior_value(uM, q, fe_eval, boundary_type, boundary_id);

        vector wM = fe_eval_transport.get_value(q);

        vector normal = fe_eval.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux_linear_transport(uM, uP, wM, wM, normal);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux, wM * normal);

        fe_eval.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector uM = fe_eval.get_value(q);
        vector uP = calculate_exterior_value(uM, q, fe_eval, boundary_type, boundary_id);

        // concerning the transport velocity w, use the same value for interior and
        // exterior states, i.e., do not prescribe boundary conditions
        vector w = fe_eval_transport.get_value(q);

        vector normal = fe_eval.get_normal_vector(q);

        vector flux_times_normal = calculate_upwind_flux_linear_transport(uM, uP, w, w, normal);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux_times_normal, w * normal);

        scalar average_normal_velocity = w * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fe_eval.submit_value(flux_times_normal - average_normal_velocity * uM, q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  // linearized operator
  template<typename FEEvaluation>
  void
  do_cell_integral_linearized_operator(FEEvaluation & fe_eval,
                                       FEEvaluation & fe_eval_linearization) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      fe_eval.evaluate(true, false, false);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector delta_u = fe_eval.get_value(q);
        vector u       = fe_eval_linearization.get_value(q);
        tensor F       = outer_product(u, delta_u);
        fe_eval.submit_gradient(-(F + transpose(F)), q); // minus sign due to integration by parts
      }
      fe_eval.integrate(false, true);
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      fe_eval.evaluate(true, true, false);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        // convective term: grad(u) * u
        vector u            = fe_eval_linearization.get_value(q);
        tensor grad_u       = fe_eval_linearization.get_gradient(q);
        vector delta_u      = fe_eval.get_value(q);
        tensor grad_delta_u = fe_eval.get_gradient(q);

        vector F = grad_u * delta_u + grad_delta_u * u;

        // plus sign since the strong formulation is used, i.e.
        // integration by parts is performed twice
        fe_eval.submit_value(F, q);
      }
      fe_eval.integrate(true, false);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename FEEvaluation>
  void
  do_face_integral_linearized_operator(FEEvaluation & fe_eval_m,
                                       FEEvaluation & fe_eval_p,
                                       FEEvaluation & fe_eval_linearization_m,
                                       FEEvaluation & fe_eval_linearization_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM = fe_eval_linearization_m.get_value(q);
        vector uP = fe_eval_linearization_p.get_value(q);

        vector delta_uM = fe_eval_m.get_value(q);
        vector delta_uP = fe_eval_p.get_value(q);

        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        fe_eval_m.submit_value(flux, q);
        fe_eval_p.submit_value(-flux, q); // minus sign since n⁺ = -n⁻
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM = fe_eval_linearization_m.get_value(q);
        vector uP = fe_eval_linearization_p.get_value(q);

        vector delta_uM = fe_eval_m.get_value(q);
        vector delta_uP = fe_eval_p.get_value(q);

        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux_times_normal =
          calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fe_eval_m.submit_value(flux_times_normal - average_normal_velocity * delta_uM -
                                 delta_average_normal_velocity * uM,
                               q);
        // opposite signs since n⁺ = - n⁻
        fe_eval_p.submit_value(-flux_times_normal + average_normal_velocity * delta_uP +
                                 delta_average_normal_velocity * uP,
                               q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename FEEvaluation>
  void
  do_face_int_integral_linearized_operator(FEEvaluation & fe_eval_m,
                                           FEEvaluation & /* fe_eval_p */,
                                           FEEvaluation & fe_eval_linearization_m,
                                           FEEvaluation & fe_eval_linearization_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM = fe_eval_linearization_m.get_value(q);
        vector uP = fe_eval_linearization_p.get_value(q);

        vector delta_uM = fe_eval_m.get_value(q);
        vector delta_uP; // set exterior value to zero

        vector normal_m = fe_eval_m.get_normal_vector(q);

        vector flux =
          calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal_m);

        fe_eval_m.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
      {
        vector uM = fe_eval_linearization_m.get_value(q);
        vector uP = fe_eval_linearization_p.get_value(q);

        vector delta_uM = fe_eval_m.get_value(q);
        vector delta_uP; // set exterior value to zero

        vector normal = fe_eval_m.get_normal_vector(q);

        vector flux_times_normal =
          calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fe_eval_m.submit_value(flux_times_normal - average_normal_velocity * delta_uM -
                                 delta_average_normal_velocity * uM,
                               q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename FEEvaluation>
  void
  do_face_ext_integral_linearized_operator(FEEvaluation & /* fe_eval_m */,
                                           FEEvaluation & fe_eval_p,
                                           FEEvaluation & fe_eval_linearization_m,
                                           FEEvaluation & fe_eval_linearization_p) const
  {
    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_p.n_q_points; ++q)
      {
        vector uM = fe_eval_linearization_m.get_value(q);
        vector uP = fe_eval_linearization_p.get_value(q);

        vector delta_uM; // set exterior value to zero
        vector delta_uP = fe_eval_p.get_value(q);

        vector normal_p = -fe_eval_p.get_normal_vector(q);

        vector flux =
          calculate_lax_friedrichs_flux_linearized(uP, uM, delta_uP, delta_uM, normal_p);

        fe_eval_p.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < fe_eval_p.n_q_points; ++q)
      {
        vector uM = fe_eval_linearization_m.get_value(q);
        vector uP = fe_eval_linearization_p.get_value(q);

        vector delta_uM; // set exterior value to zero
        vector delta_uP = fe_eval_p.get_value(q);

        vector normal_p = -fe_eval_p.get_normal_vector(q);

        vector flux_times_normal =
          calculate_upwind_flux_linearized(uP, uM, delta_uP, delta_uM, normal_p);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal_p;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal_p;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        // opposite signs since n⁺ = - n⁻
        fe_eval_p.submit_value(flux_times_normal - average_normal_velocity * delta_uP -
                                 delta_average_normal_velocity * uP,
                               q);
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  template<typename FEEvaluation>
  void
  do_boundary_integral_linearized_operator(FEEvaluation &             fe_eval,
                                           FEEvaluation &             fe_eval_linearization,
                                           types::boundary_id const & boundary_id) const
  {
    BoundaryTypeU boundary_type = operator_data.bc->get_boundary_type(boundary_id);

    if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
    {
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector uM = fe_eval_linearization.get_value(q);
        vector uP =
          calculate_exterior_value(uM, q, fe_eval_linearization, boundary_type, boundary_id);

        vector delta_uM = fe_eval.get_value(q);
        vector delta_uP = calculate_exterior_value_linearized(delta_uM, q, fe_eval, boundary_type);

        vector normal = fe_eval.get_normal_vector(q);

        vector flux = calculate_lax_friedrichs_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux, uM * normal);

        fe_eval.submit_value(flux, q);
      }
    }
    else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
    {
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        vector uM = fe_eval_linearization.get_value(q);
        vector uP =
          calculate_exterior_value(uM, q, fe_eval_linearization, boundary_type, boundary_id);

        vector delta_uM = fe_eval.get_value(q);
        vector delta_uP = calculate_exterior_value_linearized(delta_uM, q, fe_eval, boundary_type);

        vector normal = fe_eval.get_normal_vector(q);

        vector flux_times_normal =
          calculate_upwind_flux_linearized(uM, uP, delta_uM, delta_uP, normal);

        if(boundary_type == BoundaryTypeU::Neumann && operator_data.use_outflow_bc == true)
          apply_outflow_bc(flux_times_normal, uM * normal);

        scalar average_normal_velocity       = 0.5 * (uM + uP) * normal;
        scalar delta_average_normal_velocity = 0.5 * (delta_uM + delta_uP) * normal;

        // second term appears since the strong formulation is implemented (integration by parts
        // is performed twice)
        fe_eval.submit_value(flux_times_normal - average_normal_velocity * delta_uM -
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
  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_value(vector const &           uM,
                             unsigned int const       q,
                             FEEvaluation &           fe_eval,
                             BoundaryTypeU const &    boundary_type,
                             types::boundary_id const boundary_id) const
  {
    vector uP;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it;
      it                          = operator_data.bc->dirichlet_bc.find(boundary_id);
      Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

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
      vector normalM = fe_eval.get_normal_vector(q);

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
  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_value_linearized(vector &              delta_uM,
                                        unsigned int const    q,
                                        FEEvaluation &        fe_eval,
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
      vector normalM = fe_eval.get_normal_vector(q);
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
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral_nonlinear_operator(fe_eval);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_nonlinear_operator(MatrixFree<dim, Number> const & data,
                               VectorType &                    dst,
                               VectorType const &              src,
                               Range const &                   face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);
    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval.gather_evaluate(src, true, false);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      do_face_integral_nonlinear_operator(fe_eval, fe_eval_neighbor);

      fe_eval.integrate_scatter(true, false, dst);
      fe_eval_neighbor.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_nonlinear_operator(MatrixFree<dim, Number> const & data,
                                        VectorType &                    dst,
                                        VectorType const &              src,
                                        Range const &                   face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);

      fe_eval.gather_evaluate(src, true, false);

      do_boundary_integral_nonlinear_operator(fe_eval, data.get_boundary_id(face));

      fe_eval.integrate_scatter(true, false, dst);
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
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    FEEvalCellOverint fe_eval_transport(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      fe_eval_transport.reinit(cell);
      fe_eval_transport.gather_evaluate(velocity_linearization, true, false, false);

      do_cell_integral_linear_transport(fe_eval, fe_eval_transport);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_linear_transport(MatrixFree<dim, Number> const & data,
                             VectorType &                    dst,
                             VectorType const &              src,
                             Range const &                   face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    FEEvalFaceOverint fe_eval_transport(data,
                                        true,
                                        operator_data.dof_index,
                                        operator_data.quad_index);

    FEEvalFaceOverint fe_eval_transport_neighbor(data,
                                                 false,
                                                 operator_data.dof_index,
                                                 operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval.gather_evaluate(src, true, false);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      fe_eval_transport.reinit(face);
      fe_eval_transport_neighbor.reinit(face);

      fe_eval_transport.gather_evaluate(velocity_linearization, true, false);
      fe_eval_transport_neighbor.gather_evaluate(velocity_linearization, true, false);

      do_face_integral_linear_transport(fe_eval,
                                        fe_eval_neighbor,
                                        fe_eval_transport,
                                        fe_eval_transport_neighbor);

      fe_eval.integrate_scatter(true, false, dst);
      fe_eval_neighbor.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_linear_transport(MatrixFree<dim, Number> const & data,
                                      VectorType &                    dst,
                                      VectorType const &              src,
                                      Range const &                   face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_transport(data,
                                        true,
                                        operator_data.dof_index,
                                        operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval.gather_evaluate(src, true, false);

      fe_eval_transport.reinit(face);
      fe_eval_transport.gather_evaluate(velocity_linearization, true, false);

      do_boundary_integral_linear_transport(fe_eval, fe_eval_transport, data.get_boundary_id(face));

      fe_eval.integrate_scatter(true, false, dst);
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
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    FEEvalCellOverint fe_eval_linearization(data,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_linearization.reinit(cell);

      if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
        fe_eval_linearization.gather_evaluate(velocity_linearization, true, false, false);
      else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
        fe_eval_linearization.gather_evaluate(velocity_linearization, true, true, false);
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral_linearized_operator(fe_eval, fe_eval_linearization);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_linearized_operator(MatrixFree<dim, Number> const & data,
                                VectorType &                    dst,
                                VectorType const &              src,
                                Range const &                   face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization_neighbor(data,
                                                     false,
                                                     operator_data.dof_index,
                                                     operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval_linearization.reinit(face);
      fe_eval_linearization_neighbor.reinit(face);

      fe_eval.gather_evaluate(src, true, false);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      do_face_integral_linearized_operator(fe_eval,
                                           fe_eval_neighbor,
                                           fe_eval_linearization,
                                           fe_eval_linearization_neighbor);

      fe_eval.integrate_scatter(true, false, dst);
      fe_eval_neighbor.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_linearized_operator(MatrixFree<dim, Number> const & data,
                                         VectorType &                    dst,
                                         VectorType const &              src,
                                         Range const &                   face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_linearization.reinit(face);

      fe_eval.gather_evaluate(src, true, false);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      do_boundary_integral_linearized_operator(fe_eval,
                                               fe_eval_linearization,
                                               data.get_boundary_id(face));

      fe_eval.integrate_scatter(true, false, dst);
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
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    FEEvalCellOverint fe_eval_linearization(data,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_linearization.reinit(cell);

      if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
        fe_eval_linearization.gather_evaluate(velocity_linearization, true, false, false);
      else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
        fe_eval_linearization.gather_evaluate(velocity_linearization, true, true, false);
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      fe_eval.reinit(cell);

      scalar       local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        do_cell_integral_linearized_operator(fe_eval, fe_eval_linearization);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  face_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization_neighbor(data,
                                                     false,
                                                     operator_data.dof_index,
                                                     operator_data.quad_index);

    // Perform face integrals for element e⁻
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval.reinit(face);

      scalar       local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false);

        do_face_int_integral_linearized_operator(fe_eval,
                                                 fe_eval_neighbor,
                                                 fe_eval_linearization,
                                                 fe_eval_linearization_neighbor);

        fe_eval.integrate(true, false);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }

    // Perform face integrals for element e⁺
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval_neighbor.reinit(face);

      scalar       local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
      unsigned int dofs_per_cell = fe_eval_neighbor.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval_neighbor.evaluate(true, false);

        do_face_ext_integral_linearized_operator(fe_eval,
                                                 fe_eval_neighbor,
                                                 fe_eval_linearization,
                                                 fe_eval_linearization_neighbor);

        fe_eval_neighbor.integrate(true, false);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void
  boundary_face_loop_diagonal(MatrixFree<dim, Number> const & data,
                              VectorType &                    dst,
                              VectorType const &,
                              Range const & face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);

      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      scalar       local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false);

        do_boundary_integral_linearized_operator(fe_eval,
                                                 fe_eval_linearization,
                                                 data.get_boundary_id(face));

        fe_eval.integrate(true, false);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
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
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization_neighbor(data,
                                                     false,
                                                     operator_data.dof_index,
                                                     operator_data.quad_index);

    // Perform face integral for element e⁻
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval.reinit(face);
      fe_eval.gather_evaluate(src, true, false);

      do_face_int_integral_linearized_operator(fe_eval,
                                               fe_eval_neighbor,
                                               fe_eval_linearization,
                                               fe_eval_linearization_neighbor);

      fe_eval.integrate_scatter(true, false, dst);
    }

    // Perform face integral for element e⁺
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval_neighbor.reinit(face);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      do_face_ext_integral_linearized_operator(fe_eval,
                                               fe_eval_neighbor,
                                               fe_eval_linearization,
                                               fe_eval_linearization_neighbor);

      fe_eval_neighbor.integrate_scatter(true, false, dst);
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
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    FEEvalCellOverint fe_eval_linearization(data,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval_linearization.reinit(cell);

      if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
        fe_eval_linearization.gather_evaluate(velocity_linearization, true, false, false);
      else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
        fe_eval_linearization.gather_evaluate(velocity_linearization, true, true, false);
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        do_cell_integral_linearized_operator(fe_eval, fe_eval_linearization);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  void
  face_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & face_range) const
  {
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_neighbor(data,
                                       false,
                                       operator_data.dof_index,
                                       operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization_neighbor(data,
                                                     false,
                                                     operator_data.dof_index,
                                                     operator_data.quad_index);

    // Perform face integrals for element e⁻.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval.reinit(face);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false);

        do_face_int_integral_linearized_operator(fe_eval,
                                                 fe_eval_neighbor,
                                                 fe_eval_linearization,
                                                 fe_eval_linearization_neighbor);

        fe_eval.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }

    // Perform face integrals for element e⁺.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      fe_eval_linearization_neighbor.reinit(face);
      fe_eval_linearization_neighbor.gather_evaluate(velocity_linearization, true, false);

      fe_eval_neighbor.reinit(face);

      unsigned int dofs_per_cell = fe_eval_neighbor.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval_neighbor.evaluate(true, false);

        do_face_ext_integral_linearized_operator(fe_eval,
                                                 fe_eval_neighbor,
                                                 fe_eval_linearization,
                                                 fe_eval_linearization_neighbor);

        fe_eval_neighbor.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_exterior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += fe_eval_neighbor.begin_dof_values()[i][v];
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
    FEEvalFaceOverint fe_eval(data, true, operator_data.dof_index, operator_data.quad_index);

    FEEvalFaceOverint fe_eval_linearization(data,
                                            true,
                                            operator_data.dof_index,
                                            operator_data.quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);

      fe_eval_linearization.reinit(face);
      fe_eval_linearization.gather_evaluate(velocity_linearization, true, false);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false);

        do_boundary_integral_linearized_operator(fe_eval,
                                                 fe_eval_linearization,
                                                 data.get_boundary_id(face));

        fe_eval.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if(cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i = 0; i < dofs_per_cell; ++i)
              matrices[cell_number](i, j) += fe_eval.begin_dof_values()[i][v];
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
    FEEvalCellOverint fe_eval(data, operator_data.dof_index, operator_data.quad_index);
    FEEvalFaceOverint fe_eval_m(data, true, operator_data.dof_index, operator_data.quad_index);
    FEEvalFaceOverint fe_eval_p(data, false, operator_data.dof_index, operator_data.quad_index);

    FEEvalCellOverint fe_eval_linearization(data,
                                            operator_data.dof_index,
                                            operator_data.quad_index);
    FEEvalFaceOverint fe_eval_linearization_m(data,
                                              true,
                                              operator_data.dof_index,
                                              operator_data.quad_index);
    FEEvalFaceOverint fe_eval_linearization_p(data,
                                              false,
                                              operator_data.dof_index,
                                              operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      // cell integral
      unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);

      fe_eval_linearization.reinit(cell);

      if(operator_data.formulation == FormulationConvectiveTerm::DivergenceFormulation)
        fe_eval_linearization.gather_evaluate(velocity_linearization, true, false, false);
      else if(operator_data.formulation == FormulationConvectiveTerm::ConvectiveFormulation)
        fe_eval_linearization.gather_evaluate(velocity_linearization, true, true, false);
      else
        AssertThrow(false, ExcMessage("Not implemented."));

      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        do_cell_integral_linearized_operator(fe_eval, fe_eval_linearization);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }

      // loop over all faces
      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        fe_eval_linearization_m.reinit(cell, face);
        fe_eval_linearization_m.gather_evaluate(velocity_linearization, true, false);

        // TODO
        //        AssertThrow(false, ExcMessage("We have to evaluate the linearized velocity field
        //        on the neighbor. This functionality is not implemented in deal.II/matrix_free."));
        //        fe_eval_linearization_p.reinit(cell, face);
        //        fe_eval_linearization_p.gather_evaluate(velocity_linearization, true, false);

        fe_eval_m.reinit(cell, face);
        fe_eval_p.reinit(cell, face);

        auto bids        = data.get_faces_by_cells_boundary_id(cell, face);
        auto boundary_id = bids[0];

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            fe_eval_m.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
          fe_eval_m.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

          fe_eval_m.evaluate(true, false);

          if(boundary_id == numbers::internal_face_boundary_id) // internal face
          {
            // TODO
            //            do_face_int_integral_linearized_operator(fe_eval_m,
            //            fe_eval_p,fe_eval_linearization_m, fe_eval_linearization_p);

            // plug in fe_eval_linearization_m twice to avoid the above problem with accessing dofs
            // of the neighboring element
            do_face_int_integral_linearized_operator(fe_eval_m,
                                                     fe_eval_p,
                                                     fe_eval_linearization_m,
                                                     fe_eval_linearization_m);
          }
          else // boundary face
          {
            do_boundary_integral_linearized_operator(fe_eval_m,
                                                     fe_eval_linearization_m,
                                                     boundary_id);
          }

          fe_eval_m.integrate(true, false);

          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
                fe_eval_m.begin_dof_values()[i][v];
        }
      }
    }
  }

  MatrixFree<dim, Number> const * data;

  ConvectiveOperatorData<dim> operator_data;

  mutable Number eval_time;

  mutable VectorType velocity_linearization;

  unsigned int n_mpi_processes;

  std::shared_ptr<FEEvalCellOverint> fe_eval;
  std::shared_ptr<FEEvalFaceOverint> fe_eval_m;
  std::shared_ptr<FEEvalFaceOverint> fe_eval_p;

  std::shared_ptr<FEEvalCellOverint> fe_eval_linearization;
  std::shared_ptr<FEEvalFaceOverint> fe_eval_linearization_m;
  std::shared_ptr<FEEvalFaceOverint> fe_eval_linearization_p;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_CONVECTIVE_OPERATOR_H_ \
        */
