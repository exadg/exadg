/*
 * projection_operators.h
 *
 *  Created on: Jun 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATORS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATORS_H_

#include "../../solvers_and_preconditioners/util/block_jacobi_matrices.h"
#include "../../solvers_and_preconditioners/util/invert_diagonal.h"
#include "../../solvers_and_preconditioners/util/verify_calculation_of_diagonal.h"
#include "operators/base_operator.h"

namespace IncNS
{
/*
 *  Projection operator (mass matrix + divergence penalty term) where the integrals are computed
 * elementwise. This wrapper is needed to define the interface to the elementwise iterative solver.
 *
 *  Weak form:
 *
 *   (v_h, u_h)_Omega^e + delta_t * (div(v_h), tau_div * div(u_h))_Omega^e.
 *
 */
template<int dim, int fe_degree, typename value_type, typename Operator>
class ElementwiseProjectionOperatorDivergencePenalty
{
public:
  ElementwiseProjectionOperatorDivergencePenalty(Operator const & operator_in)
    : op(operator_in),
      fe_eval(1,
              FEEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type>(op.get_data(),
                                                                           op.get_dof_index(),
                                                                           op.get_quad_index()))
  {
  }

  MatrixFree<dim, value_type> const &
  get_data() const
  {
    return op.get_data();
  }

  unsigned int
  get_dof_index() const
  {
    return op.get_dof_index();
  }

  unsigned int
  get_quad_index() const
  {
    return op.get_dof_index();
  }

  void
  setup(const unsigned int cell)
  {
    fe_eval[0].reinit(cell);
  }

  unsigned int
  get_problem_size() const
  {
    return fe_eval[0].dofs_per_cell;
  }

  void
  vmult(VectorizedArray<value_type> * dst, VectorizedArray<value_type> * src) const
  {
    Elementwise::vector_init(dst, fe_eval[0].dofs_per_cell);
    op.apply_add_block_diagonal_elementwise_cell(fe_eval[0], dst, src);
  }

private:
  Operator const & op;

  mutable AlignedVector<FEEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type>> fe_eval;
};


/*
 * Projection operator (mass matrix + divergence penalty term + continuity penalty term) where the
 * integrals are computed elementwise. This wrapper is needed to define the interface to the
 * elementwise iterative solver.
 *
 *  Weak form:
 *
 *   (v_h, u_h)_Omega^e + delta_t * (div(v_h), tau_div * div(u_h))_Omega^e
 *                      + delta_t * (jump(v_h), tau_conti * jump(u_h))_Omega^e.
 *
 */
template<int dim, int fe_degree, typename value_type, typename Operator>
class ElementwiseProjectionOperator
{
public:
  ElementwiseProjectionOperator(Operator const & operator_in)
    : op(operator_in),
      current_cell(1),
      fe_eval(1,
              FEEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type>(op.get_data(),
                                                                           op.get_dof_index(),
                                                                           op.get_quad_index())),
      fe_eval_m(
        FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type>(op.get_data(),
                                                                         true,
                                                                         op.get_dof_index(),
                                                                         op.get_quad_index())),
      fe_eval_p(
        FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type>(op.get_data(),
                                                                         false,
                                                                         op.get_dof_index(),
                                                                         op.get_quad_index()))
  {
  }

  MatrixFree<dim, value_type> const &
  get_data() const
  {
    return op.get_data();
  }

  unsigned int
  get_dof_index() const
  {
    return op.get_dof_index();
  }

  unsigned int
  get_quad_index() const
  {
    return op.get_dof_index();
  }

  void
  setup(const unsigned int cell)
  {
    fe_eval[0].reinit(cell);

    current_cell = cell;
  }

  unsigned int
  get_problem_size() const
  {
    return fe_eval[0].dofs_per_cell;
  }

  void
  vmult(VectorizedArray<value_type> * dst, VectorizedArray<value_type> * src) const
  {
    Elementwise::vector_init(dst, fe_eval[0].dofs_per_cell);
    op.apply_add_block_diagonal_elementwise(
      current_cell, fe_eval[0], fe_eval_m, fe_eval_p, dst, src);
  }

private:
  Operator const & op;

  unsigned int current_cell;

  mutable AlignedVector<FEEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type>> fe_eval;
  mutable FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type>            fe_eval_m;
  mutable FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type>            fe_eval_p;
};



/*
 *  Combined divergence and continuity penalty operator: applies the operation
 *
 *   mass matrix operator + dt * divergence penalty operator + dt * continuity penalty operator .
 *
 *  The divergence and continuity penalty operators can also be applied separately. In detail
 *
 *  Mass matrix operator: ( v_h , u_h )_Omega^e where
 *   v_h : test function
 *   u_h : solution
 *
 *
 *  Divergence penalty operator: ( div(v_h) , tau_div * div(u_h) )_Omega^e where
 *   v_h : test function
 *   u_h : solution
 *   tau_div: divergence penalty factor
 *
 *            use convective term:  tau_div_conv = K * ||U||_mean * h_eff
 *
 *                                  where h_eff = h / (k_u+1) and
 *                                  h = V_e^{1/3} with the element volume V_e
 *
 *            use viscous term:     tau_div_viscous = K * nu
 *
 *            use both terms:       tau_div = tau_div_conv + tau_div_viscous
 *
 *
 *  Continuity penalty operator: ( v_h , tau_conti * jump(u_h) )_dOmega^e where
 *   v_h : test function
 *   u_h : solution
 *
 *   jump(u_h) = u_h^{-} - u_h^{+} or ( (u_h^{-} - u_h^{+})*normal ) * normal
 *
 *     where "-" denotes interior information and "+" exterior information
 *
 *   tau_conti: continuity penalty factor
 *
 *            use convective term:  tau_conti_conv = K * ||U||_mean
 *
 *            use viscous term:     tau_conti_viscous = K * nu / h
 *
 *                                  where h_eff = h / (k_u+1) and
 *                                  h = V_e^{1/3} with the element volume V_e
 *
 *            use both terms:       tau_conti = tau_conti_conv + tau_conti_viscous
 */

/*
 *  Operator data.
 */
struct CombinedDivergenceContinuityPenaltyOperatorData
{
  CombinedDivergenceContinuityPenaltyOperatorData()
    : type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
      viscosity(0.0),
      penalty_factor_div(1.0),
      penalty_factor_conti(1.0),
      which_components(ContinuityPenaltyComponents::Normal),
      implement_block_diagonal_preconditioner_matrix_free(false),
      use_cell_based_loops(false),
      preconditioner_block_jacobi(PreconditionerBlockDiagonal::InverseMassMatrix),
      block_jacobi_solver_data(SolverData(100, 1.e-12, 1.e-1))
  {
  }

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // kinematic viscosity
  double viscosity;

  // scaling factor
  double penalty_factor_div, penalty_factor_conti;

  // the continuity penalty term can be applied to all velocity components or to the normal
  // component only
  ContinuityPenaltyComponents which_components;

  // block diagonal preconditioner
  bool implement_block_diagonal_preconditioner_matrix_free;

  // use cell based loops
  bool use_cell_based_loops;

  // elementwise iterative solution of block Jacobi problems
  PreconditionerBlockDiagonal preconditioner_block_jacobi;
  SolverData                  block_jacobi_solver_data;
};

template<int dim,
         int fe_degree,
         int fe_degree_p,
         int fe_degree_xwall,
         int xwall_quad_rule,
         typename Number>
class CombinedDivergenceContinuityPenaltyOperator : public BaseOperator<dim>
{
public:
  typedef Number value_type;

  static const bool is_xwall = (xwall_quad_rule > 1) ? true : false;

  static const unsigned int n_actual_q_points_vel_linear =
    (is_xwall) ? xwall_quad_rule : fe_degree + 1;

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  typedef FEEvaluationWrapper<dim,
                              fe_degree,
                              fe_degree_xwall,
                              n_actual_q_points_vel_linear,
                              dim,
                              value_type,
                              is_xwall>
    FEEval_Velocity_Velocity_linear;

  typedef FEFaceEvaluationWrapper<dim,
                                  fe_degree,
                                  fe_degree_xwall,
                                  n_actual_q_points_vel_linear,
                                  dim,
                                  value_type,
                                  is_xwall>
    FEFaceEval_Velocity_Velocity_linear;

  typedef CombinedDivergenceContinuityPenaltyOperator<dim,
                                                      fe_degree,
                                                      fe_degree_p,
                                                      fe_degree_xwall,
                                                      xwall_quad_rule,
                                                      value_type>
    This;

  CombinedDivergenceContinuityPenaltyOperator(
    MatrixFree<dim, value_type> const &                   data_in,
    unsigned int const                                    dof_index_in,
    unsigned int const                                    quad_index_in,
    CombinedDivergenceContinuityPenaltyOperatorData const operator_data_in)
    : data(data_in),
      dof_index(dof_index_in),
      quad_index(quad_index_in),
      array_conti_penalty_parameter(0),
      array_div_penalty_parameter(0),
      time_step_size(1.0),
      operator_data(operator_data_in),
      scaling_factor_div(1.0),
      scaling_factor_conti(1.0),
      n_mpi_processes(Utilities::MPI::n_mpi_processes(MPI_COMM_WORLD)),
      block_diagonal_preconditioner_is_initialized(false)
  {
    array_conti_penalty_parameter.resize(data.n_macro_cells() + data.n_macro_ghost_cells());
    array_div_penalty_parameter.resize(data.n_macro_cells() + data.n_macro_ghost_cells());
  }

  MatrixFree<dim, value_type> const &
  get_data() const
  {
    return data;
  }

  AlignedVector<VectorizedArray<value_type>> const &
  get_array_div_penalty_parameter() const
  {
    return array_div_penalty_parameter;
  }

  FEParameters<dim> const *
  get_fe_param() const
  {
    return this->fe_param;
  }

  unsigned int
  get_dof_index() const
  {
    return dof_index;
  }

  unsigned int
  get_quad_index() const
  {
    return quad_index;
  }

  /*
   *  Set the time step size.
   */
  void
  set_time_step_size(double const & delta_t)
  {
    time_step_size = delta_t;
  }

  /*
   *  Get the time step size.
   */
  double
  get_time_step_size() const
  {
    return time_step_size;
  }

  void
  calculate_array_penalty_parameter(VectorType const & velocity)
  {
    calculate_array_div_penalty_parameter(velocity);
    calculate_array_conti_penalty_parameter(velocity);
  }

  void
  calculate_array_div_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    FEEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, dof_index);

    AlignedVector<VectorizedArray<value_type>> JxW_values(fe_eval.n_q_points);

    for(unsigned int cell = 0; cell < data.n_macro_cells() + data.n_macro_ghost_cells(); ++cell)
    {
      VectorizedArray<value_type> tau_convective = make_vectorized_array<value_type>(0.0);
      VectorizedArray<value_type> tau_viscous =
        make_vectorized_array<value_type>(operator_data.viscosity);

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm ||
         operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(velocity);
        fe_eval.evaluate(true, false);
        VectorizedArray<value_type> volume      = make_vectorized_array<value_type>(0.0);
        VectorizedArray<value_type> norm_U_mean = make_vectorized_array<value_type>(0.0);
        JxW_values.resize(fe_eval.n_q_points);
        fe_eval.fill_JxW_values(JxW_values);
        for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
        {
          volume += JxW_values[q];
          norm_U_mean += JxW_values[q] * fe_eval.get_value(q).norm();
        }
        norm_U_mean /= volume;

        tau_convective =
          norm_U_mean * std::exp(std::log(volume) / (double)dim) / (double)(fe_degree + 1);
      }

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_div_penalty_parameter[cell] = operator_data.penalty_factor_div * tau_convective;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_div_penalty_parameter[cell] = operator_data.penalty_factor_div * tau_viscous;
      }
      else if(operator_data.type_penalty_parameter ==
              TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_div_penalty_parameter[cell] =
          operator_data.penalty_factor_div * (tau_convective + tau_viscous);
      }
    }
  }

  void
  calculate_array_conti_penalty_parameter(VectorType const & velocity)
  {
    velocity.update_ghost_values();

    FEEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, dof_index);

    AlignedVector<VectorizedArray<value_type>> JxW_values(fe_eval.n_q_points);

    for(unsigned int cell = 0; cell < data.n_macro_cells() + data.n_macro_ghost_cells(); ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(velocity);
      fe_eval.evaluate(true, false);
      VectorizedArray<value_type> volume      = make_vectorized_array<value_type>(0.0);
      VectorizedArray<value_type> norm_U_mean = make_vectorized_array<value_type>(0.0);
      JxW_values.resize(fe_eval.n_q_points);
      fe_eval.fill_JxW_values(JxW_values);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        volume += JxW_values[q];
        norm_U_mean += JxW_values[q] * fe_eval.get_value(q).norm();
      }
      norm_U_mean /= volume;

      VectorizedArray<value_type> tau_convective = norm_U_mean;
      VectorizedArray<value_type> h =
        std::exp(std::log(volume) / (double)dim) / (double)(fe_degree + 1);
      VectorizedArray<value_type> tau_viscous =
        make_vectorized_array<value_type>(operator_data.viscosity) / h;

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_conti_penalty_parameter[cell] = operator_data.penalty_factor_conti * tau_convective;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_conti_penalty_parameter[cell] = operator_data.penalty_factor_conti * tau_viscous;
      }
      else if(operator_data.type_penalty_parameter ==
              TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_conti_penalty_parameter[cell] =
          operator_data.penalty_factor_conti * (tau_convective + tau_viscous);
      }
    }
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    apply(dst, src);
  }

  void
  apply(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div   = this->time_step_size;
    scaling_factor_conti = this->time_step_size;

    data.loop(&This::cell_loop,
              &This::face_loop,
              &This::boundary_face_loop_empty,
              this,
              dst,
              src,
              /*zero dst vector = */ true,
              MatrixFree<dim, value_type>::only_values,
              MatrixFree<dim, value_type>::only_values);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div   = this->time_step_size;
    scaling_factor_conti = this->time_step_size;

    data.loop(&This::cell_loop,
              &This::face_loop,
              &This::boundary_face_loop_empty,
              this,
              dst,
              src,
              /*zero dst vector = */ false,
              MatrixFree<dim, value_type>::only_values,
              MatrixFree<dim, value_type>::only_values);
  }

  void
  apply_div_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div = 1.0;

    data.cell_loop(&This::cell_loop_div_penalty,
                   this,
                   dst,
                   src,
                   /*zero dst vector = */ true);
  }

  void
  apply_add_div_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_div = 1.0;

    data.cell_loop(&This::cell_loop_div_penalty,
                   this,
                   dst,
                   src,
                   /*zero dst vector = */ false);
  }

  void
  apply_conti_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_conti = 1.0;

    data.loop(&This::cell_loop_empty,
              &This::face_loop,
              &This::boundary_face_loop_empty,
              this,
              dst,
              src,
              /*zero dst vector = */ true,
              MatrixFree<dim, value_type>::only_values,
              MatrixFree<dim, value_type>::only_values);
  }

  void
  apply_add_conti_penalty(VectorType & dst, VectorType const & src) const
  {
    scaling_factor_conti = 1.0;

    data.loop(&This::cell_loop_empty,
              &This::face_loop,
              &This::boundary_face_loop_empty,
              this,
              dst,
              src,
              /*zero dst vector = */ false,
              MatrixFree<dim, value_type>::only_values,
              MatrixFree<dim, value_type>::only_values);
  }

  /*
   *  Calculate inverse diagonal which is needed for the Jacobi preconditioner.
   */
  void
  calculate_inverse_diagonal(VectorType & diagonal) const
  {
    calculate_diagonal(diagonal);

    // verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  /*
   *  Initialize dof vector (required when using the Jacobi preconditioner).
   */
  void
  initialize_dof_vector(VectorType & vector) const
  {
    data.initialize_dof_vector(vector, dof_index);
  }

  /*
   * Block diagonal preconditioner.
   */

  // apply the inverse block diagonal operator (for matrix-based and matrix-free variants)
  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const
  {
    // matrix-free
    if(this->operator_data.implement_block_diagonal_preconditioner_matrix_free)
    {
      // Solve block Jacobi problems iteratively using an elementwise solver vectorized
      // over several elements.
      elementwise_solver->solve(dst, src);
    }
    else // matrix based
    {
      // Simply apply inverse of block matrices (using the LU factorization that has been computed
      // before).
      data.cell_loop(&This::cell_loop_apply_inverse_block_diagonal, this, dst, src);
    }
  }

  /*
   * Update block diagonal preconditioner: initialize everything related to block diagonal
   * preconditioner when this function is called the first time. Recompute block matrices in case of
   * matrix-based implementation.
   */
  void
  update_block_diagonal_preconditioner() const
  {
    // initialization

    if(!block_diagonal_preconditioner_is_initialized)
    {
      if(operator_data.implement_block_diagonal_preconditioner_matrix_free)
      {
        initialize_block_diagonal_preconditioner_matrix_free();
      }
      else // matrix-based variant
      {
        // Note that the velocity has dim components.
        unsigned int dofs_per_cell = data.get_shape_info().dofs_per_component_on_cell * dim;

        matrices.resize(data.n_macro_cells() * VectorizedArray<value_type>::n_array_elements,
                        LAPACKFullMatrix<value_type>(dofs_per_cell, dofs_per_cell));
      }

      block_diagonal_preconditioner_is_initialized = true;
    }

    // update

    // For the matrix-free variant there is nothing to do.
    // For the matrix-based variant we have to recompute the block matrices.
    if(!operator_data.implement_block_diagonal_preconditioner_matrix_free)
    {
      // clear matrices
      initialize_block_jacobi_matrices_with_zero(matrices);

      // compute block matrices and add
      this->add_block_diagonal_matrices(matrices);

      calculate_lu_factorization_block_jacobi(matrices);
    }
  }

  typedef FEEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type>     FEEvalCell;
  typedef FEFaceEvaluation<dim, fe_degree, fe_degree + 1, dim, value_type> FEEvalFace;

  void
  apply_add_block_diagonal_elementwise_cell(FEEvalCell &                          fe_eval,
                                            VectorizedArray<Number> * const       dst,
                                            VectorizedArray<Number> const * const src) const
  {
    unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      fe_eval.begin_dof_values()[i] = src[i];

    fe_eval.evaluate(true, true, false);

    do_cell_integral(fe_eval);

    fe_eval.integrate(true, true);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dst[i] += fe_eval.begin_dof_values()[i];
  }

  void
  apply_add_block_diagonal_elementwise(unsigned int const                    cell,
                                       FEEvalCell &                          fe_eval,
                                       FEEvalFace &                          fe_eval_m,
                                       FEEvalFace &                          fe_eval_p,
                                       VectorizedArray<Number> * const       dst,
                                       VectorizedArray<Number> const * const src) const
  {
    unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      fe_eval.begin_dof_values()[i] = src[i];

    fe_eval.evaluate(true, true, false);

    do_cell_integral(fe_eval);

    fe_eval.integrate(true, true);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dst[i] += fe_eval.begin_dof_values()[i];

    // face integrals
    unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
    for(unsigned int face = 0; face < n_faces; ++face)
    {
      fe_eval_m.reinit(cell, face);
      fe_eval_p.reinit(cell, face);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        fe_eval_m.begin_dof_values()[i] = src[i];

      // do not need to read dof values for fe_eval_p (already initialized with 0)

      fe_eval_m.evaluate(true, false);

      auto bids = data.get_faces_by_cells_boundary_id(cell, face);
      auto bid  = bids[0];

      if(bid == numbers::internal_face_boundary_id) // internal face
      {
        do_face_int_integral(fe_eval_m, fe_eval_p);
      }
      else // boundary face
      {
        // use same fe_eval so that the result becomes zero (only jumps involved)
        do_face_int_integral(fe_eval_m, fe_eval_m);
      }

      fe_eval_m.integrate(true, false);

      for(unsigned int i = 0; i < dofs_per_cell; ++i)
        dst[i] += fe_eval_m.begin_dof_values()[i];
    }
  }

private:
  template<typename FEEval>
  void
  do_cell_integral(FEEval & fe_eval) const
  {
    VectorizedArray<value_type> tau =
      fe_eval.read_cell_data(array_div_penalty_parameter) * scaling_factor_div;

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_value(fe_eval.get_value(q), q);
      fe_eval.submit_divergence(tau * fe_eval.get_divergence(q), q);
    }
  }

  template<typename FEEval>
  void
  do_cell_integral_div_penalty(FEEval & fe_eval) const
  {
    VectorizedArray<value_type> tau =
      fe_eval.read_cell_data(array_div_penalty_parameter) * scaling_factor_div;

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_divergence(tau * fe_eval.get_divergence(q), q);
    }
  }

  template<typename FEFaceEval>
  void
  do_face_integral(FEFaceEval & fe_eval, FEFaceEval & fe_eval_neighbor) const
  {
    VectorizedArray<value_type> tau =
      0.5 *
      (fe_eval.read_cell_data(array_conti_penalty_parameter) +
       fe_eval_neighbor.read_cell_data(array_conti_penalty_parameter)) *
      scaling_factor_conti;

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      Tensor<1, dim, VectorizedArray<value_type>> uM         = fe_eval.get_value(q);
      Tensor<1, dim, VectorizedArray<value_type>> uP         = fe_eval_neighbor.get_value(q);
      Tensor<1, dim, VectorizedArray<value_type>> jump_value = uM - uP;

      if(operator_data.which_components == ContinuityPenaltyComponents::All)
      {
        // penalize all velocity components
        fe_eval.submit_value(tau * jump_value, q);
        fe_eval_neighbor.submit_value(-tau * jump_value, q);
      }
      else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
      {
        // penalize normal components only
        Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval.get_normal_vector(q);

        fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
        fe_eval_neighbor.submit_value(-tau * (jump_value * normal) * normal, q);
      }
      else
      {
        AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                      operator_data.which_components == ContinuityPenaltyComponents::Normal,
                    ExcMessage("not implemented."));
      }
    }
  }

  template<typename FEFaceEval>
  void
  do_face_int_integral(FEFaceEval & fe_eval, FEFaceEval & fe_eval_neighbor) const
  {
    VectorizedArray<value_type> tau =
      0.5 *
      (fe_eval.read_cell_data(array_conti_penalty_parameter) +
       fe_eval_neighbor.read_cell_data(array_conti_penalty_parameter)) *
      scaling_factor_conti;

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      Tensor<1, dim, VectorizedArray<value_type>> uM = fe_eval.get_value(q);
      // set uP to zero
      Tensor<1, dim, VectorizedArray<value_type>> uP;
      Tensor<1, dim, VectorizedArray<value_type>> jump_value = uM - uP;

      if(operator_data.which_components == ContinuityPenaltyComponents::All)
      {
        // penalize all velocity components
        fe_eval.submit_value(tau * jump_value, q);
      }
      else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
      {
        // penalize normal components only
        Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval.get_normal_vector(q);
        fe_eval.submit_value(tau * (jump_value * normal) * normal, q);
      }
      else
      {
        AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                      operator_data.which_components == ContinuityPenaltyComponents::Normal,
                    ExcMessage("not implemented."));
      }
    }
  }

  template<typename FEFaceEval>
  void
  do_face_ext_integral(FEFaceEval & fe_eval, FEFaceEval & fe_eval_neighbor) const
  {
    VectorizedArray<value_type> tau =
      0.5 *
      (fe_eval.read_cell_data(array_conti_penalty_parameter) +
       fe_eval_neighbor.read_cell_data(array_conti_penalty_parameter)) *
      scaling_factor_conti;

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      // set uM to zero
      Tensor<1, dim, VectorizedArray<value_type>> uM;
      Tensor<1, dim, VectorizedArray<value_type>> uP = fe_eval_neighbor.get_value(q);
      Tensor<1, dim, VectorizedArray<value_type>> jump_value =
        uP - uM; // interior - exterior = uP - uM (neighbor!)

      if(operator_data.which_components == ContinuityPenaltyComponents::All)
      {
        // penalize all velocity components
        fe_eval_neighbor.submit_value(tau * jump_value, q);
      }
      else if(operator_data.which_components == ContinuityPenaltyComponents::Normal)
      {
        // penalize normal components only
        Tensor<1, dim, VectorizedArray<value_type>> normal = fe_eval_neighbor.get_normal_vector(q);
        fe_eval_neighbor.submit_value(tau * (jump_value * normal) * normal, q);
      }
      else
      {
        AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::All ||
                      operator_data.which_components == ContinuityPenaltyComponents::Normal,
                    ExcMessage("not implemented."));
      }
    }
  }


  void
  cell_loop(MatrixFree<dim, value_type> const &           data,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, dof_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src, true, true);

      do_cell_integral(fe_eval);

      fe_eval.integrate_scatter(true, true, dst);
    }
  }

  void
  cell_loop_div_penalty(MatrixFree<dim, value_type> const &           data,
                        VectorType &                                  dst,
                        VectorType const &                            src,
                        std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, dof_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src, false, true);

      do_cell_integral_div_penalty(fe_eval);

      fe_eval.integrate_scatter(false, true, dst);
    }
  }

  void
  cell_loop_empty(MatrixFree<dim, value_type> const & /*data*/,
                  VectorType & /*dst*/,
                  VectorType const & /*src*/,
                  std::pair<unsigned int, unsigned int> const & /*cell_range*/) const
  {
    // do nothing
  }

  void
  face_loop(MatrixFree<dim, value_type> const &           data,
            VectorType &                                  dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, true, dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data, this->fe_param, false, dof_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      fe_eval.gather_evaluate(src, true, false);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      do_face_integral(fe_eval, fe_eval_neighbor);

      fe_eval.integrate_scatter(true, false, dst);
      fe_eval_neighbor.integrate_scatter(true, false, dst);
    }
  }

  void
  boundary_face_loop_empty(MatrixFree<dim, value_type> const & /*data*/,
                           VectorType & /*dst*/,
                           VectorType const & /*src*/,
                           std::pair<unsigned int, unsigned int> const & /*face_range*/) const
  {
    // do nothing
  }

  /*
   *  This function calculates the diagonal of the projection operator including the mass matrix,
   * divergence penalty and continuity penalty operators. A prerequisite to call this function is
   * that the time step size is set correctly.
   */
  void
  calculate_diagonal(VectorType & diagonal) const
  {
    scaling_factor_div   = this->time_step_size;
    scaling_factor_conti = this->time_step_size;

    VectorType src_dummy(diagonal);
    data.loop(&This::cell_loop_diagonal,
              &This::face_loop_diagonal,
              &This::boundary_face_loop_diagonal,
              this,
              diagonal,
              src_dummy,
              true /*zero dst vector = true*/,
              MatrixFree<dim, value_type>::only_values,
              MatrixFree<dim, value_type>::only_values);
  }

  /*
   * Calculation of diagonal (cell loop).
   */
  void
  cell_loop_diagonal(MatrixFree<dim, value_type> const & data,
                     VectorType &                        dst,
                     VectorType const & /*src*/,
                     std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, dof_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j, make_vectorized_array<value_type>(1.));

        fe_eval.evaluate(true, true);

        do_cell_integral(fe_eval);

        fe_eval.integrate(true, true);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  /*
   * Calculation of diagonal (face loop).
   */
  void
  face_loop_diagonal(MatrixFree<dim, value_type> const & data,
                     VectorType &                        dst,
                     VectorType const & /*src*/,
                     std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, true, dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data, this->fe_param, false, dof_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      // element-
      unsigned int                dofs_per_cell = fe_eval.dofs_per_cell;
      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));
        fe_eval.write_cellwise_dof_value(j, make_vectorized_array<value_type>(1.));

        fe_eval.evaluate(true, false);

        do_face_int_integral(fe_eval, fe_eval_neighbor);

        fe_eval.integrate(true, false);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);

      // neighbor (element+)
      unsigned int dofs_per_cell_neighbor = fe_eval_neighbor.dofs_per_cell;
      VectorizedArray<value_type>
        local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell_neighbor; ++i)
          fe_eval_neighbor.write_cellwise_dof_value(i, make_vectorized_array<value_type>(0.));
        fe_eval_neighbor.write_cellwise_dof_value(j, make_vectorized_array<value_type>(1.));

        fe_eval_neighbor.evaluate(true, false);

        do_face_ext_integral(fe_eval, fe_eval_neighbor);

        fe_eval_neighbor.integrate(true, false);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.read_cellwise_dof_value(j);
      }
      for(unsigned int j = 0; j < dofs_per_cell_neighbor; ++j)
        fe_eval_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  /*
   * Calculation of diagonal (boundary face loop).
   */
  void
  boundary_face_loop_diagonal(MatrixFree<dim, value_type> const & /*data*/,
                              VectorType & /*dst*/,
                              VectorType const & /*src*/,
                              std::pair<unsigned int, unsigned int> const & /*face_range*/) const
  {
    // do nothing
  }

  void
  initialize_block_diagonal_preconditioner_matrix_free() const
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
      typedef Elementwise::InverseMassMatrixPreconditioner<dim, dim, fe_degree, Number>
        INVERSE_MASS;

      elementwise_preconditioner.reset(
        new INVERSE_MASS(this->get_data(), this->get_dof_index(), this->get_quad_index()));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    Elementwise::IterativeSolverData iterative_solver_data;
    iterative_solver_data.solver_type = Elementwise::SolverType::CG;
    iterative_solver_data.solver_data = this->operator_data.block_jacobi_solver_data;

    elementwise_solver.reset(new ELEMENTWISE_SOLVER(
      *std::dynamic_pointer_cast<ELEMENTWISE_OPERATOR>(elementwise_operator),
      *std::dynamic_pointer_cast<PRECONDITIONER_BASE>(elementwise_preconditioner),
      iterative_solver_data));
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<value_type>> & matrices) const
  {
    scaling_factor_div   = this->time_step_size;
    scaling_factor_conti = this->time_step_size;

    VectorType src;

    if(operator_data.use_cell_based_loops)
    {
      data.cell_loop(&This::cell_based_loop_calculate_block_diagonal, this, matrices, src);
    }
    else
    {
      AssertThrow(
        n_mpi_processes == 1,
        ExcMessage(
          "Block diagonal calculation with separate loops over cells and faces only works in serial. "
          "Use cell based loops for parallel computations."));

      data.loop(&This::cell_loop_calculate_block_diagonal,
                &This::face_loop_calculate_block_diagonal,
                &This::boundary_face_loop_calculate_block_diagonal,
                this,
                matrices,
                src);
    }
  }


  void
  cell_loop_calculate_block_diagonal(MatrixFree<dim, value_type> const &         data,
                                     std::vector<LAPACKFullMatrix<value_type>> & matrices,
                                     VectorType const &,
                                     std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, dof_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true, true);

        do_cell_integral(fe_eval);

        fe_eval.integrate(true, true);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<value_type>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  void
  face_loop_calculate_block_diagonal(MatrixFree<dim, value_type> const &         data,
                                     std::vector<LAPACKFullMatrix<value_type>> & matrices,
                                     VectorType const &,
                                     std::pair<unsigned int, unsigned int> const & face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, true, dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data, this->fe_param, false, dof_index);

    // Perform face integrals for element e⁻.
    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true, false);

        do_face_int_integral(fe_eval, fe_eval_neighbor);

        fe_eval.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
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
      fe_eval.reinit(face);
      fe_eval_neighbor.reinit(face);

      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = fe_eval_neighbor.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval_neighbor.evaluate(true, false);

        do_face_ext_integral(fe_eval, fe_eval_neighbor);

        fe_eval_neighbor.integrate(true, false);

        for(unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
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
  boundary_face_loop_calculate_block_diagonal(
    MatrixFree<dim, value_type> const &         data,
    std::vector<LAPACKFullMatrix<value_type>> & matrices,
    VectorType const &,
    std::pair<unsigned int, unsigned int> const & face_range) const
  {
    // do nothing
  }

  void
  cell_based_loop_calculate_block_diagonal(
    MatrixFree<dim, value_type> const &         data,
    std::vector<LAPACKFullMatrix<value_type>> & matrices,
    VectorType const &,
    std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_Velocity_Velocity_linear     fe_eval(data, this->fe_param, dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_m(data, this->fe_param, true, dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_p(data, this->fe_param, false, dof_index);

    // cell integral
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);

      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        fe_eval.evaluate(true, true);

        do_cell_integral(fe_eval);

        fe_eval.integrate(true, true);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < n_filled_lanes; ++v)
            matrices[cell * VectorizedArray<value_type>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }

      // loop over all faces
      unsigned int const n_faces = GeometryInfo<dim>::faces_per_cell;
      for(unsigned int face = 0; face < n_faces; ++face)
      {
        fe_eval_m.reinit(cell, face);
        fe_eval_p.reinit(cell, face);
        auto bids = data.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];

        for(unsigned int j = 0; j < dofs_per_cell; ++j)
        {
          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            fe_eval_m.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
          fe_eval_m.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

          fe_eval_m.evaluate(true, false);

          if(bid == numbers::internal_face_boundary_id) // internal face
          {
            do_face_int_integral(fe_eval_m, fe_eval_p);
          }
          else // boundary face
          {
            // use same fe_eval so that the result becomes zero (only jumps involved)
            do_face_int_integral(fe_eval_m, fe_eval_m);
          }

          fe_eval_m.integrate(true, false);

          for(unsigned int i = 0; i < dofs_per_cell; ++i)
            for(unsigned int v = 0; v < n_filled_lanes; ++v)
              matrices[cell * VectorizedArray<value_type>::n_array_elements + v](i, j) +=
                fe_eval_m.begin_dof_values()[i][v];
        }
      }
    }
  }

  /*
   * Apply inverse block diagonal:
   *
   * instead of applying the block matrix B we compute dst = B^{-1} * src (LU factorization
   * should have already been performed with the method update_inverse_block_diagonal())
   */
  void
  cell_loop_apply_inverse_block_diagonal(
    MatrixFree<dim, value_type> const &           data,
    VectorType &                                  dst,
    VectorType const &                            src,
    std::pair<unsigned int, unsigned int> const & cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data, this->fe_param, dof_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int v = 0; v < VectorizedArray<value_type>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<value_type> src_vector(dofs_per_cell);
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
          src_vector(j) = fe_eval.begin_dof_values()[j][v];

        // apply inverse matrix
        matrices[cell * VectorizedArray<value_type>::n_array_elements + v].solve(src_vector, false);

        // write solution to dst-vector
        for(unsigned int j = 0; j < dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j][v] = src_vector(j);
      }

      fe_eval.set_dof_values(dst);
    }
  }

  MatrixFree<dim, value_type> const & data;

  unsigned int const dof_index;
  unsigned int const quad_index;

  AlignedVector<VectorizedArray<value_type>> array_conti_penalty_parameter;
  AlignedVector<VectorizedArray<value_type>> array_div_penalty_parameter;

  double time_step_size;

  CombinedDivergenceContinuityPenaltyOperatorData operator_data;

  // Scaling factors for divergence and continuity penalty term:
  // Normally, the scaling factor equals the time step size when applying the combined operator
  // consisting of mass, divergence penalty and continuity penalty operators. In case that the
  // divergence and continuity penalty terms are applied separately (e.g. coupled solution approach,
  // penalty terms added to monolithic system), these scaling factors have to be set to a value
  // of 1.
  mutable double scaling_factor_div, scaling_factor_conti;

  unsigned int n_mpi_processes;

  /*
   * Vector of matrices for block-diagonal preconditioners.
   */
  mutable std::vector<LAPACKFullMatrix<Number>> matrices;

  /*
   * We want to initialize the block diagonal preconditioner (block diagonal matrices or elementwise
   * iterative solvers in case of matrix-free implementation) only once, so we store the status of
   * initialization in a variable.
   */
  mutable bool block_diagonal_preconditioner_is_initialized;


  /*
   * Block Jacobi preconditioner/smoother: matrix-free version with elementwise iterative solver
   */
  typedef ElementwiseProjectionOperator<dim, fe_degree, Number, This> ELEMENTWISE_OPERATOR;
  typedef Elementwise::PreconditionerBase<VectorizedArray<Number>>    PRECONDITIONER_BASE;
  typedef Elementwise::
    IterativeSolver<dim, dim, fe_degree, Number, ELEMENTWISE_OPERATOR, PRECONDITIONER_BASE>
      ELEMENTWISE_SOLVER;

  mutable std::shared_ptr<ELEMENTWISE_OPERATOR> elementwise_operator;
  mutable std::shared_ptr<PRECONDITIONER_BASE>  elementwise_preconditioner;
  mutable std::shared_ptr<ELEMENTWISE_SOLVER>   elementwise_solver;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATORS_H_ \
        */
