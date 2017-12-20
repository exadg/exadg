/*
 * ProjectionSolver.h
 *
 *  Created on: Jun 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATORS_AND_SOLVERS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATORS_AND_SOLVERS_H_

// TODO
#include <deal.II/base/timer.h>

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include "../../incompressible_navier_stokes/spatial_discretization/divergence_and_continuity_penalty_operators.h"
#include "operators/base_operator.h"

#include "../include/solvers_and_preconditioners/iterative_solvers.h"
#include "solvers_and_preconditioners/internal_solvers.h"
#include "solvers_and_preconditioners/inverse_mass_matrix_preconditioner.h"
#include "solvers_and_preconditioners/invert_diagonal.h"
#include "solvers_and_preconditioners/verify_calculation_of_diagonal.h"
#include "solvers_and_preconditioners/block_jacobi_matrices.h"


/*
 *  Base class for different projection operators.
 *  This class is derived from the BaseOperator where
 *  we just add the time step size as element variable
 *  along with the respective set() and get()-functions.
 */
template <int dim>
class ProjectionOperatorBase: public BaseOperator<dim>
{
public:
  ProjectionOperatorBase()
    :
    time_step_size(1.0)
  {}

  /*
   *  Set the time step size.
   */
  void set_time_step_size(double const &delta_t)
  {
    time_step_size = delta_t;
  }

  /*
   *  Get the time step size.
   */
  double get_time_step_size() const
  {
    return time_step_size;
  }

private:
  double time_step_size;
};

/*
 *  Projection operator using a divergence penalty term and
 *  an elementwise (local) and iterative solution procedure.
 *
 *  Weak form:
 *
 *   (v_h, u_h)_Omega^e + delta_t * div-div-penalty
 *
 *  where v_h = test function, u_h = solution, delta_t = time step size.
 *  See implementation of div-div penalty operator for a detailed description
 *  of this penalty term.
 */
template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorDivergencePenaltyIterative : public ProjectionOperatorBase<dim>
{
public:
  typedef FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> EvalType;

  ProjectionOperatorDivergencePenaltyIterative(DivergencePenaltyOperator<dim, fe_degree, fe_degree_p,
                                                 fe_degree_xwall, xwall_quad_rule, value_type> const &div_div_penalty)
    :
    fe_eval(1,FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type>(
                  div_div_penalty.get_data(),
                  div_div_penalty.get_dof_index(),
                  div_div_penalty.get_quad_index())),
    inverse(fe_eval[0]),
    tau(1),
    divergence_penalty_operator(&div_div_penalty)
  {
    coefficients.resize(fe_eval[0].n_q_points);
  }

  MatrixFree<dim,value_type> const & get_data() const
  {
    return divergence_penalty_operator->get_data();
  }

  unsigned int get_dof_index() const
  {
    return divergence_penalty_operator->get_dof_index();
  }

  unsigned int get_quad_index() const
  {
    return divergence_penalty_operator->get_dof_index();
  }

  void setup(const unsigned int cell)
  {
    tau[0] = divergence_penalty_operator->get_array_penalty_parameter()[cell];
    fe_eval[0].reinit(cell);
    inverse.fill_inverse_JxW_values(coefficients);
  }

  void precondition(VectorizedArray<value_type>       *dst,
                    const VectorizedArray<value_type> *src) const
  {
    inverse.apply(coefficients, dim, src, dst);
  }

  void vmult(VectorizedArray<value_type> *dst,
             VectorizedArray<value_type> *src) const
  {
    Assert(fe_eval[0].get_shape_info().element_type <=
        dealii::internal::MatrixFreeFunctions::tensor_symmetric,
           ExcNotImplemented());

    // compute matrix vector product on element
    fe_eval[0].evaluate(src, true, true, false);

    for (unsigned int q=0; q<fe_eval[0].n_q_points; ++q)
    {
      VectorizedArray<value_type> tau_times_div = tau[0] * fe_eval[0].get_divergence(q);
      fe_eval[0].submit_divergence(this->get_time_step_size()*tau_times_div, q);
      fe_eval[0].submit_value (fe_eval[0].get_value(q), q);
    }

    fe_eval[0].integrate(true, true, dst);
  }

private:
  mutable AlignedVector<FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> > fe_eval;
  AlignedVector<VectorizedArray<value_type> > coefficients;
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,dim,value_type> inverse;
  AlignedVector<VectorizedArray<value_type> > tau;
  DivergencePenaltyOperator<dim, fe_degree, fe_degree_p,
      fe_degree_xwall, xwall_quad_rule, value_type> const * divergence_penalty_operator;
};

/*
 *  TODO:
 *  Elementwise inverse mass matrix preconditioner.
 *  The Preconditioner is currently implemented in the class
 *    "ProjectionOperatorDivergencePenaltyIterative"
 */
template<typename value_type, typename Operator>
class ElementwiseInverseMassMatrixPreconditioner : public InternalSolvers::PreconditionerBase<value_type>
{
public:
  ElementwiseInverseMassMatrixPreconditioner(Operator &operator_in)
  :
    op(operator_in)
  {}

  void vmult(value_type *dst, value_type const *src) const
  {
    op.precondition(dst,src);
  }

private:
  Operator &op;
};


/*
 *  Projection operator using a divergence penalty term and
 *  an elementwise (local) and direct solution strategy.
 *
 *  Weak form:
 *
 *   (v_h, u_h)_Omega^e + delta_t * div-div-penalty
 *
 *  where v_h = test function, u_h = solution, delta_t = time step size.
 *  See implementation of div-div penalty operator for a detailed description
 *  of this penalty term.
 *
 */
template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorDivergencePenaltyDirect : public ProjectionOperatorBase<dim>
{
public:
  typedef FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> EvalType;

  ProjectionOperatorDivergencePenaltyDirect(DivergencePenaltyOperator<dim, fe_degree, fe_degree_p,
                                              fe_degree_xwall, xwall_quad_rule, value_type> const &div_div_penalty)
    :
    divergence_penalty_operator(&div_div_penalty)
  {}

  MatrixFree<dim,value_type> const & get_data() const
  {
    return divergence_penalty_operator->get_data();
  }

  unsigned int get_dof_index() const
  {
    return divergence_penalty_operator->get_dof_index();
  }

  FEParameters<dim> const * get_fe_param() const
  {
    return this->fe_param;
  }

  AlignedVector<VectorizedArray<value_type> > const & get_array_penalty_parameter() const
  {
    return divergence_penalty_operator->get_array_penalty_parameter();
  }

private:
  DivergencePenaltyOperator<dim, fe_degree, fe_degree_p,
      fe_degree_xwall, xwall_quad_rule, value_type> const * divergence_penalty_operator;
};


/*
 *  Projection operator using both a divergence and continuity penalty term.
 *
 *  Weak form:
 *
 *   (v_h, u_h)_Omega^e + delta_t * div-div-penalty + delta_t * continuity-penalty
 *
 *  where v_h = test function, u_h = solution, delta_t = time step size.
 *  See implementation of div-div/continuity penalty operator for a detailed description
 *  of these penalty terms.
 *
 */
template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorDivergenceAndContinuityPenalty : public ProjectionOperatorBase<dim>
{
public:
  typedef ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree,
      fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> This;

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,
                              dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  /*
   *  Constructor
   */
  ProjectionOperatorDivergenceAndContinuityPenalty(
      MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>  const                    &mass_matrix_operator_in,
      DivergencePenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> const &divergence_penalty_operator_in,
      ContinuityPenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> const &continuity_penalty_operator_in)
    :
    block_jacobi_matrices_have_been_initialized(false),
    mass_matrix_operator(&mass_matrix_operator_in),
    divergence_penalty_operator(&divergence_penalty_operator_in),
    continuity_penalty_operator(&continuity_penalty_operator_in),
    wall_time(0.0) //TODO
  {}

  //TODO
  double get_wall_time() const
  {
    return wall_time;
  }

  /*
   *  This function calculates the matrix-vector product for the projection operator including
   *  div-div penalty and continuity penalty terms given a vector src.
   *  A prerequisite to call this function is that the time step size is set correctly.
   */
  void vmult (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const
  {
    // TODO
    Timer timer;
    timer.restart();

    divergence_penalty_operator->apply(dst,src); // 2e9 dofs/s
    continuity_penalty_operator->apply_add(dst,src); // 8e8 dofs/s

    dst *= this->get_time_step_size(); // 3.5e9 dofs/s

    mass_matrix_operator->apply_add(dst,src); // 2.2e9 dofs/s

    // TODO
    wall_time += timer.wall_time();
  }

  /*
   *  This function calculates inhomogeneous boundary face integrals appearing on the rhs.
   *  The only operator that is relevant in this respect is the continuity penalty operator.
   *  A prerequisite to call this function is that the time step size is set correctly.
   */
  void rhs (parallel::distributed::Vector<value_type> &dst,
            double const                              eval_time) const
  {
    parallel::distributed::Vector<value_type> temp(dst);

    continuity_penalty_operator->rhs(temp,eval_time);

    dst.equ(this->get_time_step_size(),temp);
  }


  /*
   *  This function calculates inhomogeneous boundary face integrals appearing on the rhs
   *  and adds the result to the dst-vector.
   *  The only operator that is relevant in this respect is the continuity penalty operator.
   *  A prerequisite to call this function is that the time step size is set correctly.
   */
  void rhs_add (parallel::distributed::Vector<value_type> &dst,
                double const                              eval_time) const
  {
    parallel::distributed::Vector<value_type> temp(dst);

    continuity_penalty_operator->rhs(temp,eval_time);

    dst.add(this->get_time_step_size(),temp);
  }

  /*
   *  Calculate inverse diagonal which is needed for the Jacobi preconditioner.
   */
  void calculate_inverse_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    calculate_diagonal(diagonal);

    // verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  /*
   *  Apply block Jacobi preconditioner.
   */
  void apply_block_jacobi (parallel::distributed::Vector<value_type>       &dst,
                           parallel::distributed::Vector<value_type> const &src) const
  {
    this->mass_matrix_operator->get_data().cell_loop(&This::cell_loop_apply_inverse_block_jacobi_matrices, this, dst, src);
  }


  /*
   *  This function updates the block Jacobi preconditioner.
   *  Since this function also initializes the block Jacobi preconditioner,
   *  make sure that the block Jacobi matrices are allocated before calculating
   *  the matrices and the LU factorization.
   */
  void update_block_jacobi () const
  {
    if(block_jacobi_matrices_have_been_initialized == false)
    {
      // Note that the velocity has dim components.
      unsigned int dofs_per_cell = this->mass_matrix_operator->get_data().get_shape_info().dofs_per_component_on_cell*dim;

      matrices.resize(this->mass_matrix_operator->get_data().n_macro_cells()*VectorizedArray<value_type>::n_array_elements,
        LAPACKFullMatrix<value_type>(dofs_per_cell, dofs_per_cell));

      block_jacobi_matrices_have_been_initialized = true;
    }

    calculate_block_jacobi_matrices();
    calculate_lu_factorization_block_jacobi(matrices);
  }

  /*
   *  Initialize dof vector (required when using the Jacobi preconditioner).
   */
  void initialize_dof_vector(parallel::distributed::Vector<value_type> &vector) const
  {
    this->mass_matrix_operator->get_data().initialize_dof_vector(
        vector,this->mass_matrix_operator->get_operator_data().dof_index);
  }

private:
  /*
   *  This function calculates the diagonal of the projection operator including
   *  div-div penalty and continuity penalty terms.
   *  A prerequisite to call this function is that the time step size is set correctly.
   */
  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    divergence_penalty_operator->calculate_diagonal(diagonal);
    continuity_penalty_operator->add_diagonal(diagonal);

    diagonal *= this->get_time_step_size();

    mass_matrix_operator->add_diagonal(diagonal);
  }

  /*
   * This function calculates the block Jacobi matrices.
   * This is done sequentially for the different operators.
   */
  void calculate_block_jacobi_matrices() const
  {
    // initialize block Jacobi matrices with zeros
    initialize_block_jacobi_matrices_with_zero(matrices);

    divergence_penalty_operator->add_block_jacobi_matrices(matrices);
    continuity_penalty_operator->add_block_jacobi_matrices(matrices);

    for(typename std::vector<LAPACKFullMatrix<value_type> >::iterator
        it = matrices.begin(); it != matrices.end(); ++it)
    {
      (*it) *= this->get_time_step_size();
    }

    mass_matrix_operator->add_block_jacobi_matrices(matrices);
  }

  /*
   *  This function loops over all cells and applies the inverse block Jacobi matrices elementwise.
   */
  void cell_loop_apply_inverse_block_jacobi_matrices (MatrixFree<dim,value_type> const                &data,
                                                      parallel::distributed::Vector<value_type>       &dst,
                                                      parallel::distributed::Vector<value_type> const &src,
                                                      std::pair<unsigned int,unsigned int> const      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,
                                            this->mass_matrix_operator->get_fe_param(),
                                            this->mass_matrix_operator->get_operator_data().dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<value_type> src_vector(dofs_per_cell);
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          src_vector(j) = fe_eval.begin_dof_values()[j][v];

        // apply inverse matrix
        matrices[cell*VectorizedArray<value_type>::n_array_elements+v].apply_lu_factorization(src_vector,false);

        // write solution to dst-vector
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j][v] = src_vector(j);
      }

      fe_eval.set_dof_values (dst);
    }
  }

  mutable std::vector<LAPACKFullMatrix<value_type> > matrices;
  mutable bool block_jacobi_matrices_have_been_initialized;

  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>  const * mass_matrix_operator;
  DivergencePenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> const * divergence_penalty_operator;
  ContinuityPenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> const * continuity_penalty_operator;

  // TODO
  mutable double wall_time;
};


/*
 *  Projection solver in case no penalty term is involved in the
 *  projection step. Hence, we only have to apply the inverse mass
 *  matrix operator to solve the projection equation.
 */
template <int dim, int fe_degree, typename value_type>
class ProjectionSolverNoPenalty : public IterativeSolverBase<parallel::distributed::Vector<value_type> >
{
public:
  ProjectionSolverNoPenalty(MatrixFree<dim,value_type> const &data_in,
                            unsigned int const               dof_index_in,
                            unsigned int const               quad_index_in)
  {
    inverse_mass_matrix_operator.initialize(data_in,dof_index_in,quad_index_in);
  }
  unsigned int solve(parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src) const
  {
    inverse_mass_matrix_operator.apply(dst,src);

    return 0;
  }

private:
  InverseMassMatrixOperator<dim,fe_degree,value_type> inverse_mass_matrix_operator;
};

/*
 *  Projection solver for projection with divergence penalty term
 *  using a direct solution approach.
 */
template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class DirectProjectionSolverDivergencePenalty : public IterativeSolverBase<parallel::distributed::Vector<value_type> >
{
public:
  typedef ProjectionOperatorDivergencePenaltyDirect<dim, fe_degree,
      fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> PROJ_OPERATOR;

  typedef DirectProjectionSolverDivergencePenalty<dim, fe_degree,
      fe_degree_p, fe_degree_xwall, xwall_quad_rule,value_type> THIS;

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  DirectProjectionSolverDivergencePenalty(std::shared_ptr<PROJ_OPERATOR> projection_operator_in)
    :
    projection_operator(projection_operator_in)
  {}

  unsigned int solve(parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;

    projection_operator->get_data().cell_loop (&THIS::local_solve, this, dst, src);

    return 0;
  }

  void local_solve(const MatrixFree<dim,value_type>                &data,
                   parallel::distributed::Vector<value_type>       &dst,
                   const parallel::distributed::Vector<value_type> &src,
                   const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,projection_operator->get_fe_param(), projection_operator->get_dof_index());

    std::vector<LAPACKFullMatrix<value_type> > matrices(VectorizedArray<value_type>::n_array_elements);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      const unsigned int total_dofs_per_cell = fe_eval_velocity.dofs_per_cell;

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        matrices[v].reinit(total_dofs_per_cell, total_dofs_per_cell);

      // div-div penalty parameter
      // multiply penalty parameter by the time step size
      const VectorizedArray<value_type> tau = projection_operator->get_time_step_size() *
                                              projection_operator->get_array_penalty_parameter()[cell];

      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array<value_type>(0.));
        fe_eval_velocity.write_cellwise_dof_value(j,make_vectorized_array<value_type>(1.));

        fe_eval_velocity.evaluate (true,true,false);
        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          const VectorizedArray<value_type> tau_times_div = tau * fe_eval_velocity.get_divergence(q);
          Tensor<2,dim,VectorizedArray<value_type> > test;
          for (unsigned int d=0; d<dim; ++d)
            test[d][d] = tau_times_div;
          fe_eval_velocity.submit_gradient(test, q);
          fe_eval_velocity.submit_value (fe_eval_velocity.get_value(q), q);
        }
        fe_eval_velocity.integrate (true,true);

        for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        {
          if(fe_eval_velocity.component_enriched(v))
          {
            for (unsigned int i=0; i<total_dofs_per_cell; ++i)
              (matrices[v])(i,j) = (fe_eval_velocity.read_cellwise_dof_value(i))[v];
          }
          else//this is a non-enriched element
          {
            if(j<fe_eval_velocity.std_dofs_per_cell)
              for (unsigned int i=0; i<fe_eval_velocity.std_dofs_per_cell; ++i)
                (matrices[v])(i,j) = (fe_eval_velocity.read_cellwise_dof_value(i))[v];
            else //diagonal
              (matrices[v])(j,j) = 1.0;
          }
        }
      }

      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(src);

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      {
        (matrices[v]).compute_lu_factorization();
        Vector<value_type> vector_input(total_dofs_per_cell);
        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
          vector_input(j)=(fe_eval_velocity.read_cellwise_dof_value(j))[v];

        (matrices[v]).apply_lu_factorization(vector_input,false);
        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
          fe_eval_velocity.write_cellwise_dof_value(j,vector_input(j),v);
      }
      fe_eval_velocity.set_dof_values (dst);
    }
  }

  std::shared_ptr<PROJ_OPERATOR> projection_operator;
};

/*
 *  Solver data
 */
struct ProjectionSolverData
{
  ProjectionSolverData()
    :
    max_iter(1e4),
    solver_tolerance_abs(1.e-12),
    solver_tolerance_rel(1.e-6)
  {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
};

/*
 *  Projection solver for projection with divergence penalty term
 *  using an iterative solution procedure. For this solver we always
 *  use the inverse mass matrix as a preconditioner.
 */
template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule,typename value_type>
class IterativeProjectionSolverDivergencePenalty : public IterativeSolverBase<parallel::distributed::Vector<value_type> >
{
public:
  typedef ProjectionOperatorDivergencePenaltyIterative<dim, fe_degree, fe_degree_p,
      fe_degree_xwall, xwall_quad_rule, value_type> PROJ_OPERATOR;

  typedef IterativeProjectionSolverDivergencePenalty<dim, fe_degree,
      fe_degree_p, fe_degree_xwall, xwall_quad_rule,value_type> THIS;

  IterativeProjectionSolverDivergencePenalty(PROJ_OPERATOR              &projection_operator_in,
                                             ProjectionSolverData const solver_data_in)
    :
    projection_operator(projection_operator_in),
    solver_data(solver_data_in)
  {}

  virtual ~IterativeProjectionSolverDivergencePenalty(){}

  unsigned int solve(parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;

    projection_operator.get_data().cell_loop (&THIS::local_solve, this, dst, src);

    return 0;
  }

  virtual void local_solve(const MatrixFree<dim,value_type>                &data,
                           parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &src,
                           const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,
                                                                   projection_operator.get_dof_index(),
                                                                   projection_operator.get_quad_index());

    const unsigned int total_dofs_per_cell = fe_eval.dofs_per_cell;
    AlignedVector<VectorizedArray<value_type> > solution(total_dofs_per_cell);

    InternalSolvers::SolverCG<VectorizedArray<value_type> > cg_solver(total_dofs_per_cell,
                                                                      solver_data.solver_tolerance_abs,
                                                                      solver_data.solver_tolerance_rel,
                                                                      solver_data.max_iter);

    //TODO
//    InternalSolvers::SolverGMRES<VectorizedArray<value_type> > cg_solver(total_dofs_per_cell,
//                                                                         solver_data.solver_tolerance_abs,
//                                                                         solver_data.solver_tolerance_rel,
//                                                                         solver_data.max_iter);

    //TODO
//    InternalSolvers::PreconditionerIdentity<VectorizedArray<value_type> > preconditioner(total_dofs_per_cell);

    ElementwiseInverseMassMatrixPreconditioner<VectorizedArray<value_type>,PROJ_OPERATOR> preconditioner(projection_operator);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src,0);

      projection_operator.setup(cell);
      cg_solver.solve(&projection_operator,
                      solution.begin(),
                      fe_eval.begin_dof_values(),
                      &preconditioner);

      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = solution[j];
      fe_eval.set_dof_values (dst,0);
    }
  }

protected:
  PROJ_OPERATOR &projection_operator;
  ProjectionSolverData const solver_data;
};



/*
 *  Performance-optimized implementation of projection-operator including
 *  mass matrix operator, divergence penalty operator, and normal-continuity penalty operator.
 *  This implementation of the operator should only be used for performance considerations since
 *  it might not be up-to-date regarding the discretization methods currently used.
 */

/*
 *  Operator data.
 */
template<int dim>
struct OptimizedProjectionOperatorData
{
  OptimizedProjectionOperatorData()
    :
    type_penalty_parameter(TypePenaltyParameter::ConvectiveTerm),
    viscosity(0.0),
    penalty_parameter(1.0),
    which_components(ContinuityPenaltyComponents::Normal),
    use_boundary_data(false)
  {}

  // type of penalty parameter (viscous and/or convective terms)
  TypePenaltyParameter type_penalty_parameter;

  // kinematic viscosity
  double viscosity;

  // scaling factor
  double penalty_parameter;

  // the continuity penalty term can be applied
  // to all velocity components and to the normal
  // component only
  ContinuityPenaltyComponents which_components;

  // the continuity penalty term can be applied
  // on boundary faces.
  bool use_boundary_data;
  std::shared_ptr<BoundaryDescriptorNavierStokesU<dim> > bc;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorOptimized : public ProjectionOperatorBase<dim>
{
public:
  enum class BoundaryType {
    undefined,
    dirichlet,
    neumann
  };

  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;

  typedef ProjectionOperatorOptimized<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> This;

  ProjectionOperatorOptimized(MatrixFree<dim,value_type> const         &data_in,
                              unsigned int const                       dof_index_in,
                              unsigned int const                       quad_index_in,
                              OptimizedProjectionOperatorData<dim> const operator_data_in)
    :
    data(data_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    array_penalty_parameter_conti(0),
    array_penalty_parameter_div(0),
    operator_data(operator_data_in),
    eval_time(0.0),
    wall_time(0.0) //TODO
  {
    array_penalty_parameter_conti.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
    array_penalty_parameter_div.resize(data.n_macro_cells()+data.n_macro_ghost_cells());

    AssertThrow(operator_data.which_components == ContinuityPenaltyComponents::Normal &&
                operator_data.use_boundary_data == false,
                ExcMessage("Not implemented for performance-optimized projection operator"));
  }

  //TODO
  double get_wall_time() const
  {
    return wall_time;
  }

  void calculate_array_penalty_parameter(parallel::distributed::Vector<value_type> const &velocity)
  {
    calculate_array_penalty_parameter_div(velocity);
    calculate_array_penalty_parameter_conti(velocity);
  }

  void calculate_array_penalty_parameter_div(parallel::distributed::Vector<value_type> const &velocity)
  {
    velocity.update_ghost_values();

    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,dof_index);

    AlignedVector<VectorizedArray<value_type> > JxW_values(fe_eval.n_q_points);

    for (unsigned int cell=0; cell<data.n_macro_cells()+data.n_macro_ghost_cells(); ++cell)
    {
      VectorizedArray<value_type> tau_convective = make_vectorized_array<value_type>(0.0);
      VectorizedArray<value_type> tau_viscous = make_vectorized_array<value_type>(operator_data.viscosity);

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm ||
          operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms )
      {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(velocity);
        fe_eval.evaluate (true,false);
        VectorizedArray<value_type> volume = make_vectorized_array<value_type>(0.0);
        VectorizedArray<value_type> norm_U_mean = make_vectorized_array<value_type>(0.0);
        JxW_values.resize(fe_eval.n_q_points);
        fe_eval.fill_JxW_values(JxW_values);
        for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
        {
          volume += JxW_values[q];
          norm_U_mean += JxW_values[q]*fe_eval.get_value(q).norm();
        }
        norm_U_mean /= volume;

        tau_convective = norm_U_mean * std::exp(std::log(volume)/(double)dim) / (double)(fe_degree+1);
      }

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_penalty_parameter_div[cell] = operator_data.penalty_parameter * tau_convective;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_penalty_parameter_div[cell] = operator_data.penalty_parameter * tau_viscous;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_penalty_parameter_div[cell] = operator_data.penalty_parameter * (tau_convective + tau_viscous);
      }
    }
  }

  void calculate_array_penalty_parameter_conti(parallel::distributed::Vector<value_type> const &velocity)
  {
    velocity.update_ghost_values();

    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,dof_index);

    AlignedVector<VectorizedArray<value_type> > JxW_values(fe_eval.n_q_points);

    for (unsigned int cell=0; cell<data.n_macro_cells()+data.n_macro_ghost_cells(); ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(velocity);
      fe_eval.evaluate (true,false);
      VectorizedArray<value_type> volume = make_vectorized_array<value_type>(0.0);
      VectorizedArray<value_type> norm_U_mean = make_vectorized_array<value_type>(0.0);
      JxW_values.resize(fe_eval.n_q_points);
      fe_eval.fill_JxW_values(JxW_values);
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        volume += JxW_values[q];
        norm_U_mean += JxW_values[q]*fe_eval.get_value(q).norm();
      }
      norm_U_mean /= volume;

      VectorizedArray<value_type> tau_convective = norm_U_mean;
      VectorizedArray<value_type> h = std::exp(std::log(volume)/(double)dim) / (double)(fe_degree+1);
      VectorizedArray<value_type> tau_viscous = make_vectorized_array<value_type>(operator_data.viscosity) / h;

      if(operator_data.type_penalty_parameter == TypePenaltyParameter::ConvectiveTerm)
      {
        array_penalty_parameter_conti[cell] = operator_data.penalty_parameter * tau_convective;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousTerm)
      {
        array_penalty_parameter_conti[cell] = operator_data.penalty_parameter * tau_viscous;
      }
      else if(operator_data.type_penalty_parameter == TypePenaltyParameter::ViscousAndConvectiveTerms)
      {
        array_penalty_parameter_conti[cell] = operator_data.penalty_parameter * (tau_convective + tau_viscous);
      }
    }
  }

  MatrixFree<dim,value_type> const & get_data() const
  {
    return data;
  }
  AlignedVector<VectorizedArray<value_type> > const & get_array_penalty_parameter_conti() const
  {
    return array_penalty_parameter_conti;
  }
  AlignedVector<VectorizedArray<value_type> > const & get_array_penalty_parameter_div() const
  {
    return array_penalty_parameter_div;
  }
  unsigned int get_dof_index() const
  {
    return dof_index;
  }
  unsigned int get_quad_index() const
  {
    return quad_index;
  }
  FEParameters<dim> const * get_fe_param() const
  {
    return this->fe_param;
  }

  void vmult (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const
  {
    // TODO
    Timer timer;
    timer.restart();

    apply(dst,src);

    // TODO
    wall_time += timer.wall_time();
  }

  void apply (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const
  {
    this->get_data().loop(&This::cell_loop,
                          &This::face_loop,
                          &This::boundary_face_loop,
                          this, dst, src, /*zero dst vector = */ true,
                          MatrixFree<dim,value_type>::only_values,
                          MatrixFree<dim,value_type>::only_values);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  parallel::distributed::Vector<value_type> const &src) const
  {
    this->get_data().loop(&This::cell_loop,
                          &This::face_loop,
                          &This::boundary_face_loop,
                          this, dst, src, /*zero dst vector = */ false,
                          MatrixFree<dim,value_type>::only_values,
                          MatrixFree<dim,value_type>::only_values);
  }

private:
  void cell_loop (const MatrixFree<dim,value_type>                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->get_fe_param(),this->get_dof_index());

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(src,true,true);

      VectorizedArray<value_type> tau = this->get_array_penalty_parameter_div()[cell]*this->get_time_step_size();

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        fe_eval.submit_value(fe_eval.get_value(q), q);
        fe_eval.submit_divergence(tau*fe_eval.get_divergence(q), q);
      }

      fe_eval.integrate_scatter(true,true,dst);
    }
  }

  void face_loop (const MatrixFree<dim,value_type>                 &data,
                  parallel::distributed::Vector<value_type>        &dst,
                  const parallel::distributed::Vector<value_type>  &src,
                  const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->get_fe_param(),true,this->get_dof_index());
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,this->get_fe_param(),false,this->get_dof_index());

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.gather_evaluate(src, true, false);
      fe_eval_neighbor.gather_evaluate(src, true, false);

      VectorizedArray<value_type> tau = 0.5*(fe_eval.read_cell_data(this->get_array_penalty_parameter_conti())
                                             + fe_eval_neighbor.read_cell_data(this->get_array_penalty_parameter_conti()))
                                        *this->get_time_step_size();

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

        // penalize normal components only
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);

        fe_eval.submit_value(tau*(jump_value*normal)*normal,q);
        fe_eval_neighbor.submit_value(-tau*(jump_value*normal)*normal,q);
      }

      fe_eval.integrate_scatter(true,false,dst);
      fe_eval_neighbor.integrate_scatter(true,false,dst);
    }
  }

  void boundary_face_loop(const MatrixFree<dim,value_type>                &data,
                          parallel::distributed::Vector<value_type>       &dst,
                          const parallel::distributed::Vector<value_type> &src,
                          const std::pair<unsigned int,unsigned int>      &face_range) const
  {

  }

  MatrixFree<dim,value_type> const & data;
  unsigned int const dof_index;
  unsigned int const quad_index;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter_conti;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter_div;
  OptimizedProjectionOperatorData<dim> operator_data;
  double mutable eval_time;

  // TODO
  mutable double wall_time;
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATORS_AND_SOLVERS_H_ */
