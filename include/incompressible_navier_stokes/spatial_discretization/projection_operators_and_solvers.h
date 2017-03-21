/*
 * ProjectionSolver.h
 *
 *  Created on: Jun 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATORS_AND_SOLVERS_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATORS_AND_SOLVERS_H_

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
  typedef FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> EvalType;

  ProjectionOperatorDivergencePenaltyIterative(DivergencePenaltyOperator<dim, fe_degree, fe_degree_p,
                                                 fe_degree_xwall, xwall_quad_rule, value_type> const &div_div_penalty)
    :
    fe_eval(1,FEEvaluation<dim,fe_degree,fe_degree+1,dim,double>(
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

  void precondition(VectorizedArray<double>       *dst,
                    const VectorizedArray<double> *src) const
  {
    inverse.apply(coefficients, dim, src, dst);
  }

  void vmult(VectorizedArray<double> *dst,
             VectorizedArray<double> *src) const
  {
    Assert(fe_eval[0].get_shape_info().element_type <=
        dealii::internal::MatrixFreeFunctions::tensor_symmetric,
           ExcNotImplemented());

    // compute matrix vector product on element
    fe_eval[0].evaluate(src, true, true, false);

    for (unsigned int q=0; q<fe_eval[0].n_q_points; ++q)
    {
      VectorizedArray<double> tau_times_div = tau[0] * fe_eval[0].get_divergence(q);
      fe_eval[0].submit_divergence(this->get_time_step_size()*tau_times_div, q);
      fe_eval[0].submit_value (fe_eval[0].get_value(q), q);
    }

    fe_eval[0].integrate(true, true, dst);
  }

private:
  mutable AlignedVector<FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> > fe_eval;
  AlignedVector<VectorizedArray<double> > coefficients;
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,dim,double> inverse;
  AlignedVector<VectorizedArray<double> > tau;
  DivergencePenaltyOperator<dim, fe_degree, fe_degree_p,
      fe_degree_xwall, xwall_quad_rule, value_type> const * divergence_penalty_operator;
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

  /*
   *  Constructor
   */
  ProjectionOperatorDivergenceAndContinuityPenalty(
      MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>  const                    &mass_matrix_operator_in,
      DivergencePenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> const &divergence_penalty_operator_in,
      ContinuityPenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> const &continuity_penalty_operator_in)
    :
    mass_matrix_operator(&mass_matrix_operator_in),
    divergence_penalty_operator(&divergence_penalty_operator_in),
    continuity_penalty_operator(&continuity_penalty_operator_in)
  {}

  /*
   *  This function calculates the matrix-vector product for the projection operator including
   *  div-div penalty and continuity penalty terms given a vector src.
   *  A prerequisite to call this function is that the time step size is set correctly.
   */
  void vmult (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const
  {
    dst = 0;

    divergence_penalty_operator->apply_add(dst,src);
    continuity_penalty_operator->apply_add(dst,src);

    dst *= this->get_time_step_size();

    mass_matrix_operator->apply_add(dst,src);
  }

  /*
   *  Calculate inverse diagonal which is needed for the Jacobi preconditioner.
   */
  void calculate_inverse_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    calculate_diagonal(diagonal);

//    verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
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
    diagonal = 0;

    divergence_penalty_operator->add_diagonal(diagonal);
    continuity_penalty_operator->add_diagonal(diagonal);

    diagonal *= this->get_time_step_size();

    mass_matrix_operator->add_diagonal(diagonal);
  }

  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, value_type>  const * mass_matrix_operator;
  DivergencePenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> const * divergence_penalty_operator;
  ContinuityPenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> const * continuity_penalty_operator;
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

  DirectProjectionSolverDivergencePenalty(std_cxx11::shared_ptr<PROJ_OPERATOR> projection_operator_in)
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
      const unsigned int total_dofs_per_cell = fe_eval_velocity.dofs_per_cell * dim;

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        matrices[v].reinit(total_dofs_per_cell, total_dofs_per_cell);

      // div-div penalty parameter
      // multiply penalty parameter by the time step size
      const VectorizedArray<value_type> tau = projection_operator->get_time_step_size() *
                                              projection_operator->get_array_penalty_parameter()[cell];

      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval_velocity.write_cellwise_dof_value(j,make_vectorized_array(1.));

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
            if(j<fe_eval_velocity.std_dofs_per_cell*dim)
              for (unsigned int i=0; i<fe_eval_velocity.std_dofs_per_cell*dim; ++i)
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

  std_cxx11::shared_ptr<PROJ_OPERATOR> projection_operator;
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

    const unsigned int total_dofs_per_cell = fe_eval.dofs_per_cell * dim;
    AlignedVector<VectorizedArray<value_type> > solution(total_dofs_per_cell);

    InternalSolvers::SolverCG<VectorizedArray<double> > cg_solver(total_dofs_per_cell,
                                                                  solver_data.solver_tolerance_abs,
                                                                  solver_data.solver_tolerance_rel,
                                                                  solver_data.max_iter);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src,0);

      projection_operator.setup(cell);
      cg_solver.solve(&projection_operator,
                      solution.begin(),
                      fe_eval.begin_dof_values());

      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = solution[j];
      fe_eval.set_dof_values (dst,0);
    }
  }

protected:
  PROJ_OPERATOR &projection_operator;
  ProjectionSolverData const solver_data;
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_PROJECTION_OPERATORS_AND_SOLVERS_H_ */
