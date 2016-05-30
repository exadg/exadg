/*
 * ProjectionSolver.h
 *
 *  Created on: May 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PROJECTIONSOLVER_H_
#define INCLUDE_PROJECTIONSOLVER_H_

#include "PreconditionerVelocity.h"

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall> class NavierStokesOperation;

struct ProjectionOperatorData
{
  double  penalty_parameter_divergence;
  double  penalty_parameter_continuity;
  bool solve_stokes_equations;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class ProjectionSolverBase
{
public:
  static const bool is_xwall = false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? n_q_points_1d_xwall : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  ProjectionSolverBase(MatrixFree<dim,value_type> const & data_in,
                       FEParameters<value_type> & fe_param_in,
                       const unsigned int dof_index_in,
                       ProjectionOperatorData const projection_operator_data_in)
    :
    data(data_in),
    fe_param(fe_param_in),
    dof_index(dof_index_in),
    array_penalty_parameter_divergence(0),
    array_penalty_parameter_continuity(0),
    projection_operator_data(projection_operator_data_in)
  {
    array_penalty_parameter_divergence.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
    array_penalty_parameter_continuity.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
  }

  virtual ~ProjectionSolverBase(){}

  virtual unsigned int solve(parallel::distributed::BlockVector<double>       &dst,
                             const parallel::distributed::BlockVector<double> &src) const = 0;

  void calculate_array_penalty_parameter(parallel::distributed::BlockVector<value_type> const & velocity_n,
                                         double const cfl,
                                         double const time_step)
  {
    for(unsigned int d=0; d<dim; ++d)
      velocity_n.block(d).update_ghost_values();

    FEEval_Velocity_Velocity_linear fe_eval(data,fe_param,dof_index);

    AlignedVector<VectorizedArray<value_type> > JxW_values(fe_eval.n_q_points);

    for (unsigned int cell=0; cell<data.n_macro_cells()+data.n_macro_ghost_cells(); ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(velocity_n,0,dim);
      fe_eval.evaluate (true,false);
      VectorizedArray<value_type> volume = make_vectorized_array<value_type>(0.0);
      Tensor<1,dim,VectorizedArray<value_type> > U_mean;
      VectorizedArray<value_type> norm_U_mean;
      fe_eval.fill_JxW_values(JxW_values);
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        volume += JxW_values[q];
        U_mean += JxW_values[q]*fe_eval.get_value(q);
      }
      U_mean /= volume;
      norm_U_mean = U_mean.norm();

      // inverse cfl dependency does not make sense when solving the Stokes equations
      if(projection_operator_data.solve_stokes_equations == true)
      {
        array_penalty_parameter_divergence[cell] = projection_operator_data.penalty_parameter_divergence * norm_U_mean * std::exp(std::log(volume)/(double)dim) * time_step;
        array_penalty_parameter_continuity[cell] = projection_operator_data.penalty_parameter_continuity * norm_U_mean * time_step;
      }
      else
      {
        array_penalty_parameter_divergence[cell] = projection_operator_data.penalty_parameter_divergence/cfl * norm_U_mean * std::exp(std::log(volume)/(double)dim) * time_step;
        array_penalty_parameter_continuity[cell] = projection_operator_data.penalty_parameter_continuity/cfl * norm_U_mean * time_step;
      }
    }
  }

  MatrixFree<dim,value_type> const & get_data() const
  {
    return data;
  }
  AlignedVector<VectorizedArray<value_type> > const & get_array_penalty_parameter_divergence() const
  {
    return array_penalty_parameter_divergence;
  }
  AlignedVector<VectorizedArray<value_type> > const & get_array_penalty_parameter_continuity() const
  {
    return array_penalty_parameter_continuity;
  }
  unsigned int get_dof_index() const
  {
    return dof_index;
  }
  FEParameters<value_type> const & get_fe_param() const
  {
    return fe_param;
  }

private:
  MatrixFree<dim,value_type> const & data;
  FEParameters<value_type> const & fe_param;
  unsigned int const dof_index;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter_divergence;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter_continuity;
  ProjectionOperatorData projection_operator_data;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class ProjectionSolverNoPenalty : public ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>
{
public:
  ProjectionSolverNoPenalty(MatrixFree<dim,value_type> const &data_in,
                            FEParameters<value_type> & fe_param_in,
                            const unsigned int dof_index_in,
                            const unsigned int quad_index_in,
                            ProjectionOperatorData const projection_operator_data_in)
    :
    ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(data_in,fe_param_in,dof_index_in,projection_operator_data_in)
  {
    inverse_mass_matrix_operator.initialize(data_in,dof_index_in,quad_index_in);
  }
  unsigned int solve(parallel::distributed::BlockVector<double>       &dst,
                     const parallel::distributed::BlockVector<double> &src) const
  {
    inverse_mass_matrix_operator.apply_inverse_mass_matrix(src,dst);

    return 0;
  }
private:
  InverseMassMatrixOperator<dim,fe_degree,value_type> inverse_mass_matrix_operator;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class DirectProjectionSolverDivergencePenalty : public ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>
{
public:
  static const bool is_xwall = false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? n_q_points_1d_xwall : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  DirectProjectionSolverDivergencePenalty(MatrixFree<dim,value_type> const & data_in,
                                          FEParameters<value_type> & fe_param_in,
                                          const unsigned int dof_index_in,
                                          ProjectionOperatorData const projection_operator_data_in)
    :
    ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(data_in,fe_param_in,dof_index_in,projection_operator_data_in)
  {

  }

  unsigned int solve(parallel::distributed::BlockVector<double>       &dst,
                     const parallel::distributed::BlockVector<double> &src) const
  {
    dst = 0;

    this->get_data().cell_loop (&DirectProjectionSolverDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall,value_type>::local_solve, this, dst, src);

    return 0;
  }

  void local_solve(const MatrixFree<dim,value_type>                     &data,
                   parallel::distributed::BlockVector<value_type>       &dst,
                   const parallel::distributed::BlockVector<value_type> &src,
                   const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->get_fe_param(), this->get_dof_index());
    FEEval_Velocity_Velocity_linear fe_eval_phi(data,this->get_fe_param(), this->get_dof_index());

    std::vector<LAPACKFullMatrix<value_type> > matrices(VectorizedArray<value_type>::n_array_elements);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_phi.reinit(cell);
      const unsigned int total_dofs_per_cell = fe_eval_phi.dofs_per_cell * dim;

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        matrices[v].reinit(total_dofs_per_cell, total_dofs_per_cell);

      // div-div penalty parameter
      const VectorizedArray<value_type> tau = this->get_array_penalty_parameter_divergence()[cell];

      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<total_dofs_per_cell; ++i)
          fe_eval_phi.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval_phi.write_cellwise_dof_value(j,make_vectorized_array(1.));

        fe_eval_phi.evaluate (true,true,false);
        for (unsigned int q=0; q<fe_eval_phi.n_q_points; ++q)
        {
          const VectorizedArray<value_type> tau_times_div = tau * fe_eval_phi.get_divergence(q);
          Tensor<2,dim,VectorizedArray<value_type> > test;
          for (unsigned int d=0; d<dim; ++d)
            test[d][d] = tau_times_div;
          fe_eval_phi.submit_gradient(test, q);
          fe_eval_phi.submit_value (fe_eval_phi.get_value(q), q);
        }
        fe_eval_phi.integrate (true,true);

        for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
        {
          if(fe_eval_phi.component_enriched(v))
          {
            for (unsigned int i=0; i<total_dofs_per_cell; ++i)
              (matrices[v])(i,j) = (fe_eval_phi.read_cellwise_dof_value(i))[v];
          }
          else//this is a non-enriched element
          {
            if(j<fe_eval_phi.std_dofs_per_cell*dim)
              for (unsigned int i=0; i<fe_eval_phi.std_dofs_per_cell*dim; ++i)
                (matrices[v])(i,j) = (fe_eval_phi.read_cellwise_dof_value(i))[v];
            else //diagonal
              (matrices[v])(j,j) = 1.0;
          }
        }
      }

    //      for (unsigned int i=0; i<10; ++i)
    //        std::cout << std::endl;
    //      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
    //        matrices[v].print(std::cout,14,8);

      //now apply vectors to inverse matrix
    //    for (unsigned int q=0; q<fe_eval_phi.n_q_points; ++q)
    //    {
    //      fe_eval_velocity.submit_value (fe_eval_velocity.get_value(q), q);
    //    }
    //    fe_eval_velocity.integrate (true,false);

      fe_eval_velocity.reinit(cell);
//      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.read_dof_values(src,0);

      for (unsigned int v = 0; v < data.n_components_filled(cell); ++v)
      {
        (matrices[v]).compute_lu_factorization();
        Vector<value_type> vector_input(total_dofs_per_cell);
        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
          vector_input(j)=(fe_eval_velocity.read_cellwise_dof_value(j))[v];

    //        Vector<value_type> vector_result(total_dofs_per_cell);
        (matrices[v]).apply_lu_factorization(vector_input,false);
    //        (matrices[v]).vmult(vector_result,vector_input);
        for (unsigned int j=0; j<total_dofs_per_cell; ++j)
          fe_eval_velocity.write_cellwise_dof_value(j,vector_input(j),v);
      }
//      fe_eval_velocity.set_dof_values (dst,0,dim);
      fe_eval_velocity.set_dof_values (dst,0);
    }
  }
};

namespace internalCG
{
  template <typename Number, typename Number2>
  bool all_smaller (const Number a, const Number2 b)
  {
    return a<b;
  }
  template <typename Number, typename Number2>
  bool all_smaller (const VectorizedArray<Number> a, const Number2 b)
  {
    for (unsigned int i=0; i<VectorizedArray<Number>::n_array_elements; ++i)
      if (a[i] >= b)
        return false;
    return true;
  }
  template <typename Number>
  void adjust_division_by_zero (Number &)
  {}
  template <typename Number>
  void adjust_division_by_zero (VectorizedArray<Number> &x)
  {
    for (unsigned int i=0; i<VectorizedArray<Number>::n_array_elements; ++i)
      if (x[i] < 1e-30)
        x[i] = 1;
  }
}

template<typename value_type>
class SolverCGmod
{
public:
  SolverCGmod(const unsigned int unknowns,
              const double abs_tol=1.e-12,
              const double rel_tol=1.e-8,
              const unsigned int max_iter = 1e5);

  template <typename Matrix>
  void solve(const Matrix *matrix,  value_type *solution, const value_type *rhs);

private:
  const double ABS_TOL;
  const double REL_TOL;
  const unsigned int MAX_ITER;
  AlignedVector<value_type> storage;
  value_type *p,*r,*v;
  const unsigned int M;
  value_type l2_norm(const value_type *vector);

  void vector_init(value_type *dst);
  void equ(value_type *dst, const value_type scalar, const value_type *in_vector);
  void equ(value_type *dst, const value_type scalar1, const value_type *in_vector1, const value_type scalar2, const value_type *in_vector2);

  void add(value_type *dst, const value_type scalar, const value_type *in_vector);
  value_type inner_product(const value_type *vector1, const value_type *vector2);
};

template<typename value_type>
SolverCGmod<value_type>::SolverCGmod(const unsigned int unknowns,
                                     const double abs_tol,
                                     const double rel_tol,
                                     const unsigned int max_iter):
ABS_TOL(abs_tol),
REL_TOL(rel_tol),
MAX_ITER(max_iter),
M(unknowns)
{
  storage.resize(3*M);
  p = storage.begin();
  r = storage.begin()+M;
  v = storage.begin()+2*M;
}

template<typename value_type>
value_type SolverCGmod<value_type>::l2_norm(const value_type *vector)
{
  return std::sqrt(inner_product(vector, vector));
}

template<typename value_type>
void SolverCGmod<value_type>::vector_init(value_type *vector)
{
  for(unsigned int i=0;i<M;++i)
    vector[i] = 0.0;
}

template<typename value_type>
void SolverCGmod<value_type>::equ(value_type *dst, const value_type scalar, const value_type *in_vector)
{
  for(unsigned int i=0;i<M;++i)
    dst[i] = scalar*in_vector[i];
}

template<typename value_type>
void SolverCGmod<value_type>::equ(value_type *dst, const value_type scalar1, const value_type *in_vector1, const value_type scalar2, const value_type *in_vector2)
{
  for(unsigned int i=0;i<M;++i)
    dst[i] = scalar1*in_vector1[i]+scalar2*in_vector2[i];
}

template<typename value_type>
void SolverCGmod<value_type>::add(value_type *dst, const value_type scalar, const value_type *in_vector)
{
  for(unsigned int i=0;i<M;++i)
    dst[i] += scalar*in_vector[i];
}

template<typename value_type>
value_type SolverCGmod<value_type>::inner_product(const value_type *vector1, const value_type *vector2)
{
  value_type result = value_type();
  for(unsigned int i=0;i<M;++i)
    result += vector1[i]*vector2[i];

  return result;
}

template<typename value_type>
template<typename Matrix>
void SolverCGmod<value_type>::solve(const Matrix *matrix,
                                    value_type *solution,
                                    const value_type *rhs)
{
  value_type one;
  one = 1.0;

  // guess initial solution
  vector_init(solution);

  // apply matrix vector product: v = A*solution
  matrix->vmult(v,solution);

  // compute residual: r = rhs-A*solution
  equ(r,one,rhs,-one,v);
  value_type norm_r0 = l2_norm(r);

  // precondition
  matrix->precondition(p,r);

  // compute norm of residual
  value_type norm_r_abs = norm_r0;
  value_type norm_r_rel = one;
  value_type r_times_y = inner_product(p, r);

  unsigned int n_iter = 0;

  while(true)
  {
    // v = A*p
    (*matrix).vmult(v,p);

    // p_times_v = p^T*v
    value_type p_times_v = inner_product(p,v);
    internalCG::adjust_division_by_zero(p_times_v);

    // alpha = (r^T*y) / (p^T*v)
    value_type alpha = (r_times_y)/(p_times_v);

    // solution <- solution + alpha*p
    add(solution,alpha,p);

    // r <- r - alpha*v
    add(r,-alpha,v);

    // calculate residual norm
    norm_r_abs = l2_norm(r);
    norm_r_rel = norm_r_abs / norm_r0;

    // increment iteration counter
    ++n_iter;

    if (internalCG::all_smaller(norm_r_abs, ABS_TOL) ||
        internalCG::all_smaller(norm_r_rel, REL_TOL) || (n_iter > MAX_ITER))
      break;

    // precondition
    matrix->precondition(v,r);

    value_type r_times_y_new = inner_product(r,v);

    // beta = (v^T*r) / (p^T*v)
    value_type beta = r_times_y_new / r_times_y;

    // p <- r - beta*p
    equ(p,one,v,beta,p);

    r_times_y = r_times_y_new;
  }

  std::ostringstream message;
  for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; v++)
     message << " v: " << v << "  " << norm_r_abs[v] << " ";
  Assert(n_iter <= MAX_ITER,
         ExcMessage("No convergence of solver in " + Utilities::to_string(MAX_ITER)
                    + "iterations. Residual was " + message.str().c_str()));
}

template<int dim, int fe_degree>
class MatrixProjectionDivergencePenalty
{
public:
  typedef FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> EvalType;

  MatrixProjectionDivergencePenalty(const MatrixFree<dim,double> &matrix_free,
                                    const unsigned int fe_no,
                                    const unsigned int quad_no)
    :
    fe_eval(matrix_free, fe_no, quad_no),
    inverse(fe_eval)
  {
    coefficients.resize(fe_eval.n_q_points);
  }

  void setup(const unsigned int cell,
             const VectorizedArray<double> tau_grad_div_stab)
  {
    this->tau = tau_grad_div_stab;
    fe_eval.reinit(cell);
    inverse.fill_inverse_JxW_values(coefficients);
  }

  void precondition(VectorizedArray<double> *dst,
                    const VectorizedArray<double> *src) const
  {
    inverse.apply(coefficients, dim, src, dst);
  }

  void vmult(VectorizedArray<double> *dst,
             VectorizedArray<double> *src) const
  {
    Assert(fe_eval.get_shape_info().element_type <=
        dealii::internal::MatrixFreeFunctions::tensor_symmetric,
           ExcNotImplemented());

    // get internal evaluator in order to avoid copying data around
    dealii::internal::FEEvaluationImpl<dealii::internal::MatrixFreeFunctions::tensor_symmetric,
                                       dim, fe_degree, fe_degree+1, double> evaluator;
    VectorizedArray<double> *unit_values[dim], *unit_gradients[dim][dim],
      *unit_hessians[dim][dim*(dim+1)/2];
    for (unsigned int c=0; c<dim; ++c)
    {
      unit_values[c] = &fe_eval.begin_values()[c*fe_eval.n_q_points];
      for (unsigned int d=0; d<dim; ++d)
        unit_gradients[c][d] = &fe_eval.begin_gradients()[(c*dim+d)*fe_eval.n_q_points];
      for (unsigned int d=0; d<dim*(dim+1)/2; ++d)
        unit_hessians[c][d] = 0;
    }

    // compute matrix vector product on element
    for (unsigned int c=0; c<dim; ++c)
      evaluator.evaluate(fe_eval.get_shape_info(),
                         &src[c*fe_eval.dofs_per_cell], unit_values[c],
                         unit_gradients[c], unit_hessians[c],
                         true, true, false);
    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      VectorizedArray<double> tau_times_div = tau * fe_eval.get_divergence(q);
      fe_eval.submit_divergence(tau_times_div, q);
      fe_eval.submit_value (fe_eval.get_value(q), q);
    }
    for (unsigned int c=0; c<dim; ++c)
      evaluator.integrate(fe_eval.get_shape_info(),
                         &dst[c*fe_eval.dofs_per_cell], unit_values[c],
                         unit_gradients[c],
                         true, true);
  }

private:
  mutable FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> fe_eval;
  VectorizedArray<double> tau;
  AlignedVector<VectorizedArray<double> > coefficients;
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,dim,double> inverse;
};

struct ProjectionSolverData
{
  ProjectionSolverData()
    :
    max_iter(1e4),
    solver_tolerance_abs(1.e-12),
    solver_tolerance_rel(1.e-6),
    solver_projection(SolverProjection::PCG),
    preconditioner_projection(PreconditionerProjection::None)
  {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  SolverProjection solver_projection;
  PreconditionerProjection preconditioner_projection;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall,typename value_type>
class IterativeProjectionSolverDivergencePenalty : public ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall,value_type>
{
public:
  IterativeProjectionSolverDivergencePenalty(MatrixFree<dim,value_type> const & data_in,
                                             FEParameters<value_type> & fe_param_in,
                                             const unsigned int dof_index_in,
                                             const unsigned int quad_index_in,
                                             ProjectionOperatorData const projection_operator_data_in,
                                             ProjectionSolverData const solver_data_in)
    :
    ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(data_in,fe_param_in,dof_index_in,projection_operator_data_in),
    quad_index(quad_index_in),
    solver_data(solver_data_in)
  {}

  unsigned int solve(parallel::distributed::BlockVector<double>       &dst,
                     const parallel::distributed::BlockVector<double> &src) const
  {
    dst = 0;

    this->get_data().cell_loop (&IterativeProjectionSolverDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall,value_type>::local_solve, this, dst, src);

    return 0;
  }

  void local_solve(const MatrixFree<dim,value_type>                     &data,
                   parallel::distributed::BlockVector<value_type>       &dst,
                   const parallel::distributed::BlockVector<value_type> &src,
                   const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
#ifdef XWALL
    AssertThrow(false,
                ExcMessage("XWall should not arrive in iterative projection solver"));
#endif

    AssertThrow(solver_data.preconditioner_projection == PreconditionerProjection::InverseMassMatrix,
                        ExcMessage("Specified preconditioner is not available for projection_type = ProjectionType::DivergencePenalty and solver_projection = SolverProjection::PCG"));

    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type>
      fe_eval(data,this->get_dof_index(),quad_index);

    const unsigned int total_dofs_per_cell = fe_eval.dofs_per_cell * dim;

    AlignedVector<VectorizedArray<value_type> > solution(total_dofs_per_cell);
    MatrixProjectionDivergencePenalty<dim,fe_degree> matrix_projection_step(data,this->get_dof_index(),quad_index);
    SolverCGmod<VectorizedArray<double> > cg_solver(total_dofs_per_cell, solver_data.solver_tolerance_abs, solver_data.solver_tolerance_rel, solver_data.max_iter);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src,0);

      // divergence penalty parameter
      const VectorizedArray<value_type> tau = this->get_array_penalty_parameter_divergence()[cell];

      matrix_projection_step.setup(cell, tau);
      cg_solver.solve(&matrix_projection_step, solution.begin(), fe_eval.begin_dof_values());

      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = solution[j];
      fe_eval.set_dof_values (dst,0);
    }
  }

private:
  unsigned int quad_index;
  ProjectionSolverData const solver_data;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class ProjectionOperatorDivergenceAndContinuityPenalty : public Subscriptor
{
public:
  static const bool is_xwall = false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? n_q_points_1d_xwall : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;

  ProjectionOperatorDivergenceAndContinuityPenalty(
      ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> * ptr_proj_solver_in,
      MatrixFree<dim,value_type> const & data_in,
      FEParameters<value_type> & fe_param_in,
      const unsigned int dof_index_in)
    :
    ptr_proj_solver(ptr_proj_solver_in),
    data(data_in),
    fe_param(fe_param_in),
    dof_index(dof_index_in)
  {

  }

  void vmult (parallel::distributed::BlockVector<double>       &dst,
              const parallel::distributed::BlockVector<double> &src) const
  {
    apply_projection(src,dst);
  }

  void calculate_inverse_diagonal(parallel::distributed::BlockVector<value_type> &diagonal) const
  {
    diagonal = 0;

    parallel::distributed::BlockVector<value_type>  src_dummy(diagonal);

    data.loop(&ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::local_calculate_diagonal,
              &ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::local_calculate_diagonal_face,
              &ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>::local_calculate_diagonal_boundary_face,
              this, diagonal, src_dummy);

    // verify calculation of diagonal
//    parallel::distributed::BlockVector<value_type>  diagonal2(diagonal);
//    for(unsigned int d=0;d<diagonal2.n_blocks();++d)
//      diagonal2.block(d) = 0.0;
//    parallel::distributed::BlockVector<value_type>  src(diagonal2);
//    parallel::distributed::BlockVector<value_type>  dst(diagonal2);
//    for(unsigned int d=0;d<diagonal2.n_blocks();++d)
//      for (unsigned int i=0;i<diagonal.block(d).local_size();++i)
//      {
//        src.block(d).local_element(i) = 1.0;
//        apply_projection(src,dst);
//        diagonal2.block(d).local_element(i) = dst.block(d).local_element(i);
//        src.block(d).local_element(i) = 0.0;
//      }
//    //diagonal2.block(0).print(std::cout);
//
//    std::cout<<"L2 norm diagonal: "<<diagonal.l2_norm()<<std::endl;
//    std::cout<<"L2 norm diagonal2: "<<diagonal2.l2_norm()<<std::endl;
//    for(unsigned int d=0;d<diagonal2.n_blocks();++d)
//      diagonal2.block(d).add(-1.0,diagonal.block(d));
//    std::cout<<"L2 error diagonal: "<<diagonal2.l2_norm()<<std::endl;
    // verify calculation of diagonal

    //invert diagonal
    for(unsigned int d=0;d<diagonal.n_blocks();++d)
    {
      for (unsigned int i=0;i<diagonal.block(d).local_size();++i)
      {
        if( std::abs(diagonal.block(d).local_element(i)) > 1.0e-10 )
          diagonal.block(d).local_element(i) = 1.0/diagonal.block(d).local_element(i);
        else
          diagonal.block(d).local_element(i) = 1.0;
      }
    }
  }

private:

  void apply_projection (const parallel::distributed::BlockVector<value_type> &src,
                         parallel::distributed::BlockVector<value_type>       &dst) const
  {
    for(unsigned int d=0;d<dim;++d)
    {
      dst.block(d)=0;
    }
    data.loop(&ProjectionOperatorDivergenceAndContinuityPenalty::local_apply_projection,
              &ProjectionOperatorDivergenceAndContinuityPenalty::local_apply_projection_face,
              &ProjectionOperatorDivergenceAndContinuityPenalty::local_apply_projection_boundary_face,
              this, dst, src);
  }

  void local_apply_projection (const MatrixFree<dim,value_type>                 &data,
                               parallel::distributed::BlockVector<double>       &dst,
                               const parallel::distributed::BlockVector<double> &src,
                               const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate (true,true,false);

      VectorizedArray<value_type> tau = ptr_proj_solver->get_array_penalty_parameter_divergence()[cell];

      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > velocity = fe_eval_velocity.get_value(q);
        VectorizedArray<value_type > divergence = fe_eval_velocity.get_divergence(q);
        Tensor<2,dim,VectorizedArray<value_type> > unit_times_divU;
          for (unsigned int d=0; d<dim; ++d)
            unit_times_divU[d][d] = divergence;
        fe_eval_velocity.submit_value(velocity, q);
        fe_eval_velocity.submit_gradient(tau*unit_times_divU, q);
      }
      fe_eval_velocity.integrate (true,true);
      fe_eval_velocity.distribute_local_to_global (dst,0,dim);
    }
  }

  void local_apply_projection_face (const MatrixFree<dim,value_type>                  &data,
                                    parallel::distributed::BlockVector<double>        &dst,
                                    const parallel::distributed::BlockVector<double>  &src,
                                    const std::pair<unsigned int,unsigned int>        &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,true,dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_neighbor(data,fe_param,false,dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity.read_dof_values(src,0,dim);
      fe_eval_velocity.evaluate(true,false);
      fe_eval_velocity_neighbor.reinit (face);
      fe_eval_velocity_neighbor.read_dof_values(src,0,dim);
      fe_eval_velocity_neighbor.evaluate(true,false);

      VectorizedArray<value_type> tau = 0.5*(fe_eval_velocity.read_cell_data(ptr_proj_solver->get_array_penalty_parameter_continuity())
          +fe_eval_velocity_neighbor.read_cell_data(ptr_proj_solver->get_array_penalty_parameter_continuity()));

      for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_velocity_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

        fe_eval_velocity.submit_value(tau*jump_value,q);
        fe_eval_velocity_neighbor.submit_value(-tau*jump_value,q);
      }
      fe_eval_velocity.integrate(true,false);
      fe_eval_velocity.distribute_local_to_global(dst,0,dim);
      fe_eval_velocity_neighbor.integrate(true,false);
      fe_eval_velocity_neighbor.distribute_local_to_global(dst,0,dim);
    }
  }

  void local_apply_projection_boundary_face (const MatrixFree<dim,value_type>                 &,
                                             parallel::distributed::BlockVector<double>       &,
                                             const parallel::distributed::BlockVector<double> &,
                                             const std::pair<unsigned int,unsigned int>       &) const
  {

  }

  void local_calculate_diagonal (const MatrixFree<dim,value_type>                 &data,
                                 parallel::distributed::BlockVector<double>       &dst,
                                 const parallel::distributed::BlockVector<double> &src,
                                 const std::pair<unsigned int,unsigned int>       &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,dof_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval_velocity.reinit(cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval_velocity.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
      {
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval_velocity.write_cellwise_dof_value(j,make_vectorized_array(1.));

        // copied from local_apply_projection
        fe_eval_velocity.evaluate (true,true,false);

        VectorizedArray<value_type> tau = ptr_proj_solver->get_array_penalty_parameter_divergence()[cell];

        for (unsigned int q=0; q<fe_eval_velocity.n_q_points; ++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > velocity = fe_eval_velocity.get_value(q);
          VectorizedArray<value_type > divergence = fe_eval_velocity.get_divergence(q);
          Tensor<2,dim,VectorizedArray<value_type> > unit_times_divU;
            for (unsigned int d=0; d<dim; ++d)
              unit_times_divU[d][d] = divergence;
          fe_eval_velocity.submit_value(velocity, q);
          fe_eval_velocity.submit_gradient(tau*unit_times_divU, q);
        }
        // copied from local_apply_projection
        fe_eval_velocity.integrate (true,true);
        local_diagonal_vector[j] = fe_eval_velocity.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
        fe_eval_velocity.write_cellwise_dof_value(j,local_diagonal_vector[j]);
      fe_eval_velocity.distribute_local_to_global (dst);
    }
  }

  void local_calculate_diagonal_face (const MatrixFree<dim,value_type>                  &data,
                                      parallel::distributed::BlockVector<double>        &dst,
                                      const parallel::distributed::BlockVector<double>  &src,
                                      const std::pair<unsigned int,unsigned int>        &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity(data,fe_param,true,dof_index);
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_neighbor(data,fe_param,false,dof_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval_velocity.reinit (face);
      fe_eval_velocity_neighbor.reinit (face);

      // element-
      VectorizedArray<value_type> local_diagonal_vector[fe_eval_velocity.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval_velocity.write_cellwise_dof_value(j,make_vectorized_array(1.));
        // set all dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_velocity_neighbor.write_cellwise_dof_value(i, make_vectorized_array(0.));

        // copied from local_apply_projection_face (note that fe_eval_neighbor.submit... has to be removed)
        fe_eval_velocity.evaluate(true,false);
        fe_eval_velocity_neighbor.evaluate(true,false);

        VectorizedArray<value_type> tau = 0.5*(fe_eval_velocity.read_cell_data(ptr_proj_solver->get_array_penalty_parameter_continuity())
            +fe_eval_velocity_neighbor.read_cell_data(ptr_proj_solver->get_array_penalty_parameter_continuity()));

        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_velocity_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

          fe_eval_velocity.submit_value(tau*jump_value,q);
        }
        // copied from local_apply_projection_face (note that fe_eval_neighbor.submit... has to be removed)

        // integrate on element-
        fe_eval_velocity.integrate(true,false);
        local_diagonal_vector[j] = fe_eval_velocity.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity.dofs_per_cell*dim; ++j)
        fe_eval_velocity.write_cellwise_dof_value(j, local_diagonal_vector[j]);
      fe_eval_velocity.distribute_local_to_global(dst);

      // neighbor (element+)
      VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_velocity_neighbor.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++j)
      {
        // set all dof values of element- to zero
        for (unsigned int i=0; i<fe_eval_velocity.dofs_per_cell*dim; ++i)
          fe_eval_velocity.write_cellwise_dof_value(i,make_vectorized_array(0.));
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_velocity_neighbor.write_cellwise_dof_value(i, make_vectorized_array(0.));
        fe_eval_velocity_neighbor.write_cellwise_dof_value(j,make_vectorized_array(1.));

        // copied from local_apply_projection_face (note that fe_eval.submit... has to be removed)
        fe_eval_velocity.evaluate(true,false);
        fe_eval_velocity_neighbor.evaluate(true,false);

        VectorizedArray<value_type> tau = 0.5*(fe_eval_velocity.read_cell_data(ptr_proj_solver->get_array_penalty_parameter_continuity())
            +fe_eval_velocity_neighbor.read_cell_data(ptr_proj_solver->get_array_penalty_parameter_continuity()));

        for(unsigned int q=0;q<fe_eval_velocity.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval_velocity.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_velocity_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

          fe_eval_velocity_neighbor.submit_value(-tau*jump_value,q);
        }
        // copied from local_apply_projection_face (note that fe_eval.submit... has to be removed)

        // integrate on element+
        fe_eval_velocity_neighbor.integrate(true,false);
        local_diagonal_vector_neighbor[j] = fe_eval_velocity_neighbor.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_velocity_neighbor.dofs_per_cell*dim; ++j)
        fe_eval_velocity_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);
      fe_eval_velocity_neighbor.distribute_local_to_global(dst);
    }
  }

  void local_calculate_diagonal_boundary_face (const MatrixFree<dim,value_type>                 &,
                                               parallel::distributed::BlockVector<double>       &,
                                               const parallel::distributed::BlockVector<double> &,
                                               const std::pair<unsigned int,unsigned int>       &) const
  {

  }

  ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> * ptr_proj_solver;
  MatrixFree<dim,value_type> const & data;
  FEParameters<value_type> const & fe_param;
  unsigned int dof_index;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
class IterativeProjectionSolverDivergenceAndContinuityPenalty : public ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>
{
public:
  IterativeProjectionSolverDivergenceAndContinuityPenalty(
      MatrixFree<dim,value_type> const & data_in,
      FEParameters<value_type> & fe_param_in,
      const unsigned int dof_index_in,
      const unsigned int quad_index_in,
      ProjectionOperatorData const projection_operator_data_in,
      ProjectionSolverData const solver_data_in)
    :
    ProjectionSolverBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(data_in,fe_param_in,dof_index_in,projection_operator_data_in),
    solver_data(solver_data_in),
    global_matrix(this,data_in,fe_param_in,dof_index_in),
    preconditioner(nullptr)
  {
    if(solver_data.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
      preconditioner = new InverseMassMatrixPreconditionerVelocity<dim,fe_degree,value_type>(data_in,dof_index_in,quad_index_in);
    else if(solver_data.preconditioner_projection == PreconditionerProjection::Jacobi)
      preconditioner = new JacobiPreconditionerVelocity<dim,value_type,ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> >
                                (data_in,dof_index_in,global_matrix);
  }

  ~IterativeProjectionSolverDivergenceAndContinuityPenalty()
  {
    delete preconditioner;
  }

  unsigned int solve(parallel::distributed::BlockVector<double>       &dst,
                     const parallel::distributed::BlockVector<double> &src) const
  {
    ReductionControl solver_control (solver_data.max_iter,
                                     solver_data.solver_tolerance_abs,
                                     solver_data.solver_tolerance_rel);

    try
    {
      if(solver_data.solver_projection == SolverProjection::PCG)
      {
        SolverCG<parallel::distributed::BlockVector<double> > solver (solver_control);
        if(solver_data.preconditioner_projection == PreconditionerProjection::None)
          solver.solve (global_matrix, dst, src, PreconditionIdentity());
        else if(solver_data.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
          solver.solve (global_matrix, dst, src, *preconditioner);
        else if(solver_data.preconditioner_projection == PreconditionerProjection::Jacobi)
        {
          // recalculate diagonal since the diagonal depends on the penalty parameter which itself depends on
          // the velocity field
          JacobiPreconditionerVelocity<dim,value_type,ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> >
            *jacobi_preconditioner = dynamic_cast<JacobiPreconditionerVelocity<dim,value_type,ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> > *>(preconditioner);
          jacobi_preconditioner->recalculate_diagonal(global_matrix);
          solver.solve (global_matrix, dst, src, *preconditioner);
        }
        else
          AssertThrow(solver_data.preconditioner_projection == PreconditionerProjection::None ||
                      solver_data.preconditioner_projection == PreconditionerProjection::InverseMassMatrix ||
                      solver_data.preconditioner_projection == PreconditionerProjection::Jacobi,
                      ExcMessage("Specified preconditioner of projection solver not available for projection_type = ProjectionType::DivergenceAndContinuityPenalty."));
      }
      else
        AssertThrow(solver_data.solver_projection == SolverProjection::PCG,
            ExcMessage("Specified projection solver not available for projection_type = ProjectionType::DivergenceAndContinuityPenalty."));
    }
    catch (SolverControl::NoConvergence &)
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
        std::cout<<"Viscous solver failed to solve to given tolerance." << std::endl;
    }
    return solver_control.last_step();
  }

private:
  ProjectionSolverData const solver_data;
  ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type> global_matrix;
  PreconditionerVelocityBase *preconditioner;
};


#endif /* INCLUDE_PROJECTIONSOLVER_H_ */
