/*
 * ProjectionSolver.h
 *
 *  Created on: Jun 17, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_PROJECTIONSOLVER_H_
#define INCLUDE_PROJECTIONSOLVER_H_

#include <deal.II/base/vectorization.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>

#include "JacobiPreconditioner.h"
#include "InverseMassMatrixPreconditioner.h"

#include "InverseMassMatrixXWall.h"
#include "BaseOperator.h"

#include "VerifyCalculationOfDiagonal.h"
#include "InvertDiagonal.h"

#include "InternalSolvers.h"

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule> class NavierStokesOperation;

struct ProjectionOperatorData
{
  double  penalty_parameter_divergence;
  double  penalty_parameter_continuity;
  bool solve_stokes_equations;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorBase: public BaseOperator<dim>
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  ProjectionOperatorBase(MatrixFree<dim,value_type> const &data_in,
                         const unsigned int               dof_index_in,
                         const unsigned int               quad_index_in,
                         ProjectionOperatorData const     projection_operator_data_in)
    :
    data(data_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    array_penalty_parameter_divergence(0),
    array_penalty_parameter_continuity(0),
    projection_operator_data(projection_operator_data_in)
  {
    array_penalty_parameter_divergence.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
    array_penalty_parameter_continuity.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
  }

  void calculate_array_penalty_parameter(parallel::distributed::Vector<value_type> const &velocity_n,
                                         double const                                    cfl,
                                         double const                                    time_step)
  {
    velocity_n.update_ghost_values();

    FEEval_Velocity_Velocity_linear fe_eval(data,this->fe_param,dof_index);

    AlignedVector<VectorizedArray<value_type> > JxW_values(fe_eval.n_q_points);

    for (unsigned int cell=0; cell<data.n_macro_cells()+data.n_macro_ghost_cells(); ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(velocity_n);
      fe_eval.evaluate (true,false);
      VectorizedArray<value_type> volume = make_vectorized_array<value_type>(0.0);
      Tensor<1,dim,VectorizedArray<value_type> > U_mean;
      VectorizedArray<value_type> norm_U_mean;
      JxW_values.resize(fe_eval.n_q_points);
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
//        array_penalty_parameter_divergence[cell] = projection_operator_data.penalty_parameter_divergence * norm_U_mean * std::exp(std::log(volume)/(double)dim) * time_step;
//        array_penalty_parameter_continuity[cell] = projection_operator_data.penalty_parameter_continuity * norm_U_mean * time_step;

        array_penalty_parameter_divergence[cell] = projection_operator_data.penalty_parameter_divergence * norm_U_mean * std::exp(std::log(volume)/(double)dim);
        array_penalty_parameter_continuity[cell] = projection_operator_data.penalty_parameter_continuity * norm_U_mean;
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
  unsigned int get_quad_index() const
  {
    return quad_index;
  }
  FEParameters<dim> const * get_fe_param() const
  {
    return this->fe_param;
  }

private:
  MatrixFree<dim,value_type> const & data;
  unsigned int const dof_index;
  unsigned int const quad_index;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter_divergence;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter_continuity;
  ProjectionOperatorData projection_operator_data;
};


struct DivAndContiPenaltyOperatorData
{
  DivAndContiPenaltyOperatorData()
    :
    div_div_penalty(false),
    continuity_penalty(false),
    penalty_parameter_divergence(1.0),
    penalty_parameter_continuity(1.0)
  {}

  bool div_div_penalty;
  bool continuity_penalty;
  double penalty_parameter_divergence;
  double penalty_parameter_continuity;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class DivergenceAndContinuityPenaltyOperator : public BaseOperator<dim>
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;

  typedef DivergenceAndContinuityPenaltyOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> This;

  DivergenceAndContinuityPenaltyOperator(MatrixFree<dim,value_type> const     &data_in,
                                         unsigned int const                   dof_index_in,
                                         unsigned int const                   quad_index_in,
                                         DivAndContiPenaltyOperatorData const operator_data_in)
    :
    data(data_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    array_penalty_parameter_divergence(0),
    array_penalty_parameter_continuity(0),
    operator_data(operator_data_in)
  {
    array_penalty_parameter_divergence.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
    array_penalty_parameter_continuity.resize(data.n_macro_cells()+data.n_macro_ghost_cells());
  }

  void calculate_array_penalty_parameter(parallel::distributed::Vector<value_type> const &velocity,
                                         double const                                    time_step = 1.0)
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
      Tensor<1,dim,VectorizedArray<value_type> > U_mean;
      VectorizedArray<value_type> norm_U_mean;
      JxW_values.resize(fe_eval.n_q_points);
      fe_eval.fill_JxW_values(JxW_values);
      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        volume += JxW_values[q];
        U_mean += JxW_values[q]*fe_eval.get_value(q);
      }
      U_mean /= volume;
      norm_U_mean = U_mean.norm();

      array_penalty_parameter_divergence[cell] = operator_data.penalty_parameter_divergence * norm_U_mean * std::exp(std::log(volume)/(double)dim);// * time_step;
      array_penalty_parameter_continuity[cell] = operator_data.penalty_parameter_continuity * norm_U_mean; // * time_step;
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
    apply(dst,src);
  }

  void apply (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const
  {
    dst = 0;

    apply_add(dst,src);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  parallel::distributed::Vector<value_type> const &src) const
  {
    this->get_data().loop(&This::cell_loop,&This::face_loop,
                          &This::boundary_face_loop,this, dst, src);
  }

private:
  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation       &fe_eval,
                               unsigned int const cell) const
  {
    fe_eval.evaluate (false,true,false);

    VectorizedArray<value_type> tau = this->get_array_penalty_parameter_divergence()[cell];

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      VectorizedArray<value_type > divergence = fe_eval.get_divergence(q);
      Tensor<2,dim,VectorizedArray<value_type> > unit_times_divU;
      for (unsigned int d=0; d<dim; ++d)
      {
        unit_times_divU[d][d] = divergence;
      }
      fe_eval.submit_gradient(tau*unit_times_divU, q);
    }

    fe_eval.integrate (false,true);
  }

  void cell_loop (const MatrixFree<dim,value_type>                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    if(operator_data.div_div_penalty == true)
    {
      FEEval_Velocity_Velocity_linear fe_eval(data,this->get_fe_param(),this->get_dof_index());

      for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
      {
        fe_eval.reinit(cell);
        fe_eval.read_dof_values(src);

        do_cell_integral(fe_eval,cell);

        fe_eval.distribute_local_to_global (dst);
      }
    }
  }

  void face_loop (const MatrixFree<dim,value_type>                 &data,
                  parallel::distributed::Vector<value_type>        &dst,
                  const parallel::distributed::Vector<value_type>  &src,
                  const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    if(operator_data.continuity_penalty == true)
    {
      FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->get_fe_param(),true,this->get_dof_index());
      FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,this->get_fe_param(),false,this->get_dof_index());

      for(unsigned int face=face_range.first; face<face_range.second; face++)
      {
        fe_eval.reinit (face);
        fe_eval_neighbor.reinit (face);

        fe_eval.read_dof_values(src);
        fe_eval_neighbor.read_dof_values(src);

        fe_eval.evaluate(true,false);
        fe_eval_neighbor.evaluate(true,false);

        VectorizedArray<value_type> tau = 0.5*(fe_eval.read_cell_data(this->get_array_penalty_parameter_continuity())
                                               + fe_eval_neighbor.read_cell_data(this->get_array_penalty_parameter_continuity()));

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

          fe_eval.submit_value(tau*jump_value,q);
          fe_eval_neighbor.submit_value(-tau*jump_value,q);
        }
        fe_eval.integrate(true,false);
        fe_eval_neighbor.integrate(true,false);

        fe_eval.distribute_local_to_global(dst);
        fe_eval_neighbor.distribute_local_to_global(dst);
      }
    }
  }

  void boundary_face_loop (const MatrixFree<dim,value_type>                &,
                           parallel::distributed::Vector<value_type>       &,
                           const parallel::distributed::Vector<value_type> &,
                           const std::pair<unsigned int,unsigned int>      &) const
  {
    // no continuity penalty on boundary faces
  }

  MatrixFree<dim,value_type> const & data;
  unsigned int const dof_index;
  unsigned int const quad_index;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter_divergence;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter_continuity;
  DivAndContiPenaltyOperatorData operator_data;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorDivergencePenalty : public ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
{
public:
  typedef FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> EvalType;

  ProjectionOperatorDivergencePenalty(MatrixFree<dim,value_type> const &data_in,
                                      unsigned int const               dof_index_in,
                                      unsigned int const               quad_index_in,
                                      ProjectionOperatorData const     projection_operator_data_in)
    :
    ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>(data_in, dof_index_in,quad_index_in,projection_operator_data_in),
    fe_eval(1,FEEvaluation<dim,fe_degree,fe_degree+1,dim,double>(data_in, dof_index_in, quad_index_in)),
    inverse(fe_eval[0]),
    tau(1)
  {
    coefficients.resize(fe_eval[0].n_q_points);
  }

  void setup(const unsigned int cell)
  {
    tau[0] = this->get_array_penalty_parameter_divergence()[cell];
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
      fe_eval[0].submit_divergence(tau_times_div, q);
      fe_eval[0].submit_value (fe_eval[0].get_value(q), q);
    }

    fe_eval[0].integrate(true, true, dst);
  }

private:
  mutable AlignedVector<FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> > fe_eval;
  AlignedVector<VectorizedArray<double> > coefficients;
  MatrixFreeOperators::CellwiseInverseMassMatrix<dim,fe_degree,dim,double> inverse;
  AlignedVector<VectorizedArray<double> > tau;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorDivergencePenaltyXWall : public ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  ProjectionOperatorDivergencePenaltyXWall(MatrixFree<dim,value_type> const & data_in,
                                           FEParameters<dim> * fe_param_in,
                                           const unsigned int dof_index_in,
                                           const unsigned int quad_index_in,
                                           ProjectionOperatorData const projection_operator_data_in,
                                           std_cxx11::shared_ptr< InverseMassMatrixXWallOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type> > inv_mass_xw)
    :
    ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>(data_in, dof_index_in,quad_index_in,projection_operator_data_in),
    fe_eval(1,FEEval_Velocity_Velocity_linear(data_in, fe_param_in, dof_index_in, quad_index_in)),
    inverse_mass_matrix_operator_xwall(inv_mass_xw),
    tau(1),
    curr_cell(0)
  {
  }

  void setup(const unsigned int cell)
  {
    curr_cell = cell;
    tau[0] = this->get_array_penalty_parameter_divergence()[cell];
    fe_eval[0].reinit(cell);
  }

  void precondition(VectorizedArray<double>       *dst,
                    const VectorizedArray<double> *src) const
  {
    inverse_mass_matrix_operator_xwall->local_apply_inverse_mass_matrix(curr_cell, dst, src);
  }

  void vmult(VectorizedArray<double> *dst,
             VectorizedArray<double> *src) const
  {
    Assert(fe_eval[0].get_shape_info().element_type <=
        dealii::internal::MatrixFreeFunctions::tensor_symmetric,
           ExcNotImplemented());


    // compute matrix vector product on element
    fe_eval[0].evaluate(src, true, true);

    for (unsigned int q=0; q<fe_eval[0].n_q_points; ++q)
    {
      VectorizedArray<double> tau_times_div = tau[0] * fe_eval[0].get_divergence(q);
      Tensor<2,dim,VectorizedArray<value_type> > unit_times_divU;
        for (unsigned int d=0; d<dim; ++d)
          unit_times_divU[d][d] = tau_times_div;
        fe_eval[0].submit_gradient(unit_times_divU, q);
      fe_eval[0].submit_value (fe_eval[0].get_value(q), q);
    }

    fe_eval[0].integrate(true, true, dst);
  }

private:
  mutable AlignedVector<FEEval_Velocity_Velocity_linear> fe_eval;
  std_cxx11::shared_ptr< InverseMassMatrixXWallOperator<dim,fe_degree,fe_degree_xwall,xwall_quad_rule,value_type> > inverse_mass_matrix_operator_xwall;
  AlignedVector<VectorizedArray<double> > tau;
  unsigned int curr_cell;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class ProjectionOperatorDivergenceAndContinuityPenalty : public ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  typedef FEFaceEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEFaceEval_Velocity_Velocity_linear;

  typedef ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> This;

  ProjectionOperatorDivergenceAndContinuityPenalty(MatrixFree<dim,value_type> const &data_in,
                                                   unsigned int const               dof_index_in,
                                                   unsigned int const               quad_index_in,
                                                   ProjectionOperatorData const     projection_operator_data_in)
    :
    ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type>(data_in, dof_index_in,quad_index_in,projection_operator_data_in)
  {}

  void vmult (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const
  {
    dst = 0;

    this->get_data().loop(&This::cell_loop,&This::face_loop,
                          &This::boundary_face_loop,this, dst, src);
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    diagonal = 0;

    parallel::distributed::Vector<value_type>  src_dummy(diagonal);

    this->get_data().loop(&This::cell_loop_diagonal,&This::face_loop_diagonal,
                          &This::boundary_face_loop_diagonal,this, diagonal, src_dummy);
  }

  void calculate_inverse_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    calculate_diagonal(diagonal);

    // verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  void initialize_dof_vector(parallel::distributed::Vector<value_type> &diagonal) const
  {
    this->get_data().initialize_dof_vector(diagonal,this->get_dof_index());
  }

private:
  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation       &fe_eval,
                               unsigned int const cell) const
  {
    fe_eval.evaluate (true,true,false);

    VectorizedArray<value_type> tau = this->get_array_penalty_parameter_divergence()[cell];

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > velocity = fe_eval.get_value(q);
      VectorizedArray<value_type > divergence = fe_eval.get_divergence(q);
      Tensor<2,dim,VectorizedArray<value_type> > unit_times_divU;
      for (unsigned int d=0; d<dim; ++d)
      {
        unit_times_divU[d][d] = divergence;
      }
      fe_eval.submit_value(velocity, q);
      fe_eval.submit_gradient(tau*unit_times_divU, q);
    }
    fe_eval.integrate (true,true);
  }

  void cell_loop (const MatrixFree<dim,value_type>                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  const parallel::distributed::Vector<value_type> &src,
                  const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->get_fe_param(),this->get_dof_index());

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral(fe_eval,cell);

      fe_eval.distribute_local_to_global (dst);
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

      fe_eval.read_dof_values(src);
      fe_eval_neighbor.read_dof_values(src);

      fe_eval.evaluate(true,false);
      fe_eval_neighbor.evaluate(true,false);

      VectorizedArray<value_type> tau = 0.5*(fe_eval.read_cell_data(this->get_array_penalty_parameter_continuity())
                                             + fe_eval_neighbor.read_cell_data(this->get_array_penalty_parameter_continuity()));

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

        fe_eval.submit_value(tau*jump_value,q);
        fe_eval_neighbor.submit_value(-tau*jump_value,q);
      }
      fe_eval.integrate(true,false);
      fe_eval_neighbor.integrate(true,false);

      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop (const MatrixFree<dim,value_type>                &,
                           parallel::distributed::Vector<value_type>       &,
                           const parallel::distributed::Vector<value_type> &,
                           const std::pair<unsigned int,unsigned int>      &) const
  {

  }

  void cell_loop_diagonal (const MatrixFree<dim,value_type>                &data,
                           parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &,
                           const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    FEEval_Velocity_Velocity_linear fe_eval(data,this->get_fe_param(),this->get_dof_index());

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array(1.));

        do_cell_integral(fe_eval,cell);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j,local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop_diagonal (const MatrixFree<dim,value_type>                 &data,
                           parallel::distributed::Vector<value_type>        &dst,
                           const parallel::distributed::Vector<value_type>  &,
                           const std::pair<unsigned int,unsigned int>       &face_range) const
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval(data,this->get_fe_param(),true,this->get_dof_index());
    FEFaceEval_Velocity_Velocity_linear fe_eval_neighbor(data,this->get_fe_param(),false,this->get_dof_index());

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      // element-
      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell*dim; ++i)
          fe_eval.write_cellwise_dof_value(i,make_vectorized_array(0.));
        fe_eval.write_cellwise_dof_value(j,make_vectorized_array(1.));

        fe_eval.evaluate(true,false);

        VectorizedArray<value_type> tau = 0.5*(fe_eval.read_cell_data(this->get_array_penalty_parameter_continuity())
                                               + fe_eval_neighbor.read_cell_data(this->get_array_penalty_parameter_continuity()));

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
          // set uP to zero
          Tensor<1,dim,VectorizedArray<value_type> > uP;
          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;

          fe_eval.submit_value(tau*jump_value,q);
        }

        fe_eval.integrate(true,false);

        local_diagonal_vector[j] = fe_eval.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell*dim; ++j)
        fe_eval.write_cellwise_dof_value(j, local_diagonal_vector[j]);

      fe_eval.distribute_local_to_global(dst);

      // neighbor (element+)
      VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell*dim];
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell*dim; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell*dim; ++i)
          fe_eval_neighbor.write_cellwise_dof_value(i, make_vectorized_array(0.));
        fe_eval_neighbor.write_cellwise_dof_value(j,make_vectorized_array(1.));

        fe_eval_neighbor.evaluate(true,false);

        VectorizedArray<value_type> tau = 0.5*(fe_eval.read_cell_data(this->get_array_penalty_parameter_continuity())
                                               + fe_eval_neighbor.read_cell_data(this->get_array_penalty_parameter_continuity()));

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          // set uM to zero
          Tensor<1,dim,VectorizedArray<value_type> > uM;
          Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
          Tensor<1,dim,VectorizedArray<value_type> > jump_value = uP - uM; // interior - exterior = uP - uM (neighbor!)

          fe_eval_neighbor.submit_value(tau*jump_value,q);
        }
        fe_eval_neighbor.integrate(true,false);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.read_cellwise_dof_value(j);
      }
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell*dim; ++j)
        fe_eval_neighbor.write_cellwise_dof_value(j, local_diagonal_vector_neighbor[j]);

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void boundary_face_loop_diagonal (const MatrixFree<dim,value_type>                &,
                                    parallel::distributed::Vector<value_type>       &,
                                    const parallel::distributed::Vector<value_type> &,
                                    const std::pair<unsigned int,unsigned int>      &) const
  {

  }
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

template<typename value_type>
class ProjectionSolverBase
{
public:
  ProjectionSolverBase(){}

  virtual ~ProjectionSolverBase(){}

  virtual unsigned int solve(parallel::distributed::Vector<value_type>       &dst,
                             const parallel::distributed::Vector<value_type> &src) const = 0;
};

template <int dim, int fe_degree, typename value_type>
class ProjectionSolverNoPenalty : public ProjectionSolverBase<value_type>
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

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class DirectProjectionSolverDivergencePenalty : public ProjectionSolverBase<value_type>
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;

  DirectProjectionSolverDivergencePenalty(ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> * projection_operator_in)
    :
    projection_operator(projection_operator_in)
  {}

  unsigned int solve(parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src) const
  {
    dst = 0;

    projection_operator->get_data().cell_loop (&DirectProjectionSolverDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule,value_type>::local_solve, this, dst, src);

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
      const VectorizedArray<value_type> tau = projection_operator->get_array_penalty_parameter_divergence()[cell];

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

  ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> * projection_operator;
};


template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule,typename value_type>
class IterativeProjectionSolverDivergencePenalty : public ProjectionSolverBase<value_type>
{
public:
  IterativeProjectionSolverDivergencePenalty(ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> * projection_operator_in,
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

    projection_operator->get_data().cell_loop (&IterativeProjectionSolverDivergencePenalty<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule,value_type>::local_solve, this, dst, src);

    return 0;
  }

  virtual void local_solve(const MatrixFree<dim,value_type>                &data,
                           parallel::distributed::Vector<value_type>       &dst,
                           const parallel::distributed::Vector<value_type> &src,
                           const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    AssertThrow(solver_data.preconditioner_projection == PreconditionerProjection::InverseMassMatrix,
        ExcMessage("Specified preconditioner is not implemented!"));

    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,
                                                                   projection_operator->get_dof_index(),
                                                                   projection_operator->get_quad_index());

    const unsigned int total_dofs_per_cell = fe_eval.dofs_per_cell * dim;
    AlignedVector<VectorizedArray<value_type> > solution(total_dofs_per_cell);

    InternalSolvers::SolverCG<VectorizedArray<double> > cg_solver(total_dofs_per_cell,
                                                                  solver_data.solver_tolerance_abs,
                                                                  solver_data.solver_tolerance_rel,
                                                                  solver_data.max_iter);

    typedef ProjectionOperatorDivergencePenalty<dim, fe_degree, fe_degree_p,
        fe_degree_xwall, xwall_quad_rule, value_type> PROJ_OPERATOR;
    PROJ_OPERATOR *projection_operator_div = static_cast<PROJ_OPERATOR *>(projection_operator);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src,0);

      projection_operator_div->setup(cell);
      cg_solver.solve(projection_operator_div,
                      solution.begin(),
                      fe_eval.begin_dof_values());

      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = solution[j];
      fe_eval.set_dof_values (dst,0);
    }
  }

protected:
  ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> * projection_operator;
  ProjectionSolverData const solver_data;
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule,typename value_type>
class IterativeProjectionSolverDivergencePenaltyXWall: public IterativeProjectionSolverDivergencePenalty<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>
{
public:
  static const bool is_xwall = (xwall_quad_rule>1) ? true : false;
  static const unsigned int n_actual_q_points_vel_linear = (is_xwall) ? xwall_quad_rule : fe_degree+1;
  typedef FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_actual_q_points_vel_linear,dim,value_type,is_xwall> FEEval_Velocity_Velocity_linear;
  IterativeProjectionSolverDivergencePenaltyXWall(ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> * projection_operator_in,
                                                  ProjectionSolverData const solver_data_in)
    :
      IterativeProjectionSolverDivergencePenalty<dim,fe_degree,fe_degree_p,fe_degree_xwall,xwall_quad_rule,value_type>(projection_operator_in,solver_data_in)
  {}

  void local_solve(const MatrixFree<dim,value_type>                &data,
                   parallel::distributed::Vector<value_type>       &dst,
                   const parallel::distributed::Vector<value_type> &src,
                   const std::pair<unsigned int,unsigned int>      &cell_range) const
  {
    AssertThrow(this->solver_data.preconditioner_projection == PreconditionerProjection::InverseMassMatrix,
                        ExcMessage("Specified preconditioner is not available for projection_type = ProjectionType::DivergencePenalty and solver_projection = SolverProjection::PCG"));

    FEEval_Velocity_Velocity_linear fe_eval(data,this->projection_operator->get_fe_param(),this->projection_operator->get_dof_index(),this->projection_operator->get_quad_index());

    ProjectionOperatorDivergencePenaltyXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> *
          projection_operator_div = static_cast<ProjectionOperatorDivergencePenaltyXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> * >(this->projection_operator);

    AlignedVector<VectorizedArray<value_type> > source;
    AlignedVector<VectorizedArray<value_type> > solution;
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      const unsigned int total_dofs_per_cell = fe_eval.dofs_per_cell * dim;
      source.resize(total_dofs_per_cell);
      solution.resize(total_dofs_per_cell);
      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
        source[j] = fe_eval.read_cellwise_dof_value(j);

      InternalSolvers::SolverCG<VectorizedArray<double> > cg_solver(total_dofs_per_cell,
                                                                    this->solver_data.solver_tolerance_abs,
                                                                    this->solver_data.solver_tolerance_rel,
                                                                    this->solver_data.max_iter);

      projection_operator_div->setup(cell);

      cg_solver.solve(projection_operator_div, solution.begin(), source.begin());

      for (unsigned int j=0; j<total_dofs_per_cell; ++j)
        fe_eval.write_cellwise_dof_value(j,solution[j]);
      fe_eval.set_dof_values (dst);
    }
  }
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule, typename value_type>
class IterativeProjectionSolverDivergenceAndContinuityPenalty : public ProjectionSolverBase<value_type>
{
public:
  typedef ProjectionOperatorDivergenceAndContinuityPenalty<dim, fe_degree,
      fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> PROJ_OPERATOR;

  IterativeProjectionSolverDivergenceAndContinuityPenalty(
      ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, value_type> * projection_operator_in,
      ProjectionSolverData const solver_data_in)
    :
    projection_operator(nullptr),
    solver_data(solver_data_in),
    preconditioner(nullptr)
  {
    projection_operator = static_cast<PROJ_OPERATOR *>(projection_operator_in);

    if(solver_data.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
    {
      preconditioner = new InverseMassMatrixPreconditioner<dim,fe_degree,value_type>
         (projection_operator->get_data(),
          projection_operator->get_dof_index(),
          projection_operator->get_quad_index());
    }
    else if(solver_data.preconditioner_projection == PreconditionerProjection::Jacobi)
    {
      preconditioner = new JacobiPreconditioner<value_type,PROJ_OPERATOR>(*projection_operator);
    }
  }

  ~IterativeProjectionSolverDivergenceAndContinuityPenalty()
  {
    delete preconditioner;
    preconditioner = nullptr;
  }

  unsigned int solve(parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src) const
  {
    ReductionControl solver_control (solver_data.max_iter,
                                     solver_data.solver_tolerance_abs,
                                     solver_data.solver_tolerance_rel);

    try
    {
      if(solver_data.solver_projection == SolverProjection::PCG)
      {
        SolverCG<parallel::distributed::Vector<value_type> > solver (solver_control);
        if(solver_data.preconditioner_projection == PreconditionerProjection::None)
          solver.solve (*projection_operator, dst, src, PreconditionIdentity());
        else if(solver_data.preconditioner_projection == PreconditionerProjection::InverseMassMatrix)
          solver.solve (*projection_operator, dst, src, *preconditioner);
        else if(solver_data.preconditioner_projection == PreconditionerProjection::Jacobi)
        {
          // recalculate diagonal since the diagonal depends on the penalty
          // parameter which itself depends on the velocity field
          preconditioner->update(projection_operator);
          solver.solve (*projection_operator, dst, src, *preconditioner);
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
        std::cout << "Viscous solver failed to solve to given tolerance." << std::endl;
    }
    return solver_control.last_step();
  }

private:
  PROJ_OPERATOR * projection_operator;
  ProjectionSolverData const solver_data;
  PreconditionerBase<value_type> *preconditioner;
};


#endif /* INCLUDE_PROJECTIONSOLVER_H_ */
