/*
 * nonlinear_operator_base.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_NONLINEAR_OPERATOR_BASE_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_NONLINEAR_OPERATOR_BASE_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation_notemplate.h>
#include <deal.II/matrix_free/matrix_free.h>

#include "../../../functionalities/lazy_ptr.h"

using namespace dealii;

template<int dim, typename Number, int n_components = 1>
class NonLinearOperatorBase
{
public:
  typedef NonLinearOperatorBase<dim, Number, n_components> This;
  typedef LinearAlgebra::distributed::Vector<Number>       VectorType;
  typedef CellIntegrator<dim, n_components, Number>        IntegratorCell;
  typedef std::pair<unsigned int, unsigned int>            Range;

  virtual ~NonLinearOperatorBase()
  {
  }

  void
  reinit(MatrixFree<dim, Number> const & mf_data) const;

  void
  set_solution_linearization(VectorType & vector);

  void
  evaluate_nonlinear_residual(VectorType & dst, VectorType const & src) const;

private:
  void
  cell_residuum_loop(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const &              src,
                     Range const &                   range) const;

protected:
  virtual void
  do_cell_residuum_integral(IntegratorCell & integrator, unsigned int const cell) const = 0;

  void
  do_initialize_dof_vector(VectorType & vector) const;

  IntegratorCell &
  get_linerization_point(unsigned int const cell) const;

protected:
  mutable lazy_ptr<MatrixFree<dim, Number>> matrix_free;
  mutable bool                              is_dg;
  mutable std::shared_ptr<IntegratorCell>   integrator_prev;
  mutable VectorType                        prev;
};

template<int dim, typename Number, int n_components>
void
NonLinearOperatorBase<dim, Number, n_components>::reinit(
  MatrixFree<dim, Number> const & matrix_free_) const
{
  matrix_free.reset(matrix_free_);
  is_dg =
    (matrix_free->get_dof_handler(0 /*this->operator_data.dof_index*/).get_fe().dofs_per_vertex ==
     0);
  integrator_prev.reset(new IntegratorCell(matrix_free_));
  do_initialize_dof_vector(this->prev);
  this->prev.update_ghost_values();
}

template<int dim, typename Number, int n_components>
void
NonLinearOperatorBase<dim, Number, n_components>::set_solution_linearization(VectorType & vector)
{
  this->prev = vector;
  this->prev.update_ghost_values();
}

template<int dim, typename Number, int n_components>
void
NonLinearOperatorBase<dim, Number, n_components>::evaluate_nonlinear_residual(
  VectorType &       dst,
  VectorType const & src) const
{
  if(is_dg /*&& this->do_eval_faces*/)
  {
    // TODO
  }
  else
  {
    matrix_free->cell_loop(&This::cell_residuum_loop, this, dst, src, true);
  }
}

template<int dim, typename Number, int n_components>
void
NonLinearOperatorBase<dim, Number, n_components>::cell_residuum_loop(
  MatrixFree<dim, Number> const & data,
  VectorType &                    dst,
  VectorType const &              src,
  Range const &                   range) const
{
  IntegratorCell fe_eval(data /*, this->operator_data.dof_index, this->operator_data.quad_index*/);

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    fe_eval.reinit(cell);
    fe_eval.read_dof_values_plain(src);
    fe_eval.evaluate(false, true, false); // TODO

    this->do_cell_residuum_integral(fe_eval, cell);
    fe_eval.integrate_scatter(false, true, dst); // TODO
  }
}

template<int dim, typename Number, int n_components>
void
NonLinearOperatorBase<dim, Number, n_components>::do_initialize_dof_vector(
  VectorType & vector) const
{
  matrix_free->initialize_dof_vector(vector, 0 /*operator_data.dof_index*/);
}

template<int dim, typename Number, int n_components>
typename NonLinearOperatorBase<dim, Number, n_components>::IntegratorCell &
NonLinearOperatorBase<dim, Number, n_components>::get_linerization_point(
  unsigned int const cell) const

{
  integrator_prev->reinit(cell);
  integrator_prev->read_dof_values_plain(this->prev);
  integrator_prev->evaluate(false, true); // TODO

  return *this->integrator_prev;
}

#endif
