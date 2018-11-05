/*
 * mass_matrix_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation.h>

using namespace dealii;

namespace IncNS
{
struct MassMatrixOperatorData
{
  MassMatrixOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;
};

template<int dim, int degree, typename Number>
class MassMatrixOperator
{
public:
  typedef MassMatrixOperator<dim, degree, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number> scalar;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef FEEvaluation<dim, degree, degree + 1, dim, Number> FEEval;

  MassMatrixOperator() : data(nullptr), scaling_factor(1.0)
  {
  }

  MassMatrixOperatorData const &
  get_operator_data() const
  {
    return operator_data;
  }

  MatrixFree<dim, Number> const &
  get_data() const
  {
    return *data;
  }

  void
  initialize(MatrixFree<dim, Number> const & mf_data,
             MassMatrixOperatorData const &  operator_data_in)
  {
    this->data          = &mf_data;
    this->operator_data = operator_data_in;

    // Block Jacobi elementwise
    fe_eval.reset(new FEEval(*data, this->operator_data.dof_index, this->operator_data.quad_index));
  }

  // apply matrix vector multiplication
  void
  apply(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    data->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);
  }

  void
  apply_scale(VectorType & dst, Number const & factor, VectorType const & src) const
  {
    scaling_factor = factor;

    data->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);

    scaling_factor = 1.0;
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    data->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);
  }

  void
  apply_scale_add(VectorType & dst, Number const & factor, VectorType const & src) const
  {
    scaling_factor = factor;

    data->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);

    scaling_factor = 1.0;
  }

  void
  calculate_diagonal(VectorType & diagonal) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;
    data->cell_loop(
      &This::cell_loop_diagonal, this, diagonal, src, true /*zero_dst_vector = true*/);
  }

  void
  add_diagonal(VectorType & diagonal) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;
    data->cell_loop(
      &This::cell_loop_diagonal, this, diagonal, src, false /*zero_dst_vector = false*/);
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<Number>> & matrices) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;

    data->cell_loop(&This::cell_loop_calculate_block_diagonal, this, matrices, src);
  }

  void
  apply_add_block_diagonal_elementwise(unsigned int const   cell,
                                       scalar * const       dst,
                                       scalar const * const src,
                                       unsigned int const   problem_size = 1) const
  {
    unsigned int dofs_per_cell = fe_eval->dofs_per_cell;

    fe_eval->reinit(cell);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      fe_eval->begin_dof_values()[i] = src[i];

    fe_eval->evaluate(true, false, false);

    do_cell_integral(*fe_eval);

    fe_eval->integrate(true, false);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dst[i] += fe_eval->begin_dof_values()[i];
  }

private:
  template<typename FEEvaluation>
  void
  do_cell_integral(FEEvaluation & fe_eval) const
  {
    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_value(scaling_factor * fe_eval.get_value(q), q);
    }
  }

  void
  cell_loop(MatrixFree<dim, Number> const & data,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    FEEval fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      fe_eval.gather_evaluate(src, true, false, false);

      do_cell_integral(fe_eval);

      fe_eval.integrate_scatter(true, false, dst);
    }
  }

  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & data,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & cell_range) const
  {
    FEEval fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;
      // tensor_dofs_per_cell >= dofs_per_cell
      VectorizedArray<Number> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false, false);

        do_cell_integral(fe_eval);

        fe_eval.integrate(true, false);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }

  void
  cell_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         data,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & cell_range) const
  {
    FEEval fe_eval(data, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      unsigned int dofs_per_cell = fe_eval.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        fe_eval.evaluate(true, false, false);

        do_cell_integral(fe_eval);

        fe_eval.integrate(true, false);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  MatrixFree<dim, Number> const * data;

  MassMatrixOperatorData operator_data;

  mutable Number scaling_factor;

  // required for elementwise block Jacobi operation
  std::shared_ptr<FEEval> fe_eval;
};

} // namespace IncNS



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_ \
        */
