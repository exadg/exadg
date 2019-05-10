/*
 * mass_matrix_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

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

template<int dim, typename Number>
class MassMatrixOperator
{
public:
  typedef MassMatrixOperator<dim, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number> scalar;

  typedef std::pair<unsigned int, unsigned int> Range;

  typedef CellIntegrator<dim, dim, Number> Integrator;

  MassMatrixOperator() : matrix_free(nullptr), scaling_factor(1.0)
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
    return *matrix_free;
  }

  void
  initialize(MatrixFree<dim, Number> const & matrix_free_in,
             MassMatrixOperatorData const &  operator_data_in)
  {
    this->matrix_free   = &matrix_free_in;
    this->operator_data = operator_data_in;

    // Block Jacobi elementwise
    integrator.reset(
      new Integrator(*matrix_free, this->operator_data.dof_index, this->operator_data.quad_index));
  }

  // apply matrix vector multiplication
  void
  apply(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    matrix_free->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);
  }

  void
  apply_scale(VectorType & dst, Number const & factor, VectorType const & src) const
  {
    scaling_factor = factor;

    matrix_free->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);

    scaling_factor = 1.0;
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    matrix_free->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);
  }

  void
  apply_scale_add(VectorType & dst, Number const & factor, VectorType const & src) const
  {
    scaling_factor = factor;

    matrix_free->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);

    scaling_factor = 1.0;
  }

  void
  calculate_diagonal(VectorType & diagonal) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;
    matrix_free->cell_loop(
      &This::cell_loop_diagonal, this, diagonal, src, true /*zero_dst_vector = true*/);
  }

  void
  add_diagonal(VectorType & diagonal) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;
    matrix_free->cell_loop(
      &This::cell_loop_diagonal, this, diagonal, src, false /*zero_dst_vector = false*/);
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<Number>> & matrices) const
  {
    AssertThrow(std::abs(scaling_factor - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;

    matrix_free->cell_loop(&This::cell_loop_calculate_block_diagonal, this, matrices, src);
  }

  void
  apply_add_block_diagonal_elementwise(unsigned int const   cell,
                                       scalar * const       dst,
                                       scalar const * const src,
                                       unsigned int const   problem_size = 1) const
  {
    (void)problem_size;

    unsigned int dofs_per_cell = integrator->dofs_per_cell;

    integrator->reinit(cell);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      integrator->begin_dof_values()[i] = src[i];

    integrator->evaluate(true, false, false);

    do_cell_integral(*integrator);

    integrator->integrate(true, false);

    for(unsigned int i = 0; i < dofs_per_cell; ++i)
      dst[i] += integrator->begin_dof_values()[i];
  }

private:
  template<typename Integrator>
  void
  do_cell_integral(Integrator & integrator) const
  {
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      integrator.submit_value(scaling_factor * integrator.get_value(q), q);
    }
  }

  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const &              src,
            Range const &                   cell_range) const
  {
    Integrator integrator(matrix_free, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      integrator.gather_evaluate(src, true, false, false);

      do_cell_integral(integrator);

      integrator.integrate_scatter(true, false, dst);
    }
  }

  void
  cell_loop_diagonal(MatrixFree<dim, Number> const & matrix_free,
                     VectorType &                    dst,
                     VectorType const &,
                     Range const & cell_range) const
  {
    Integrator integrator(matrix_free, operator_data.dof_index, operator_data.quad_index);

    unsigned int const                     dofs_per_cell = integrator.dofs_per_cell;
    AlignedVector<VectorizedArray<Number>> local_diagonal_vector(dofs_per_cell);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(true, false, false);

        do_cell_integral(integrator);

        integrator.integrate(true, false);

        local_diagonal_vector[j] = integrator.begin_dof_values()[j];
      }
      for(unsigned int j = 0; j < dofs_per_cell; ++j)
        integrator.begin_dof_values()[j] = local_diagonal_vector[j];

      integrator.distribute_local_to_global(dst);
    }
  }

  void
  cell_loop_calculate_block_diagonal(MatrixFree<dim, Number> const &         matrix_free,
                                     std::vector<LAPACKFullMatrix<Number>> & matrices,
                                     VectorType const &,
                                     Range const & cell_range) const
  {
    Integrator integrator(matrix_free, operator_data.dof_index, operator_data.quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      unsigned int dofs_per_cell = integrator.dofs_per_cell;

      for(unsigned int j = 0; j < dofs_per_cell; ++j)
      {
        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          integrator.begin_dof_values()[i] = make_vectorized_array<Number>(0.);
        integrator.begin_dof_values()[j] = make_vectorized_array<Number>(1.);

        integrator.evaluate(true, false, false);

        do_cell_integral(integrator);

        integrator.integrate(true, false);

        for(unsigned int i = 0; i < dofs_per_cell; ++i)
          for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
            matrices[cell * VectorizedArray<Number>::n_array_elements + v](i, j) +=
              integrator.begin_dof_values()[i][v];
      }
    }
  }

  MatrixFree<dim, Number> const * matrix_free;

  MassMatrixOperatorData operator_data;

  mutable Number scaling_factor;

  // required for elementwise block Jacobi operation
  std::shared_ptr<Integrator> integrator;
};

} // namespace IncNS



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_ \
        */
