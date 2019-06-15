/*
 * mass_matrix_operator.h
 *
 *  Created on: Nov 5, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_

#include <deal.II/matrix_free/fe_evaluation_notemplate.h>

// TODO can be removed once MassMatrixOperator derives from OperatorBase
#include "../../../operators/integrator_flags.h"
#include "../../../operators/mapping_flags.h"

using namespace dealii;

namespace IncNS
{
namespace Operators
{
template<int dim, typename Number>
class MassMatrixKernel
{
public:
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  MassMatrixKernel() : scaling_factor(1.0)
  {
  }

  void
  reinit(double const & factor) const
  {
    set_scaling_factor(factor);
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(true, false, false);
    flags.cell_integrate = CellFlags(true, false, false);

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = update_JxW_values;

    // no face integrals

    return flags;
  }

  Number
  get_scaling_factor() const
  {
    return scaling_factor;
  }

  void
  set_scaling_factor(Number const & number) const
  {
    scaling_factor = number;
  }

  /*
   * Volume flux, i.e., the term occurring in the volume integral
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    get_volume_flux(vector const & value) const
  {
    return scaling_factor * value;
  }

private:
  mutable Number scaling_factor;
};

} // namespace Operators

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

  MassMatrixOperator() : matrix_free(nullptr)
  {
  }

  void
  set_scaling_factor(Number const & number)
  {
    kernel.set_scaling_factor(number);
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
  reinit(MatrixFree<dim, Number> const & matrix_free_in,
         MassMatrixOperatorData const &  operator_data_in)
  {
    matrix_free   = &matrix_free_in;
    operator_data = operator_data_in;

    // Block Jacobi elementwise
    integrator.reset(
      new Integrator(*matrix_free, this->operator_data.dof_index, this->operator_data.quad_index));
  }

  // apply matrix vector multiplication
  void
  apply(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(std::abs(kernel.get_scaling_factor() - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    matrix_free->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);
  }

  // TODO can be removed once MassMatrixOperator derives from OperatorBase
  void
  apply_scale(VectorType & dst, Number const & factor, VectorType const & src) const
  {
    kernel.set_scaling_factor(factor);

    matrix_free->cell_loop(&This::cell_loop, this, dst, src, true /*zero_dst_vector = true*/);

    kernel.set_scaling_factor(1.0);
  }

  void
  apply_add(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(std::abs(kernel.get_scaling_factor() - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    matrix_free->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);
  }

  // TODO can be removed once MassMatrixOperator derives from OperatorBase
  void
  apply_scale_add(VectorType & dst, Number const & factor, VectorType const & src) const
  {
    kernel.set_scaling_factor(factor);

    matrix_free->cell_loop(&This::cell_loop, this, dst, src, false /*zero_dst_vector = false*/);

    kernel.set_scaling_factor(1.0);
  }

  void
  calculate_diagonal(VectorType & diagonal) const
  {
    AssertThrow(std::abs(kernel.get_scaling_factor() - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;
    matrix_free->cell_loop(
      &This::cell_loop_diagonal, this, diagonal, src, true /*zero_dst_vector = true*/);
  }

  void
  add_diagonal(VectorType & diagonal) const
  {
    AssertThrow(std::abs(kernel.get_scaling_factor() - 1.0) < 1.e-12,
                ExcMessage("Invalid parameter scaling_factor."));

    VectorType src;
    matrix_free->cell_loop(
      &This::cell_loop_diagonal, this, diagonal, src, false /*zero_dst_vector = false*/);
  }

  void
  add_block_diagonal_matrices(std::vector<LAPACKFullMatrix<Number>> & matrices) const
  {
    AssertThrow(std::abs(kernel.get_scaling_factor() - 1.0) < 1.e-12,
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
      integrator.submit_value(kernel.get_volume_flux(integrator.get_value(q)), q);
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

  // TODO can be removed once MassMatrixOperator derives from OperatorBase
  // required for elementwise block Jacobi operation
  std::shared_ptr<Integrator> integrator;

  Operators::MassMatrixKernel<dim, Number> kernel;
};

} // namespace IncNS



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_OPERATORS_MASS_MATRIX_OPERATOR_H_ \
        */
