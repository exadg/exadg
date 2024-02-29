/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/fe/fe_tools.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/matrix_free/tensor_product_kernels.h>

// ExaDG
#include <exadg/incompressible_navier_stokes/preconditioners/block_preconditioner_momentum.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/momentum_operator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
BlockPreconditionerMomentum<dim, Number>::BlockPreconditionerMomentum(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  PDEOperator const &                     pde_operator)
  : matrix_free(&matrix_free), pde_operator(&pde_operator)
{
  AssertThrow(
    matrix_free.get_dof_handler(0).get_triangulation().all_reference_cells_are_hyper_cube(),
    dealii::ExcNotImplemented());
  AssertThrow(matrix_free.get_dof_handler(0).get_fe().dofs_per_face == 0,
              dealii::ExcNotImplemented("Only implemented for discontinuous elements"));
}

template<int dim, typename Number>
void
BlockPreconditionerMomentum<dim, Number>::update()
{
  dealii::FiniteElement<dim> const & fe =
    matrix_free->get_dof_handler(pde_operator->get_dof_index()).get_fe();
  unsigned int const degree = fe.degree;

  dealii::LAPACKFullMatrix<double> mass_matrix_1d(degree + 1, degree + 1);
  dealii::LAPACKFullMatrix<double> laplace_matrix_1d(degree + 1, degree + 1);
  {
    // convert multi-d element to FiniteElement<1> by changing the template
    // argument from 'd' (2,3) to '1'.
    std::string       name_fe    = fe.base_element(0).get_name();
    std::size_t const pos_modify = name_fe.find_first_of("<") + 1;
    name_fe[pos_modify]          = '1';
    std::unique_ptr<dealii::FiniteElement<1>> fe_1d =
      dealii::FETools::get_fe_by_name<1, 1>(name_fe);
    AssertDimension(fe_1d->degree, degree);

    dealii::QGauss<1> quad(degree + 1);
    for(unsigned int i = 0; i < degree + 1; ++i)
      for(unsigned int j = 0; j < degree + 1; ++j)
      {
        double sum_m = 0, sum_l = 0;
        for(unsigned int q = 0; q < quad.size(); ++q)
        {
          sum_m += (fe_1d->shape_value(i, quad.point(q)) * fe_1d->shape_value(j, quad.point(q))) *
                   quad.weight(q);
          sum_l +=
            (fe_1d->shape_grad(i, quad.point(q))[0] * fe_1d->shape_grad(j, quad.point(q))[0]) *
            quad.weight(q);
        }
        mass_matrix_1d(i, j) = sum_m;
        dealii::Point<1> p0, p1(1.0);
        const double     penalty =
          (degree + 1) * (degree + 1) * pde_operator->get_viscous_kernel_data().IP_factor;
        sum_l += (fe_1d->shape_value(i, p0) * fe_1d->shape_value(j, p0) * penalty +
                  0.5 * fe_1d->shape_grad(i, p0)[0] * fe_1d->shape_value(j, p0) +
                  0.5 * fe_1d->shape_grad(j, p0)[0] * fe_1d->shape_value(i, p0));
        sum_l += (1. * fe_1d->shape_value(i, p1) * fe_1d->shape_value(j, p1) * penalty -
                  0.5 * fe_1d->shape_grad(i, p1)[0] * fe_1d->shape_value(j, p1) -
                  0.5 * fe_1d->shape_grad(j, p1)[0] * fe_1d->shape_value(i, p1));
        laplace_matrix_1d(i, j) = pde_operator->get_viscous_kernel_data().viscosity * sum_l;
      }
  }

  std::vector<dealii::Vector<double>> eigenvecs(degree + 1);
  laplace_matrix_1d.compute_generalized_eigenvalues_symmetric(mass_matrix_1d, eigenvecs);

  transformation_matrix.reinit(degree + 1, degree + 1);
  for(unsigned int i = 0; i < eigenvecs.size(); ++i)
    for(unsigned int j = 0; j < eigenvecs[0].size(); ++j)
      transformation_matrix(i, j) = eigenvecs[i](j);

  bool transformation_is_hierarchical = true;
  for(unsigned int i = 0; i < degree + 1; ++i)
  {
    for(unsigned int j = 0; j < (degree + 1) / 2; ++j)
    {
      if(i % 2 == 1 &&
         std::abs(transformation_matrix(i, j) + transformation_matrix(i, degree - j)) >
           1000. * std::numeric_limits<Number>::epsilon())
        transformation_is_hierarchical = false;
      if(i % 2 == 0 &&
         std::abs(transformation_matrix(i, j) - transformation_matrix(i, degree - j)) >
           1000. * std::numeric_limits<Number>::epsilon())
        transformation_is_hierarchical = false;
    }
  }
  AssertThrow(transformation_is_hierarchical,
              dealii::ExcNotImplemented(
                "Expected a hierarchical transformation for preconditioner"));

  transformation_eigenvalues.resize(degree + 1);
  for(unsigned int i = 0; i < eigenvecs.size(); ++i)
    transformation_eigenvalues[i] = laplace_matrix_1d.eigenvalue(i).real();

  coefficients.resize(matrix_free->n_cell_batches());
  for(unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
  {
    dealii::Tensor<2, dim, scalar> const jacobian =
      matrix_free->get_mapping_info()
        .cell_data[0]
        .jacobians[0][matrix_free->get_mapping_info().cell_data[0].data_index_offsets[cell]];
    coefficients[cell][0] = 1. / dealii::determinant(jacobian);
    for(unsigned int d = 0; d < dim; ++d)
    {
      coefficients[cell][d + 1] = jacobian[d][d];
      for(unsigned int e = 0; e < dim; ++e)
        if(d != e)
          coefficients[cell][d + 1] /= jacobian[e][e];
    }
  }

  this->update_needed = false;
}

template<int dim, typename Number>
void
BlockPreconditionerMomentum<dim, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  vmult(dst, src, {}, {});
}

template<int dim, typename Number>
void
BlockPreconditionerMomentum<dim, Number>::vmult(
  VectorType &                                                        dst,
  VectorType const &                                                  src,
  const std::function<void(const unsigned int, const unsigned int)> & before_loop,
  const std::function<void(const unsigned int, const unsigned int)> & after_loop) const
{
  unsigned int const degree =
    matrix_free->get_dof_handler(pde_operator->get_dof_index()).get_fe().degree;
  if(degree == 2)
    do_vmult<2>(dst, src, before_loop, after_loop);
  else if(degree == 3)
    do_vmult<3>(dst, src, before_loop, after_loop);
  else if(degree == 4)
    do_vmult<4>(dst, src, before_loop, after_loop);
  else if(degree == 5)
    do_vmult<5>(dst, src, before_loop, after_loop);
  else if(degree == 6)
    do_vmult<6>(dst, src, before_loop, after_loop);
  else if(degree == 7)
    do_vmult<7>(dst, src, before_loop, after_loop);
  else if(degree == 8)
    do_vmult<8>(dst, src, before_loop, after_loop);
  else if(degree == 9)
    do_vmult<9>(dst, src, before_loop, after_loop);
  else
    do_vmult<-1>(dst, src, before_loop, after_loop);
}

template<int dim, typename Number>
template<int template_degree>
void
BlockPreconditionerMomentum<dim, Number>::do_vmult(
  VectorType &                                                        global_dst,
  VectorType const &                                                  global_src,
  const std::function<void(const unsigned int, const unsigned int)> & before_loop,
  const std::function<void(const unsigned int, const unsigned int)> & after_loop) const
{
  unsigned int const degree =
    template_degree > -1 ?
      template_degree :
      matrix_free->get_dof_handler(pde_operator->get_dof_index()).get_fe().degree;
  const unsigned int            nn = dealii::Utilities::pow(degree + 1, dim);
  dealii::AlignedVector<scalar> tmp(nn);
  Number const inverse_time_step = pde_operator->get_scaling_factor_mass_operator();

  dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_symmetric_hierarchical,
                                           dim,
                                           template_degree + 1,
                                           template_degree + 1,
                                           scalar,
                                           Number>
    eval(&transformation_matrix(0, 0), {}, {}, degree + 1, degree + 1);

  for(unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
  {
    unsigned int const * dof_indices =
      matrix_free->get_dof_info(pde_operator->get_dof_index()).dof_indices_contiguous[2].data() +
      cell * scalar::size();
    for(unsigned int v = 0; v < scalar::size(); ++v)
      AssertThrow(dof_indices[v] < global_dst.locally_owned_size(), dealii::ExcNotImplemented());

    {
      bool all_indices_contiguous = true;
      for(unsigned int i = 1; i < scalar::size(); ++i)
        if(dof_indices[i] != dof_indices[0] + i * nn * dim)
        {
          all_indices_contiguous = false;
          break;
        }
      if(all_indices_contiguous)
        {
          if (before_loop)
            before_loop(dof_indices[0], dof_indices[0] + scalar::size() * nn * dim);
        }
      else
        for(unsigned int i = 0; i < scalar::size(); ++i)
          {
            if (before_loop)
              before_loop(dof_indices[i], dof_indices[i] + nn * dim);
          }
    }

    for(unsigned int d = 0; d < dim; ++d)
    {
      dealii::vectorized_load_and_transpose(nn,
                                            global_src.begin() + nn * d,
                                            dof_indices,
                                            tmp.begin());
      eval.template values<0, 0, 0>(tmp.begin(), tmp.begin());
      if(dim > 1)
        eval.template values<1, 0, 0>(tmp.begin(), tmp.begin());
      if(dim > 2)
        eval.template values<2, 0, 0>(tmp.begin(), tmp.begin());

      const std::array<scalar, dim + 1> coefficient = coefficients[cell];
      if(dim == 2)
        for(unsigned int i1 = 0, i = 0; i1 < degree + 1; ++i1)
          for(unsigned int i0 = 0; i0 < degree + 1; ++i0, ++i)
            tmp[i] /= inverse_time_step * coefficient[0] +
                      coefficient[1] * transformation_eigenvalues[i1] +
                      coefficient[2] * transformation_eigenvalues[i0];
      else if(dim == 3)
        for(unsigned int i2 = 0, i = 0; i2 < degree + 1; ++i2)
          for(unsigned int i1 = 0; i1 < degree + 1; ++i1)
          {
            scalar const partial_sum = inverse_time_step * coefficient[0] +
                                       coefficient[3] * transformation_eigenvalues[i2] +
                                       coefficient[2] * transformation_eigenvalues[i1];
            for(unsigned int i0 = 0; i0 < degree + 1; ++i0, ++i)
              tmp[i] /= partial_sum + coefficient[1] * transformation_eigenvalues[i0];
          }

      eval.template values<0, 1, 0>(tmp.begin(), tmp.begin());
      if(dim > 1)
        eval.template values<1, 1, 0>(tmp.begin(), tmp.begin());
      if(dim > 2)
        eval.template values<2, 1, 0>(tmp.begin(), tmp.begin());

      dealii::vectorized_transpose_and_store(
        false, nn, tmp.begin(), dof_indices, global_dst.begin() + nn * d);
    }

    {
      bool all_indices_contiguous = true;
      for(unsigned int i = 1; i < scalar::size(); ++i)
        if(dof_indices[i] != dof_indices[0] + i * nn * dim)
        {
          all_indices_contiguous = false;
          break;
        }
      if(all_indices_contiguous)
        {
          if (after_loop)
            after_loop(dof_indices[0], dof_indices[0] + scalar::size() * nn * dim);
        }
      else
        for(unsigned int i = 0; i < scalar::size(); ++i)
          {
            if (after_loop)
              after_loop(dof_indices[i], dof_indices[i] + nn * dim);
          }
    }
  }
}


template class BlockPreconditionerMomentum<2, float>;
template class BlockPreconditionerMomentum<3, float>;

template class BlockPreconditionerMomentum<2, double>;
template class BlockPreconditionerMomentum<3, double>;

} // namespace IncNS
} // namespace ExaDG
