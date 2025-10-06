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
#include <exadg/incompressible_navier_stokes/preconditioners/block_preconditioner_projection.h>
#include <exadg/incompressible_navier_stokes/spatial_discretization/operators/projection_operator.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
BlockPreconditionerProjection<dim, Number>::BlockPreconditionerProjection(
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

/*
 * Starting with the given inverse mass matrix, compute the inverse of the
 * matrix obtained by updating the (0, 0) and (n, n) entries in the mass
 * matrix via the Woodbury matrix identity.
 */
template<typename Number, typename Number2, typename MatrixType>
void
compute_inverse_mass_matrix_augmented(dealii::FullMatrix<Number2> const & inv_mass_matrix,
                                      std::array<Number, 2> const         extra_entries,
                                      Number                              h_inv,
                                      MatrixType &                        inverse_matrix)
{
  unsigned int const           degree = inv_mass_matrix.m() - 1;
  dealii::Tensor<2, 2, Number> woodbury_matrix;
  for(unsigned int f = 0; f < 2; ++f)
    for(unsigned int e = 0; e < 2; ++e)
      woodbury_matrix[f][e] = Number((double)(f == e)) + extra_entries[f] * extra_entries[e] *
                                                           h_inv *
                                                           inv_mass_matrix(f * degree, e * degree);
  woodbury_matrix = invert(woodbury_matrix);
  for(unsigned int d = 0; d < 2; ++d)
    for(unsigned int e = 0; e < 2; ++e)
      woodbury_matrix[d][e] *= h_inv * extra_entries[d] * extra_entries[e];
  const Number2 * inv_mass_0 = &inv_mass_matrix(0, 0);
  const Number2 * inv_mass_d = &inv_mass_matrix(degree, 0);
  for(unsigned int i = 0; i < degree + 1; ++i)
  {
    Number const tmp_0 =
      inv_mass_0[i] * woodbury_matrix[0][0] + inv_mass_d[i] * woodbury_matrix[1][0];
    Number const tmp_1 =
      inv_mass_0[i] * woodbury_matrix[0][1] + inv_mass_d[i] * woodbury_matrix[1][1];
    Number *        matrix_i   = &inverse_matrix(i, 0);
    const Number2 * inv_mass_i = &inv_mass_matrix(i, 0);
    for(unsigned int j = 0; j < degree + 1; ++j)
    {
      matrix_i[j] = h_inv * (inv_mass_i[j] - inv_mass_0[j] * tmp_0 - inv_mass_d[j] * tmp_1);
    }
  }
}

template<int dim, typename Number>
void
BlockPreconditionerProjection<dim, Number>::update()
{
  penalty_parameters = pde_operator->get_penalty_coefficients();
  for(auto & entry : penalty_parameters)
  {
    entry.first = std::sqrt(entry.first * pde_operator->get_time_step_size());
    for(unsigned int d = 0; d < 2 * dim; ++d)
      entry.second[d] = std::sqrt(entry.second[d] * pde_operator->get_time_step_size());
  }

  dealii::FiniteElement<dim> const & fe =
    matrix_free->get_dof_handler(pde_operator->get_dof_index()).get_fe();
  unsigned int const degree = fe.degree;

  dealii::FullMatrix<double> shape_matrix(degree + 1, degree + 1);
  dealii::FullMatrix<double> deriv_matrix(degree + 1, degree + 1);
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
    {
      for(unsigned int q = 0; q < quad.size(); ++q)
      {
        deriv_matrix(q, i) = fe_1d->shape_grad(i, quad.point(q))[0] * std::sqrt(quad.weight(q));
        shape_matrix(q, i) = fe_1d->shape_value(i, quad.point(q)) * std::sqrt(quad.weight(q));
      }
    }
  }

  dealii::FullMatrix<double> inv_mass_matrix(degree + 1, degree + 1);
  shape_matrix.gauss_jordan();
  shape_matrix.mTmult(inv_mass_matrix, shape_matrix);
  inverse_mass_matrix = inv_mass_matrix;

  dealii::FullMatrix<double>       inverse_mass_matrix_penalty(degree + 1, degree + 1);
  dealii::FullMatrix<double>       u_matrix_1d(degree + 1, degree + 1);
  dealii::FullMatrix<double>       v_matrix_1d(degree + 1, degree + 1);
  dealii::FullMatrix<double>       w_matrix_1d(degree + 1, degree + 1);
  dealii::LAPACKFullMatrix<double> lapack;
  dealii::Vector<double>           eigvalues;
  dealii::FullMatrix<double>       eigvectors;
  transformation_value.resize(matrix_free->n_cell_batches() * dim * (degree + 1) * (degree + 1));
  transformation_deriv.resize(matrix_free->n_cell_batches() * dim * (degree + 1) * (degree + 1));
  transformation_eigenvalues.resize(matrix_free->n_cell_batches() * dim * (degree + 1));
  for(unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
  {
    dealii::Tensor<2, dim, scalar> const jacobian =
      matrix_free->get_mapping_info()
        .cell_data[0]
        .jacobians[0][matrix_free->get_mapping_info().cell_data[0].data_index_offsets[cell]];
    for(unsigned int v = 0; v < matrix_free->n_active_entries_per_cell_batch(cell); ++v)
      for(unsigned int d = 0; d < dim; ++d)
      {
        const double          h_inv = jacobian[d][d][v];
        std::array<double, 2> penalty{{penalty_parameters[cell].second[2 * d][v],
                                       penalty_parameters[cell].second[2 * d + 1][v]}};
        compute_inverse_mass_matrix_augmented(inv_mass_matrix,
                                              penalty,
                                              h_inv,
                                              inverse_mass_matrix_penalty);
        deriv_matrix.mmult(v_matrix_1d, inverse_mass_matrix_penalty);
        v_matrix_1d *= penalty_parameters[cell].first[v] * std::sqrt(h_inv);
        v_matrix_1d.mTmult(w_matrix_1d, deriv_matrix);
        w_matrix_1d *= penalty_parameters[cell].first[v] * std::sqrt(h_inv);

        lapack.copy_from(w_matrix_1d);
        lapack.compute_eigenvalues_symmetric(-1e-6 * penalty_parameters[cell].first[v] * h_inv,
                                             1000 *
                                               std::max(penalty_parameters[cell].first[v] * h_inv,
                                                        h_inv),
                                             0.,
                                             eigvalues,
                                             eigvectors);

        eigvectors.TmTmult(u_matrix_1d, shape_matrix);
        std::size_t start_idx = (cell * dim + d) * (degree + 1) * (degree + 1);
        for(unsigned int i = 0; i < degree + 1; ++i)
          for(unsigned int j = 0; j < degree + 1; ++j)
            transformation_value[start_idx + i * (degree + 1) + j][v] =
              u_matrix_1d(i, j) * std::sqrt(h_inv);

        eigvectors.Tmmult(u_matrix_1d, v_matrix_1d);
        for(unsigned int i = 0; i < degree + 1; ++i)
          for(unsigned int j = 0; j < degree + 1; ++j)
            transformation_deriv[start_idx + i * (degree + 1) + j][v] = u_matrix_1d(i, j);

        for(unsigned int i = 0; i < degree + 1; ++i)
          transformation_eigenvalues[(cell * dim + d) * (degree + 1) + i][v] = eigvalues(i);
      }
  }

  this->update_needed = false;
}

template<int dim, typename Number>
void
BlockPreconditionerProjection<dim, Number>::vmult(VectorType & dst, VectorType const & src) const
{
  vmult(dst, src, {}, {});
}

template<int dim, typename Number>
void
BlockPreconditionerProjection<dim, Number>::vmult(
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
BlockPreconditionerProjection<dim, Number>::do_vmult(
  VectorType &                                                        global_dst,
  VectorType const &                                                  global_src,
  const std::function<void(const unsigned int, const unsigned int)> & before_loop,
  const std::function<void(const unsigned int, const unsigned int)> & after_loop) const
{
  unsigned int const degree =
    template_degree > -1 ?
      template_degree :
      matrix_free->get_dof_handler(pde_operator->get_dof_index()).get_fe().degree;
  const unsigned int            nn  = dealii::Utilities::pow(degree + 1, dim);
  const unsigned int            nn2 = 2 * nn;
  dealii::AlignedVector<scalar> src(nn * dim), dst(nn * dim), tmp(nn * 2);
  dealii::Table<2, scalar>      inverse_mass_matrix_penalty(degree + 1, degree + 1);

  for(unsigned int cell = 0; cell < matrix_free->n_cell_batches(); ++cell)
  {
    dealii::Tensor<2, dim, scalar> const jacobian =
      matrix_free->get_mapping_info()
        .cell_data[0]
        .jacobians[0][matrix_free->get_mapping_info().cell_data[0].data_index_offsets[cell]];
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
        if(before_loop)
          before_loop(dof_indices[0], dof_indices[0] + scalar::size() * nn * dim);
      }
      else
        for(unsigned int i = 0; i < scalar::size(); ++i)
        {
          if(before_loop)
            before_loop(dof_indices[i], dof_indices[i] + nn * dim);
        }
    }

    dealii::vectorized_load_and_transpose(nn * dim, global_src.begin(), dof_indices, src.begin());

    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_general,
                                             dim,
                                             template_degree + 1,
                                             template_degree + 1,
                                             scalar>
      eval({}, {}, {}, degree + 1, degree + 1);
    dealii::internal::EvaluatorTensorProduct<dealii::internal::evaluate_general,
                                             dim,
                                             template_degree + 1,
                                             template_degree + 1,
                                             scalar,
                                             Number>
      evalm({}, {}, {}, degree + 1, degree + 1);

    Number const * mass = &inverse_mass_matrix(0, 0);
    scalar const * b    = &inverse_mass_matrix_penalty(0, 0);

    std::array<scalar, 2> penalty{
      {penalty_parameters[cell].second[0], penalty_parameters[cell].second[1]}};
    compute_inverse_mass_matrix_augmented(inverse_mass_matrix,
                                          penalty,
                                          jacobian[0][0],
                                          inverse_mass_matrix_penalty);

    eval.template apply<0, 1, 0>(b, src.begin(), dst.begin());
    evalm.template apply<1, 1, 0>(mass, dst.begin(), dst.begin());
    if(dim > 2)
    {
      evalm.template apply<2, 1, 0>(mass, dst.begin(), dst.begin());
      for(unsigned int i = 0; i < nn; ++i)
        dst[i] *= jacobian[1][1] * jacobian[2][2];
    }
    else
      for(unsigned int i = 0; i < nn; ++i)
        dst[i] *= jacobian[1][1];

    evalm.template apply<0, 1, 0>(mass, src.begin() + nn, dst.begin() + nn);
    penalty = {{penalty_parameters[cell].second[2], penalty_parameters[cell].second[3]}};
    compute_inverse_mass_matrix_augmented(inverse_mass_matrix,
                                          penalty,
                                          jacobian[1][1],
                                          inverse_mass_matrix_penalty);
    eval.template apply<1, 1, 0>(b, dst.begin() + nn, dst.begin() + nn);

    if(dim > 2)
    {
      evalm.template apply<2, 1, 0>(mass, dst.begin() + nn, dst.begin() + nn);
      for(unsigned int i = 0; i < nn; ++i)
        dst[i + nn] *= jacobian[0][0] * jacobian[2][2];
      evalm.template apply<0, 1, 0>(mass, src.begin() + nn2, dst.begin() + nn2);
      evalm.template apply<1, 1, 0>(mass, dst.begin() + nn2, dst.begin() + nn2);
      penalty = {{penalty_parameters[cell].second[4], penalty_parameters[cell].second[5]}};
      compute_inverse_mass_matrix_augmented(inverse_mass_matrix,
                                            penalty,
                                            jacobian[2][2],
                                            inverse_mass_matrix_penalty);
      eval.template apply<2, 1, 0>(b, dst.begin() + nn2, dst.begin() + nn2);
      for(unsigned int i = 0; i < nn; ++i)
        dst[i + nn2] *= jacobian[0][0] * jacobian[1][1];
    }
    else
      for(unsigned int i = 0; i < nn; ++i)
        dst[i + nn] *= jacobian[0][0];

    std::size_t const start_idx = cell * dim * (degree + 1) * (degree + 1);
    scalar const *    val0      = transformation_value.data() + start_idx;
    scalar const *    val1      = val0 + (degree + 1) * (degree + 1);
    scalar const *    val2      = val1 + (degree + 1) * (degree + 1);
    scalar const *    grad0     = transformation_deriv.data() + start_idx;
    scalar const *    grad1     = grad0 + (degree + 1) * (degree + 1);
    scalar const *    grad2     = grad1 + (degree + 1) * (degree + 1);

    eval.template apply<0, 0, 0>(grad0, src.begin(), tmp.begin());
    eval.template apply<1, 0, 0>(val1, tmp.begin(), tmp.begin());
    eval.template apply<0, 0, 0>(val0, src.begin() + nn, tmp.begin() + nn);
    eval.template apply<1, 0, 1>(grad1, tmp.begin() + nn, tmp.begin());
    if(dim > 2)
    {
      eval.template apply<2, 0, 0>(val2, tmp.begin(), tmp.begin());
      eval.template apply<0, 0, 0>(val0, src.begin() + nn2, tmp.begin() + nn);
      eval.template apply<1, 0, 0>(val1, tmp.begin() + nn, tmp.begin() + nn);
      eval.template apply<2, 0, 1>(grad2, tmp.begin() + nn, tmp.begin());
    }

    scalar const * eigenvalues0 = transformation_eigenvalues.data() + cell * (degree + 1) * dim;
    scalar const * eigenvalues1 = eigenvalues0 + (degree + 1);
    scalar const * eigenvalues2 = eigenvalues1 + (degree + 1);
    if(dim == 2)
      for(unsigned int i1 = 0, i = 0; i1 < degree + 1; ++i1)
        for(unsigned int i0 = 0; i0 < degree + 1; ++i0, ++i)
          tmp[i] /= -(scalar(1.0) + eigenvalues1[i1] + eigenvalues0[i0]);
    else if(dim == 3)
      for(unsigned int i2 = 0, i = 0; i2 < degree + 1; ++i2)
        for(unsigned int i1 = 0; i1 < degree + 1; ++i1)
        {
          scalar const partial_sum = -(scalar(1.0) + eigenvalues2[i2] + eigenvalues1[i1]);
          for(unsigned int i0 = 0; i0 < degree + 1; ++i0, ++i)
            tmp[i] /= partial_sum - eigenvalues0[i0];
        }

    if(dim > 2)
    {
      eval.template apply<2, 1, 0>(grad2, tmp.begin(), tmp.begin() + nn);
      eval.template apply<1, 1, 0>(val1, tmp.begin() + nn, tmp.begin() + nn);
      eval.template apply<0, 1, 1>(val0, tmp.begin() + nn, dst.begin() + nn2);
      eval.template apply<2, 1, 0>(val2, tmp.begin(), tmp.begin());
    }
    eval.template apply<1, 1, 0>(grad1, tmp.begin(), tmp.begin() + nn);
    eval.template apply<0, 1, 1>(val0, tmp.begin() + nn, dst.begin() + nn);
    eval.template apply<1, 1, 0>(val1, tmp.begin(), tmp.begin());
    eval.template apply<0, 1, 1>(grad0, tmp.begin(), dst.begin());

    dealii::vectorized_transpose_and_store(
      false, nn * dim, dst.begin(), dof_indices, global_dst.begin());

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
        if(after_loop)
          after_loop(dof_indices[0], dof_indices[0] + scalar::size() * nn * dim);
      }
      else
        for(unsigned int i = 0; i < scalar::size(); ++i)
        {
          if(after_loop)
            after_loop(dof_indices[i], dof_indices[i] + nn * dim);
        }
    }
  }
}


template class BlockPreconditionerProjection<2, float>;
template class BlockPreconditionerProjection<3, float>;

template class BlockPreconditionerProjection<2, double>;
template class BlockPreconditionerProjection<3, double>;

} // namespace IncNS
} // namespace ExaDG
