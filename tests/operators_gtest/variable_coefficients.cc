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
#include <gtest/gtest.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/grid/grid_generator.h>
#include <exadg/operators/variable_coefficients.h>

namespace
{
unsigned int const dim = 3;
using Number           = double;

Number
extract(dealii::VectorizedArray<Number> const & scalar_in, unsigned int const v)
{
  return scalar_in[v];
}

dealii::Tensor<2, dim, Number>
extract(dealii::Tensor<2, dim, dealii::VectorizedArray<Number>> const & dyadic_in,
        unsigned int const                                              v)
{
  dealii::Tensor<2, dim, Number> dyadic;
  for(unsigned int d1 = 0; d1 < dim; ++d1)
    for(unsigned int d2 = 0; d2 < dim; ++d2)
      dyadic[d1][d2] = dyadic_in[d1][d2][v];
  return dyadic;
}

template<typename coefficient_t>
class VariableCoefficientsTest : public ::testing::Test
{
private:
  static unsigned int const degree               = 2;
  static unsigned int const n_global_refinements = 1;

protected:
  void
  SetUp() override
  {
    dealii::Triangulation<dim> triangulation;
    dealii::GridGenerator::hyper_cube(triangulation);
    triangulation.refine_global(n_global_refinements);

    dealii::FE_DGQ<dim> const fe(degree);

    dealii::DoFHandler<dim> dof_handler(triangulation);
    dof_handler.distribute_dofs(fe);

    dealii::AffineConstraints<Number> affine_constraints;
    affine_constraints.close();

    dealii::MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags                = dealii::update_JxW_values;
    additional_data.mapping_update_flags_inner_faces    = dealii::update_JxW_values;
    additional_data.mapping_update_flags_boundary_faces = dealii::update_JxW_values;

    matrix_free.reinit(dealii::MappingQ<dim>(degree),
                       dof_handler,
                       affine_constraints,
                       dealii::QGauss<1>(degree + 1),
                       additional_data);

    n_cell_batches = matrix_free.n_cell_batches();
    n_face_batches = matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches();
    n_inner_face_batches = matrix_free.n_inner_face_batches();

    n_q_points_cell = matrix_free.get_n_q_points(0);
    n_q_points_face = matrix_free.get_n_q_points_face(0);

    vectorization_length = dealii::VectorizedArray<double>::size();

    coefficients.initialize(matrix_free, 0, coefficient_t{});
  }

  dealii::MatrixFree<dim, Number>            matrix_free;
  ExaDG::VariableCoefficients<coefficient_t> coefficients;

  unsigned int n_cell_batches;
  unsigned int n_face_batches;
  unsigned int n_inner_face_batches;

  unsigned int n_q_points_cell;
  unsigned int n_q_points_face;

  unsigned int vectorization_length;
};

using coefficient_types =
  ::testing::Types<dealii::VectorizedArray<Number>,
                   dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>,
                   dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>>;
TYPED_TEST_SUITE(VariableCoefficientsTest, coefficient_types);

TYPED_TEST(VariableCoefficientsTest, InitializesWithConstant)
{
  TypeParam coefficient;

  // Create a nice scalar coefficient
  if constexpr(std::is_same_v<TypeParam, dealii::VectorizedArray<Number>>)
  {
    coefficient = 3.14159;
  }
  // Create a nice dyadic coefficient
  else if constexpr(std::is_same_v<TypeParam,
                                   dealii::Tensor<2, dim, dealii::VectorizedArray<Number>>>)
  {
    coefficient[0][0]       = 1.41421;
    coefficient[0][dim - 1] = 1.73205;
  }
  // Create a nice symmetric dyadic coefficient
  else if constexpr(std::is_same_v<
                      TypeParam,
                      dealii::SymmetricTensor<2, dim, dealii::VectorizedArray<Number>>>)
  {
    coefficient = 2.71828 * dealii::unit_symmetric_tensor<dim, dealii::VectorizedArray<Number>>();
  }

  // Call initialize to reinit (resize) and fill with coefficient
  this->coefficients.initialize(this->matrix_free, 0, coefficient);

  // Check cells
  for(unsigned int cell = 0; cell < this->n_cell_batches; ++cell)
    for(unsigned int q = 0; q < this->n_q_points_cell; ++q)
    {
      auto coeff_cell = this->coefficients.get_coefficient_cell(cell, q);
      for(unsigned int v = 0; v < this->vectorization_length; ++v)
        EXPECT_EQ(extract(coefficient, v), extract(coeff_cell, v));
    }

  // Check faces
  for(unsigned int face = 0; face < this->n_face_batches; ++face)
    for(unsigned int q = 0; q < this->n_q_points_face; ++q)
    {
      auto const coeff_face = this->coefficients.get_coefficient_face(face, q);
      for(unsigned int v = 0; v < this->vectorization_length; ++v)
        EXPECT_EQ(extract(coefficient, v), extract(coeff_face, v));
    }

  // Check neighbor faces
  for(unsigned int face = 0; face < this->n_inner_face_batches; ++face)
    for(unsigned int q = 0; q < this->n_q_points_face; ++q)
    {
      auto const coeff_face_neighbor = this->coefficients.get_coefficient_face_neighbor(face, q);
      for(unsigned int v = 0; v < this->vectorization_length; ++v)
        EXPECT_EQ(extract(coefficient, v), extract(coeff_face_neighbor, v));
    }
}
} // namespace
