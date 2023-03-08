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
#ifndef INCLUDE_EXADG_OPERATORS_VARIABLE_COEFFICIENTS_H_
#define INCLUDE_EXADG_OPERATORS_VARIABLE_COEFFICIENTS_H_

#include <deal.II/matrix_free/matrix_free.h>

namespace ExaDG
{
template<typename coefficient_t>
class VariableCoefficientsCells
{
public:
  template<int dim, typename Number>
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             unsigned int const                      quad_index,
             coefficient_t const &                   constant_coefficient)
  {
    reinit(matrix_free, quad_index);

    fill(constant_coefficient);
  }

  coefficient_t
  get_coefficient(unsigned int const cell, unsigned int const q) const
  {
    return coefficients_cell[cell][q];
  }

  void
  set_coefficient(unsigned int const cell, unsigned int const q, coefficient_t const & value)
  {
    coefficients_cell[cell][q] = value;
  }

private:
  template<int dim, typename Number>
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free, unsigned int const quad_index)
  {
    coefficients_cell.reinit(matrix_free.n_cell_batches(), matrix_free.get_n_q_points(quad_index));
  }

  void
  fill(coefficient_t const & constant_coefficient)
  {
    coefficients_cell.fill(constant_coefficient);
  }

  // variable coefficients
  dealii::Table<2, coefficient_t> coefficients_cell;
};

template<typename coefficient_t>
class VariableCoefficients
{
public:
  template<int dim, typename Number>
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             unsigned int const                      quad_index,
             coefficient_t const &                   constant_coefficient)
  {
    reinit(matrix_free, quad_index);

    fill(constant_coefficient);
  }

  coefficient_t
  get_coefficient_cell(unsigned int const cell, unsigned int const q) const
  {
    return coefficients_cell[cell][q];
  }

  void
  set_coefficient_cell(unsigned int const cell, unsigned int const q, coefficient_t const & value)
  {
    coefficients_cell[cell][q] = value;
  }

  coefficient_t
  get_coefficient_face(unsigned int const face, unsigned int const q) const
  {
    return coefficients_face[face][q];
  }

  void
  set_coefficient_face(unsigned int const face, unsigned int const q, coefficient_t const & value)
  {
    coefficients_face[face][q] = value;
  }

  coefficient_t
  get_coefficient_face_neighbor(unsigned int const face, unsigned int const q) const
  {
    return coefficients_face_neighbor[face][q];
  }

  void
  set_coefficient_face_neighbor(unsigned int const    face,
                                unsigned int const    q,
                                coefficient_t const & value)
  {
    coefficients_face_neighbor[face][q] = value;
  }

  // TODO
  //
  //  coefficient_t
  //  get_coefficient_cell_based(unsigned int const face,
  //                             unsigned int const q) const
  //  {
  //    return coefficients_face_cell_based[face][q];
  //  }
  //
  //  void
  //  set_coefficient_cell_based(unsigned int const    face,
  //                             unsigned int const    q,
  //                             coefficient_t const & value)
  //  {
  //    coefficients_face_cell_based[face][q] = value;
  //  }

private:
  template<int dim, typename Number>
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free, unsigned int const quad_index)
  {
    coefficients_cell.reinit(matrix_free.n_cell_batches(), matrix_free.get_n_q_points(quad_index));

    coefficients_face.reinit(matrix_free.n_inner_face_batches() +
                               matrix_free.n_boundary_face_batches(),
                             matrix_free.get_n_q_points_face(quad_index));

    coefficients_face_neighbor.reinit(matrix_free.n_inner_face_batches(),
                                      matrix_free.get_n_q_points_face(quad_index));

    // TODO
    // // cell-based face loops
    // coefficients_face_cell_based.reinit(matrix_free.n_cell_batches()*2*dim,
    // matrix_free.get_n_q_points_face(quad_index));
  }

  void
  fill(coefficient_t const & constant_coefficient)
  {
    coefficients_cell.fill(constant_coefficient);
    coefficients_face.fill(constant_coefficient);
    coefficients_face_neighbor.fill(constant_coefficient);

    // TODO
    // coefficients_face_cell_based.fill(constant_coefficient);
  }

  // variable coefficients

  // cell
  dealii::Table<2, coefficient_t> coefficients_cell;

  // face-based loops
  dealii::Table<2, coefficient_t> coefficients_face;
  dealii::Table<2, coefficient_t> coefficients_face_neighbor;

  // TODO
  //  // cell-based face loops
  //  dealii::Table<2, coefficient_t> coefficients_face_cell_based;
};

} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_VARIABLE_COEFFICIENTS_H_ */
