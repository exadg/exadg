/*
 * variable_coefficients.h
 *
 *  Created on: Jun 15, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_OPERATORS_VARIABLE_COEFFICIENTS_H_
#define INCLUDE_OPERATORS_VARIABLE_COEFFICIENTS_H_

using namespace dealii;

template<int dim, typename Number>
class VariableCoefficients
{
private:
  typedef VectorizedArray<Number> scalar;

public:
  void
  initialize(MatrixFree<dim, Number> const & matrix_free,
             unsigned int const              degree,
             Number const &                  constant_coefficient)
  {
    unsigned int const points_per_cell = Utilities::pow(degree + 1, dim);
    unsigned int const points_per_face = Utilities::pow(degree + 1, dim - 1);

    // cells
    coefficients_cell.reinit(matrix_free.n_cell_batches(), points_per_cell);

    coefficients_cell.fill(make_vectorized_array<Number>(constant_coefficient));

    // face-based loops
    coefficients_face.reinit(matrix_free.n_inner_face_batches() +
                               matrix_free.n_boundary_face_batches(),
                             points_per_face);

    coefficients_face.fill(make_vectorized_array<Number>(constant_coefficient));

    coefficients_face_neighbor.reinit(matrix_free.n_inner_face_batches(), points_per_face);

    coefficients_face_neighbor.fill(make_vectorized_array<Number>(constant_coefficient));

    // TODO cell-based face loops
    //    coefficients_face_cell_based.reinit(matrix_free.n_cell_batches()*2*dim,
    //        points_per_face);
    //
    //    coefficients_face_cell_based.fill(make_vectorized_array<Number>(constant_coefficient));
  }

  scalar
  get_coefficient_cell(unsigned int const cell, unsigned int const q)
  {
    return coefficients_cell[cell][q];
  }

  void
  set_coefficient_cell(unsigned int const cell, unsigned int const q, scalar const & value)
  {
    coefficients_cell[cell][q] = value;
  }

  scalar
  get_coefficient_face(unsigned int const face, unsigned int const q)
  {
    return coefficients_face[face][q];
  }

  void
  set_coefficient_face(unsigned int const face, unsigned int const q, scalar const & value)
  {
    coefficients_face[face][q] = value;
  }

  scalar
  get_coefficient_face_neighbor(unsigned int const face, unsigned int const q)
  {
    return coefficients_face_neighbor[face][q];
  }

  void
  set_coefficient_face_neighbor(unsigned int const face, unsigned int const q, scalar const & value)
  {
    coefficients_face_neighbor[face][q] = value;
  }

  // TODO
  //  scalar
  //  get_coefficient_cell_based(unsigned int const face,
  //                             unsigned int const q)
  //  {
  //    return coefficients_face_cell_based[face][q];
  //  }
  //
  //  void
  //  set_coefficient_cell_based(unsigned int const face,
  //                             unsigned int const q,
  //                             scalar const &     value)
  //  {
  //    coefficients_face_cell_based[face][q] = value;
  //  }

private:
  // variable coefficients

  // cell
  Table<2, scalar> coefficients_cell;

  // face-based loops
  Table<2, scalar> coefficients_face;
  Table<2, scalar> coefficients_face_neighbor;

  // TODO
  //  // cell-based face loops
  //  Table<2, scalar> coefficients_face_cell_based;
};


#endif /* INCLUDE_OPERATORS_VARIABLE_COEFFICIENTS_H_ */
