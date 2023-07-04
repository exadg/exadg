/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2023 by the ExaDG authors
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

#include <exadg/functions_and_boundary_conditions/evaluate_functions.h>

namespace ExaDG
{
/**
 * This class serves as a manager of several table objects, which hold quadrature-point-level
 * coefficient information for the different matrix-free loop types and access methods.
 *
 * The class provides data structures for cell loops, separate face loops, and cell-based face
 * loops. It is responsible solely for the storage of coefficients. No generic way of computing
 * the coefficients is provided here. This task is left to the owner of a `VariableCoefficients`
 * object.
 *
 * @tparam coefficient_type Type of coefficient stored in the tables. In most cases, this is
 * a `dealii::Tensor` or a similar object.
 */
template<typename coefficient_type>
class VariableCoefficients
{
public:
  /**
   * Initializes the variable coefficients object based on @p matrix_free and @p quad_index
   * according to the desired properties @p use_faces_in, and @p use_cell_based_face_loops_in,
   * i.e. resizes the desired tables to the required number of cells, faces, and quadrature points.
   *
   * @param matrix_free Underlying matrix-free description
   * @param quad_index Quadrature index in the matrix-free description to create the
   * coefficient tables according to the quadrature points
   * @param store_face_data_in Boolean switch to use a coefficient table for faces (and neighbors)
   * @param store_cell_based_face_data_in Boolean switch to use a coefficient table for cell-based
   * face access. This is an additional and optional way of accessing the face coefficients and
   * should be `false` if @p store_face_data_in is `false`.
   */
  template<int dim, typename Number>
  void
  initialize(dealii::MatrixFree<dim, Number> const & matrix_free,
             unsigned int const                      quad_index,
             bool const                              store_face_data_in,
             bool const                              store_cell_based_face_data_in)
  {
    if(not store_face_data_in)
      AssertThrow(not store_cell_based_face_data_in,
                  dealii::ExcMessage("Storing only cell-based face data does not make sense"
                                     " if storing face data is disabled."));

    store_face_data            = store_face_data_in;
    store_cell_based_face_data = store_cell_based_face_data_in;

    reinit(matrix_free, quad_index);
  }

  /**
   * Sets the coefficient @p constant_coefficient everywhere.
   */
  void
  set_coefficients(coefficient_type const & constant_coefficient)
  {
    fill(constant_coefficient);
  }

  /**
   * Returns the coefficient in the entry (@p cell , @p q) in the cell coefficient table.
   */
  coefficient_type
  get_coefficient_cell(unsigned int const cell, unsigned int const q) const
  {
    return coefficients_cell[cell][q];
  }

  /**
   * Sets the specific entry (@p cell , @p q) in the cell coefficient table to @p
   * coefficient.
   */
  void
  set_coefficient_cell(unsigned int const       cell,
                       unsigned int const       q,
                       coefficient_type const & coefficient)
  {
    coefficients_cell[cell][q] = coefficient;
  }

  /**
   * Returns the coefficient in the entry (@p face , @p q) in the face coefficient table.
   */
  coefficient_type
  get_coefficient_face(unsigned int const face, unsigned int const q) const
  {
    return coefficients_face[face][q];
  }

  /**
   * Sets the specific entry (@p face , @p q) in the face coefficient table to @p coefficient.
   */
  void
  set_coefficient_face(unsigned int const       face,
                       unsigned int const       q,
                       coefficient_type const & coefficient)
  {
    coefficients_face[face][q] = coefficient;
  }

  /**
   * Returns the coefficient in the entry (@p face , @p q) in the neighbor face coefficient table.
   */
  coefficient_type
  get_coefficient_face_neighbor(unsigned int const face, unsigned int const q) const
  {
    return coefficients_face_neighbor[face][q];
  }

  /**
   * Sets the specific entry (@p face , @p q) in the neighbor face coefficient table to
   * @p coefficient.
   */
  void
  set_coefficient_face_neighbor(unsigned int const       face,
                                unsigned int const       q,
                                coefficient_type const & coefficient)
  {
    coefficients_face_neighbor[face][q] = coefficient;
  }

  /**
   * Returns the coefficient in the entry (@p cell_based_face , @p q) in the cell-based face
   * coefficient table.
   */
  coefficient_type
  get_coefficient_face_cell_based(unsigned int const cell_based_face, unsigned int const q) const
  {
    return coefficients_face_cell_based[cell_based_face][q];
  }

  /**
   * Sets the specific entry (@p cell_based_face , @p q) in the cell-based face coefficient table
   * to @p coefficient.
   */
  void
  set_coefficient_face_cell_based(unsigned int const       cell_based_face,
                                  unsigned int const       q,
                                  coefficient_type const & coefficient)
  {
    coefficients_face_cell_based[cell_based_face][q] = coefficient;
  }

  bool
  stores_cell_based_face_data() const
  {
    return store_cell_based_face_data;
  }

private:
  /**
   * Reinitializes the tables in use. This resizes the tables and initializes the coefficients with
   * the default constructed @p coefficient_type.
   */
  template<int dim, typename Number>
  void
  reinit(dealii::MatrixFree<dim, Number> const & matrix_free, unsigned int const quad_index)
  {
    coefficients_cell.reinit(matrix_free.n_cell_batches(), matrix_free.get_n_q_points(quad_index));

    if(store_face_data)
    {
      coefficients_face.reinit(matrix_free.n_inner_face_batches() +
                                 matrix_free.n_boundary_face_batches(),
                               matrix_free.get_n_q_points_face(quad_index));
      coefficients_face_neighbor.reinit(matrix_free.n_inner_face_batches(),
                                        matrix_free.get_n_q_points_face(quad_index));

      if(store_cell_based_face_data)
      {
        unsigned int const n_faces_per_cell =
          matrix_free.get_dof_handler().get_triangulation().get_reference_cells()[0].n_faces();

        coefficients_face_cell_based.reinit(matrix_free.n_cell_batches() * n_faces_per_cell,
                                            matrix_free.get_n_q_points_face(quad_index));
      }
    }
  }

  /**
   * Fills the tables with a constant coefficient.
   */
  void
  fill(coefficient_type const & constant_coefficient)
  {
    coefficients_cell.fill(constant_coefficient);

    if(store_face_data)
    {
      coefficients_face.fill(constant_coefficient);
      coefficients_face_neighbor.fill(constant_coefficient);

      if(store_cell_based_face_data)
        coefficients_face_cell_based.fill(constant_coefficient);
    }
  }

  //! Coefficient table for cells
  dealii::Table<2, coefficient_type> coefficients_cell;

  //! Coefficient table for faces
  dealii::Table<2, coefficient_type> coefficients_face;

  //! Coefficient table for neighbor faces
  dealii::Table<2, coefficient_type> coefficients_face_neighbor;

  //! Coefficient table for faces with cell-based access
  dealii::Table<2, coefficient_type> coefficients_face_cell_based;

  //! Boolean switch to use a coefficient table for faces
  bool store_face_data{false};

  /**
   * @brief Boolean switch to use a coefficient table for cell-based face access
   *
   * This is an additional/optional data structure which can be used when cell-based face access
   * is required alongside the separate face access.
   */
  bool store_cell_based_face_data{false};
};

namespace VarCoeffUtils
{
template<int dim, typename Number, int coefficient_rank>
class AnalyticalTensorCoefficientFunctionEvaluator
{
  using This = AnalyticalTensorCoefficientFunctionEvaluator<dim, Number, coefficient_rank>;

  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;

  using Scalar          = dealii::VectorizedArray<Number>;
  using CoefficientType = dealii::Tensor<coefficient_rank, dim, Scalar>;

  using IntegratorCell = CellIntegrator<dim, 1, Number>;
  using IntegratorFace = FaceIntegrator<dim, 1, Number>;

  using Range = std::pair<unsigned int, unsigned int>;

public:
  AnalyticalTensorCoefficientFunctionEvaluator(
    dealii::MatrixFree<dim, Number> const & matrix_free,
    unsigned int const                      quad_index,
    dealii::Function<dim> &                 coefficient_function,
    double const                            time,
    VariableCoefficients<CoefficientType> & variable_coefficients)
    : matrix_free(matrix_free),
      quad_index(quad_index),
      coefficient_function(coefficient_function),
      time(time),
      variable_coefficients(variable_coefficients)
  {
  }

  void
  calculate_coefficients()
  {
    VectorType dummy;

    matrix_free.loop(&This::cell_loop_set_coefficients,
                     &This::face_loop_set_coefficients,
                     &This::face_loop_set_coefficients,
                     this,
                     dummy,
                     dummy);

    if(variable_coefficients.stores_cell_based_face_data())
      matrix_free.loop_cell_centric(&This::cell_based_loop_set_coefficients, this, dummy, dummy);
  };

private:
  /**
   * Performs cell and quadrature point loops to calculate and store the variable coefficients.
   */
  void
  cell_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                             VectorType &,
                             VectorType const &,
                             Range const & cell_range)
  {
    IntegratorCell integrator(matrix_free, {}, quad_index);

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      integrator.reinit(cell);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        auto const coefficient =
          FunctionEvaluator<coefficient_rank, dim, Number>::value(coefficient_function,
                                                                  integrator.quadrature_point(q),
                                                                  time);

        variable_coefficients.set_coefficient_cell(cell, q, coefficient);
      }
    }
  };

  /**
   * Performs face and quadrature point loops to calculate and store the variable coefficients.
   */
  void
  face_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                             VectorType &,
                             VectorType const &,
                             Range const & face_range)
  {
    IntegratorFace integrator(matrix_free, true, {}, quad_index);

    for(unsigned int face = face_range.first; face < face_range.second; face++)
    {
      integrator.reinit(face);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        auto const coefficient =
          FunctionEvaluator<coefficient_rank, dim, Number>::value(coefficient_function,
                                                                  integrator.quadrature_point(q),
                                                                  time);

        variable_coefficients.set_coefficient_face(face, q, coefficient);
      }
    }
  };

  /**
   * Performs cell-based face and quadrature loops to calculate and store the variable coefficients.
   */
  void
  cell_based_loop_set_coefficients(dealii::MatrixFree<dim, Number> const & matrix_free,
                                   VectorType &,
                                   VectorType const &,
                                   Range const & cell_range)
  {
    IntegratorFace integrator(matrix_free, true, {}, quad_index);

    unsigned int const n_faces_per_cell =
      matrix_free.get_dof_handler().get_triangulation().get_reference_cells()[0].n_faces();

    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      for(unsigned int face = 0; face < n_faces_per_cell; ++face)
      {
        integrator.reinit(cell, face);

        unsigned int const cell_based_face_index = integrator.get_current_cell_index();

        for(unsigned int q = 0; q < integrator.n_q_points; ++q)
        {
          auto const coefficient =
            FunctionEvaluator<coefficient_rank, dim, Number>::value(coefficient_function,
                                                                    integrator.quadrature_point(q),
                                                                    time);

          variable_coefficients.set_coefficient_face_cell_based(cell_based_face_index,
                                                                q,
                                                                coefficient);
        }
      }
    }
  };

  dealii::MatrixFree<dim, Number> const & matrix_free;
  unsigned int const                      quad_index;
  dealii::Function<dim> &                 coefficient_function;
  double const                            time;
  VariableCoefficients<CoefficientType> & variable_coefficients;
};

/**
 * Visitor to access the coefficient on cells to use with variants either holding a single constant
 * coefficient value or a VariableCoefficients object,
 *
 * i.e. visitor for std::variant<CoefficientType, VariableCoefficients<CoefficientType>>
 */
template<typename CoefficientType>
struct GetCoefficientCell
{
  unsigned int const cell;
  unsigned int const q;

  CoefficientType
  operator()(CoefficientType const & coeff)
  {
    return coeff;
  }

  CoefficientType
  operator()(VariableCoefficients<CoefficientType> const & coeff)
  {
    return coeff.get_coefficient_cell(cell, q);
  }
};

/**
 * Visitor to access the coefficient on faces in variable coefficient variants, see
 * @ExaDG::VarCoeffUtils::GetCoefficientCell
 */
template<typename CoefficientType>
struct GetCoefficientFace
{
  unsigned int const face;
  unsigned int const q;

  CoefficientType
  operator()(CoefficientType const & coeff)
  {
    return coeff;
  }

  CoefficientType
  operator()(VariableCoefficients<CoefficientType> const & coeff)
  {
    return coeff.get_coefficient_cell(face, q);
  }
};

/**
 * Visitor to access the coefficient on faces with cell-based access in variable coefficient
 * variants, see @ExaDG::VarCoeffUtils::GetCoefficientCell
 */
template<typename CoefficientType>
struct GetCoefficientFaceCellBased
{
  unsigned int const cell_based_face;
  unsigned int const q;

  CoefficientType
  operator()(CoefficientType const & coeff)
  {
    return coeff;
  }

  CoefficientType
  operator()(VariableCoefficients<CoefficientType> const & coeff)
  {
    return coeff.get_coefficient_face_cell_based(cell_based_face, q);
  }
};

} // namespace VarCoeffUtils
} // namespace ExaDG

#endif /* INCLUDE_EXADG_OPERATORS_VARIABLE_COEFFICIENTS_H_ */
