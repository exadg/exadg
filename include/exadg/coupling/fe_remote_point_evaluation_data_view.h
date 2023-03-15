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

#ifndef INCLUDE_COUPLING_FE_REMOTE_POINT_EVALUATION_DATA_VIEW_H_
#define INCLUDE_COUPLING_FE_REMOTE_POINT_EVALUATION_DATA_VIEW_H_

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

namespace ExaDG
{
template<int dim, typename Number>
class FERemotePointEvaluationDataView
{
public:
  template<typename VectorizedArrayType>
  void
  initialize_volume(const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
                    const unsigned int                                           this_dof_index,
                    const unsigned int                                           this_quad_index)
  {
    // TODO: can we access quadrature points without FEFEval?
    dealii::FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_m(this_matrix_free,
                                                                           this_dof_index,
                                                                           this_quad_index);

    // determine relevant boundary faces and determine how much space each needs
    cell_ptrs.resize(this_matrix_free.n_cell_batches() + 1);
    cell_ptrs[0] = 0;

    for(unsigned int cell = 0; cell < this_matrix_free.n_cell_batches(); ++cell)
    {
      phi_m.reinit(cell);
      cell_ptrs[cell + 1] = cell_ptrs[cell] + phi_m.n_q_points;
    }

    cell_start = 0;
  }

  template<typename VectorizedArrayType>
  void
  initialize_inner_faces(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
    const unsigned int                                           this_dof_index,
    const unsigned int                                           this_quad_index)
  {
    // TODO: can we access quadrature points without FEFaceEval?
    dealii::FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_m(this_matrix_free,
                                                                               true,
                                                                               this_dof_index,
                                                                               this_quad_index);

    // determine relevant boundary faces and determine how much space each needs
    cell_ptrs.resize(this_matrix_free.n_inner_face_batches() + 1);
    cell_ptrs[0] = 0;

    for(unsigned int face = 0; face < this_matrix_free.n_inner_face_batches(); ++face)
    {
      phi_m.reinit(face);
      cell_ptrs[face + 1] = cell_ptrs[face] + phi_m.n_q_points;
    }

    cell_start = 0;
  }

  template<typename VectorizedArrayType>
  void
  initialize_boundary_faces(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
    const unsigned int                                           this_dof_index,
    const unsigned int                                           this_quad_index)
  {
    // TODO: can we access quadrature points without FEFaceEval?
    dealii::FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_m(this_matrix_free,
                                                                               true,
                                                                               this_dof_index,
                                                                               this_quad_index);

    // determine relevant boundary faces and determine how much space each needs
    cell_ptrs.resize(this_matrix_free.n_boundary_face_batches() + 1);
    cell_ptrs[0] = 0;

    for(unsigned int bface = 0; bface < this_matrix_free.n_boundary_face_batches(); ++bface)
    {
      const unsigned int face = bface + this_matrix_free.n_inner_face_batches();

      phi_m.reinit(face);
      cell_ptrs[bface + 1] = cell_ptrs[bface] + phi_m.n_q_points;
    }

    cell_start = this_matrix_free.n_inner_face_batches();
  }

  template<typename VectorizedArrayType>
  void
  initialize_faces(const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
                   const unsigned int                                           this_dof_index,
                   const unsigned int                                           this_quad_index,
                   const std::set<dealii::types::boundary_id> &                 faces)
  {
    // TODO: can we access quadrature points without FEFaceEval?
    dealii::FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_m(this_matrix_free,
                                                                               true,
                                                                               this_dof_index,
                                                                               this_quad_index);

    // determine relevant boundary faces and determine how much space each needs
    cell_ptrs.resize(this_matrix_free.n_boundary_face_batches() + 1);
    cell_ptrs[0] = 0;

    for(unsigned int bface = 0; bface < this_matrix_free.n_boundary_face_batches(); ++bface)
    {
      const unsigned int face = bface + this_matrix_free.n_inner_face_batches();

      if(faces.find(this_matrix_free.get_boundary_id(face)) != faces.end())
      {
        phi_m.reinit(face);
        cell_ptrs[bface + 1] = cell_ptrs[bface] + phi_m.n_q_points;
      }
      else
      {
        cell_ptrs[bface + 1] = cell_ptrs[bface];
      }
    }

    cell_start = this_matrix_free.n_inner_face_batches();
  }

  unsigned int
  get_shift(const unsigned int cell) const
  {
    Assert(cell_start <= cell, dealii::ExcInternalError());
    AssertIndexRange(cell - cell_start, cell_ptrs.size());
    return cell_ptrs[cell - cell_start];
  }

  unsigned int
  size() const
  {
    Assert(cell_ptrs.size() > 0, dealii::ExcInternalError());
    return cell_ptrs.back();
  }

private:
  std::vector<unsigned int> cell_ptrs;

  unsigned int cell_start;
};
} // namespace ExaDG
#endif /*INCLUDE_COUPLING_FE_REMOTE_POINT_EVALUATION_DATA_VIEW_H_*/
