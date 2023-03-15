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

#ifndef INCLUDE_COUPLING_FE_REMOTE_POINT_EVALUATION_COMMUNICATOR_H_
#define INCLUDE_COUPLING_FE_REMOTE_POINT_EVALUATION_COMMUNICATOR_H_

#include <deal.II/base/point.h>
#include <deal.II/base/smartpointer.h>

#include <deal.II/distributed/tria.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>

#include <deal.II/numerics/vector_tools.h>

#include <exadg/coupling/fe_remote_point_evaluation_data.h>
#include <exadg/coupling/fe_remote_point_evaluation_data_view.h>

namespace ExaDG
{
template<int dim, typename Number, typename VectorizedArrayType>
std::tuple<std::vector<std::pair<unsigned int, unsigned int>>, std::vector<dealii::Point<dim>>>
get_points_of_cells(const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
                    const unsigned int                                           this_dof_index,
                    const unsigned int                                           this_quad_index)
{
  std::vector<std::pair<unsigned int, unsigned int>> cells;
  std::vector<dealii::Point<dim>>                    points;

  // TODO: can we access quadrature points without FEEval?
  dealii::FEEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_m(this_matrix_free,
                                                                         this_dof_index,
                                                                         this_quad_index);

  for(unsigned int cell = 0; cell < this_matrix_free.n_cell_batches(); ++cell)
  {
    phi_m.reinit(cell);
    cells.emplace_back(cell, this_matrix_free.n_active_entries_per_cell_batch(cell));

    for(unsigned int v = 0; v < this_matrix_free.n_active_entries_per_cell_batch(cell); ++v)
    {
      for(unsigned int q = 0; q < phi_m.n_q_points; ++q)
      {
        const auto         point = phi_m.quadrature_point(q);
        dealii::Point<dim> temp;
        for(unsigned int i = 0; i < dim; ++i)
          temp[i] = point[i][v];

        points.emplace_back(temp);
      }
    }
  }

  return {cells, points};
}

template<int dim, typename Number, typename VectorizedArrayType>
std::tuple<std::vector<std::pair<unsigned int, unsigned int>>, std::vector<dealii::Point<dim>>>
get_points_of_inner_faces(
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
  const unsigned int                                           this_dof_index,
  const unsigned int                                           this_quad_index)
{
  std::vector<std::pair<unsigned int, unsigned int>> faces;
  std::vector<dealii::Point<dim>>                    points;

  // TODO: can we access quadrature points without FEFaceEval?
  dealii::FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_m(this_matrix_free,
                                                                             true,
                                                                             this_dof_index,
                                                                             this_quad_index);

  for(unsigned int face = 0; face < this_matrix_free.n_inner_face_batches(); ++face)
  {
    phi_m.reinit(face);
    faces.emplace_back(face, this_matrix_free.n_active_entries_per_face_batch(face));

    for(unsigned int v = 0; v < this_matrix_free.n_active_entries_per_face_batch(face); ++v)
    {
      for(unsigned int q = 0; q < phi_m.n_q_points; ++q)
      {
        const auto         point = phi_m.quadrature_point(q);
        dealii::Point<dim> temp;
        for(unsigned int i = 0; i < dim; ++i)
          temp[i] = point[i][v];

        points.emplace_back(temp);
      }
    }
  }
  return {faces, points};
}

template<int dim, typename Number, typename VectorizedArrayType>
std::tuple<std::vector<std::pair<unsigned int, unsigned int>>, std::vector<dealii::Point<dim>>>
get_points_of_boundary_faces(
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
  const unsigned int                                           this_dof_index,
  const unsigned int                                           this_quad_index)
{
  std::vector<std::pair<unsigned int, unsigned int>> faces;
  std::vector<dealii::Point<dim>>                    points;

  // TODO: can we access quadrature points without FEFaceEval?
  dealii::FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_m(this_matrix_free,
                                                                             true,
                                                                             this_dof_index,
                                                                             this_quad_index);

  for(unsigned int bface = 0; bface < this_matrix_free.n_boundary_face_batches(); ++bface)
  {
    const unsigned int face = bface + this_matrix_free.n_inner_face_batches();

    phi_m.reinit(face);
    faces.emplace_back(face, this_matrix_free.n_active_entries_per_face_batch(face));

    for(unsigned int v = 0; v < this_matrix_free.n_active_entries_per_face_batch(face); ++v)
    {
      for(unsigned int q = 0; q < phi_m.n_q_points; ++q)
      {
        const auto         point = phi_m.quadrature_point(q);
        dealii::Point<dim> temp;
        for(unsigned int i = 0; i < dim; ++i)
          temp[i] = point[i][v];

        points.emplace_back(temp);
      }
    }
  }
  return {faces, points};
}


template<int dim, typename Number, typename VectorizedArrayType>
std::tuple<std::vector<std::pair<unsigned int, unsigned int>>, std::vector<dealii::Point<dim>>>
get_points_on_boundary_face(
  const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
  const unsigned int                                           this_dof_index,
  const unsigned int                                           this_quad_index,
  const dealii::types::boundary_id                             primary_id)
{
  std::vector<std::pair<unsigned int, unsigned int>> faces;
  std::vector<dealii::Point<dim>>                    points;

  // TODO: can we access quadrature points without FEFaceEval?
  dealii::FEFaceEvaluation<dim, -1, 0, 1, Number, VectorizedArrayType> phi_m(this_matrix_free,
                                                                             true,
                                                                             this_dof_index,
                                                                             this_quad_index);

  for(unsigned int bface = 0; bface < this_matrix_free.n_boundary_face_batches(); ++bface)
  {
    const unsigned int face = bface + this_matrix_free.n_inner_face_batches();

    if(this_matrix_free.get_boundary_id(face) == primary_id)
    {
      phi_m.reinit(face);
      faces.emplace_back(face, this_matrix_free.n_active_entries_per_face_batch(face));

      for(unsigned int v = 0; v < this_matrix_free.n_active_entries_per_face_batch(face); ++v)
      {
        for(unsigned int q = 0; q < phi_m.n_q_points; ++q)
        {
          const auto         point = phi_m.quadrature_point(q);
          dealii::Point<dim> temp;
          for(unsigned int i = 0; i < dim; ++i)
            temp[i] = point[i][v];

          points.emplace_back(temp);
        }
      }
    }
  }

  return {faces, points};
}

template<int dim>
class MarkVerticesAtBoundaryID
{
public:
  MarkVerticesAtBoundaryID(const dealii::Triangulation<dim> & tria, dealii::types::boundary_id id)
    : tria(tria), id(id)
  {
  }

  std::vector<bool>
  operator()()
  {
    std::vector<bool> mask(tria.n_vertices(), false);

    for(const auto & face : tria.active_face_iterators())
      if(face->at_boundary() && face->boundary_id() == id)
        for(const auto v : face->vertex_indices())
          mask[face->vertex_index(v)] = true;

    return mask;
  }

private:
  const dealii::Triangulation<dim> & tria;
  const dealii::types::boundary_id   id;
};

template<int dim>
class MarkVerticesAtBoundaryIDs
{
public:
  MarkVerticesAtBoundaryIDs(const dealii::Triangulation<dim> &           tria,
                            const std::set<dealii::types::boundary_id> & ids)
    : tria(tria), ids(ids)
  {
  }

  std::vector<bool>
  operator()()
  {
    std::vector<bool> mask(tria.n_vertices(), false);

    for(const auto & face : tria.active_face_iterators())
      if(face->at_boundary() && ids.find(face->boundary_id()) != ids.end())
        for(const auto v : face->vertex_indices())
          mask[face->vertex_index(v)] = true;

    return mask;
  }

private:
  const dealii::Triangulation<dim> &         tria;
  const std::set<dealii::types::boundary_id> ids;
};

/**
 * A class to fill the fields in FERemotePointEvaluationData.
 */
template<int dim, typename Number>
class FERemotePointEvaluationCommunicator : public dealii::Subscriptor
{
public:
  template<typename VectorizedArrayType>
  void
  initialize_volume_coupling(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
    const unsigned int                                           this_dof_index,
    const unsigned int                                           this_quad_index,
    const dealii::Triangulation<dim> &                           other_tria,
    const dealii::Mapping<dim> &                                 other_mapping,
    const double                                                 tol = 1e-6)
  {
    view.initialize_volume(this_matrix_free, this_dof_index, this_quad_index);

    const auto [cells, points] =
      get_points_of_cells(this_matrix_free, this_dof_index, this_quad_index);

    auto rpe = std::make_shared<dealii::Utilities::MPI::RemotePointEvaluation<dim>>(tol, false, 0);

    rpe->reinit(points, other_tria, other_mapping);

    communication_objects.emplace_back(rpe, cells);
  }

  template<typename VectorizedArrayType>
  void
  initialize_inner_face_coupling(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
    const unsigned int                                           this_dof_index,
    const unsigned int                                           this_quad_index,
    const dealii::Triangulation<dim> &                           other_tria,
    const dealii::Mapping<dim> &                                 other_mapping,
    const double                                                 tol = 1e-6)
  {
    view.initialize_inner_faces(this_matrix_free, this_dof_index, this_quad_index);

    const auto [faces, points] =
      get_points_of_inner_faces(this_matrix_free, this_dof_index, this_quad_index);

    auto rpe = std::make_shared<dealii::Utilities::MPI::RemotePointEvaluation<dim>>(tol, false, 0);

    rpe->reinit(points, other_tria, other_mapping);

    communication_objects.emplace_back(rpe, faces);
  }

  template<typename VectorizedArrayType>
  void
  initialize_boundary_face_coupling(
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
    const unsigned int                                           this_dof_index,
    const unsigned int                                           this_quad_index,
    const dealii::Triangulation<dim> &                           other_tria,
    const dealii::Mapping<dim> &                                 other_mapping,
    const double                                                 tol = 1e-6)
  {
    view.initialize_boundary_faces(this_matrix_free, this_dof_index, this_quad_index);

    const auto [faces, points] =
      get_points_of_boundary_faces(this_matrix_free, this_dof_index, this_quad_index);

    auto rpe = std::make_shared<dealii::Utilities::MPI::RemotePointEvaluation<dim>>(tol, false, 0);

    rpe->reinit(points, other_tria, other_mapping);

    communication_objects.emplace_back(rpe, faces);
  }

  bool
  all_points_found() const
  {
    for(auto const [rpe, _] : communication_objects)
      if(!rpe->all_points_found())
        return false;

    return true;
  }


  template<typename VectorizedArrayType>
  void
  initialize_face_pairs(
    const std::vector<std::pair<dealii::types::boundary_id, dealii::types::boundary_id>> &
                                                                 face_pairs,
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
    const unsigned int                                           this_dof_index,
    const unsigned int                                           this_quad_index,
    const dealii::Triangulation<dim> &                           other_tria,
    const dealii::Mapping<dim> &                                 other_mapping,
    const double                                                 tol = 1e-6)
  {
    std::set<dealii::types::boundary_id> faces;
    for(const auto & face_pair : face_pairs)
      faces.insert(face_pair.first);

    view.initialize_faces(this_matrix_free, this_dof_index, this_quad_index, faces);

    // loop could be eliminated if we would use a single RPE
    for(const auto & face_pair : face_pairs)
    {
      const unsigned int primary_id   = face_pair.first;
      const unsigned int secondary_id = face_pair.second;

      const auto [faces, points] =
        get_points_on_boundary_face(this_matrix_free, this_dof_index, this_quad_index, primary_id);

      auto rpe = std::make_shared<dealii::Utilities::MPI::RemotePointEvaluation<dim>>(
        tol, false, 0, MarkVerticesAtBoundaryID(other_tria, secondary_id));

      rpe->reinit(points, other_tria, other_mapping);

      AssertThrow(rpe->all_points_found(), dealii::ExcMessage("Not all remote points found."));

      communication_objects.emplace_back(rpe, faces);
    }
  }

  template<typename VectorizedArrayType>
  void
  reinit(const std::vector<std::pair<dealii::types::boundary_id, dealii::types::boundary_id>> &
                                                                      face_pairs,
         const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
         const unsigned int                                           this_dof_index,
         const unsigned int                                           this_quad_index,
         const dealii::Triangulation<dim> &                           other_tria,
         const dealii::Mapping<dim> &                                 other_mapping)
  {
    AssertThrow(face_pairs.size() == communication_objects.size(),
                dealii::ExcMessage("FacePairs and Comm Objects have to fit!"));

    auto face_pair = face_pairs.begin();
    for(auto & [rpe, faces_old] : communication_objects)
    {
      AssertThrow(&rpe->get_triangulation() == &other_tria,
                  dealii::ExcMessage("Triangulations in RemotePointEvaluation can not change."));

      const unsigned int primary_id = face_pair->first;

      const auto [faces_new, points] =
        get_points_on_boundary_face(this_matrix_free, this_dof_index, this_quad_index, primary_id);

      rpe->reinit(points, other_tria, other_mapping);

      AssertThrow(rpe->all_points_found(), dealii::ExcMessage("Not all remote points found."));

      faces_old = std::move(faces_new);

      ++face_pair;
    }
  }


  template<typename VectorizedArrayType>
  void
  initialize_face_pairs(
    const std::vector<std::pair<dealii::types::boundary_id, dealii::types::boundary_id>> &
                                                                 face_pairs,
    const dealii::MatrixFree<dim, Number, VectorizedArrayType> & this_matrix_free,
    const unsigned int                                           this_dof_index,
    const unsigned int                                           this_quad_index,
    const double                                                 tol = 1e-6)
  {
    this->initialize_face_pairs(
      face_pairs,
      this_matrix_free,
      this_dof_index,
      this_quad_index,
      this_matrix_free.get_dof_handler(this_dof_index).get_triangulation(),
      *this_matrix_free.get_mapping_info().mapping,
      tol);
  }

  template<int n_components,
           typename VectorizedArrayType,
           typename VectorType,
           template<int>
           class MeshType>
  void
  update_ghost_values(
    FERemotePointEvaluationData<dim, n_components, Number, VectorizedArrayType> & dst,
    const MeshType<dim> &                                                         mesh,
    const VectorType &                                                            src,
    const dealii::EvaluationFlags::EvaluationFlags                                eval_flags,
    const dealii::VectorTools::EvaluationFlags::EvaluationFlags                   vec_flags) const
  {
    const bool has_ghost_elements = src.has_ghost_elements();

    if(has_ghost_elements == false)
      src.update_ghost_values();

    // loop could be eliminitated if we would use a single RPE
    for(const auto & communication_object : communication_objects)
    {
      if(eval_flags & dealii::EvaluationFlags::values)
      {
        copy_data(dst.values,
                  dealii::VectorTools::point_values<n_components>(
                    *communication_object.first, mesh, src, vec_flags),
                  communication_object.second);
      }

      if(eval_flags & dealii::EvaluationFlags::gradients)
      {
        copy_data(dst.gradients,
                  dealii::VectorTools::point_gradients<n_components>(
                    *communication_object.first, mesh, src, vec_flags),
                  communication_object.second);
      }

      Assert(!(eval_flags & dealii::EvaluationFlags::hessians), dealii::ExcNotImplemented());
    }

    if(has_ghost_elements == false)
      zero_out_ghost_values(src); // TODO: move to deal.II
  }

  unsigned int
  get_shift(const unsigned int cell) const
  {
    return view.get_shift(cell);
  }

private:
  FERemotePointEvaluationDataView<dim, Number> view;
  std::vector<std::pair<std::shared_ptr<dealii::Utilities::MPI::RemotePointEvaluation<dim>>,
                        std::vector<std::pair<unsigned int, unsigned int>>>>
    communication_objects;

  template<typename T>
  static void
  zero_out_ghost_values(const dealii::Vector<T> &)
  {
    // nothing to do
  }

  template<typename T>
  static void
  zero_out_ghost_values(const dealii::LinearAlgebra::distributed::Vector<T> & vec)
  {
    vec.zero_out_ghost_values();
  }

  template<typename T1, std::size_t n_lanes>
  void
  copy_data(dealii::VectorizedArray<T1, n_lanes> & dst, const unsigned int v, const T1 & src) const
  {
    AssertIndexRange(v, n_lanes);

    dst[v] = src;
  }

  template<typename T1, int rank_, std::size_t n_lanes, int dim_>
  void
  copy_data(dealii::Tensor<rank_, dim_, dealii::VectorizedArray<T1, n_lanes>> & dst,
            const unsigned int                                                  v,
            const dealii::Tensor<rank_, dim_, T1> &                             src) const
  {
    AssertIndexRange(v, n_lanes);

    if constexpr(rank_ == 1)
    {
      for(unsigned int i = 0; i < dim_; ++i)
        dst[i][v] = src[i];
    }
    else
    {
      for(unsigned int i = 0; i < rank_; ++i)
        for(unsigned int j = 0; j < dim_; ++j)
          dst[i][j][v] = src[i][j];
    }
  }

  template<typename T1, int rank_, std::size_t n_lanes, int n_components_, int dim_>
  void
  copy_data(dealii::Tensor<rank_,
                           n_components_,
                           dealii::Tensor<rank_, dim_, dealii::VectorizedArray<T1, n_lanes>>> & dst,
            const unsigned int                                                                  v,
            const dealii::Tensor<rank_, n_components_, dealii::Tensor<rank_, dim_, T1>> & src) const
  {
    if constexpr(rank_ == 1)
    {
      for(unsigned int i = 0; i < n_components_; ++i)
        copy_data(dst[i], v, src[i]);
    }
    else
    {
      for(unsigned int i = 0; i < rank_; ++i)
        for(unsigned int j = 0; j < n_components_; ++j)
          dst[i][j][v] = src[i][j];
    }
  }

  template<typename T1, typename T2>
  void
  copy_data(std::vector<T1> &                                          dst,
            const std::vector<T2> &                                    src,
            const std::vector<std::pair<unsigned int, unsigned int>> & data_ptrs) const
  {
    dst.resize(view.size());

    unsigned int c = 0;
    for(const auto data_ptr : data_ptrs)
    {
      const unsigned int bface     = data_ptr.first;
      const unsigned int n_entries = data_ptr.second;

      for(unsigned int v = 0; v < n_entries; ++v)
        for(unsigned int j = view.get_shift(bface); j < view.get_shift(bface + 1); ++j, ++c)
        {
          AssertIndexRange(j, dst.size());
          AssertIndexRange(c, src.size());

          copy_data(dst[j], v, src[c]);
        }
    }
  }
};
} // namespace ExaDG
#endif /*INCLUDE_COUPLING_FE_REMOTE_POINT_EVALUATION_COMMUNICATOR_H_*/
