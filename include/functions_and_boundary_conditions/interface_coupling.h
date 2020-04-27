/*
 * interface_coupling.h
 *
 *  Created on: Mar 5, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_
#define INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_

#include "../postprocessor/evaluate_solution_in_given_point.h"
#include "function_interpolation.h"

#include "../matrix_free/matrix_free_wrapper.h"

using namespace dealii;

template<int dim, int n_components, typename Number>
class InterfaceCoupling
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : numbers::invalid_unsigned_int);

  typedef InterfaceCoupling<dim, n_components, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef FaceIntegrator<dim, n_components, Number>  Integrator;
  typedef std::pair<unsigned int, unsigned int>      Range;

  typedef std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/> Id;
  typedef std::pair<unsigned int, std::vector<Number>>         DofIndexAndShapeValues;
  typedef std::map<Id, std::vector<DofIndexAndShapeValues>>    ArrayBookmarks;
  typedef std::map<Id, std::vector<Tensor<rank, dim, Number>>> ArraySolution;

public:
  InterfaceCoupling(MPI_Comm const & mpi_comm)
    : dof_index_dst(0), dof_handler_src(nullptr), mapping_src(nullptr)
  {
    AssertThrow(Utilities::MPI::n_mpi_processes(mpi_comm) == 1,
                ExcMessage("InterfaceCoupling is currently only implemented for serial case."));
  }

  void
  setup(
    std::shared_ptr<MatrixFree<dim, Number>>     matrix_free_dst_in,
    std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data_dst_in,
    std::vector<std::string> const &             quadrature_rules_dst_in,
    unsigned int &                               dof_index_dst_in,
    std::map<types::boundary_id, std::shared_ptr<FunctionInterpolation<rank, dim>>> const & bc_in,
    DoFHandler<dim> const & dof_handler_src_in,
    Mapping<dim> const &    mapping_src_in,
    VectorType const &      dof_vector_src)
  {
    matrix_free_dst      = matrix_free_dst_in;
    matrix_free_data_dst = matrix_free_data_dst_in;
    quadrature_rules_dst = quadrature_rules_dst_in;
    dof_index_dst        = dof_index_dst_in;
    bc                   = bc_in;

    dof_handler_src = &dof_handler_src_in;
    mapping_src     = &mapping_src_in;

    VectorType dst_dummy;
    matrix_free_dst->loop(&This::cell_loop_empty,
                          &This::face_loop_empty,
                          &This::boundary_face_loop,
                          this,
                          dst_dummy,
                          dof_vector_src);
  }

  void
  update_data(VectorType const & dof_vector_src)
  {
    for(auto boundary = bc.begin(); boundary != bc.end(); ++boundary)
    {
      std::map<unsigned int, ArrayBookmarks> & map_bookmarks =
        global_map_bookmarks.find(boundary->first)->second;
      std::map<unsigned int, ArraySolution> & map_solution =
        global_map_solution.find(boundary->first)->second;

      for(auto quad = quadrature_rules_dst.begin(); quad != quadrature_rules_dst.end(); ++quad)
      {
        unsigned int quad_index = matrix_free_data_dst->get_quad_index(*quad);

        ArrayBookmarks & array_bookmarks = map_bookmarks.find(quad_index)->second;
        ArraySolution &  array_solution  = map_solution.find(quad_index)->second;

        for(auto q_point = array_solution.begin(); q_point != array_solution.end(); ++q_point)
        {
          Id id = q_point->first;

          std::vector<Tensor<rank, dim, Number>> & solution = array_solution.find(id)->second;
          std::vector<DofIndexAndShapeValues> &    bookmark = array_bookmarks.find(id)->second;

          for(unsigned int i = 0; i < solution.size(); ++i)
          {
            // interpolate solution from dof vector and overwrite data
            solution[i] = Interpolator<rank, dim, Number>::value(*dof_handler_src,
                                                                 dof_vector_src,
                                                                 bookmark[i].first,
                                                                 bookmark[i].second);
          }
        }
      }
    }
  }

private:
  void
  cell_loop_empty(MatrixFree<dim, Number> const &,
                  VectorType &,
                  VectorType const &,
                  Range const &) const
  {
  }

  void
  face_loop_empty(MatrixFree<dim, Number> const &,
                  VectorType &,
                  VectorType const &,
                  Range const &) const
  {
  }

  void
  boundary_face_loop(MatrixFree<dim, Number> const & matrix_free,
                     VectorType &                    dst,
                     VectorType const &              dof_vector_src,
                     Range const &                   face_range) const
  {
    (void)dst;

    for(auto boundary = bc.begin(); boundary != bc.end(); ++boundary)
    {
      std::map<unsigned int, ArrayBookmarks> map_bookmarks;
      std::map<unsigned int, ArraySolution>  map_solution;

      for(auto quad = quadrature_rules_dst.begin(); quad != quadrature_rules_dst.end(); ++quad)
      {
        unsigned int quad_index = matrix_free_data_dst->get_quad_index(*quad);

        Integrator integrator(matrix_free, true, dof_index_dst, quad_index);

        ArrayBookmarks array_bookmarks;
        ArraySolution  array_solution;
        for(unsigned int face = face_range.first; face < face_range.second; face++)
        {
          integrator.reinit(face);

          for(unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            Point<dim, VectorizedArray<Number>> q_points = integrator.quadrature_point(q);

            for(unsigned int v = 0; v < VectorizedArray<Number>::n_array_elements; ++v)
            {
              Point<dim> q_point;
              for(unsigned int d = 0; d < dim; ++d)
                q_point[d] = q_points[d][v];

              std::vector<DofIndexAndShapeValues> bookmark;
              get_dof_index_and_shape_values(
                *dof_handler_src, *mapping_src, dof_vector_src, q_point, bookmark);

              std::vector<Tensor<rank, dim, Number>> solution;
              solution.resize(bookmark.size());

              Id id = std::make_tuple(face, q, v);
              array_bookmarks.emplace(id, bookmark);
              array_solution.emplace(id, solution);
            }
          }
        }

        map_bookmarks.emplace(quad_index, array_bookmarks);
        map_solution.emplace(quad_index, array_solution);
      }

      boundary->second->set_data_pointer(map_solution);

      global_map_bookmarks.emplace(boundary->first, map_bookmarks);
      global_map_solution.emplace(boundary->first, map_solution);
    }
  }

  std::shared_ptr<MatrixFree<dim, Number>>     matrix_free_dst;
  std::shared_ptr<MatrixFreeData<dim, Number>> matrix_free_data_dst;
  std::vector<std::string>                     quadrature_rules_dst;
  unsigned int                                 dof_index_dst;

  mutable std::map<types::boundary_id, std::shared_ptr<FunctionInterpolation<rank, dim>>> bc;

  DoFHandler<dim> const * dof_handler_src;
  Mapping<dim> const *    mapping_src;

  std::map<unsigned int, std::map<unsigned int, ArrayBookmarks>> global_map_bookmarks;
  std::map<unsigned int, std::map<unsigned int, ArraySolution>>  global_map_solution;
};



#endif /* INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_ */
