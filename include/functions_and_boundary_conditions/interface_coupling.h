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

using namespace dealii;

template<int dim, int n_components, typename Number>
class InterfaceCoupling
{
private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : numbers::invalid_unsigned_int);

  typedef InterfaceCoupling<dim, n_components, Number> This;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;
  typedef FaceIntegrator<dim, n_components, Number>  Integrator;
  typedef std::pair<unsigned int, unsigned int>      Range;

  typedef std::pair<std::vector<types::global_dof_index>, std::vector<double>> InterpolationData;
  typedef std::vector<std::vector<InterpolationData>> ArrayInterpolationData;

  typedef std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/> Id;
  typedef std::map<Id, types::global_dof_index>                                     MapVectorIndex;
  typedef std::vector<Point<dim>>                ArrayQuadraturePoints;
  typedef std::vector<Tensor<rank, dim, double>> ArraySolution;

  typedef unsigned int quad_index;

public:
  InterfaceCoupling(MPI_Comm const & mpi_comm)
    : dof_index_dst(0), dof_handler_src(nullptr), mapping_src(nullptr)
  {
    AssertThrow(Utilities::MPI::n_mpi_processes(mpi_comm) == 1,
                ExcMessage("InterfaceCoupling is currently only implemented for serial case."));
  }

  void
  setup(std::shared_ptr<MatrixFree<dim, Number>> matrix_free_dst_in,
        unsigned int const                       dof_index_dst_in,
        std::vector<quad_index> const &          quad_indices_dst_in,
        std::map<types::boundary_id,
                 std::shared_ptr<FunctionInterpolation<rank, dim, double>>> const & map_bc_in,
        DoFHandler<dim> const & dof_handler_src_in,
        Mapping<dim> const &    mapping_src_in,
        VectorType const &      dof_vector_src_in)
  {
    matrix_free_dst  = matrix_free_dst_in;
    dof_index_dst    = dof_index_dst_in;
    quad_indices_dst = quad_indices_dst_in;
    map_bc           = map_bc_in;
    dof_handler_src  = &dof_handler_src_in;
    mapping_src      = &mapping_src_in;

    // 1. Setup: create map "ID <-> vector_index" and fill array of quadrature points
    for(auto quad = quad_indices_dst.begin(); quad != quad_indices_dst.end(); ++quad)
    {
      quad_index const quad_id = *quad;

      // initialize maps
      global_map_vector_index.emplace(quad_id, MapVectorIndex());
      map_quadrature_points.emplace(quad_id, ArrayQuadraturePoints());
      map_interpolation_data.emplace(quad_id, ArrayInterpolationData());
      map_solution.emplace(quad_id, ArraySolution());

      MapVectorIndex &        map_vector_index = global_map_vector_index.find(quad_id)->second;
      ArrayQuadraturePoints & array_q_points   = map_quadrature_points.find(quad_id)->second;

      for(unsigned int face = matrix_free_dst->n_inner_face_batches();
          face <
          matrix_free_dst->n_inner_face_batches() + matrix_free_dst->n_boundary_face_batches();
          ++face)
      {
        // only consider relevant boundary IDs
        if(map_bc.find(matrix_free_dst->get_boundary_id(face)) != map_bc.end())
        {
          Integrator integrator(*matrix_free_dst, true, dof_index_dst, quad_id);
          integrator.reinit(face);

          for(unsigned int q = 0; q < integrator.n_q_points; ++q)
          {
            Point<dim, VectorizedArray<Number>> q_points = integrator.quadrature_point(q);

            for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
            {
              Point<dim> q_point;
              for(unsigned int d = 0; d < dim; ++d)
                q_point[d] = q_points[d][v];

              Id                      id    = std::make_tuple(face, q, v);
              types::global_dof_index index = array_q_points.size();
              map_vector_index.emplace(id, index);
              array_q_points.push_back(q_point);
            }
          }
        }
      }
    }

    /*
     * TODO
     * 2. Communication: receive and cache quadrature points of other ranks,
     *    redundantly store own q-points (those that are needed)
     */

    /*
     * 3. Compute dof indices and shape values for all quadrature points, and initialize solution
     */
    VectorTypeDouble         dof_vector_src_double_copy;
    VectorTypeDouble const * dof_vector_src_double_ptr;
    if(std::is_same<double, Number>::value)
    {
      dof_vector_src_double_ptr = reinterpret_cast<VectorTypeDouble const *>(&dof_vector_src_in);
    }
    else
    {
      dof_vector_src_double_copy = dof_vector_src_in;
      dof_vector_src_double_ptr  = &dof_vector_src_double_copy;
    }

    for(auto quad = quad_indices_dst.begin(); quad != quad_indices_dst.end(); ++quad)
    {
      quad_index const quad_id = *quad;

      MapVectorIndex &         map_vector_index = global_map_vector_index.find(quad_id)->second;
      ArrayQuadraturePoints &  array_q_points   = map_quadrature_points.find(quad_id)->second;
      ArrayInterpolationData & array_interpolation_data =
        map_interpolation_data.find(quad_id)->second;
      ArraySolution & array_solution = map_solution.find(quad_id)->second;

      array_interpolation_data.resize(map_vector_index.size());
      array_solution.resize(map_vector_index.size());

      for(auto iter = map_vector_index.begin(); iter != map_vector_index.end(); ++iter)
      {
        types::global_dof_index index = iter->second;

        std::vector<InterpolationData> interpolation_data;
        get_dof_indices_and_shape_values(*dof_handler_src,
                                         *mapping_src,
                                         *dof_vector_src_double_ptr,
                                         array_q_points[index],
                                         interpolation_data);

        AssertThrow(
          interpolation_data.size() > 0,
          ExcMessage("interpolation_data is empty. Check why no adjacent points have been found."));

        array_interpolation_data[index] = interpolation_data;
        array_solution[index]           = Tensor<rank, dim, double>();
      }
    }

    /*
     * 4. Communication: get results for all local quadrature points on the dst-side
     */

    // finally, give boundary condition access to the data
    for(auto boundary = map_bc.begin(); boundary != map_bc.end(); ++boundary)
    {
      boundary->second->set_data_pointer(global_map_vector_index, map_solution);
    }
  }

  void
  update_data(VectorType const & dof_vector_src)
  {
    VectorTypeDouble         dof_vector_src_double_copy;
    VectorTypeDouble const * dof_vector_src_double_ptr;
    if(std::is_same<double, Number>::value)
    {
      dof_vector_src_double_ptr = reinterpret_cast<VectorTypeDouble const *>(&dof_vector_src);
    }
    else
    {
      dof_vector_src_double_copy = dof_vector_src;
      dof_vector_src_double_ptr  = &dof_vector_src_double_copy;
    }

    for(auto quad = quad_indices_dst.begin(); quad != quad_indices_dst.end(); ++quad)
    {
      quad_index const quad_id = *quad;

      MapVectorIndex &         map_vector_index = global_map_vector_index.find(quad_id)->second;
      ArrayInterpolationData & array_interpolation_data =
        map_interpolation_data.find(quad_id)->second;
      ArraySolution & array_solution = map_solution.find(quad_id)->second;

      for(auto iter = map_vector_index.begin(); iter != map_vector_index.end(); ++iter)
      {
        types::global_dof_index const index = iter->second;

        Tensor<rank, dim, double> &      solution = array_solution[index];
        std::vector<InterpolationData> & vector_interpolation_data =
          array_interpolation_data[index];

        // interpolate solution from dof vector using cached data
        // and average over all adjacent points
        unsigned int counter = 0;
        // init with zeros since we accumulate into this variable
        solution = Tensor<rank, dim, double>();
        for(unsigned int i = 0; i < vector_interpolation_data.size(); ++i)
        {
          solution += Interpolator<rank, dim, double>::value(*dof_handler_src,
                                                             *dof_vector_src_double_ptr,
                                                             vector_interpolation_data[i].first,
                                                             vector_interpolation_data[i].second);
          ++counter;
        }

        solution *= 1.0 / (double)counter;
      }
    }
  }

private:
  std::shared_ptr<MatrixFree<dim, Number>> matrix_free_dst;
  unsigned int                             dof_index_dst;
  std::vector<quad_index>                  quad_indices_dst;

  mutable std::map<types::boundary_id, std::shared_ptr<FunctionInterpolation<rank, dim, double>>>
    map_bc;

  DoFHandler<dim> const * dof_handler_src;
  Mapping<dim> const *    mapping_src;

  mutable std::map<quad_index, MapVectorIndex>         global_map_vector_index;
  mutable std::map<quad_index, ArrayQuadraturePoints>  map_quadrature_points;
  mutable std::map<quad_index, ArrayInterpolationData> map_interpolation_data;
  mutable std::map<quad_index, ArraySolution>          map_solution;
};



#endif /* INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_ */
