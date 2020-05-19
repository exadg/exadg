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

  typedef unsigned int quad_index;
  typedef unsigned int mpi_rank;

  typedef std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/> Id;

  typedef std::map<Id, types::global_dof_index> MapIndex;

  typedef std::pair<std::vector<types::global_dof_index>, std::vector<double>> Cache;

  typedef std::vector<Point<dim>>                             ArrayQuadraturePoints;
  typedef std::vector<std::vector<Cache>>                     ArrayVectorCache;
  typedef std::vector<std::vector<Tensor<rank, dim, double>>> ArrayVectorTensor;

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
    matrix_free_dst = matrix_free_dst_in;
    dof_index_dst   = dof_index_dst_in;
    quad_rules_dst  = quad_indices_dst_in;
    map_bc          = map_bc_in;
    dof_handler_src = &dof_handler_src_in;
    mapping_src     = &mapping_src_in;

    // implementation needs Number = double
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

    for(auto quadrature = quad_rules_dst.begin(); quadrature != quad_rules_dst.end(); ++quadrature)
    {
      // initialize maps
      map_index_dst.emplace(*quadrature, MapIndex());
      map_q_points_dst.emplace(*quadrature, ArrayQuadraturePoints());
      map_solution_dst.emplace(*quadrature, ArrayVectorTensor());

      MapIndex &              map_index          = map_index_dst.find(*quadrature)->second;
      ArrayQuadraturePoints & array_q_points_dst = map_q_points_dst.find(*quadrature)->second;
      ArrayVectorTensor &     array_solution_dst = map_solution_dst.find(*quadrature)->second;


      /*
       * 1. Setup: create map "ID <-> vector_index" and fill array of quadrature points
       */
      for(unsigned int face = matrix_free_dst->n_inner_face_batches();
          face <
          matrix_free_dst->n_inner_face_batches() + matrix_free_dst->n_boundary_face_batches();
          ++face)
      {
        // only consider relevant boundary IDs
        if(map_bc.find(matrix_free_dst->get_boundary_id(face)) != map_bc.end())
        {
          Integrator integrator(*matrix_free_dst, true, dof_index_dst, *quadrature);
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
              types::global_dof_index index = array_q_points_dst.size();
              map_index.emplace(id, index);
              array_q_points_dst.push_back(q_point);
            }
          }
        }
      }


      /*
       * TODO
       * 2. Communication: receive and cache quadrature points of other ranks,
       *    redundantly store own q-points (those that are needed)
       */
      map_q_points_src.emplace(*quadrature, std::map<mpi_rank, ArrayQuadraturePoints>());
      map_cache_src.emplace(*quadrature, std::map<mpi_rank, ArrayVectorCache>());
      map_solution_src.emplace(*quadrature, std::map<mpi_rank, ArrayVectorTensor>());

      std::map<mpi_rank, ArrayQuadraturePoints> & map_mpi_q_points_src =
        map_q_points_src.find(*quadrature)->second;
      std::map<mpi_rank, ArrayVectorCache> & map_mpi_cache_src =
        map_cache_src.find(*quadrature)->second;
      std::map<mpi_rank, ArrayVectorTensor> & map_mpi_solution_src =
        map_solution_src.find(*quadrature)->second;

      // for the serial case, simply copy array of quadrature points
      map_mpi_q_points_src.emplace(0, array_q_points_dst);

      map_mpi_cache_src.emplace(0, ArrayVectorCache());

      map_mpi_solution_src.emplace(0, ArrayVectorTensor());


      /*
       * 3. Compute dof indices and shape values for all quadrature points
       */
      for(auto it_mpi = map_mpi_q_points_src.begin(); it_mpi != map_mpi_q_points_src.end();
          ++it_mpi)
      {
        mpi_rank const          proc               = it_mpi->first;
        ArrayQuadraturePoints & array_q_points_src = it_mpi->second;
        ArrayVectorCache &      array_cache_src    = map_mpi_cache_src.find(proc)->second;
        ArrayVectorTensor &     array_solution_src = map_mpi_solution_src.find(proc)->second;

        array_cache_src.resize(array_q_points_src.size());
        array_solution_src.resize(array_q_points_src.size());

        for(types::global_dof_index q = 0; q < array_q_points_src.size(); ++q)
        {
          std::vector<Cache> cache;
          get_dof_indices_and_shape_values(*dof_handler_src,
                                           *mapping_src,
                                           *dof_vector_src_double_ptr,
                                           array_q_points_src[q],
                                           cache);

          AssertThrow(cache.size() > 0,
                      ExcMessage("cache is empty. Check why no adjacent points have been found."));

          array_cache_src[q] = cache;
          array_solution_src[q] =
            std::vector<Tensor<rank, dim, double>>(cache.size(), Tensor<rank, dim, double>());
        }
      }


      /*
       * 4. Communication: transfer results back to dst-side
       */
      // serial case: mpi_rank = 0, simply copy data
      array_solution_dst = map_mpi_solution_src.find(0)->second;
    }

    // finally, give boundary condition access to the data
    for(auto boundary = map_bc.begin(); boundary != map_bc.end(); ++boundary)
    {
      boundary->second->set_data_pointer(map_index_dst, map_solution_dst);
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

    for(auto quadrature = quad_rules_dst.begin(); quadrature != quad_rules_dst.end(); ++quadrature)
    {
      ArrayVectorTensor & array_solution_dst = map_solution_dst.find(*quadrature)->second;

      std::map<mpi_rank, ArrayVectorCache> & map_mpi_cache_src =
        map_cache_src.find(*quadrature)->second;
      std::map<mpi_rank, ArrayVectorTensor> & map_mpi_solution_src =
        map_solution_src.find(*quadrature)->second;

      for(auto it_mpi = map_mpi_cache_src.begin(); it_mpi != map_mpi_cache_src.end(); ++it_mpi)
      {
        mpi_rank const      proc               = it_mpi->first;
        ArrayVectorCache &  array_cache_src    = map_mpi_cache_src.find(proc)->second;
        ArrayVectorTensor & array_solution_src = map_mpi_solution_src.find(proc)->second;

        for(types::global_dof_index q = 0; q < array_cache_src.size(); ++q)
        {
          std::vector<Cache> &                     vector_cache    = array_cache_src[q];
          std::vector<Tensor<rank, dim, double>> & vector_solution = array_solution_src[q];

          // interpolate solution from dof vector using cached data
          for(unsigned int i = 0; i < vector_cache.size(); ++i)
          {
            vector_solution[i] = Interpolator<rank, dim, double>::value(*dof_handler_src,
                                                                        *dof_vector_src_double_ptr,
                                                                        vector_cache[i].first,
                                                                        vector_cache[i].second);
          }
        }
      }

      /*
       * Communication: transfer results back to dst-side
       */
      // serial case: mpi_rank = 0, simply copy data
      array_solution_dst = map_mpi_solution_src.find(0)->second;
    }
  }

private:
  /*
   * dst-side
   */
  std::shared_ptr<MatrixFree<dim, Number>> matrix_free_dst;
  unsigned int                             dof_index_dst;
  std::vector<quad_index>                  quad_rules_dst;

  mutable std::map<quad_index, MapIndex>              map_index_dst;
  mutable std::map<quad_index, ArrayQuadraturePoints> map_q_points_dst;
  mutable std::map<quad_index, ArrayVectorTensor>     map_solution_dst;

  mutable std::map<types::boundary_id, std::shared_ptr<FunctionInterpolation<rank, dim, double>>>
    map_bc;

  /*
   * src-side
   */
  DoFHandler<dim> const * dof_handler_src;
  Mapping<dim> const *    mapping_src;

  mutable std::map<quad_index, std::map<mpi_rank, ArrayQuadraturePoints>> map_q_points_src;
  mutable std::map<quad_index, std::map<mpi_rank, ArrayVectorCache>>      map_cache_src;
  mutable std::map<quad_index, std::map<mpi_rank, ArrayVectorTensor>>     map_solution_src;
};



#endif /* INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_ */
