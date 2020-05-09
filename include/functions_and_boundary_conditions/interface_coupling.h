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

  typedef std::tuple<unsigned int /*face*/, unsigned int /*q*/, unsigned int /*v*/> Id;
  typedef std::pair<std::vector<types::global_dof_index>, std::vector<double>> InterpolationData;
  typedef std::map<Id, std::vector<InterpolationData>>         ArrayInterpolationData;
  typedef std::map<Id, std::vector<Tensor<rank, dim, double>>> ArraySolution;

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

    // initialize maps
    for(auto boundary = map_bc.begin(); boundary != map_bc.end(); ++boundary)
    {
      types::boundary_id const boundary_id = boundary->first;

      global_map_interpolation_data.emplace(boundary_id,
                                            std::map<quad_index, ArrayInterpolationData>());
      global_map_solution.emplace(boundary_id, std::map<quad_index, ArraySolution>());

      std::map<quad_index, ArrayInterpolationData> & map_interpolation_data =
        global_map_interpolation_data.find(boundary_id)->second;
      std::map<quad_index, ArraySolution> & map_solution =
        global_map_solution.find(boundary_id)->second;

      for(auto quad = quad_indices_dst.begin(); quad != quad_indices_dst.end(); ++quad)
      {
        quad_index const quad_id = *quad;

        map_interpolation_data.emplace(quad_id, ArrayInterpolationData());
        map_solution.emplace(quad_id, ArraySolution());
      }
    }

    // fill maps with data
    VectorType dst_dummy;
    matrix_free_dst->loop(&This::cell_loop_empty,
                          &This::face_loop_empty,
                          &This::boundary_face_loop,
                          this,
                          dst_dummy,
                          dof_vector_src_in);


    // give boundary condition access to the data
    for(auto boundary = map_bc.begin(); boundary != map_bc.end(); ++boundary)
    {
      boundary->second->set_data_pointer(global_map_solution.find(boundary->first)->second);
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

    for(auto boundary = map_bc.begin(); boundary != map_bc.end(); ++boundary)
    {
      types::boundary_id const boundary_id = boundary->first;

      std::map<quad_index, ArrayInterpolationData> & map_interpolation_data =
        global_map_interpolation_data.find(boundary_id)->second;
      std::map<quad_index, ArraySolution> & map_solution =
        global_map_solution.find(boundary_id)->second;

      for(auto quad = quad_indices_dst.begin(); quad != quad_indices_dst.end(); ++quad)
      {
        quad_index const quad_id = *quad;

        ArrayInterpolationData & array_interpolation_data =
          map_interpolation_data.find(quad_id)->second;
        ArraySolution & array_solution = map_solution.find(quad_id)->second;

        for(auto q_point = array_solution.begin(); q_point != array_solution.end(); ++q_point)
        {
          Id id = q_point->first;

          std::vector<Tensor<rank, dim, double>> & solution = array_solution.find(id)->second;
          std::vector<InterpolationData> &         interpolation_data =
            array_interpolation_data.find(id)->second;

          for(unsigned int i = 0; i < solution.size(); ++i)
          {
            // interpolate solution from dof vector using cached data
            solution[i] = Interpolator<rank, dim, double>::value(*dof_handler_src,
                                                                 *dof_vector_src_double_ptr,
                                                                 interpolation_data[i].first,
                                                                 interpolation_data[i].second);
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

    (void)dst;

    for(auto boundary = map_bc.begin(); boundary != map_bc.end(); ++boundary)
    {
      types::boundary_id const boundary_id = boundary->first;

      std::map<quad_index, ArrayInterpolationData> & map_interpolation_data =
        global_map_interpolation_data.find(boundary_id)->second;
      std::map<quad_index, ArraySolution> & map_solution =
        global_map_solution.find(boundary_id)->second;

      for(auto quad = quad_indices_dst.begin(); quad != quad_indices_dst.end(); ++quad)
      {
        quad_index const quad_id = *quad;

        ArrayInterpolationData & array_interpolation_data =
          map_interpolation_data.find(quad_id)->second;
        ArraySolution & array_solution = map_solution.find(quad_id)->second;

        Integrator integrator(matrix_free, true, dof_index_dst, quad_id);

        for(unsigned int face = face_range.first; face < face_range.second; face++)
        {
          integrator.reinit(face);

          if(boundary_id == matrix_free.get_boundary_id(face))
          {
            for(unsigned int q = 0; q < integrator.n_q_points; ++q)
            {
              Point<dim, VectorizedArray<Number>> q_points = integrator.quadrature_point(q);

              for(unsigned int v = 0; v < VectorizedArray<Number>::size(); ++v)
              {
                Point<dim> q_point;
                for(unsigned int d = 0; d < dim; ++d)
                  q_point[d] = q_points[d][v];

                std::vector<InterpolationData> interpolation_data;
                get_dof_indices_and_shape_values(*dof_handler_src,
                                                 *mapping_src,
                                                 *dof_vector_src_double_ptr,
                                                 q_point,
                                                 interpolation_data);

                AssertThrow(
                  interpolation_data.size() > 0,
                  ExcMessage(
                    "interpolation_data is empty. Check why no adjacent points have been found."));

                std::vector<Tensor<rank, dim, double>> solution(interpolation_data.size(),
                                                                Tensor<rank, dim, double>());

                Id id = std::make_tuple(face, q, v);
                array_interpolation_data.emplace(id, interpolation_data);
                array_solution.emplace(id, solution);
              }
            }
          }
        }
      }
    }
  }

  std::shared_ptr<MatrixFree<dim, Number>> matrix_free_dst;
  unsigned int                             dof_index_dst;
  std::vector<quad_index>                  quad_indices_dst;

  mutable std::map<types::boundary_id, std::shared_ptr<FunctionInterpolation<rank, dim, double>>>
    map_bc;

  DoFHandler<dim> const * dof_handler_src;
  Mapping<dim> const *    mapping_src;

  mutable std::map<types::boundary_id, std::map<quad_index, ArrayInterpolationData>>
                                                                            global_map_interpolation_data;
  mutable std::map<types::boundary_id, std::map<quad_index, ArraySolution>> global_map_solution;
};



#endif /* INCLUDE_FUNCTIONALITIES_INTERFACE_COUPLING_H_ */
