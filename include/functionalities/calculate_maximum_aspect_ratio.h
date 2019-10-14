/*
 * calculate_maximum_aspect_ratio.h
 *
 *  Created on: Sep 2, 2019
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_CALCULATE_MAXIMUM_ASPECT_RATIO_H_
#define INCLUDE_FUNCTIONALITIES_CALCULATE_MAXIMUM_ASPECT_RATIO_H_

#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_nothing.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/numerics/vector_tools.h>

using namespace dealii;

template<int dim, typename Number>
double
calculate_aspect_ratio_jacobian(MatrixFree<dim, Number> const & matrix_free,
                                DoFHandler<dim> const &         dof_handler,
                                Mapping<dim> const &            mapping)
{
  double aspect_ratio_global = 0.0;

  unsigned int const fe_degree = dof_handler.get_fe().degree;
  QGauss<dim>        gauss(fe_degree + 1);

  FEValues<dim> fe_values(mapping, dof_handler.get_fe().base_element(0), gauss, update_jacobians);

  // loop over cells of processor
  for(unsigned int cell = 0; cell < matrix_free.n_macro_cells(); ++cell)
  {
    double aspect_ratio_cell = 0.0;

    for(unsigned int v = 0; v < matrix_free.n_components_filled(cell); v++)
    {
      typename DoFHandler<dim>::cell_iterator cell_iter = matrix_free.get_cell_iterator(cell, v);

      fe_values.reinit(cell_iter);

      // loop over quadrature points
      for(unsigned int q = 0; q < gauss.size(); ++q)
      {
        Tensor<2, dim, double>   jacobian = Tensor<2, dim, double>(fe_values.jacobian(q));
        LAPACKFullMatrix<double> J        = LAPACKFullMatrix<double>(dim);
        for(unsigned int i = 0; i < dim; i++)
          for(unsigned int j = 0; j < dim; j++)
            J(i, j) = jacobian[i][j];

        J.compute_svd();

        double const max_sv = J.singular_value(0);
        double const min_sv = J.singular_value(dim - 1);
        double const ar     = max_sv / min_sv;

        aspect_ratio_cell = std::max(aspect_ratio_cell, ar);
      }
    }

    aspect_ratio_global = std::max(aspect_ratio_global, aspect_ratio_cell);
  }

  // find maximum over all processors
  aspect_ratio_global = Utilities::MPI::max(aspect_ratio_global, MPI_COMM_WORLD);

  return aspect_ratio_global;
}

template<int dim>
Vector<double>
calculate_aspect_ratio_of_cells(Triangulation<dim> const & triangulation,
                                Mapping<dim> const &       mapping,
                                Quadrature<dim> const &    quadrature)
{
  FE_Nothing<dim> fe;
  FEValues<dim>   fe_values(mapping, fe, quadrature, update_jacobians);

  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
                                                    endc = triangulation.end();

  Vector<double> aspect_ratio_vector(triangulation.n_active_cells());

  // loop over cells of processor
  for(; cell != endc; ++cell)
  {
    if(cell->is_locally_owned())
    {
      double aspect_ratio_cell = 0.0;

      fe_values.reinit(cell);

      // loop over quadrature points
      for(unsigned int q = 0; q < quadrature.size(); ++q)
      {
        Tensor<2, dim, double>   jacobian = Tensor<2, dim, double>(fe_values.jacobian(q));
        LAPACKFullMatrix<double> J        = LAPACKFullMatrix<double>(dim);
        for(unsigned int i = 0; i < dim; i++)
          for(unsigned int j = 0; j < dim; j++)
            J(i, j) = jacobian[i][j];

        // We intentionally do not want to throw an exception in case of inverted elements
        // since this is not the task of this function. Instead, inf is written into the vector in
        // case of inverted elements.
        if(determinant(jacobian) <= 0)
          aspect_ratio_cell = std::numeric_limits<double>::infinity();
        else
        {
          J.compute_svd();

          double const max_sv = J.singular_value(0);
          double const min_sv = J.singular_value(dim - 1);
          double const ar     = max_sv / min_sv;

          aspect_ratio_cell = std::max(aspect_ratio_cell, ar);
        }
      }

      // fill vector
      aspect_ratio_vector(cell->active_cell_index()) = aspect_ratio_cell;
    }
  }

  return aspect_ratio_vector;
}

template<int dim>
double
calculate_maximum_aspect_ratio(Triangulation<dim> const & triangulation,
                               Mapping<dim> const &       mapping,
                               Quadrature<dim> const &    quadrature)
{
  Vector<double> aspect_ratio_vector =
    calculate_aspect_ratio_of_cells(triangulation, mapping, quadrature);
  return VectorTools::compute_global_error(triangulation,
                                           aspect_ratio_vector,
                                           VectorTools::Linfty_norm);
}

template<int dim>
inline double
calculate_maximum_vertex_distance(typename Triangulation<dim>::active_cell_iterator & cell)
{
  double maximum_vertex_distance = 0.0;

  for(unsigned int i = 0; i < GeometryInfo<dim>::vertices_per_cell; ++i)
  {
    Point<dim> & ref_vertex = cell->vertex(i);
    // start the loop with the second vertex!
    for(unsigned int j = 0; j < GeometryInfo<dim>::vertices_per_cell; ++j)
    {
      if(j != i)
      {
        Point<dim> & vertex     = cell->vertex(j);
        maximum_vertex_distance = std::max(maximum_vertex_distance, vertex.distance(ref_vertex));
      }
    }
  }

  return maximum_vertex_distance;
}

template<int dim>
inline double
calculate_aspect_ratio_vertex_distance(Triangulation<dim> const & triangulation)
{
  typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
                                                    endc = triangulation.end();

  double max_aspect_ratio = 0.0;

  for(; cell != endc; ++cell)
  {
    if(cell->is_locally_owned())
    {
      double minimum_vertex_distance = cell->minimum_vertex_distance();
      double maximum_vertex_distance = calculate_maximum_vertex_distance<dim>(cell);
      // normalize so that a uniform Cartesian mesh has aspect ratio = 1
      double const aspect_ratio =
        (maximum_vertex_distance / minimum_vertex_distance) / std::sqrt(dim);

      max_aspect_ratio = std::max(aspect_ratio, max_aspect_ratio);
    }
  }

  double const global_max_aspect_ratio = Utilities::MPI::max(max_aspect_ratio, MPI_COMM_WORLD);

  return global_max_aspect_ratio;
}



#endif /* INCLUDE_FUNCTIONALITIES_CALCULATE_MAXIMUM_ASPECT_RATIO_H_ */
