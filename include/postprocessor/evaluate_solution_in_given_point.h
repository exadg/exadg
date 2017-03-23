/*
 * evaluate_solution_in_given_point.h
 *
 *  Created on: Mar 15, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_EVALUATE_SOLUTION_IN_GIVEN_POINT_H_
#define INCLUDE_POSTPROCESSOR_EVALUATE_SOLUTION_IN_GIVEN_POINT_H_

#include <deal.II/lac/parallel_vector.h>

template<int dim>
void my_point_value(const Mapping<dim>                                &mapping,
                    const DoFHandler<dim>                             &dof_handler,
                    const parallel::distributed::Vector<double>       &solution,
                    typename DoFHandler<dim>::active_cell_iterator const &cell,
                    Point<dim>  const                                 &point_in_ref_coord,
                    Vector<double>                                    &value)
{
  Assert(GeometryInfo<dim>::distance_to_unit_cell(point_in_ref_coord) < 1e-10,ExcInternalError());

  const FiniteElement<dim> &fe = dof_handler.get_fe();

  const Quadrature<dim> quadrature (GeometryInfo<dim>::project_to_unit_cell(point_in_ref_coord));

  FEValues<dim> fe_values(mapping, fe, quadrature, update_values);
  fe_values.reinit(cell);

  // then use this to get the values of the given fe_function at this point
  std::vector<Vector<double> > solution_value(1, Vector<double> (fe.n_components()));
  fe_values.get_function_values(solution, solution_value);
  value = solution_value[0];
}

template<int dim>
void evaluate_solution_in_point(DoFHandler<dim> const                       &dof_handler,
                                Mapping<dim> const                          &mapping,
                                parallel::distributed::Vector<double> const &numerical_solution,
                                Point<dim> const                            &point,
                                double                                      &solution_value)
{
  // processor local variables: initialize with zeros since we add values to these variables
  unsigned int counter = 0;
  solution_value = 0.0;

  // find adjacent cells to specified point by calculating the closest vertex and the cells
  // surrounding this vertex, make sure that at least one cell is found
  unsigned int vertex_id = GridTools::find_closest_vertex(dof_handler, point);
  std::vector<typename DoFHandler<dim>::active_cell_iterator> adjacent_cells_tmp
    = GridTools::find_cells_adjacent_to_vertex(dof_handler,vertex_id);
  Assert(adjacent_cells_tmp.size()>0, ExcMessage("No adjacent cells found for given point."));

  // copy adjacent cells into a set
  std::set<typename DoFHandler<dim>::active_cell_iterator> adjacent_cells (adjacent_cells_tmp.begin(),adjacent_cells_tmp.end());

  // loop over all adjacent cells
  for (typename std::set<typename DoFHandler<dim>::active_cell_iterator>::iterator cell = adjacent_cells.begin(); cell != adjacent_cells.end(); ++cell)
  {
    // go on only if cell is owned by the processor
    if((*cell)->is_locally_owned())
    {
      // this is a safety factor and might be insufficient for strongly distorted elements
      double const factor = 1.1;
      Point<dim> point_in_ref_coord;
      // This if() is needed because the function transform_real_to_unit_cell() throws exception
      // if the point is too far away from the cell.
      // Hence, we make sure that the cell is only considered if the point is close to the cell.
      if((*cell)->center().distance(point) < factor * (*cell)->center().distance(dof_handler.get_triangulation().get_vertices()[vertex_id]))
      {
        try
        {
          point_in_ref_coord = mapping.transform_real_to_unit_cell(*cell, point);
        }
        catch(...)
        {
          std::cerr << std::endl
                    << "Could not transform point from real to unit cell. "
                       "Probably, the specified point is too far away from the cell."
                    << std::endl;
        }

        const double distance = GeometryInfo<dim>::distance_to_unit_cell(point_in_ref_coord);

        // if point lies on the current cell
        if(distance < 1.0e-10)
        {
          Vector<double> value(1);
          my_point_value(mapping,
                         dof_handler,
                         numerical_solution,
                         *cell,
                         point_in_ref_coord,
                         value);

          solution_value += value(0);
          ++counter;
        }
      }
    }
  }

  // parallel computations: add results of all processors and calculate mean value
  counter = Utilities::MPI::sum(counter,MPI_COMM_WORLD);
  Assert(counter>0,ExcMessage("No points found."));

  solution_value = Utilities::MPI::sum(solution_value,MPI_COMM_WORLD);
  solution_value /= (double)counter;
}

///*
// *  Find all active cells around a given point in real space. The return value
// *  is a std::vector of std::pair<cell,point_in_ref_coordinates>.
// *  A <cell,point_in_ref_coordinates> pair is inserted in this vector
// *  if the distance of the point to the cell (in reference space)
// *  is smaller than a tolerance. This function does not ensure that the point
// *  in reference space lies on the cell.
// */
//template<int dim, template<int,int> class MeshType, int spacedim = dim>
//std::vector<std::pair<typename MeshType<dim, spacedim>::active_cell_iterator, Point<dim> > >
//find_all_active_cells_around_point(Mapping<dim> const           &mapping,
//                                   MeshType<dim,spacedim> const &mesh,
//                                   Point<dim> const             &p)
//{
//  std::vector<std::pair<typename MeshType<dim, spacedim>::active_cell_iterator, Point<dim> > > cells;
//
//  // find adjacent cells to specified point by calculating the closest vertex and the cells
//  // surrounding this vertex, make sure that at least one cell is found
//  unsigned int vertex_id = GridTools::find_closest_vertex(mesh, p);
//  std::vector<typename MeshType<dim,spacedim>::active_cell_iterator>
//      adjacent_cells_tmp = GridTools::find_cells_adjacent_to_vertex(mesh,vertex_id);
//  Assert(adjacent_cells_tmp.size()>0, ExcMessage("No adjacent cells found for given point."));
//
//  // copy adjacent cells into a set
//  std::set<typename MeshType<dim,spacedim>::active_cell_iterator> adjacent_cells (adjacent_cells_tmp.begin(),adjacent_cells_tmp.end());
//
//  // loop over all adjacent cells
//  typename std::set<typename MeshType<dim,spacedim>::active_cell_iterator>::iterator cell = adjacent_cells.begin(), endc = adjacent_cells.end();
//  for (; cell != endc; ++cell)
//  {
//    // this is a safety factor and might be insufficient for strongly distorted elements
//    double const factor = 1.1;
//    Point<dim> point_in_ref_coord;
//    // This if() is needed because the function transform_real_to_unit_cell() throws exception
//    // if the point is too far away from the cell.
//    // Hence, we make sure that the cell is only considered if the point is "close to the cell".
//    if((*cell)->center().distance(p) < factor * (*cell)->center().distance(mesh.get_triangulation().get_vertices()[vertex_id]))
//    {
//      try
//      {
//        point_in_ref_coord = mapping.transform_real_to_unit_cell(*cell, p);
//      }
//      catch(...)
//      {
//        // A point that does not lie on the reference cell.
//        point_in_ref_coord = Point<dim>(-1.0,-1.0);
//
//        std::cerr << std::endl
//                  << "Could not transform point from real to unit cell. "
//                     "Probably, the specified point is too far away from the cell."
//                  << std::endl;
//      }
//
//      const double distance = GeometryInfo<dim>::distance_to_unit_cell(point_in_ref_coord);
//
//      // insert cell into vector if point lies on the current cell
//      double const tol = 1.0e-10;
//      if(distance < tol)
//      {
//        cells.push_back(std::make_pair(*cell,point_in_ref_coord));
//      }
//    }
//  }
//}
//
//template<int dim>
//void evaluate_solution_in_point(DoFHandler<dim> const                       &dof_handler,
//                                Mapping<dim> const                          &mapping,
//                                parallel::distributed::Vector<double> const &numerical_solution,
//                                Point<dim> const                            &point,
//                                double                                      &solution_value)
//{
//  // processor local variables: initialize with zeros since we add values to these variables
//  unsigned int counter = 0;
//  solution_value = 0.0;
//
//  typedef std::pair<typename DoFHandler<dim>::active_cell_iterator, Point<dim> > MY_PAIR;
//  std::vector<MY_PAIR> adjacent_cells = find_all_active_cells_around_point(mapping,dof_handler,point);
//
//  // loop over all adjacent cells
//  for (typename std::vector<MY_PAIR>::iterator cell = adjacent_cells.begin(); cell != adjacent_cells.end(); ++cell)
//  {
//    // go on only if cell is owned by the processor
//    if(cell->first->is_locally_owned())
//    {
//        Vector<double> value(1);
//        my_point_value(mapping,
//                       dof_handler,
//                       numerical_solution,
//                       cell->first,
//                       cell->second,
//                       value);
//
//        solution_value += value(0);
//        ++counter;
//    }
//  }
//
//  // parallel computations: add results of all processors and calculate mean value
//  counter = Utilities::MPI::sum(counter,MPI_COMM_WORLD);
//  Assert(counter>0,ExcMessage("No points found."));
//
//  solution_value = Utilities::MPI::sum(solution_value,MPI_COMM_WORLD);
//  solution_value /= (double)counter;
//}


#endif /* INCLUDE_POSTPROCESSOR_EVALUATE_SOLUTION_IN_GIVEN_POINT_H_ */
