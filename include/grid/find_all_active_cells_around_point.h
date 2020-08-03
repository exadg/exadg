/*
 * find_all_active_cells_around_point.h
 *
 *  Created on: Jul 28, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_GRID_FIND_ALL_ACTIVE_CELLS_AROUND_POINT_H_
#define INCLUDE_GRID_FIND_ALL_ACTIVE_CELLS_AROUND_POINT_H_


template<int dim>
std::vector<std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>>>
find_all_active_cells_around_point(Mapping<dim> const &                                mapping,
                                   Triangulation<dim> const &                          tria,
                                   Point<dim> const &                                  point,
                                   double const                                        tolerance,
                                   typename Triangulation<dim>::active_cell_iterator & cell_hint,
                                   std::vector<bool> const &          marked_vertices,
                                   GridTools::Cache<dim, dim> const & cache)
{
  typedef std::pair<typename Triangulation<dim>::active_cell_iterator, Point<dim>> Pair;

  std::vector<Pair> adjacent_cells;

  try
  {
    Pair first_cell =
      GridTools::find_active_cell_around_point(cache, point, cell_hint, marked_vertices, tolerance);

    // update cell_hint to have a good hint when the function is called next time
    cell_hint = first_cell.first;

    adjacent_cells =
      GridTools::find_all_active_cells_around_point(mapping, tria, point, tolerance, first_cell);
  }
  catch(...)
  {
  }

  return adjacent_cells;
}

template<int dim>
unsigned int
n_locally_owned_active_cells_around_point(
  Triangulation<dim> const &                               tria,
  Mapping<dim> const &                                     mapping,
  Point<dim> const &                                       point,
  double const                                             tolerance,
  typename Triangulation<dim, dim>::active_cell_iterator & cell_hint,
  std::vector<bool> const &                                marked_vertices,
  GridTools::Cache<dim, dim> const &                       cache)
{
  auto adjacent_cells = find_all_active_cells_around_point(
    mapping, tria, point, tolerance, cell_hint, marked_vertices, cache);

  // count locally owned active cells
  unsigned int counter = 0;
  for(auto cell : adjacent_cells)
  {
    if(cell.first->is_locally_owned())
    {
      ++counter;
    }
  }

  return counter;
}


#endif /* INCLUDE_GRID_FIND_ALL_ACTIVE_CELLS_AROUND_POINT_H_ */
