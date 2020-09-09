/*
 * verify_boundary_conditions.h
 *
 *  Created on: Mar 9, 2020
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_VERIFY_BOUNDARY_CONDITIONS_H_
#define INCLUDE_FUNCTIONALITIES_VERIFY_BOUNDARY_CONDITIONS_H_

#include <deal.II/grid/tria.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename BoundaryDescriptor>
void
verify_boundary_conditions(
  BoundaryDescriptor const & boundary_descriptor,
  Triangulation<dim> const & triangulation,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
    periodic_face_pairs)
{
  // fill set with periodic boundary ids
  std::set<types::boundary_id> periodic_boundary_ids;
  for(auto periodic_pair = periodic_face_pairs.begin(); periodic_pair != periodic_face_pairs.end();
      ++periodic_pair)
  {
    AssertThrow(periodic_pair->cell[0]->level() == 0,
                ExcMessage("Received periodic face pair on non-zero level"));
    periodic_boundary_ids.insert(
      periodic_pair->cell[0]->face(periodic_pair->face_idx[0])->boundary_id());
    periodic_boundary_ids.insert(
      periodic_pair->cell[1]->face(periodic_pair->face_idx[1])->boundary_id());
  }

  // Make sure that each boundary face has exactly one boundary type
  for(auto cell = triangulation.begin(); cell != triangulation.end(); ++cell)
  {
    for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      if(cell->at_boundary(f))
      {
        types::boundary_id const bid = cell->face(f)->boundary_id();
        boundary_descriptor.verify_boundary_conditions(bid, periodic_boundary_ids);
      }
    }
  }
}

} // namespace ExaDG


#endif /* INCLUDE_FUNCTIONALITIES_VERIFY_BOUNDARY_CONDITIONS_H_ */
