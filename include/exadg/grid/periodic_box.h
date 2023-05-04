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

#ifndef APPLICATIONS_GRID_TOOLS_PERIODIC_BOX_H_
#define APPLICATIONS_GRID_TOOLS_PERIODIC_BOX_H_

// deal.II
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>

// ExaDG
#include <exadg/grid/deformed_cube_manifold.h>

namespace ExaDG
{
template<int dim>
void
create_periodic_box(std::shared_ptr<dealii::Triangulation<dim>>              triangulation,
                    unsigned int const                                       n_refine_space,
                    std::vector<dealii::GridTools::PeriodicFacePair<
                      typename dealii::Triangulation<dim>::cell_iterator>> & periodic_faces,
                    unsigned int const                                       n_subdivisions,
                    double const                                             left,
                    double const                                             right,
                    bool const   curvilinear_mesh = false,
                    double const deformation      = 0.1)
{
  dealii::GridGenerator::subdivided_hyper_cube(*triangulation, n_subdivisions, left, right);

  if(curvilinear_mesh)
  {
    unsigned int const               frequency = 2;
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequency);
    triangulation->set_all_manifold_ids(1);
    triangulation->set_manifold(1, manifold);

    std::vector<bool> vertex_touched(triangulation->n_vertices(), false);

    for(auto const & cell : triangulation->cell_iterators())
    {
      for(unsigned int const v : cell->vertex_indices())
      {
        if(vertex_touched[cell->vertex_index(v)] == false)
        {
          dealii::Point<dim> & vertex           = cell->vertex(v);
          dealii::Point<dim>   new_point        = manifold.push_forward(vertex);
          vertex                                = new_point;
          vertex_touched[cell->vertex_index(v)] = true;
        }
      }
    }
  }

  for(auto const & cell : triangulation->cell_iterators())
  {
    for(unsigned int const face_number : cell->face_indices())
    {
      // x-direction
      if((std::fabs(cell->face(face_number)->center()(0) - left) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(0);
      else if((std::fabs(cell->face(face_number)->center()(0) - right) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(1);
      // y-direction
      else if((std::fabs(cell->face(face_number)->center()(1) - left) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(2);
      else if((std::fabs(cell->face(face_number)->center()(1) - right) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(3);
      // z-direction
      else if(dim == 3 and (std::fabs(cell->face(face_number)->center()(2) - left) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(4);
      else if(dim == 3 and (std::fabs(cell->face(face_number)->center()(2) - right) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(5);
    }
  }

  dealii::GridTools::collect_periodic_faces(
    *triangulation, 0, 1, 0 /*x-direction*/, periodic_faces);
  dealii::GridTools::collect_periodic_faces(
    *triangulation, 2, 3, 1 /*y-direction*/, periodic_faces);
  if(dim == 3)
    dealii::GridTools::collect_periodic_faces(
      *triangulation, 4, 5, 2 /*z-direction*/, periodic_faces);

  triangulation->add_periodicity(periodic_faces);

  // perform global refinements
  triangulation->refine_global(n_refine_space);
}

} // namespace ExaDG

#endif /* APPLICATIONS_GRID_TOOLS_PERIODIC_BOX_H_ */
