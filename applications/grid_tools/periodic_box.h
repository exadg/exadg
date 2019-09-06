/*
 * periodic_box.h
 *
 *  Created on: Sep 1, 2019
 *      Author: fehn
 */

#ifndef APPLICATIONS_GRID_TOOLS_PERIODIC_BOX_H_
#define APPLICATIONS_GRID_TOOLS_PERIODIC_BOX_H_

#include "deformed_cube_manifold.h"

template<int dim>
void
create_periodic_box(
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
  unsigned int const                                n_refine_space,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                     periodic_faces,
  unsigned int const n_subdivisions,
  double const       left,
  double const       right,
  bool const         curvilinear_mesh = false,
  double const       deformation      = 0.1)
{
  GridGenerator::subdivided_hyper_cube(*triangulation, n_subdivisions, left, right);

  if(curvilinear_mesh)
  {
    unsigned int const               frequency = 2;
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequency);
    triangulation->set_all_manifold_ids(1);
    triangulation->set_manifold(1, manifold);

    std::vector<bool> vertex_touched(triangulation->n_vertices(), false);

    for(typename Triangulation<dim>::cell_iterator cell = triangulation->begin();
        cell != triangulation->end();
        ++cell)
    {
      for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
      {
        if(vertex_touched[cell->vertex_index(v)] == false)
        {
          Point<dim> & vertex                   = cell->vertex(v);
          Point<dim>   new_point                = manifold.push_forward(vertex);
          vertex                                = new_point;
          vertex_touched[cell->vertex_index(v)] = true;
        }
      }
    }
  }

  typename Triangulation<dim>::cell_iterator cell = triangulation->begin(),
                                             endc = triangulation->end();
  for(; cell != endc; ++cell)
  {
    for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
        ++face_number)
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
      else if((std::fabs(cell->face(face_number)->center()(2) - left) < 1e-12) && dim == 3)
        cell->face(face_number)->set_all_boundary_ids(4);
      else if((std::fabs(cell->face(face_number)->center()(2) - right) < 1e-12) && dim == 3)
        cell->face(face_number)->set_all_boundary_ids(5);
    }
  }

  auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0, 1, 0 /*x-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 2, 3, 1 /*y-direction*/, periodic_faces);
  if(dim == 3)
    GridTools::collect_periodic_faces(*tria, 4, 5, 2 /*z-direction*/, periodic_faces);

  triangulation->add_periodicity(periodic_faces);

  // perform global refinements
  triangulation->refine_global(n_refine_space);
}


#endif /* APPLICATIONS_GRID_TOOLS_PERIODIC_BOX_H_ */
