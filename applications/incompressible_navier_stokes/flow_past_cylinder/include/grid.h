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

#ifndef APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_CYLINDER_H_
#define APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_CYLINDER_H_

// boost
#include <boost/math/special_functions/sign.hpp>

// ExaDG
#include <exadg/grid/one_sided_cylindrical_manifold.h>

namespace ExaDG
{
namespace FlowPastCylinder
{
// physical dimensions (diameter D and center coordinate Y_C can be varied)
double const X_0 = 0.0;  // origin (x-coordinate)
double const Y_0 = 0.0;  // origin (y-coordinate)
double const L1  = 0.3;  // x-coordinate of inflow boundary (2d test cases)
double const L2  = 2.5;  // x-coordinate of outflow boundary (=length for 3d test cases)
double const H   = 0.41; // height of channel
double const D   = 0.1;  // cylinder diameter
double const X_C = 0.5;  // center of cylinder (x-coordinate)
double const Y_C = 0.2;  // center of cylinder (y-coordinate)

namespace CircularCylinder
{
double const X_1 = L1;      // left x-coordinate of mesh block around the cylinder
double const X_2 = 0.7;     // right x-coordinate of mesh block around the cylinder
double const R   = D / 2.0; // cylinder radius

// MeshType
// Type1: no refinement around cylinder surface (coarsest mesh has 34 elements in 2D)
// Type2: two layers of spherical cells around cylinder (used in Fehn et al. (JCP, 2017, "On the
// stability of projection methods ...")),
//        (coarsest mesh has 50 elements in 2D)
// Type3: coarse mesh has only one element in direction perpendicular to flow direction,
//        one layer of spherical cells around cylinder for coarsest mesh (coarsest mesh has 12
//        elements in 2D)
// Type4: no refinement around cylinder, coarsest mesh consists of 4 cells for the block that
//        that surrounds the cylinder (coarsest mesh has 8 elements in 2D)
enum class MeshType
{
  Type1,
  Type2,
  Type3,
  Type4
};
MeshType const MESH_TYPE = MeshType::Type2;

// needed for mesh type 2 with two layers of spherical cells around cylinder
double const R_1 = 1.2 * R;
double const R_2 = 1.7 * R;

// needed for mesh type 3 with one layers of spherical cells around cylinder
double const R_3 = 1.75 * R;

// ManifoldType
// Surface manifold: when refining the mesh only the cells close to the manifold-surface are curved
// (should not be used!) Volume manifold: when refining the mesh all child cells are curved since it
// is a volume manifold
enum class ManifoldType
{
  SurfaceManifold,
  VolumeManifold
};
ManifoldType const MANIFOLD_TYPE = ManifoldType::VolumeManifold;

// manifold ID of spherical manifold
unsigned int const MANIFOLD_ID = 10;

// vectors of manifold_ids and face_ids
std::vector<unsigned int> manifold_ids;
std::vector<unsigned int> face_ids;

template<int dim>
void
set_boundary_ids(dealii::Triangulation<dim> & tria, bool compute_in_2d)
{
  // Set the cylinder boundary to 2, outflow to 1, the rest to 0.
  for(typename dealii::Triangulation<dim>::active_cell_iterator cell = tria.begin();
      cell != tria.end();
      ++cell)
  {
    for(unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) // loop over cells
    {
      if(cell->face(f)->at_boundary())
      {
        dealii::Point<dim> point_on_centerline;
        point_on_centerline[0] = X_C;
        point_on_centerline[1] = Y_C;
        if(dim == 3)
          point_on_centerline[dim - 1] = cell->face(f)->center()[2];

        if(std::abs(cell->face(f)->center()[0] - (compute_in_2d ? L1 : X_0)) < 1e-12)
          cell->face(f)->set_all_boundary_ids(0);
        else if(std::abs(cell->face(f)->center()[0] - L2) < 1e-12)
          cell->face(f)->set_all_boundary_ids(1);
        else if(point_on_centerline.distance(cell->face(f)->center()) <= R)
          cell->face(f)->set_all_boundary_ids(2);
        else
          cell->face(f)->set_all_boundary_ids(0);
      }
    }
  }
}

void create_triangulation(dealii::Triangulation<2> & tria, bool const compute_in_2d = true)
{
  AssertThrow(std::abs((X_2 - X_1) - 2.0 * (X_C - X_1)) < 1.0e-12,
              dealii::ExcMessage("Geometry parameters X_1, X_2, X_C invalid!"));

  dealii::Point<2> center = dealii::Point<2>(X_C, Y_C);

  if(MESH_TYPE == MeshType::Type1)
  {
    dealii::Triangulation<2> left, middle, right, tmp, tmp2;

    // left part (only needed for 3D problem)
    std::vector<unsigned int> ref_1(2, 2);
    ref_1[1] = 2;
    dealii::GridGenerator::subdivided_hyper_rectangle(
      left, ref_1, dealii::Point<2>(X_0, Y_0), dealii::Point<2>(X_1, H), false);

    // right part (2D and 3D)
    std::vector<unsigned int> ref_2(2, 9);
    ref_2[1] = 2;
    dealii::GridGenerator::subdivided_hyper_rectangle(
      right, ref_2, dealii::Point<2>(X_2, Y_0), dealii::Point<2>(L2, H), false);

    // create middle part first as a hyper shell
    double const       outer_radius   = (X_2 - X_1) / 2.0;
    unsigned int const n_cells        = 4;
    dealii::Point<2>   current_center = dealii::Point<2>((X_1 + X_2) / 2.0, outer_radius);
    dealii::GridGenerator::hyper_shell(middle, current_center, R, outer_radius, n_cells, true);
    MyCylindricalManifold<2> boundary(current_center);
    middle.set_manifold(0, boundary);
    middle.refine_global(1);

    // then move the vertices to the points where we want them to be to create a slightly asymmetric
    // cube with a hole
    for(dealii::Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
    {
      for(unsigned int v = 0; v < dealii::GeometryInfo<2>::vertices_per_cell; ++v)
      {
        dealii::Point<2> & vertex = cell->vertex(v);
        if(std::abs(vertex[0] - X_2) < 1e-10 && std::abs(vertex[1] - current_center[1]) < 1e-10)
          vertex = dealii::Point<2>(X_2, H / 2.0);
        else if(std::abs(vertex[0] - (current_center[0] + outer_radius / std::sqrt(2.0))) < 1e-10 &&
                std::abs(vertex[1] - (current_center[1] + outer_radius / std::sqrt(2.0))) < 1e-10)
          vertex = dealii::Point<2>(X_2, H);
        else if(std::abs(vertex[0] - (current_center[0] + outer_radius / std::sqrt(2.0))) < 1e-10 &&
                std::abs(vertex[1] - (current_center[1] - outer_radius / std::sqrt(2.0))) < 1e-10)
          vertex = dealii::Point<2>(X_2, Y_0);
        else if(std::abs(vertex[0] - current_center[0]) < 1e-10 &&
                std::abs(vertex[1] - (X_2 - X_1)) < 1e-10)
          vertex = dealii::Point<2>(current_center[0], H);
        else if(std::abs(vertex[0] - current_center[0]) < 1e-10 &&
                std::abs(vertex[1] - X_0) < 1e-10)
          vertex = dealii::Point<2>(current_center[0], X_0);
        else if(std::abs(vertex[0] - (current_center[0] - outer_radius / std::sqrt(2.0))) < 1e-10 &&
                std::abs(vertex[1] - (current_center[1] + outer_radius / std::sqrt(2.0))) < 1e-10)
          vertex = dealii::Point<2>(X_1, H);
        else if(std::abs(vertex[0] - (current_center[0] - outer_radius / std::sqrt(2.0))) < 1e-10 &&
                std::abs(vertex[1] - (current_center[1] - outer_radius / std::sqrt(2.0))) < 1e-10)
          vertex = dealii::Point<2>(X_1, Y_0);
        else if(std::abs(vertex[0] - X_1) < 1e-10 &&
                std::abs(vertex[1] - current_center[1]) < 1e-10)
          vertex = dealii::Point<2>(X_1, H / 2.0);
      }
    }

    // the same for the inner circle
    for(dealii::Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
    {
      for(unsigned int v = 0; v < dealii::GeometryInfo<2>::vertices_per_cell; ++v)
      {
        dealii::Point<2> & vertex = cell->vertex(v);

        // allow to shift cylinder center
        if(std::abs(vertex.distance(current_center) - R) < 1.e-10 ||
           std::abs(vertex.distance(current_center) - (R + (outer_radius - R) / 2.0)) < 1.e-10)
        {
          vertex[0] += center[0] - current_center[0];
          vertex[1] += center[1] - current_center[1];
        }
      }
    }

    // we have to copy the triangulation because we cannot merge triangulations with refinement ...
    dealii::GridGenerator::flatten_triangulation(middle, tmp2);

    if(compute_in_2d)
    {
      dealii::GridGenerator::merge_triangulations(tmp2, right, tria);
    }
    else
    {
      dealii::GridGenerator::merge_triangulations(left, tmp2, tmp);
      dealii::GridGenerator::merge_triangulations(tmp, right, tria);
    }

    if(compute_in_2d)
    {
      // set manifold ID's
      tria.set_all_manifold_ids(0);

      for(dealii::Triangulation<2>::active_cell_iterator cell = tria.begin(); cell != tria.end();
          ++cell)
      {
        if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
        {
          for(unsigned int f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f)
          {
            if(cell->face(f)->at_boundary() && center.distance(cell->face(f)->center()) <= R)
            {
              cell->face(f)->set_all_manifold_ids(MANIFOLD_ID);
            }
          }
        }
        else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
        {
          for(unsigned int f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f)
          {
            bool face_at_sphere_boundary = true;
            for(unsigned int v = 0; v < dealii::GeometryInfo<2 - 1>::vertices_per_cell; ++v)
            {
              if(std::abs(center.distance(cell->face(f)->vertex(v)) - R) > 1e-12)
                face_at_sphere_boundary = false;
            }
            if(face_at_sphere_boundary)
            {
              face_ids.push_back(f);
              unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
              cell->set_all_manifold_ids(manifold_id);
              manifold_ids.push_back(manifold_id);
            }
          }
        }
        else
        {
          AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold ||
                        MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                      dealii::ExcMessage("Specified manifold type not implemented"));
        }
      }
    }
  }
  else if(MESH_TYPE == MeshType::Type2)
  {
    MyCylindricalManifold<2> cylinder_manifold(center);

    dealii::Triangulation<2> left, circle_1, circle_2, circle_tmp, middle, middle_tmp, middle_tmp2,
      right, tmp_3D;

    // left part (only needed for 3D problem)
    std::vector<unsigned int> ref_1(2, 2);
    ref_1[1] = 2;
    dealii::GridGenerator::subdivided_hyper_rectangle(
      left, ref_1, dealii::Point<2>(X_0, Y_0), dealii::Point<2>(X_1, H), false);

    // right part (2D and 3D)
    std::vector<unsigned int> ref_2(2, 9);
    ref_2[1] = 2;
    dealii::GridGenerator::subdivided_hyper_rectangle(
      right, ref_2, dealii::Point<2>(X_2, Y_0), dealii::Point<2>(L2, H), false);

    // create middle part first as a hyper shell
    double const       outer_radius = (X_2 - X_1) / 2.0;
    unsigned int const n_cells      = 4;
    dealii::GridGenerator::hyper_shell(middle, center, R_2, outer_radius, n_cells, true);
    middle.set_all_manifold_ids(MANIFOLD_ID);
    middle.set_manifold(MANIFOLD_ID, cylinder_manifold);
    middle.refine_global(1);

    // two inner circles in order to refine towards the cylinder surface
    unsigned int const n_cells_circle = 8;
    dealii::GridGenerator::hyper_shell(circle_1, center, R, R_1, n_cells_circle, true);
    dealii::GridGenerator::hyper_shell(circle_2, center, R_1, R_2, n_cells_circle, true);

    // then move the vertices to the points where we want them to be to create a slightly asymmetric
    // cube with a hole
    for(dealii::Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
    {
      for(unsigned int v = 0; v < dealii::GeometryInfo<2>::vertices_per_cell; ++v)
      {
        dealii::Point<2> & vertex = cell->vertex(v);
        if(std::abs(vertex[0] - X_2) < 1e-10 && std::abs(vertex[1] - Y_C) < 1e-10)
        {
          vertex = dealii::Point<2>(X_2, H / 2.0);
        }
        else if(std::abs(vertex[0] - (X_C + (X_2 - X_1) / 2.0 / std::sqrt(2))) < 1e-10 &&
                std::abs(vertex[1] - (Y_C + (X_2 - X_1) / 2.0 / std::sqrt(2))) < 1e-10)
        {
          vertex = dealii::Point<2>(X_2, H);
        }
        else if(std::abs(vertex[0] - (X_C + (X_2 - X_1) / 2.0 / std::sqrt(2))) < 1e-10 &&
                std::abs(vertex[1] - (Y_C - (X_2 - X_1) / 2.0 / std::sqrt(2))) < 1e-10)
        {
          vertex = dealii::Point<2>(X_2, Y_0);
        }
        else if(std::abs(vertex[0] - X_C) < 1e-10 &&
                std::abs(vertex[1] - (Y_C + (X_2 - X_1) / 2.0)) < 1e-10)
        {
          vertex = dealii::Point<2>(X_C, H);
        }
        else if(std::abs(vertex[0] - X_C) < 1e-10 &&
                std::abs(vertex[1] - (Y_C - (X_2 - X_1) / 2.0)) < 1e-10)
        {
          vertex = dealii::Point<2>(X_C, Y_0);
        }
        else if(std::abs(vertex[0] - (X_C - (X_2 - X_1) / 2.0 / std::sqrt(2))) < 1e-10 &&
                std::abs(vertex[1] - (Y_C + (X_2 - X_1) / 2.0 / std::sqrt(2))) < 1e-10)
        {
          vertex = dealii::Point<2>(X_1, H);
        }
        else if(std::abs(vertex[0] - (X_C - (X_2 - X_1) / 2.0 / std::sqrt(2))) < 1e-10 &&
                std::abs(vertex[1] - (Y_C - (X_2 - X_1) / 2.0 / std::sqrt(2))) < 1e-10)
        {
          vertex = dealii::Point<2>(X_1, Y_0);
        }
        else if(std::abs(vertex[0] - X_1) < 1e-10 && std::abs(vertex[1] - Y_C) < 1e-10)
        {
          vertex = dealii::Point<2>(X_1, H / 2.0);
        }
      }
    }

    // we have to copy the triangulation because we cannot merge triangulations with refinement ...
    dealii::GridGenerator::flatten_triangulation(middle, middle_tmp);

    dealii::GridGenerator::merge_triangulations(circle_1, circle_2, circle_tmp);
    dealii::GridGenerator::merge_triangulations(middle_tmp, circle_tmp, middle_tmp2);

    if(compute_in_2d)
    {
      dealii::GridGenerator::merge_triangulations(middle_tmp2, right, tria);
    }
    else // 3D
    {
      dealii::GridGenerator::merge_triangulations(left, middle_tmp2, tmp_3D);
      dealii::GridGenerator::merge_triangulations(tmp_3D, right, tria);
    }

    if(compute_in_2d)
    {
      // set manifold ID's
      tria.set_all_manifold_ids(0);

      for(dealii::Triangulation<2>::active_cell_iterator cell = tria.begin(); cell != tria.end();
          ++cell)
      {
        if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
        {
          if(center.distance(cell->center()) <= R_2)
            cell->set_all_manifold_ids(MANIFOLD_ID);
        }
        else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
        {
          if(center.distance(cell->center()) <= R_2)
            cell->set_all_manifold_ids(MANIFOLD_ID);
          else
          {
            for(unsigned int f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f)
            {
              bool face_at_sphere_boundary = true;
              for(unsigned int v = 0; v < dealii::GeometryInfo<2 - 1>::vertices_per_cell; ++v)
              {
                if(std::abs(center.distance(cell->face(f)->vertex(v)) - R_2) > 1e-12)
                  face_at_sphere_boundary = false;
              }
              if(face_at_sphere_boundary)
              {
                face_ids.push_back(f);
                unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
                cell->set_all_manifold_ids(manifold_id);
                manifold_ids.push_back(manifold_id);
              }
            }
          }
        }
      }
    }
  }
  else if(MESH_TYPE == MeshType::Type3)
  {
    dealii::Triangulation<2> left, middle, circle, middle_tmp, right, tmp_3D;

    // left part (only needed for 3D problem)
    std::vector<unsigned int> ref_1(2, 1);
    dealii::GridGenerator::subdivided_hyper_rectangle(
      left, ref_1, dealii::Point<2>(X_0, Y_0), dealii::Point<2>(X_1, H), false);

    // right part (2D and 3D)
    std::vector<unsigned int> ref_2(2, 4);
    ref_2[1] = 1;
    dealii::GridGenerator::subdivided_hyper_rectangle(
      right, ref_2, dealii::Point<2>(X_2, Y_0), dealii::Point<2>(L2, H), false);

    // middle part
    double const       outer_radius = (X_2 - X_1) / 2.0;
    unsigned int const n_cells      = 4;
    dealii::Point<2>   origin;

    // inner circle around cylinder
    dealii::GridGenerator::hyper_shell(circle, origin, R, R_3, n_cells, true);
    dealii::GridTools::rotate(dealii::numbers::PI / 4, circle);
    dealii::GridTools::shift(dealii::Point<2>(outer_radius + X_1, outer_radius), circle);

    // create middle part first as a hyper shell
    dealii::GridGenerator::hyper_shell(
      middle, origin, R_3, outer_radius * std::sqrt(2.0), n_cells, true);
    dealii::GridTools::rotate(dealii::numbers::PI / 4, middle);
    dealii::GridTools::shift(dealii::Point<2>(outer_radius + X_1, outer_radius), middle);

    // then move the vertices to the points where we want them to be
    for(dealii::Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
    {
      for(unsigned int v = 0; v < dealii::GeometryInfo<2>::vertices_per_cell; ++v)
      {
        dealii::Point<2> & vertex = cell->vertex(v);

        // shift two points at the top to a height of H
        if(std::abs(vertex[0] - X_1) < 1e-10 && std::abs(vertex[1] - (X_2 - X_1)) < 1e-10)
        {
          vertex = dealii::Point<2>(X_1, H);
        }
        else if(std::abs(vertex[0] - X_2) < 1e-10 && std::abs(vertex[1] - (X_2 - X_1)) < 1e-10)
        {
          vertex = dealii::Point<2>(X_2, H);
        }

        // allow to shift cylinder center
        dealii::Point<2> current_center = dealii::Point<2>((X_1 + X_2) / 2.0, (X_2 - X_1) / 2.0);
        if(std::abs(vertex.distance(current_center) - R_3) < 1.e-10)
        {
          vertex[0] += center[0] - current_center[0];
          vertex[1] += center[1] - current_center[1];
        }
      }
    }

    // the same for the inner circle
    for(dealii::Triangulation<2>::cell_iterator cell = circle.begin(); cell != circle.end(); ++cell)
    {
      for(unsigned int v = 0; v < dealii::GeometryInfo<2>::vertices_per_cell; ++v)
      {
        dealii::Point<2> & vertex = cell->vertex(v);

        // allow to shift cylinder center
        dealii::Point<2> current_center = dealii::Point<2>((X_1 + X_2) / 2.0, (X_2 - X_1) / 2.0);
        if(std::abs(vertex.distance(current_center) - R) < 1.e-10 ||
           std::abs(vertex.distance(current_center) - R_3) < 1.e-10)
        {
          vertex[0] += center[0] - current_center[0];
          vertex[1] += center[1] - current_center[1];
        }
      }
    }

    dealii::GridGenerator::merge_triangulations(circle, middle, middle_tmp);

    if(compute_in_2d)
    {
      dealii::GridGenerator::merge_triangulations(middle_tmp, right, tria);
    }
    else // 3D
    {
      dealii::GridGenerator::merge_triangulations(left, middle_tmp, tmp_3D);
      dealii::GridGenerator::merge_triangulations(tmp_3D, right, tria);
    }

    if(compute_in_2d)
    {
      // set manifold ID's
      tria.set_all_manifold_ids(0);

      for(dealii::Triangulation<2>::active_cell_iterator cell = tria.begin(); cell != tria.end();
          ++cell)
      {
        if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
        {
          if(center.distance(cell->center()) <= R_3)
          {
            cell->set_all_manifold_ids(MANIFOLD_ID);
          }
          else
          {
            for(unsigned int f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f)
            {
              bool face_at_sphere_boundary = true;
              for(unsigned int v = 0; v < dealii::GeometryInfo<2 - 1>::vertices_per_cell; ++v)
              {
                if(std::abs(center.distance(cell->face(f)->vertex(v)) - R_3) > 1e-12)
                  face_at_sphere_boundary = false;
              }
              if(face_at_sphere_boundary)
              {
                face_ids.push_back(f);
                unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
                cell->set_all_manifold_ids(manifold_id);
                manifold_ids.push_back(manifold_id);
              }
            }
          }
        }
        else
        {
          AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                      dealii::ExcMessage("Specified manifold type not implemented."));
        }
      }
    }
  }
  else if(MESH_TYPE == MeshType::Type4)
  {
    dealii::Triangulation<2> left, middle, circle, right, tmp_3D;

    // left part (only needed for 3D problem)
    std::vector<unsigned int> ref_1(2, 1);
    dealii::GridGenerator::subdivided_hyper_rectangle(
      left, ref_1, dealii::Point<2>(X_0, Y_0), dealii::Point<2>(X_1, H), false);

    // right part (2D and 3D)
    std::vector<unsigned int> ref_2(2, 4);
    ref_2[1] = 1; // only one cell over channel height
    dealii::GridGenerator::subdivided_hyper_rectangle(
      right, ref_2, dealii::Point<2>(X_2, Y_0), dealii::Point<2>(L2, H), false);

    // middle part
    double const       outer_radius = (X_2 - X_1) / 2.0;
    unsigned int const n_cells      = 4;
    dealii::Point<2>   origin;

    // create middle part first as a hyper shell
    dealii::GridGenerator::hyper_shell(
      middle, origin, R, outer_radius * std::sqrt(2.0), n_cells, true);
    dealii::GridTools::rotate(dealii::numbers::PI / 4, middle);
    dealii::GridTools::shift(dealii::Point<2>(outer_radius + X_1, outer_radius), middle);

    // then move the vertices to the points where we want them to be
    for(dealii::Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
    {
      for(unsigned int v = 0; v < dealii::GeometryInfo<2>::vertices_per_cell; ++v)
      {
        dealii::Point<2> & vertex = cell->vertex(v);

        // shift two points at the top to a height of H
        if(std::abs(vertex[0] - X_1) < 1e-10 && std::abs(vertex[1] - (X_2 - X_1)) < 1e-10)
        {
          vertex = dealii::Point<2>(X_1, H);
        }
        else if(std::abs(vertex[0] - X_2) < 1e-10 && std::abs(vertex[1] - (X_2 - X_1)) < 1e-10)
        {
          vertex = dealii::Point<2>(X_2, H);
        }

        // allow to shift cylinder center
        dealii::Point<2> current_center = dealii::Point<2>((X_1 + X_2) / 2.0, (X_2 - X_1) / 2.0);
        if(std::abs(vertex.distance(current_center) - R) < 1.e-10)
        {
          vertex[0] += center[0] - current_center[0];
          vertex[1] += center[1] - current_center[1];
        }
      }
    }


    if(compute_in_2d)
    {
      dealii::GridGenerator::merge_triangulations(middle, right, tria);
    }
    else // 3D
    {
      dealii::GridGenerator::merge_triangulations(left, middle, tmp_3D);
      dealii::GridGenerator::merge_triangulations(tmp_3D, right, tria);
    }

    if(compute_in_2d)
    {
      // set manifold ID's
      tria.set_all_manifold_ids(0);

      for(dealii::Triangulation<2>::active_cell_iterator cell = tria.begin(); cell != tria.end();
          ++cell)
      {
        if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
        {
          for(unsigned int f = 0; f < dealii::GeometryInfo<2>::faces_per_cell; ++f)
          {
            bool face_at_sphere_boundary = true;
            for(unsigned int v = 0; v < dealii::GeometryInfo<2 - 1>::vertices_per_cell; ++v)
            {
              if(std::abs(center.distance(cell->face(f)->vertex(v)) - R) > 1e-12)
                face_at_sphere_boundary = false;
            }
            if(face_at_sphere_boundary)
            {
              face_ids.push_back(f);
              unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
              cell->set_all_manifold_ids(manifold_id);
              manifold_ids.push_back(manifold_id);
            }
          }
        }
        else
        {
          AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                      dealii::ExcMessage("Specified manifold type not implemented."));
        }
      }
    }
  }
  else
  {
    AssertThrow(MESH_TYPE == MeshType::Type1 || MESH_TYPE == MeshType::Type2 ||
                  MESH_TYPE == MeshType::Type3 || MESH_TYPE == MeshType::Type4,
                dealii::ExcMessage("Specified mesh type not implemented"));
  }

  if(compute_in_2d == true)
  {
    // Set boundary ID's
    set_boundary_ids<2>(tria, compute_in_2d);
  }
}


void create_triangulation(dealii::Triangulation<3> & tria)
{
  dealii::Triangulation<2> tria_2d;
  create_triangulation(tria_2d, false);

  if(MESH_TYPE == MeshType::Type1)
  {
    dealii::GridGenerator::extrude_triangulation(tria_2d, 3, H, tria);

    // set manifold ID's
    tria.set_all_manifold_ids(0);

    if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
    {
      for(dealii::Triangulation<3>::active_cell_iterator cell = tria.begin(); cell != tria.end();
          ++cell)
      {
        for(unsigned int f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f)
        {
          if(cell->face(f)->at_boundary() && dealii::Point<3>(X_C, Y_C, cell->face(f)->center()[2])
                                                 .distance(cell->face(f)->center()) <= R)
          {
            cell->face(f)->set_all_manifold_ids(MANIFOLD_ID);
          }
        }
      }
    }
    else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
    {
      for(dealii::Triangulation<3>::active_cell_iterator cell = tria.begin(); cell != tria.end();
          ++cell)
      {
        for(unsigned int f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f)
        {
          bool face_at_sphere_boundary = true;
          for(unsigned int v = 0; v < dealii::GeometryInfo<3 - 1>::vertices_per_cell; ++v)
          {
            if(std::abs(dealii::Point<3>(X_C, Y_C, cell->face(f)->vertex(v)[2])
                          .distance(cell->face(f)->vertex(v)) -
                        R) > 1e-12)
              face_at_sphere_boundary = false;
          }
          if(face_at_sphere_boundary)
          {
            face_ids.push_back(f);
            unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
            cell->set_all_manifold_ids(manifold_id);
            manifold_ids.push_back(manifold_id);
          }
        }
      }
    }
    else
    {
      AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold ||
                    MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                  dealii::ExcMessage("Specified manifold type not implemented"));
    }
  }
  else if(MESH_TYPE == MeshType::Type2)
  {
    dealii::GridGenerator::extrude_triangulation(tria_2d, 3, H, tria);

    // set manifold ID's
    tria.set_all_manifold_ids(0);

    if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
    {
      for(dealii::Triangulation<3>::active_cell_iterator cell = tria.begin(); cell != tria.end();
          ++cell)
      {
        if(dealii::Point<3>(X_C, Y_C, cell->center()[2]).distance(cell->center()) <= R_2)
          cell->set_all_manifold_ids(MANIFOLD_ID);
      }
    }
    else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
    {
      for(dealii::Triangulation<3>::active_cell_iterator cell = tria.begin(); cell != tria.end();
          ++cell)
      {
        if(dealii::Point<3>(X_C, Y_C, cell->center()[2]).distance(cell->center()) <= R_2)
          cell->set_all_manifold_ids(MANIFOLD_ID);
        else
        {
          for(unsigned int f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f)
          {
            bool face_at_sphere_boundary = true;
            for(unsigned int v = 0; v < dealii::GeometryInfo<3 - 1>::vertices_per_cell; ++v)
            {
              if(std::abs(dealii::Point<3>(X_C, Y_C, cell->face(f)->vertex(v)[2])
                            .distance(cell->face(f)->vertex(v)) -
                          R_2) > 1e-12)
                face_at_sphere_boundary = false;
            }
            if(face_at_sphere_boundary)
            {
              face_ids.push_back(f);
              unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
              cell->set_all_manifold_ids(manifold_id);
              manifold_ids.push_back(manifold_id);
            }
          }
        }
      }
    }
    else
    {
      AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold ||
                    MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                  dealii::ExcMessage("Specified manifold type not implemented"));
    }
  }
  else if(MESH_TYPE == MeshType::Type3)
  {
    dealii::GridGenerator::extrude_triangulation(tria_2d, 2, H, tria);

    // set manifold ID's
    tria.set_all_manifold_ids(0);

    if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
    {
      for(dealii::Triangulation<3>::active_cell_iterator cell = tria.begin(); cell != tria.end();
          ++cell)
      {
        if(dealii::Point<3>(X_C, Y_C, cell->center()[2]).distance(cell->center()) <= R_3)
          cell->set_all_manifold_ids(MANIFOLD_ID);
        else
        {
          for(unsigned int f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f)
          {
            bool face_at_sphere_boundary = true;
            for(unsigned int v = 0; v < dealii::GeometryInfo<3 - 1>::vertices_per_cell; ++v)
            {
              if(std::abs(dealii::Point<3>(X_C, Y_C, cell->face(f)->vertex(v)[2])
                            .distance(cell->face(f)->vertex(v)) -
                          R_3) > 1e-12)
                face_at_sphere_boundary = false;
            }
            if(face_at_sphere_boundary)
            {
              face_ids.push_back(f);
              unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
              cell->set_all_manifold_ids(manifold_id);
              manifold_ids.push_back(manifold_id);
            }
          }
        }
      }
    }
    else
    {
      AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                  dealii::ExcMessage("Specified manifold type not implemented"));
    }
  }
  else if(MESH_TYPE == MeshType::Type4)
  {
    dealii::GridGenerator::extrude_triangulation(tria_2d, 2, H, tria);

    // set manifold ID's
    tria.set_all_manifold_ids(0);

    if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
    {
      for(dealii::Triangulation<3>::active_cell_iterator cell = tria.begin(); cell != tria.end();
          ++cell)
      {
        for(unsigned int f = 0; f < dealii::GeometryInfo<3>::faces_per_cell; ++f)
        {
          bool face_at_sphere_boundary = true;
          for(unsigned int v = 0; v < dealii::GeometryInfo<3 - 1>::vertices_per_cell; ++v)
          {
            if(std::abs(dealii::Point<3>(X_C, Y_C, cell->face(f)->vertex(v)[2])
                          .distance(cell->face(f)->vertex(v)) -
                        R) > 1e-12)
              face_at_sphere_boundary = false;
          }
          if(face_at_sphere_boundary)
          {
            face_ids.push_back(f);
            unsigned int manifold_id = MANIFOLD_ID + manifold_ids.size() + 1;
            cell->set_all_manifold_ids(manifold_id);
            manifold_ids.push_back(manifold_id);
          }
        }
      }
    }
    else
    {
      AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                  dealii::ExcMessage("Specified manifold type not implemented"));
    }
  }
  else
  {
    AssertThrow(MESH_TYPE == MeshType::Type1 || MESH_TYPE == MeshType::Type2 ||
                  MESH_TYPE == MeshType::Type3 || MESH_TYPE == MeshType::Type4,
                dealii::ExcMessage("Specified mesh type not implemented"));
  }

  // Set boundary ID's
  set_boundary_ids<3>(tria, false);
}

template<int dim>
void
do_create_grid(dealii::Triangulation<dim> &                             triangulation,
               unsigned int const                                       n_refine_space,
               std::vector<dealii::GridTools::PeriodicFacePair<
                 typename dealii::Triangulation<dim>::cell_iterator>> & periodic_faces)
{
  (void)periodic_faces;

  dealii::Point<dim> center;
  center[0] = X_C;
  center[1] = Y_C;

  dealii::Point<3> center_cyl_manifold;
  center_cyl_manifold[0] = center[0];
  center_cyl_manifold[1] = center[1];

  // apply this manifold for all mesh types
  dealii::Point<3> direction;
  direction[2] = 1.;

  static std::shared_ptr<dealii::Manifold<dim>> cylinder_manifold;

  if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
  {
    cylinder_manifold = std::shared_ptr<dealii::Manifold<dim>>(
      dim == 2 ? static_cast<dealii::Manifold<dim> *>(new dealii::SphericalManifold<dim>(center)) :
                 reinterpret_cast<dealii::Manifold<dim> *>(
                   new dealii::CylindricalManifold<3>(direction, center_cyl_manifold)));
  }
  else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
  {
    cylinder_manifold = std::shared_ptr<dealii::Manifold<dim>>(
      static_cast<dealii::Manifold<dim> *>(new MyCylindricalManifold<dim>(center)));
  }
  else
  {
    AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold ||
                  MANIFOLD_TYPE == ManifoldType::VolumeManifold,
                dealii::ExcMessage("Specified manifold type not implemented"));
  }

  create_triangulation(triangulation);
  triangulation.set_manifold(MANIFOLD_ID, *cylinder_manifold);

  // generate vector of manifolds and apply manifold to all cells that have been marked
  static std::vector<std::shared_ptr<dealii::Manifold<dim>>> manifold_vec;
  manifold_vec.resize(manifold_ids.size());

  for(unsigned int i = 0; i < manifold_ids.size(); ++i)
  {
    for(typename dealii::Triangulation<dim>::cell_iterator cell = triangulation.begin();
        cell != triangulation.end();
        ++cell)
    {
      if(cell->manifold_id() == manifold_ids[i])
      {
        manifold_vec[i] =
          std::shared_ptr<dealii::Manifold<dim>>(static_cast<dealii::Manifold<dim> *>(
            new OneSidedCylindricalManifold<dim>(cell, face_ids[i], center)));
        triangulation.set_manifold(manifold_ids[i], *(manifold_vec[i]));
      }
    }
  }

  triangulation.refine_global(n_refine_space);
} // function do_create_grid

} // namespace CircularCylinder

namespace SquareCylinder
{
double const L   = L2;            // length of 3D cylinder
double const I_x = X_C - 0.5 * D; // position of cylinder in x-direction (bottom left corner)
double const I_y = Y_C - 0.5 * D; // position of cylinder in y-direction (bottom left corner)

// left x-coordinate of mesh block around the cylinder
double const X_1 = L1;

// shift nodes in a trapez like manner to the square cylinder
bool const adaptive_mesh_shift = false;

// = 1 for adaptive_mesh_shift == true
// > 1 extends the region of a finer mesh to the right of the cylinder
// (allowed only in the case adaptive_mesh_shift == false)
unsigned int const FAC_X_2 = 2;

// right x-coordinate of mesh block around the cylinder
double const X_2 = X_C + 0.5 * D + FAC_X_2 * I_y;

double const Y_1 = Y_0 + I_y; // y level of bottom of square cylinder
double const Y_2 = Y_1 + D;   // y level of top of square cylinder

// number of elements in y direction
unsigned int const nele_y_bottom = 3;
unsigned int const nele_y_top    = 3;
unsigned int const nele_y_middle = 2;

// number of elements in x direction
unsigned int const nele_x_left          = 2;
unsigned int const nele_x_right         = 10;
unsigned int const nele_x_middle_middle = 2;
unsigned int const nele_x_middle_left   = 3;
unsigned int const nele_x_middle_right  = FAC_X_2 * 3;

// number of elements in z direction
unsigned int const nele_z = 5;

double const h_y_2 = (H - Y_2) / nele_y_top;
double const h_y_1 = (Y_2 - Y_1) / nele_y_middle;
double const h_y_0 = (Y_1 - Y_0) / nele_y_bottom;

double const h_x_2 = (L - X_2) / nele_x_right;
double const h_x_1 = D / nele_x_middle_middle;
double const h_x_0 = (X_1 - X_0) / nele_x_left;

void create_trapezoid(dealii::Triangulation<2> & tria,
                      std::vector<unsigned int>  ref,
                      dealii::Point<2> const     x_0,
                      double const               length,
                      double const               height,
                      double const               max_shift,
                      double const               min_shift)
{
  dealii::Triangulation<2> tmp;

  dealii::Point<2> x_1 = x_0 + dealii::Point<2>(length, height);

  dealii::GridGenerator::subdivided_hyper_rectangle(tmp, ref, x_0, x_1, false);

  for(dealii::Triangulation<2>::vertex_iterator v = tmp.begin_vertex(); v != tmp.end_vertex(); ++v)
  {
    dealii::Point<2> & vertex = v->vertex();

    if(0 < vertex[0] && vertex[0] < length)
      vertex = dealii::Point<2>(vertex[0] + 0.75 * length / ref[0] * vertex[0] / length, vertex[1]);

    double const m = (max_shift - min_shift) / height;

    double const b = max_shift - m * x_1[1];

    vertex =
      dealii::Point<2>(vertex[0], vertex[1] + (m * vertex[1] + b) * (length - vertex[0]) / length);
  }

  tria.copy_triangulation(tmp);
}


template<int dim>
void
set_boundary_ids(dealii::Triangulation<dim> & tria)
{
  // Set the wall boundary and inflow to 0, cylinder boundary to 2, outflow to 1
  for(typename dealii::Triangulation<dim>::active_cell_iterator cell = tria.begin();
      cell != tria.end();
      ++cell)
  {
    for(unsigned int f = 0; f < dealii::GeometryInfo<dim>::faces_per_cell; ++f) // loop over cells
    {
      if(cell->face(f)->at_boundary())
      {
        dealii::Point<dim> point_on_centerline;
        point_on_centerline[0] = X_C;
        point_on_centerline[1] = Y_C;
        if(dim == 3)
          point_on_centerline[dim - 1] = cell->face(f)->center()[2];

        if(dim == 3 ? std::abs(cell->face(f)->center()[0] - X_0) < 1e-12 :
                      std::abs(cell->face(f)->center()[0] - X_1) < 1e-12) // inflow
          cell->face(f)->set_all_boundary_ids(0);
        else if(std::abs(cell->face(f)->center()[0] - (X_0 + L)) < 1e-12) // outflow
          cell->face(f)->set_all_boundary_ids(1);
        else if(std::abs(cell->face(f)->center()[0] - point_on_centerline[0]) <=
                  (0.5 * D + 1e-12) &&
                std::abs(cell->face(f)->center()[1] - point_on_centerline[1]) <=
                  (0.5 * D + 1e-12)) // square cylinder walls
          cell->face(f)->set_all_boundary_ids(2);
        else
          cell->face(f)->set_all_boundary_ids(0); // domain walls
      }
    }
  }
}

template<unsigned int dim>
void create_triangulation(dealii::Triangulation<2> & triangulation, bool is_2d = true)
{
  dealii::Triangulation<2> left, left_bottom, left_middle, left_top, middle, middle_top,
    middle_bottom, middle_left, middle_right, middle_left_top, middle_left_bottom, middle_right_top,
    middle_right_bottom, right, right_bottom, right_middle, right_top, tmp;

  // left
  std::vector<unsigned int> ref_left_top    = {nele_x_left, nele_y_top};
  std::vector<unsigned int> ref_left_bottom = {nele_x_left, nele_y_bottom};
  std::vector<unsigned int> ref_left_middle = {nele_x_left, nele_y_middle};

  // right
  std::vector<unsigned int> ref_right_top    = {nele_x_right, nele_y_top};
  std::vector<unsigned int> ref_right_bottom = {nele_x_right, nele_y_bottom};
  std::vector<unsigned int> ref_right_middle = {nele_x_right, nele_y_middle};

  // middle
  std::vector<unsigned int> ref_middle_top       = {nele_x_middle_middle, nele_y_top};
  std::vector<unsigned int> ref_middle_top_right = {nele_x_middle_right, nele_y_top};
  std::vector<unsigned int> ref_middle_top_left  = {nele_x_middle_left, nele_y_top};

  std::vector<unsigned int> ref_middle_bottom_right = {nele_x_middle_right, nele_y_bottom};
  std::vector<unsigned int> ref_middle_bottom_left  = {nele_x_middle_left, nele_y_bottom};
  std::vector<unsigned int> ref_middle_bottom       = {nele_x_middle_middle, nele_y_bottom};

  std::vector<unsigned int> ref_middle_right = {nele_x_middle_right, nele_y_middle};
  std::vector<unsigned int> ref_middle_left  = {nele_x_middle_left, nele_y_middle};

  // left part
  dealii::GridGenerator::subdivided_hyper_rectangle(
    left_bottom, ref_left_bottom, dealii::Point<2>(X_0, Y_0), dealii::Point<2>(X_1, Y_1), false);

  dealii::GridGenerator::subdivided_hyper_rectangle(
    left_middle, ref_left_middle, dealii::Point<2>(X_0, Y_1), dealii::Point<2>(X_1, Y_2), false);

  dealii::GridGenerator::subdivided_hyper_rectangle(
    left_top, ref_left_top, dealii::Point<2>(X_0, Y_2), dealii::Point<2>(X_1, H), false);

  // merge left triangulations
  dealii::GridGenerator::merge_triangulations(left_bottom, left_middle, tmp);
  dealii::GridGenerator::merge_triangulations(tmp, left_top, left);

  // right part
  dealii::GridGenerator::subdivided_hyper_rectangle(
    right_bottom, ref_right_bottom, dealii::Point<2>(X_2, Y_0), dealii::Point<2>(L, Y_1), false);

  dealii::GridGenerator::subdivided_hyper_rectangle(
    right_middle, ref_right_middle, dealii::Point<2>(X_2, Y_1), dealii::Point<2>(L, Y_2), false);

  dealii::GridGenerator::subdivided_hyper_rectangle(
    right_top, ref_right_top, dealii::Point<2>(X_2, Y_2), dealii::Point<2>(L, H), false);

  // merge right triangulations
  dealii::GridGenerator::merge_triangulations(right_bottom, right_middle, tmp);
  dealii::GridGenerator::merge_triangulations(tmp, right_top, right);

  // middle part
  if(!adaptive_mesh_shift)
  {
    // create middle bottom part
    dealii::GridGenerator::subdivided_hyper_rectangle(middle_bottom,
                                                      ref_middle_bottom,
                                                      dealii::Point<2>(X_0 + I_x, Y_0),
                                                      dealii::Point<2>(X_0 + I_x + D, Y_1),
                                                      false);

    // create middle top part
    dealii::GridGenerator::subdivided_hyper_rectangle(middle_top,
                                                      ref_middle_top,
                                                      dealii::Point<2>(X_0 + I_x, Y_2),
                                                      dealii::Point<2>(X_0 + I_x + D, H),
                                                      false);

    // create middle left part
    dealii::GridGenerator::subdivided_hyper_rectangle(middle_left,
                                                      ref_middle_left,
                                                      dealii::Point<2>(X_1, Y_1),
                                                      dealii::Point<2>(X_0 + I_x, Y_2),
                                                      false);

    dealii::GridGenerator::subdivided_hyper_rectangle(middle_left_top,
                                                      ref_middle_top_left,
                                                      dealii::Point<2>(X_1, Y_2),
                                                      dealii::Point<2>(X_0 + I_x, H),
                                                      false);

    dealii::GridGenerator::subdivided_hyper_rectangle(middle_left_bottom,
                                                      ref_middle_bottom_left,
                                                      dealii::Point<2>(X_1, Y_0),
                                                      dealii::Point<2>(X_0 + I_x, Y_1),
                                                      false);

    // create middle right part
    dealii::GridGenerator::subdivided_hyper_rectangle(middle_right,
                                                      ref_middle_right,
                                                      dealii::Point<2>(X_0 + I_x + D, Y_1),
                                                      dealii::Point<2>(X_2, Y_2),
                                                      false);

    dealii::GridGenerator::subdivided_hyper_rectangle(middle_right_top,
                                                      ref_middle_top_right,
                                                      dealii::Point<2>(X_0 + I_x + D, Y_2),
                                                      dealii::Point<2>(X_2, H),
                                                      false);

    dealii::GridGenerator::subdivided_hyper_rectangle(middle_right_bottom,
                                                      ref_middle_bottom_right,
                                                      dealii::Point<2>(X_0 + I_x + D, Y_0),
                                                      dealii::Point<2>(X_2, Y_1),
                                                      false);
  }
  else
  {
    dealii::Triangulation<2> middle_right_right;

    unsigned int nele_trapezoid = 4;

    std::vector<unsigned int> ref_trapezoid_left_bottom = {nele_trapezoid, 4};
    std::vector<unsigned int> ref_trapezoid_left_middle = {nele_trapezoid, 1};
    std::vector<unsigned int> ref_trapezoid_left_top    = {nele_trapezoid, 3};

    std::vector<unsigned int> ref_trapezoid_right_bottom = {nele_trapezoid, 4};
    std::vector<unsigned int> ref_trapezoid_right_middle = {nele_trapezoid, 1};
    std::vector<unsigned int> ref_trapezoid_right_top    = {nele_trapezoid, 3};

    std::vector<unsigned int> ref_trapezoid_top    = {nele_trapezoid, 6};
    std::vector<unsigned int> ref_trapezoid_bottom = {nele_trapezoid, 6};

    // create middle left trapez
    create_trapezoid(middle_left_top,
                     ref_trapezoid_left_top,
                     dealii::Point<2>(0, 0),
                     I_x - X_1,
                     3.0 * D / 8.0,
                     H - Y_2,
                     3.0 * D / 8.0);

    dealii::Tensor<1, 2> shift_middle_left_top;
    shift_middle_left_top[0] = X_1;
    shift_middle_left_top[1] = Y_1 + D / 2 + D / 8;
    dealii::GridTools::shift(shift_middle_left_top, middle_left_top);

    create_trapezoid(middle_left,
                     ref_trapezoid_left_middle,
                     dealii::Point<2>(0, 0),
                     I_x - X_1,
                     1.0 * D / 8.0,
                     3.0 * D / 8.0,
                     0);

    dealii::Tensor<1, 2> shift_middle_left;
    shift_middle_left[0] = X_1;
    shift_middle_left[1] = Y_1 + D / 2;
    dealii::GridTools::shift(shift_middle_left, middle_left);

    create_trapezoid(middle_left_bottom,
                     ref_trapezoid_left_bottom,
                     dealii::Point<2>(0, 0),
                     I_x - X_1,
                     D / 2.0,
                     0,
                     -Y_1);

    dealii::Tensor<1, 2> shift_middle_left_bottom;
    shift_middle_left_bottom[0] = X_1;
    shift_middle_left_bottom[1] = Y_1;
    dealii::GridTools::shift(shift_middle_left_bottom, middle_left_bottom);

    // middle right trapez
    create_trapezoid(middle_right_top,
                     ref_trapezoid_right_top,
                     dealii::Point<2>(0, 0),
                     I_y,
                     3.0 * D / 8.0,
                     -3.0 * D / 8.0,
                     -(H - Y_2));

    dealii::Tensor<1, 2> shift_middle_right_top;
    shift_middle_right_top[0] = I_x + D + I_y;
    shift_middle_right_top[1] = Y_2;
    dealii::GridTools::rotate(M_PI, middle_right_top);
    dealii::GridTools::shift(shift_middle_right_top, middle_right_top);

    create_trapezoid(middle_right,
                     ref_trapezoid_right_middle,
                     dealii::Point<2>(0, 0),
                     I_y,
                     1.0 * D / 8.0,
                     0,
                     -3.0 * D / 8.0);

    dealii::Tensor<1, 2> shift_middle_right;
    shift_middle_right[0] = I_x + D + I_y;
    shift_middle_right[1] = Y_1 + D / 2 + D / 8;
    dealii::GridTools::rotate(M_PI, middle_right);
    dealii::GridTools::shift(shift_middle_right, middle_right);

    create_trapezoid(middle_right_bottom,
                     ref_trapezoid_right_bottom,
                     dealii::Point<2>(0, 0),
                     I_y,
                     D / 2.0,
                     Y_1,
                     0);

    dealii::Tensor<1, 2> shift_middle_right_bottom;
    shift_middle_right_bottom[0] = I_x + D + I_y;
    shift_middle_right_bottom[1] = Y_1 + D / 2;
    dealii::GridTools::rotate(M_PI, middle_right_bottom);
    dealii::GridTools::shift(shift_middle_right_bottom, middle_right_bottom);

    // create top trapez
    create_trapezoid(
      middle_top, ref_trapezoid_top, dealii::Point<2>(0, 0), H - Y_2, D, I_y, -(I_x - X_1));

    dealii::Tensor<1, 2> shift_middle_top;
    shift_middle_top[0] = I_x;
    shift_middle_top[1] = H;
    dealii::GridTools::rotate(-M_PI / 2.0, middle_top);
    dealii::GridTools::shift(shift_middle_top, middle_top);

    // create bottom trapez
    create_trapezoid(
      middle_bottom, ref_trapezoid_bottom, dealii::Point<2>(0, 0), Y_1, D, I_y, -(I_x - X_1));

    dealii::Tensor<1, 2> shift_middle_bottom;
    shift_middle_bottom[0] = I_x + D;
    shift_middle_bottom[1] = 0;
    dealii::GridTools::rotate(M_PI / 2.0, middle_bottom);
    dealii::GridTools::shift(shift_middle_bottom, middle_bottom);
  }

  // merge middle left triangulations
  dealii::GridGenerator::merge_triangulations(middle_left, middle_left_top, tmp);
  dealii::GridGenerator::merge_triangulations(tmp, middle_left_bottom, middle_left);

  // merge middle right triangulations
  dealii::GridGenerator::merge_triangulations(middle_right, middle_right_top, tmp);
  dealii::GridGenerator::merge_triangulations(tmp, middle_right_bottom, middle_right);


  // merge middle triangulations
  dealii::GridGenerator::merge_triangulations(
    {&middle_bottom, &middle_left, &middle_top, &middle_right}, middle);

  // merge middle and right together
  dealii::GridGenerator::merge_triangulations(right, middle, tmp);

  // merge left to middle and right in 3D case
  if(is_2d)
    triangulation.copy_triangulation(tmp);
  else
    dealii::GridGenerator::merge_triangulations(tmp, left, triangulation);
}

template<unsigned int dim>
void create_triangulation(dealii::Triangulation<3> & triangulation)
{
  dealii::Triangulation<2> tria_2D;
  create_triangulation<2>(tria_2D, false);

  dealii::GridGenerator::extrude_triangulation(tria_2D, nele_z, H, triangulation);
}

template<unsigned int dim>
void
do_create_grid(dealii::Triangulation<dim> & triangulation, unsigned int n_global_refinements)
{
  create_triangulation<dim>(triangulation);

  // set boundary ids
  set_boundary_ids<dim>(triangulation);

  triangulation.refine_global(n_global_refinements);
}

} // namespace SquareCylinder

enum CylinderType
{
  circular,
  square
} cylinder_type;

void
select_cylinder_type(std::string cylinder_type_string)
{
  if(cylinder_type_string == "circular")
    cylinder_type = circular;
  else if(cylinder_type_string == "square")
    cylinder_type = square;
  else
    AssertThrow(false, dealii::ExcNotImplemented());
}

template<unsigned int dim>
void
create_cylinder_grid(dealii::Triangulation<dim> &                             triangulation,
                     unsigned int const                                       n_refine_space,
                     std::vector<dealii::GridTools::PeriodicFacePair<
                       typename dealii::Triangulation<dim>::cell_iterator>> & periodic_faces,
                     std::string                                              cylinder_type_string)
{
  select_cylinder_type(cylinder_type_string);

  switch(cylinder_type)
  {
    case circular:
      CircularCylinder::do_create_grid<dim>(triangulation, n_refine_space, periodic_faces);
      break;
    case square:
      SquareCylinder::do_create_grid<dim>(triangulation, n_refine_space);
      break;
    default:
      AssertThrow(false, dealii::ExcNotImplemented());
  }
}

} // namespace FlowPastCylinder
} // namespace ExaDG

#endif /* APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_CYLINDER_H_ */
