/*
 * mesh_flow_past_cylinder.h
 *
 *  Created on: Mar 1, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_CYLINDER_H_
#define APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_CYLINDER_H_

#include "../../include/functionalities/one_sided_cylindrical_manifold.h"

// physical dimensions (diameter D and center coordinate Y_C can be varied)
double const X_0 = 0.0; // origin (x-coordinate)
double const Y_0 = 0.0; // origin (y-coordinate)
double const L1 = 0.3; // x-coordinate of inflow boundary (2d test cases)
double const L2 = 2.5; // x-coordinate of outflow boundary (=length for 3d test cases)
double const H = 0.41; // height of channel
double const X_1 = L1; // left x-coordinate of mesh block around the cylinder
double const X_2 = 0.7; // right x-coordinate of mesh block around the cylinder
double const R = D/2.0; // cylinder radius
double const X_C = (X_2+X_1)/2.0; // center of cylinder (x-coordinate)

// MeshType
// Type1: no refinement around cylinder surface (coarsest mesh has 34 elements in 2D)
// Type2: two layers of spherical cells around cylinder (used in Fehn et al. (JCP, 2017, "On the stability of projection methods ...")),
//        (coarsest mesh has 50 elements in 2D)
// Type3: coarse mesh has only one element in direction perpendicular to flow direction,
//        one layer of spherical cells around cylinder for coarsest mesh (coarsest mesh has 12 elements in 2D)
// Type4: no refinement around cylinder, coarsest mesh consists of 4 cells for the block that
//        that surrounds the cylinder (coarsest mesh has 8 elements in 2D)
enum class MeshType{ Type1, Type2, Type3, Type4 };
MeshType const MESH_TYPE = MeshType::Type2;

// needed for mesh type 2 with two layers of spherical cells around cylinder
double const R_1 = 1.2*R;
double const R_2 = 1.7*R;

// neded for mesh type 3 with one layers of spherical cells around cylinder
double const R_3 = 1.75*R;

// ManifoldType
// Surface manifold: when refining the mesh only the cells close to the manifold-surface are curved (should not be used!)
// Volume manifold: when refining the mesh all child cells are curved since it is a volume manifold
enum class ManifoldType{ SurfaceManifold, VolumeManifold };
ManifoldType const MANIFOLD_TYPE = ManifoldType::VolumeManifold;

// manifold ID of spherical manifold
unsigned int const MANIFOLD_ID = 10;

// vectors of manifold_ids and face_ids
std::vector<unsigned int> manifold_ids;
std::vector<unsigned int> face_ids;

template<int dim>
void set_boundary_ids(Triangulation<dim> &tria, bool compute_in_2d)
{
 // Set the cylinder boundary to 2, outflow to 1, the rest to 0.
 for (typename Triangulation<dim>::active_cell_iterator cell=tria.begin(); cell != tria.end(); ++cell)
 {
   for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)// loop over cells
   {
     if (cell->face(f)->at_boundary())
     {
       Point<dim> point_on_centerline;
       point_on_centerline[0] = X_C;
       point_on_centerline[1] = Y_C;
       if(dim==3)
         point_on_centerline[dim-1] = cell->face(f)->center()[2];

       if (std::abs(cell->face(f)->center()[0] - (compute_in_2d ? L1 : X_0)) < 1e-12)
         cell->face(f)->set_all_boundary_ids(0);
       else if (std::abs(cell->face(f)->center()[0]-L2) < 1e-12)
         cell->face(f)->set_all_boundary_ids(1);
       else if (point_on_centerline.distance(cell->face(f)->center()) <= R)
         cell->face(f)->set_all_boundary_ids(2);
       else
         cell->face(f)->set_all_boundary_ids(0);
     }
   }
 }
}

void create_triangulation(Triangulation<2> &tria, const bool compute_in_2d = true)
{
 AssertThrow(std::abs((X_2-X_1) - 2.0*(X_C-X_1))<1.0e-12, ExcMessage("Geometry parameters X_1, X_2, X_C invalid!"));

 Point<2> center = Point<2>(X_C,Y_C);

 if(MESH_TYPE == MeshType::Type1)
 {
   Triangulation<2> left, middle, right, tmp, tmp2;

   // left part (only needed for 3D problem)
   std::vector<unsigned int> ref_1(2, 2);
   ref_1[1] = 2;
   GridGenerator::subdivided_hyper_rectangle(left, ref_1 ,Point<2>(X_0,Y_0), Point<2>(X_1, H), false);

   // right part (2D and 3D)
   std::vector<unsigned int> ref_2(2, 9);
   ref_2[1] = 2;
   GridGenerator::subdivided_hyper_rectangle(right, ref_2,Point<2>(X_2,Y_0), Point<2>(L2, H), false);

   // create middle part first as a hyper shell
   const double outer_radius = (X_2-X_1)/2.0;
   const unsigned int n_cells = 4;
   Point<2> current_center = Point<2>((X_1+X_2)/2.0, outer_radius);
   GridGenerator::hyper_shell(middle, current_center, R, outer_radius, n_cells, true);
   MyCylindricalManifold<2> boundary(current_center);
   middle.set_manifold(0, boundary);
   middle.refine_global(1);

   // then move the vertices to the points where we want them to be to create a slightly asymmetric cube with a hole
   for (Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
   {
    for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
    {
      Point<2> &vertex = cell->vertex(v);
      if (std::abs(vertex[0] - X_2) < 1e-10 && std::abs(vertex[1] - current_center[1]) < 1e-10)
        vertex = Point<2>(X_2, H/2.0);
      else if (std::abs(vertex[0] - (current_center[0] + outer_radius/std::sqrt(2.0))) < 1e-10 &&
               std::abs(vertex[1] - (current_center[1] + outer_radius/std::sqrt(2.0))) < 1e-10)
        vertex = Point<2>(X_2, H);
      else if (std::abs(vertex[0] - (current_center[0] + outer_radius/std::sqrt(2.0))) < 1e-10 &&
               std::abs(vertex[1] - (current_center[1] - outer_radius/std::sqrt(2.0))) < 1e-10)
        vertex = Point<2>(X_2, Y_0);
      else if (std::abs(vertex[0] - current_center[0]) < 1e-10 && std::abs(vertex[1] - (X_2-X_1)) < 1e-10)
        vertex = Point<2>(current_center[0], H);
      else if (std::abs(vertex[0] - current_center[0]) < 1e-10 && std::abs(vertex[1] - X_0) < 1e-10)
        vertex = Point<2>(current_center[0], X_0);
      else if (std::abs(vertex[0] - (current_center[0] - outer_radius/std::sqrt(2.0))) < 1e-10 &&
               std::abs(vertex[1] - (current_center[1] + outer_radius/std::sqrt(2.0))) < 1e-10)
        vertex = Point<2>(X_1, H);
      else if (std::abs(vertex[0] - (current_center[0] - outer_radius/std::sqrt(2.0))) < 1e-10 &&
               std::abs(vertex[1] - (current_center[1] - outer_radius/std::sqrt(2.0))) < 1e-10)
        vertex = Point<2>(X_1, Y_0);
      else if (std::abs(vertex[0] - X_1) < 1e-10 && std::abs(vertex[1] - current_center[1]) < 1e-10)
        vertex = Point<2>(X_1, H/2.0);
    }
   }

   // the same for the inner circle
   for (Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
   {
     for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
     {
       Point<2> &vertex = cell->vertex(v);

       // allow to shift cylinder center
       if(std::abs(vertex.distance(current_center)-R) < 1.e-10 ||
          std::abs(vertex.distance(current_center)-(R + (outer_radius-R)/2.0)) < 1.e-10)
       {
         vertex[0] += center[0] - current_center[0];
         vertex[1] += center[1] - current_center[1];
       }
     }
   }

   // we have to copy the triangulation because we cannot merge triangulations with refinement ...
   GridGenerator::flatten_triangulation(middle, tmp2);

   if (compute_in_2d)
   {
    GridGenerator::merge_triangulations (tmp2, right, tria);
   }
   else
   {
    GridGenerator::merge_triangulations (left, tmp2, tmp);
    GridGenerator::merge_triangulations (tmp, right, tria);
   }

   if (compute_in_2d)
   {
     // set manifold ID's
     tria.set_all_manifold_ids(0);

     for (Triangulation<2>::active_cell_iterator cell=tria.begin(); cell != tria.end(); ++cell)
     {
       if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
       {
         for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
         {
           if (cell->face(f)->at_boundary() && center.distance(cell->face(f)->center())<=R)
           {
             cell->face(f)->set_all_manifold_ids(MANIFOLD_ID);
           }
         }
       }
       else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
       {
         for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
         {
           bool face_at_sphere_boundary = true;
           for (unsigned int v=0; v<GeometryInfo<2-1>::vertices_per_cell; ++v)
           {
             if (std::abs(center.distance(cell->face(f)->vertex(v)) - R) > 1e-12)
               face_at_sphere_boundary = false;
           }
           if (face_at_sphere_boundary)
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
         AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold || MANIFOLD_TYPE == ManifoldType::VolumeManifold,
             ExcMessage("Specified manifold type not implemented"));
       }
     }
   }
 }
 else if(MESH_TYPE == MeshType::Type2)
 {
   MyCylindricalManifold<2> cylinder_manifold(center);

   Triangulation<2> left, circle_1, circle_2, circle_tmp, middle, middle_tmp, middle_tmp2, right, tmp_3D;

   // left part (only needed for 3D problem)
   std::vector<unsigned int> ref_1(2, 2);
   ref_1[1] = 2;
   GridGenerator::subdivided_hyper_rectangle(left, ref_1 ,Point<2>(X_0,Y_0), Point<2>(X_1, H), false);

   // right part (2D and 3D)
   std::vector<unsigned int> ref_2(2, 9);
   ref_2[1] = 2;
   GridGenerator::subdivided_hyper_rectangle(right, ref_2, Point<2>(X_2, Y_0), Point<2>(L2, H), false);

   // create middle part first as a hyper shell
   const double outer_radius = (X_2-X_1)/2.0;
   const unsigned int n_cells = 4;
   GridGenerator::hyper_shell(middle, center, R_2, outer_radius, n_cells, true);
   middle.set_all_manifold_ids(MANIFOLD_ID);
   middle.set_manifold(MANIFOLD_ID, cylinder_manifold);
   middle.refine_global(1);

   // two inner circles in order to refine towards the cylinder surface
   const unsigned int n_cells_circle = 8;
   GridGenerator::hyper_shell(circle_1, center, R, R_1, n_cells_circle, true);
   GridGenerator::hyper_shell(circle_2, center, R_1, R_2, n_cells_circle, true);

   // then move the vertices to the points where we want them to be to create a slightly asymmetric cube with a hole
   for (Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
   {
     for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
     {
       Point<2> &vertex = cell->vertex(v);
       if (std::abs(vertex[0] - X_2) < 1e-10 && std::abs(vertex[1] - Y_C) < 1e-10)
       {
         vertex = Point<2>(X_2, H/2.0);
       }
       else if (std::abs(vertex[0] - (X_C + (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10 && std::abs(vertex[1] - (Y_C + (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10)
       {
         vertex = Point<2>(X_2, H);
       }
       else if (std::abs(vertex[0] - (X_C + (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10 && std::abs(vertex[1] - (Y_C - (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10)
       {
         vertex = Point<2>(X_2, Y_0);
       }
       else if (std::abs(vertex[0] - X_C) < 1e-10 && std::abs(vertex[1] - (Y_C +(X_2-X_1)/2.0)) < 1e-10)
       {
         vertex = Point<2>(X_C, H);
       }
       else if (std::abs(vertex[0] - X_C) < 1e-10 && std::abs(vertex[1] - (Y_C-(X_2-X_1)/2.0)) < 1e-10)
       {
         vertex = Point<2>(X_C, Y_0);
       }
       else if (std::abs(vertex[0] - (X_C - (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10 && std::abs(vertex[1] - (Y_C + (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10)
       {
         vertex = Point<2>(X_1, H);
       }
       else if (std::abs(vertex[0] - (X_C - (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10 && std::abs(vertex[1] - (Y_C - (X_2-X_1)/2.0/std::sqrt(2))) < 1e-10)
       {
         vertex = Point<2>(X_1, Y_0);
       }
       else if (std::abs(vertex[0] - X_1) < 1e-10 && std::abs(vertex[1] - Y_C) < 1e-10)
       {
         vertex = Point<2>(X_1, H/2.0);
       }
     }
   }

   // we have to copy the triangulation because we cannot merge triangulations with refinement ...
   GridGenerator::flatten_triangulation(middle, middle_tmp);

   GridGenerator::merge_triangulations(circle_1,circle_2,circle_tmp);
   GridGenerator::merge_triangulations(middle_tmp,circle_tmp,middle_tmp2);

   if (compute_in_2d)
   {
     GridGenerator::merge_triangulations(middle_tmp2,right,tria);
   }
   else // 3D
   {
     GridGenerator::merge_triangulations (left, middle_tmp2, tmp_3D);
     GridGenerator::merge_triangulations (tmp_3D, right, tria);
   }

   if (compute_in_2d)
   {
     // set manifold ID's
     tria.set_all_manifold_ids(0);

     for (Triangulation<2>::active_cell_iterator cell=tria.begin(); cell != tria.end(); ++cell)
     {
       if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
       {
         if(center.distance(cell->center())<= R_2)
           cell->set_all_manifold_ids(MANIFOLD_ID);
       }
       else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
       {
         if(center.distance(cell->center())<= R_2)
           cell->set_all_manifold_ids(MANIFOLD_ID);
         else
         {
           for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
           {
             bool face_at_sphere_boundary = true;
             for (unsigned int v=0; v<GeometryInfo<2-1>::vertices_per_cell; ++v)
             {
               if (std::abs(center.distance(cell->face(f)->vertex(v)) - R_2) > 1e-12)
                 face_at_sphere_boundary = false;
             }
             if (face_at_sphere_boundary)
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
   Triangulation<2> left, middle, circle, middle_tmp, right, tmp_3D;

   // left part (only needed for 3D problem)
   std::vector<unsigned int> ref_1(2, 1);
   GridGenerator::subdivided_hyper_rectangle(left, ref_1 ,Point<2>(X_0,Y_0), Point<2>(X_1, H), false);

   // right part (2D and 3D)
   std::vector<unsigned int> ref_2(2, 4);
   ref_2[1] = 1;
   GridGenerator::subdivided_hyper_rectangle(right, ref_2, Point<2>(X_2, Y_0), Point<2>(L2, H), false);

   // middle part
   const double outer_radius = (X_2-X_1)/2.0;
   const unsigned int n_cells = 4;
   Point<2> origin;

   // inner circle around cylinder
   GridGenerator::hyper_shell(circle, origin, R, R_3, n_cells, true);
   GridTools::rotate(numbers::PI/4, circle);
   GridTools::shift(Point<2>(outer_radius+X_1,outer_radius),circle);

   // create middle part first as a hyper shell
   GridGenerator::hyper_shell(middle, origin, R_3, outer_radius*std::sqrt(2.0), n_cells, true);
   GridTools::rotate(numbers::PI/4, middle);
   GridTools::shift(Point<2>(outer_radius+X_1,outer_radius),middle);

   // then move the vertices to the points where we want them to be
   for (Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
   {
     for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
     {
       Point<2> &vertex = cell->vertex(v);

       // shift two points at the top to a height of H
       if (std::abs(vertex[0] - X_1) < 1e-10 && std::abs(vertex[1] - (X_2-X_1)) < 1e-10)
       {
         vertex = Point<2>(X_1, H);
       }
       else if (std::abs(vertex[0] - X_2) < 1e-10 && std::abs(vertex[1] - (X_2-X_1)) < 1e-10)
       {
         vertex = Point<2>(X_2, H);
       }

       // allow to shift cylinder center
       Point<2> current_center = Point<2>((X_1+X_2)/2.0, (X_2-X_1)/2.0);
       if(std::abs(vertex.distance(current_center)-R_3) < 1.e-10)
       {
         vertex[0] += center[0] - current_center[0];
         vertex[1] += center[1] - current_center[1];
       }
     }
   }

   // the same for the inner circle
   for (Triangulation<2>::cell_iterator cell = circle.begin(); cell != circle.end(); ++cell)
   {
     for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
     {
       Point<2> &vertex = cell->vertex(v);

       // allow to shift cylinder center
       Point<2> current_center = Point<2>((X_1+X_2)/2.0, (X_2-X_1)/2.0);
       if(std::abs(vertex.distance(current_center)-R) < 1.e-10 || std::abs(vertex.distance(current_center)-R_3) < 1.e-10)
       {
         vertex[0] += center[0] - current_center[0];
         vertex[1] += center[1] - current_center[1];
       }
     }
   }

   GridGenerator::merge_triangulations(circle,middle,middle_tmp);

   if (compute_in_2d)
   {
     GridGenerator::merge_triangulations(middle_tmp,right,tria);
   }
   else // 3D
   {
     GridGenerator::merge_triangulations (left, middle_tmp, tmp_3D);
     GridGenerator::merge_triangulations (tmp_3D, right, tria);
   }

   if (compute_in_2d)
   {
     // set manifold ID's
     tria.set_all_manifold_ids(0);

     for (Triangulation<2>::active_cell_iterator cell=tria.begin(); cell != tria.end(); ++cell)
     {
       if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
       {
         if(center.distance(cell->center()) <= R_3)
         {
           cell->set_all_manifold_ids(MANIFOLD_ID);
         }
         else
         {
           for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
           {
             bool face_at_sphere_boundary = true;
             for (unsigned int v=0; v<GeometryInfo<2-1>::vertices_per_cell; ++v)
             {
               if (std::abs(center.distance(cell->face(f)->vertex(v)) - R_3) > 1e-12)
                 face_at_sphere_boundary = false;
             }
             if (face_at_sphere_boundary)
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
         AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold, ExcMessage("Specified manifold type not implemented."));
       }
     }
   }
 }
 else if(MESH_TYPE == MeshType::Type4)
 {
   Triangulation<2> left, middle, circle, right, tmp_3D;

   // left part (only needed for 3D problem)
   std::vector<unsigned int> ref_1(2, 1);
   GridGenerator::subdivided_hyper_rectangle(left, ref_1 ,Point<2>(X_0,Y_0), Point<2>(X_1, H), false);

   // right part (2D and 3D)
   std::vector<unsigned int> ref_2(2, 4);
   ref_2[1] = 1; //only one cell over channel height
   GridGenerator::subdivided_hyper_rectangle(right, ref_2, Point<2>(X_2, Y_0), Point<2>(L2, H), false);

   // middle part
   const double outer_radius = (X_2-X_1)/2.0;
   const unsigned int n_cells = 4;
   Point<2> origin;

   // create middle part first as a hyper shell
   GridGenerator::hyper_shell(middle, origin, R, outer_radius*std::sqrt(2.0), n_cells, true);
   GridTools::rotate(numbers::PI/4, middle);
   GridTools::shift(Point<2>(outer_radius+X_1,outer_radius),middle);

   // then move the vertices to the points where we want them to be
   for (Triangulation<2>::cell_iterator cell = middle.begin(); cell != middle.end(); ++cell)
   {
     for (unsigned int v=0; v < GeometryInfo<2>::vertices_per_cell; ++v)
     {
       Point<2> &vertex = cell->vertex(v);

       // shift two points at the top to a height of H
       if (std::abs(vertex[0] - X_1) < 1e-10 && std::abs(vertex[1] - (X_2-X_1)) < 1e-10)
       {
         vertex = Point<2>(X_1, H);
       }
       else if (std::abs(vertex[0] - X_2) < 1e-10 && std::abs(vertex[1] - (X_2-X_1)) < 1e-10)
       {
         vertex = Point<2>(X_2, H);
       }

       // allow to shift cylinder center
       Point<2> current_center = Point<2>((X_1+X_2)/2.0, (X_2-X_1)/2.0);
       if(std::abs(vertex.distance(current_center)-R) < 1.e-10)
       {
         vertex[0] += center[0] - current_center[0];
         vertex[1] += center[1] - current_center[1];
       }
     }
   }


   if (compute_in_2d)
   {
     GridGenerator::merge_triangulations(middle,right,tria);
   }
   else // 3D
   {
     GridGenerator::merge_triangulations (left, middle, tmp_3D);
     GridGenerator::merge_triangulations (tmp_3D, right, tria);
   }

   if (compute_in_2d)
   {
     // set manifold ID's
     tria.set_all_manifold_ids(0);

     for (Triangulation<2>::active_cell_iterator cell=tria.begin(); cell != tria.end(); ++cell)
     {
       if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
       {
         for (unsigned int f=0; f<GeometryInfo<2>::faces_per_cell; ++f)
         {
           bool face_at_sphere_boundary = true;
           for (unsigned int v=0; v<GeometryInfo<2-1>::vertices_per_cell; ++v)
           {
             if (std::abs(center.distance(cell->face(f)->vertex(v)) - R) > 1e-12)
               face_at_sphere_boundary = false;
           }
           if (face_at_sphere_boundary)
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
         AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold, ExcMessage("Specified manifold type not implemented."));
       }
     }
   }
 }
 else
 {
   AssertThrow(MESH_TYPE == MeshType::Type1 || MESH_TYPE == MeshType::Type2 || MESH_TYPE == MeshType::Type3 || MESH_TYPE == MeshType::Type4,
       ExcMessage("Specified mesh type not implemented"));
 }

 if(compute_in_2d == true)
 {
   // Set boundary ID's
   set_boundary_ids<2>(tria, compute_in_2d);
 }
}


void create_triangulation(Triangulation<3> &tria)
{
Triangulation<2> tria_2d;
create_triangulation(tria_2d, false);

if(MESH_TYPE == MeshType::Type1)
{
  GridGenerator::extrude_triangulation(tria_2d, 3, H, tria);

  // set manifold ID's
  tria.set_all_manifold_ids(0);

  if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
  {
    for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
    {
      for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
      {
        if (cell->face(f)->at_boundary() && Point<3>(X_C,Y_C,cell->face(f)->center()[2]).distance(cell->face(f)->center()) <= R)
        {
          cell->face(f)->set_all_manifold_ids(MANIFOLD_ID);
        }
      }
    }
  }
  else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
  {
    for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
    {
      for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
      {
        bool face_at_sphere_boundary = true;
        for (unsigned int v=0; v<GeometryInfo<3-1>::vertices_per_cell; ++v)
        {
          if (std::abs(Point<3>(X_C,Y_C,cell->face(f)->vertex(v)[2]).distance(cell->face(f)->vertex(v)) - R) > 1e-12)
            face_at_sphere_boundary = false;
        }
        if (face_at_sphere_boundary)
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
    AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold || MANIFOLD_TYPE == ManifoldType::VolumeManifold,
        ExcMessage("Specified manifold type not implemented"));
  }
}
else if(MESH_TYPE == MeshType::Type2)
{
  GridGenerator::extrude_triangulation(tria_2d, 3, H, tria);

  // set manifold ID's
  tria.set_all_manifold_ids(0);

  if(MANIFOLD_TYPE == ManifoldType::SurfaceManifold)
  {
    for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
    {
      if(Point<3>(X_C,Y_C,cell->center()[2]).distance(cell->center()) <= R_2)
        cell->set_all_manifold_ids(MANIFOLD_ID);
    }
  }
  else if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
  {
    for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
    {
      if(Point<3>(X_C,Y_C,cell->center()[2]).distance(cell->center())<= R_2)
        cell->set_all_manifold_ids(MANIFOLD_ID);
      else
      {
        for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
        {
          bool face_at_sphere_boundary = true;
          for (unsigned int v=0; v<GeometryInfo<3-1>::vertices_per_cell; ++v)
          {
            if (std::abs(Point<3>(X_C,Y_C,cell->face(f)->vertex(v)[2]).distance(cell->face(f)->vertex(v)) - R_2) > 1e-12)
              face_at_sphere_boundary = false;
          }
          if (face_at_sphere_boundary)
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
    AssertThrow(MANIFOLD_TYPE == ManifoldType::SurfaceManifold || MANIFOLD_TYPE == ManifoldType::VolumeManifold,
        ExcMessage("Specified manifold type not implemented"));
  }
}
else if(MESH_TYPE == MeshType::Type3)
{
  GridGenerator::extrude_triangulation(tria_2d, 2, H, tria);

  // set manifold ID's
  tria.set_all_manifold_ids(0);

  if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
  {
   for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
   {
     if(Point<3>(X_C,Y_C,cell->center()[2]).distance(cell->center())<= R_3)
       cell->set_all_manifold_ids(MANIFOLD_ID);
     else
     {
       for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
       {
         bool face_at_sphere_boundary = true;
         for (unsigned int v=0; v<GeometryInfo<3-1>::vertices_per_cell; ++v)
         {
           if (std::abs(Point<3>(X_C,Y_C,cell->face(f)->vertex(v)[2]).distance(cell->face(f)->vertex(v)) - R_3) > 1e-12)
             face_at_sphere_boundary = false;
         }
         if (face_at_sphere_boundary)
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
   AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold, ExcMessage("Specified manifold type not implemented"));
 }
}
else if(MESH_TYPE == MeshType::Type4)
{
  GridGenerator::extrude_triangulation(tria_2d, 2, H, tria);

  // set manifold ID's
  tria.set_all_manifold_ids(0);

  if(MANIFOLD_TYPE == ManifoldType::VolumeManifold)
  {
   for (Triangulation<3>::active_cell_iterator cell=tria.begin();cell != tria.end(); ++cell)
   {
     for (unsigned int f=0; f<GeometryInfo<3>::faces_per_cell; ++f)
     {
       bool face_at_sphere_boundary = true;
       for (unsigned int v=0; v<GeometryInfo<3-1>::vertices_per_cell; ++v)
       {
         if (std::abs(Point<3>(X_C,Y_C,cell->face(f)->vertex(v)[2]).distance(cell->face(f)->vertex(v)) - R) > 1e-12)
           face_at_sphere_boundary = false;
       }
       if (face_at_sphere_boundary)
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
   AssertThrow(MANIFOLD_TYPE == ManifoldType::VolumeManifold, ExcMessage("Specified manifold type not implemented"));
 }
}
else
{
  AssertThrow(MESH_TYPE == MeshType::Type1 || MESH_TYPE == MeshType::Type2 || MESH_TYPE == MeshType::Type3 || MESH_TYPE == MeshType::Type4,
      ExcMessage("Specified mesh type not implemented"));
}

 // Set boundary ID's
 set_boundary_ids<3>(tria, false);
}

#endif /* APPLICATIONS_GRID_TOOLS_MESH_FLOW_PAST_CYLINDER_H_ */
