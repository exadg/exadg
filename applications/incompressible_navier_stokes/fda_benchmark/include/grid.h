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

#ifndef APPLICATIONS_GRID_TOOLS_FDA_BENCHMARK_NOZZLE_H_
#define APPLICATIONS_GRID_TOOLS_FDA_BENCHMARK_NOZZLE_H_

// ExaDG
#include <exadg/grid/one_sided_cylindrical_manifold.h>

namespace ExaDG
{
namespace FDANozzle
{
// radius
double const R       = 0.002;
double const R_INNER = R;
double const R_OUTER = 3.0 * R;
double const D       = 2.0 * R_OUTER;

// lengths (dimensions in flow direction z)
double const LENGTH_PRECURSOR = 8.0 * R_OUTER;
double const LENGTH_INFLOW    = 8.0 * R_OUTER;
double const LENGTH_CONE = (R_OUTER - R_INNER) / std::tan(20.0 / 2.0 * dealii::numbers::PI / 180.0);
double const LENGTH_THROAT  = 0.04;
double const LENGTH_OUTFLOW = 20.0 * R_OUTER;
double const OFFSET         = 2.0 * R_OUTER;

// mesh parameters
unsigned int const N_CELLS_AXIAL           = 2;
unsigned int const N_CELLS_AXIAL_PRECURSOR = 4 * N_CELLS_AXIAL;
unsigned int const N_CELLS_AXIAL_INFLOW    = 4 * N_CELLS_AXIAL;
unsigned int const N_CELLS_AXIAL_CONE      = 2 * N_CELLS_AXIAL;
unsigned int const N_CELLS_AXIAL_THROAT    = 4 * N_CELLS_AXIAL;
unsigned int const N_CELLS_AXIAL_OUTFLOW   = 10 * N_CELLS_AXIAL;

// manifold IDs
unsigned int const MANIFOLD_ID_CYLINDER    = 1234;
unsigned int const MANIFOLD_ID_OFFSET_CONE = 7890;

// z-coordinates
double const Z2_OUTFLOW = LENGTH_OUTFLOW;
double const Z1_OUTFLOW = 0.0;

double const Z2_THROAT = 0.0;
double const Z1_THROAT = -LENGTH_THROAT;

double const Z2_CONE = -LENGTH_THROAT;
double const Z1_CONE = -LENGTH_THROAT - LENGTH_CONE;

double const Z2_INFLOW = -LENGTH_THROAT - LENGTH_CONE;
double const Z1_INFLOW = -LENGTH_THROAT - LENGTH_CONE - LENGTH_INFLOW;

double const Z2_PRECURSOR = -LENGTH_THROAT - LENGTH_CONE - LENGTH_INFLOW - OFFSET;
double const Z1_PRECURSOR =
  -LENGTH_THROAT - LENGTH_CONE - LENGTH_INFLOW - OFFSET - LENGTH_PRECURSOR;

/*
 *  This function returns the radius of the cross-section at a
 *  specified location z in streamwise direction.
 */
double
radius_function(double const z)
{
  double radius = R_OUTER;

  if(z >= Z1_INFLOW and z <= Z2_INFLOW)
    radius = R_OUTER;
  else if(z >= Z1_CONE and z <= Z2_CONE)
    radius = R_OUTER * (1.0 - (z - Z1_CONE) / (Z2_CONE - Z1_CONE) * (R_OUTER - R_INNER) / R_OUTER);
  else if(z >= Z1_THROAT and z <= Z2_THROAT)
    radius = R_INNER;
  else if(z > Z1_OUTFLOW and z <= Z2_OUTFLOW)
    radius = R_OUTER;

  return radius;
}

template<int dim>
void
create_grid_and_set_boundary_ids_nozzle(
  dealii::Triangulation<dim> & triangulation,
  unsigned int const           n_refine_space,
  std::vector<dealii::GridTools::PeriodicFacePair<
    typename dealii::Triangulation<dim>::cell_iterator>> & /*periodic_faces*/)
{
  /*
   *   Inflow
   */
  dealii::Triangulation<2>   tria_2d_inflow;
  dealii::Triangulation<dim> tria_inflow;
  dealii::GridGenerator::hyper_ball(tria_2d_inflow, dealii::Point<2>(), R_OUTER);

  dealii::GridGenerator::extrude_triangulation(tria_2d_inflow,
                                               N_CELLS_AXIAL_INFLOW + 1,
                                               LENGTH_INFLOW,
                                               tria_inflow);
  dealii::Tensor<1, dim> offset_inflow;
  offset_inflow[2] = Z1_INFLOW;
  dealii::GridTools::shift(offset_inflow, tria_inflow);

  /*
   *   Cone
   */
  dealii::Triangulation<2>   tria_2d_cone;
  dealii::Triangulation<dim> tria_cone;
  dealii::GridGenerator::hyper_ball(tria_2d_cone, dealii::Point<2>(), R_OUTER);

  dealii::GridGenerator::extrude_triangulation(tria_2d_cone,
                                               N_CELLS_AXIAL_CONE + 1,
                                               LENGTH_CONE,
                                               tria_cone);
  dealii::Tensor<1, dim> offset_cone;
  offset_cone[2] = Z1_CONE;
  dealii::GridTools::shift(offset_cone, tria_cone);

  // apply conical geometry: stretch vertex positions according to z-coordinate
  for(auto cell : tria_cone.cell_iterators())
  {
    for(auto const & v : cell->vertex_indices())
    {
      if(cell->vertex(v)[2] > Z1_CONE + 1.e-10)
      {
        dealii::Point<dim> point_2d;
        double const       z = cell->vertex(v)[2];
        point_2d[2]          = z;

        // note that this value is only valid for the current dealii implementation of hyper_ball!
        if(std::abs((cell->vertex(v) - point_2d).norm() - 2.485281374239e-03 / 6.0e-3 * R_OUTER) <
             1.e-10 or
           std::abs((cell->vertex(v) - point_2d).norm() - R_OUTER) < 1.e-10)
        {
          cell->vertex(v)[0] *= 1.0 - (cell->vertex(v)[2] - Z1_CONE) / (Z2_CONE - Z1_CONE) *
                                        (R_OUTER - R_INNER) / R_OUTER;
          cell->vertex(v)[1] *= 1.0 - (cell->vertex(v)[2] - Z1_CONE) / (Z2_CONE - Z1_CONE) *
                                        (R_OUTER - R_INNER) / R_OUTER;
        }
      }
    }
  }

  /*
   *   Throat
   */
  dealii::Triangulation<2>   tria_2d_throat;
  dealii::Triangulation<dim> tria_throat;
  dealii::GridGenerator::hyper_ball(tria_2d_throat, dealii::Point<2>(), R_INNER);

  dealii::GridGenerator::extrude_triangulation(tria_2d_throat,
                                               N_CELLS_AXIAL_THROAT + 1,
                                               LENGTH_THROAT,
                                               tria_throat);
  dealii::Tensor<1, dim> offset_throat;
  offset_throat[2] = Z1_THROAT;
  dealii::GridTools::shift(offset_throat, tria_throat);

  /*
   *   OUTFLOW
   */
  unsigned int const n_cells_circle = 4;
  double const       R_1            = R_INNER + 1.0 / 3.0 * (R_OUTER - R_INNER);
  double const       R_2            = R_INNER + 2.0 / 3.0 * (R_OUTER - R_INNER);

  dealii::Triangulation<2> tria_2d_outflow_inner, circle_1, circle_2, circle_3, tria_tmp_2d_1,
    tria_tmp_2d_2, tria_2d_outflow;
  dealii::GridGenerator::hyper_ball(tria_2d_outflow_inner, dealii::Point<2>(), R_INNER);

  dealii::GridGenerator::hyper_shell(
    circle_1, dealii::Point<2>(), R_INNER, R_1, n_cells_circle, true);
  dealii::GridTools::rotate(dealii::numbers::PI / 4, circle_1);
  dealii::GridGenerator::hyper_shell(circle_2, dealii::Point<2>(), R_1, R_2, n_cells_circle, true);
  dealii::GridTools::rotate(dealii::numbers::PI / 4, circle_2);
  dealii::GridGenerator::hyper_shell(
    circle_3, dealii::Point<2>(), R_2, R_OUTER, n_cells_circle, true);
  dealii::GridTools::rotate(dealii::numbers::PI / 4, circle_3);

  // merge 2d triangulations
  dealii::GridGenerator::merge_triangulations(tria_2d_outflow_inner, circle_1, tria_tmp_2d_1);
  dealii::GridGenerator::merge_triangulations(circle_2, circle_3, tria_tmp_2d_2);
  dealii::GridGenerator::merge_triangulations(tria_tmp_2d_1, tria_tmp_2d_2, tria_2d_outflow);

  // extrude in z-direction
  dealii::Triangulation<dim> tria_outflow;
  dealii::GridGenerator::extrude_triangulation(tria_2d_outflow,
                                               N_CELLS_AXIAL_OUTFLOW + 1,
                                               LENGTH_OUTFLOW,
                                               tria_outflow);
  dealii::Tensor<1, dim> offset_outflow;
  offset_outflow[2] = Z1_OUTFLOW;
  dealii::GridTools::shift(offset_outflow, tria_outflow);

  /*
   *  MERGE TRIANGULATIONS
   */
  dealii::Triangulation<dim> tria_tmp, tria_tmp2;
  dealii::GridGenerator::merge_triangulations(tria_inflow, tria_cone, tria_tmp);
  dealii::GridGenerator::merge_triangulations(tria_tmp, tria_throat, tria_tmp2);
  dealii::GridGenerator::merge_triangulations(tria_tmp2, tria_outflow, triangulation);

  /*
   *  MANIFOLDS
   */
  dealii::Triangulation<dim> * current_tria = &triangulation;
  current_tria->set_all_manifold_ids(0);

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids;
  std::vector<unsigned int> face_ids;

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids_cone;
  std::vector<unsigned int> face_ids_cone;
  std::vector<double>       radius_0_cone;
  std::vector<double>       radius_1_cone;

  for(auto cell : current_tria->cell_iterators())
  {
    // INFLOW
    if(cell->center()[2] < Z2_INFLOW)
    {
      for(auto const & f : cell->face_indices())
      {
        if(cell->face(f)->at_boundary())
        {
          bool face_at_sphere_boundary = true;
          for(auto const & v : cell->face(f)->vertex_indices())
          {
            dealii::Point<dim> point = dealii::Point<dim>(0, 0, cell->face(f)->vertex(v)[2]);
            if(std::abs((cell->face(f)->vertex(v) - point).norm() - R_OUTER) > 1e-12)
            {
              face_at_sphere_boundary = false;
              break;
            }
          }

          if(face_at_sphere_boundary)
          {
            face_ids.push_back(f);
            unsigned int manifold_id = manifold_ids.size() + 1;
            cell->set_all_manifold_ids(manifold_id);
            manifold_ids.push_back(manifold_id);
            break;
          }
        }
      }
    }
    // CONE
    else if(cell->center()[2] > Z1_CONE and cell->center()[2] < Z2_CONE)
    {
      for(auto const & f : cell->face_indices())
      {
        if(cell->face(f)->at_boundary())
        {
          bool   face_at_sphere_boundary = true;
          double min_z                   = std::numeric_limits<double>::max();
          double max_z                   = -std::numeric_limits<double>::max();

          for(auto const & v : cell->face(f)->vertex_indices())
          {
            double const z = cell->face(f)->vertex(v)[2];
            if(z > max_z)
              max_z = z;
            if(z < min_z)
              min_z = z;

            dealii::Point<dim> point = dealii::Point<dim>(0, 0, z);
            if(std::abs((cell->face(f)->vertex(v) - point).norm() - radius_function(z)) > 1e-12)
            {
              face_at_sphere_boundary = false;
              break;
            }
          }

          if(face_at_sphere_boundary)
          {
            face_ids_cone.push_back(f);
            unsigned int manifold_id = MANIFOLD_ID_OFFSET_CONE + manifold_ids_cone.size() + 1;
            cell->set_all_manifold_ids(manifold_id);
            manifold_ids_cone.push_back(manifold_id);
            radius_0_cone.push_back(radius_function(min_z));
            radius_1_cone.push_back(radius_function(max_z));
            break;
          }
        }
      }
    }
    // THROAT
    else if(cell->center()[2] > Z1_THROAT and cell->center()[2] < Z2_THROAT)
    {
      for(auto const & f : cell->face_indices())
      {
        if(cell->face(f)->at_boundary())
        {
          bool face_at_sphere_boundary = true;
          for(auto const & v : cell->face(f)->vertex_indices())
          {
            dealii::Point<dim> point = dealii::Point<dim>(0, 0, cell->face(f)->vertex(v)[2]);
            if(std::abs((cell->face(f)->vertex(v) - point).norm() - R_INNER) > 1e-12)
            {
              face_at_sphere_boundary = false;
              break;
            }
          }

          if(face_at_sphere_boundary)
          {
            face_ids.push_back(f);
            unsigned int manifold_id = manifold_ids.size() + 1;
            cell->set_all_manifold_ids(manifold_id);
            manifold_ids.push_back(manifold_id);
            break;
          }
        }
      }
    }
    // OUTFLOW
    else if(cell->center()[2] > Z1_OUTFLOW and cell->center()[2] < Z2_OUTFLOW)
    {
      dealii::Point<dim> point2 = dealii::Point<dim>(0, 0, cell->center()[2]);

      // cylindrical manifold for outer cell layers
      if((cell->center() - point2).norm() > R_INNER / std::sqrt(2.0))
        cell->set_all_manifold_ids(MANIFOLD_ID_CYLINDER);

      // one-sided cylindrical manifold for core region
      for(auto const & f : cell->face_indices())
      {
        if(cell->face(f)->at_boundary())
        {
          bool face_at_sphere_boundary = true;
          for(auto const & v : cell->face(f)->vertex_indices())
          {
            dealii::Point<dim> point = dealii::Point<dim>(0, 0, cell->face(f)->vertex(v)[2]);
            if(std::abs((cell->face(f)->vertex(v) - point).norm() - R_INNER) > 1e-12 or
               (cell->center() - point2).norm() > R_INNER / std::sqrt(2.0))
            {
              face_at_sphere_boundary = false;
              break;
            }
          }

          if(face_at_sphere_boundary)
          {
            face_ids.push_back(f);
            unsigned int manifold_id = manifold_ids.size() + 1;
            cell->set_all_manifold_ids(manifold_id);
            manifold_ids.push_back(manifold_id);
            break;
          }
        }
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Should not arrive here."));
    }
  }

  // one-sided spherical manifold
  // generate vector of manifolds and apply manifold to all cells that have been marked
  static std::vector<std::shared_ptr<dealii::Manifold<dim>>> manifold_vec;
  manifold_vec.resize(manifold_ids.size());

  for(unsigned int i = 0; i < manifold_ids.size(); ++i)
  {
    for(auto cell : current_tria->cell_iterators())
    {
      if(cell->manifold_id() == manifold_ids[i])
      {
        dealii::Point<dim> center = dealii::Point<dim>();
        manifold_vec[i] =
          std::shared_ptr<dealii::Manifold<dim>>(static_cast<dealii::Manifold<dim> *>(
            new OneSidedCylindricalManifold<dim>(*current_tria, cell, face_ids[i], center)));
        current_tria->set_manifold(manifold_ids[i], *(manifold_vec[i]));
      }
    }
  }

  // conical manifold
  static std::vector<std::shared_ptr<dealii::Manifold<dim>>> manifold_vec_cone;
  manifold_vec_cone.resize(manifold_ids_cone.size());

  for(unsigned int i = 0; i < manifold_ids_cone.size(); ++i)
  {
    for(auto cell : current_tria->cell_iterators())
    {
      if(cell->manifold_id() == manifold_ids_cone[i])
      {
        dealii::Point<dim> center = dealii::Point<dim>();
        manifold_vec_cone[i]      = std::shared_ptr<dealii::Manifold<dim>>(
          static_cast<dealii::Manifold<dim> *>(new OneSidedConicalManifold<dim>(
            *current_tria, cell, face_ids_cone[i], center, radius_0_cone[i], radius_1_cone[i])));
        current_tria->set_manifold(manifold_ids_cone[i], *(manifold_vec_cone[i]));
      }
    }
  }

  // set cylindrical manifold
  static std::shared_ptr<dealii::Manifold<dim>> cylinder_manifold;
  cylinder_manifold = std::shared_ptr<dealii::Manifold<dim>>(
    static_cast<dealii::Manifold<dim> *>(new MyCylindricalManifold<dim>(dealii::Point<dim>())));
  current_tria->set_manifold(MANIFOLD_ID_CYLINDER, *cylinder_manifold);

  /*
   *  BOUNDARY ID's
   */
  for(auto cell : triangulation.cell_iterators())
  {
    for(auto const & f : cell->face_indices())
    {
      // inflow boundary on the left has ID = 1
      if((std::fabs(cell->face(f)->center()[2] - Z1_INFLOW) < 1e-12))
      {
        cell->face(f)->set_boundary_id(1);
      }

      // outflow boundary on the right has ID = 2
      if((std::fabs(cell->face(f)->center()[2] - Z2_OUTFLOW) < 1e-12))
      {
        cell->face(f)->set_boundary_id(2);
      }
    }
  }

  // perform global refinements
  triangulation.refine_global(n_refine_space);
}

template<int dim>
void
create_grid_and_set_boundary_ids_precursor(
  dealii::Triangulation<dim> & tria,
  unsigned int const           n_refine_space,
  std::vector<
    dealii::GridTools::PeriodicFacePair<typename dealii::Triangulation<dim>::cell_iterator>> &
    periodic_face_pairs)
{
  dealii::Triangulation<2> tria_2d;
  dealii::GridGenerator::hyper_ball(tria_2d, dealii::Point<2>(), R_OUTER);
  dealii::GridGenerator::extrude_triangulation(tria_2d,
                                               N_CELLS_AXIAL_PRECURSOR + 1,
                                               LENGTH_PRECURSOR,
                                               tria);
  dealii::Tensor<1, dim> offset = dealii::Tensor<1, dim>();
  offset[2]                     = Z1_PRECURSOR;
  dealii::GridTools::shift(offset, tria);

  /*
   *  MANIFOLDS
   */
  tria.set_all_manifold_ids(0);

  // first fill vectors of manifold_ids and face_ids
  std::vector<unsigned int> manifold_ids;
  std::vector<unsigned int> face_ids;

  for(auto cell : tria.cell_iterators())
  {
    for(auto const & f : cell->face_indices())
    {
      bool face_at_sphere_boundary = true;
      for(auto const & v : cell->face(f)->vertex_indices())
      {
        dealii::Point<dim> point = dealii::Point<dim>(0, 0, cell->face(f)->vertex(v)[2]);

        if(std::abs((cell->face(f)->vertex(v) - point).norm() - R_OUTER) > 1e-12)
          face_at_sphere_boundary = false;
      }
      if(face_at_sphere_boundary)
      {
        face_ids.push_back(f);
        unsigned int manifold_id = manifold_ids.size() + 1;
        cell->set_all_manifold_ids(manifold_id);
        manifold_ids.push_back(manifold_id);
      }
    }
  }

  // generate vector of manifolds and apply manifold to all cells that have been marked
  static std::vector<std::shared_ptr<dealii::Manifold<dim>>> manifold_vec;
  manifold_vec.resize(manifold_ids.size());

  for(unsigned int i = 0; i < manifold_ids.size(); ++i)
  {
    for(auto cell : tria.cell_iterators())
    {
      if(cell->manifold_id() == manifold_ids[i])
      {
        manifold_vec[i] =
          std::shared_ptr<dealii::Manifold<dim>>(static_cast<dealii::Manifold<dim> *>(
            new OneSidedCylindricalManifold<dim>(tria, cell, face_ids[i], dealii::Point<dim>())));
        tria.set_manifold(manifold_ids[i], *(manifold_vec[i]));
      }
    }
  }

  /*
   *  BOUNDARY ID's
   */
  for(auto cell : tria.cell_iterators())
  {
    for(auto const & face : cell->face_indices())
    {
      // left boundary
      if((std::fabs(cell->face(face)->center()[2] - Z1_PRECURSOR) < 1e-12))
      {
        cell->face(face)->set_boundary_id(0 + 10);
      }

      // right boundary
      if((std::fabs(cell->face(face)->center()[2] - Z2_PRECURSOR) < 1e-12))
      {
        cell->face(face)->set_boundary_id(1 + 10);
      }
    }
  }

  dealii::GridTools::collect_periodic_faces(tria, 0 + 10, 1 + 10, 2, periodic_face_pairs);
  tria.add_periodicity(periodic_face_pairs);

  // perform global refinements
  tria.refine_global(n_refine_space);
}

} // namespace FDANozzle
} // namespace ExaDG


#endif /* APPLICATIONS_GRID_TOOLS_FDA_BENCHMARK_NOZZLE_H_ */
