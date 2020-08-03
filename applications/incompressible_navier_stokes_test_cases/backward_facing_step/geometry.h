/*
 * geometry.h
 *
 *  Created on: 30.03.2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_GEOMETRY_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_GEOMETRY_H_

namespace IncNS
{
namespace BackwardFacingStep
{
namespace Geometry
{
double const PI = numbers::PI;

// Height H
double const H = 0.041;

// channel
double const LENGTH_CHANNEL = 2.0 * PI * H;
double const HEIGHT_CHANNEL = 2.0 * H;
double const WIDTH_CHANNEL  = 4.0 * H;

// use a gap between both geometries for visualization purposes
double const GAP_CHANNEL_BFS = 2.0 * H;

// backward facing step geometry
double const LENGTH_BFS_DOWN   = 20.0 * H;
double const LENGTH_BFS_UP     = 2.0 * H;
double const HEIGHT_BFS_STEP   = H;
double const HEIGHT_BFS_INFLOW = HEIGHT_CHANNEL;
double const WIDTH_BFS         = WIDTH_CHANNEL;

double const X1_COORDINATE_INFLOW          = -LENGTH_BFS_UP;
double const X1_COORDINATE_OUTFLOW         = LENGTH_BFS_DOWN;
double const X1_COORDINATE_OUTFLOW_CHANNEL = -LENGTH_BFS_UP - GAP_CHANNEL_BFS;

// mesh stretching parameters
bool use_grid_stretching_in_y_direction = true;

double const GAMMA_LOWER = 60.0;
double const GAMMA_UPPER = 40.0;

/*
 *  maps eta in [-H, 2*H] --> y in [-H,2*H]
 */
double
grid_transform_y(const double & eta)
{
  double y = 0.0;
  double gamma, xi;
  if(eta < 0.0)
  {
    gamma = GAMMA_LOWER;
    xi    = -0.5 * H;
  }
  else
  {
    gamma = GAMMA_UPPER;
    xi    = H;
  }
  y = xi * (1.0 - (std::tanh(gamma * (xi - eta)) / std::tanh(gamma * xi)));
  return y;
}

/*
 *  grid transform function for turbulent channel statistics
 *  requires that the input parameter is 0 < xi < 1
 */
double
grid_transform_turb_channel(const double & xi)
{
  // map xi in [0,1] --> eta in [0, 2H]
  double eta = HEIGHT_CHANNEL * xi;
  return grid_transform_y(eta);
}

/*
 * inverse mapping:
 *
 *  maps y in [-H,2*H] --> eta in [-H,2*H]
 */
double
inverse_grid_transform_y(const double & y)
{
  double eta = 0.0;
  double gamma, xi;
  if(y < 0.0)
  {
    gamma = GAMMA_LOWER;
    xi    = -0.5 * H;
  }
  else
  {
    gamma = GAMMA_UPPER;
    xi    = H;
  }
  eta = xi - (1.0 / gamma) * std::atanh((1.0 - y / xi) * std::tanh(gamma * xi));
  return eta;
}

template<int dim>
class MyManifold : public ChartManifold<dim, dim, dim>
{
public:
  MyManifold()
  {
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates
   *  to point x in physical coordinates
   */
  Point<dim>
  push_forward(const Point<dim> & xi) const
  {
    Point<dim> x = xi;
    x[1]         = grid_transform_y(xi[1]);

    return x;
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates
   */
  Point<dim>
  pull_back(const Point<dim> & x) const
  {
    Point<dim> xi = x;
    xi[1]         = inverse_grid_transform_y(x[1]);

    return xi;
  }

  std::unique_ptr<Manifold<dim>>
  clone() const override
  {
    return std::make_unique<MyManifold<dim>>();
  }
};

template<int dim>
void
create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
            unsigned int const                                n_refine_space,
            std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
              periodic_faces)
{
  AssertThrow(dim == 3, ExcMessage("NotImplemented"));

  Triangulation<dim> tria_1, tria_2, tria_3;

  // inflow part of BFS
  GridGenerator::subdivided_hyper_rectangle(tria_1,
                                            std::vector<unsigned int>({1, 1, 1}),
                                            Point<dim>(-LENGTH_BFS_UP, 0.0, -WIDTH_BFS / 2.0),
                                            Point<dim>(0.0, HEIGHT_BFS_INFLOW, WIDTH_BFS / 2.0));

  // downstream part of BFS (upper)
  GridGenerator::subdivided_hyper_rectangle(tria_2,
                                            std::vector<unsigned int>({10, 1, 1}),
                                            Point<dim>(0.0, 0.0, -WIDTH_BFS / 2.0),
                                            Point<dim>(LENGTH_BFS_DOWN,
                                                       HEIGHT_BFS_INFLOW,
                                                       WIDTH_BFS / 2.0));

  // downstream part of BFS (lower = step)
  GridGenerator::subdivided_hyper_rectangle(tria_3,
                                            std::vector<unsigned int>({10, 1, 1}),
                                            Point<dim>(0.0, 0.0, -WIDTH_BFS / 2.0),
                                            Point<dim>(LENGTH_BFS_DOWN,
                                                       -HEIGHT_BFS_STEP,
                                                       WIDTH_BFS / 2.0));

  Triangulation<dim> tmp1;
  GridGenerator::merge_triangulations(tria_1, tria_2, tmp1);
  GridGenerator::merge_triangulations(tmp1, tria_3, *triangulation);


  // set boundary ID's
  for(auto cell : triangulation->active_cell_iterators())
  {
    for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      // outflow boundary on the right has ID = 1
      if((std::fabs(cell->face(f)->center()(0) - X1_COORDINATE_OUTFLOW) < 1.e-12))
        cell->face(f)->set_boundary_id(1);
      // inflow boundary on the left has ID = 2
      if((std::fabs(cell->face(f)->center()(0) - X1_COORDINATE_INFLOW) < 1.e-12))
        cell->face(f)->set_boundary_id(2);

      // periodicity in z-direction
      if((std::fabs(cell->face(f)->center()(2) - WIDTH_BFS / 2.0) < 1.e-12))
        cell->face(f)->set_all_boundary_ids(2 + 10);
      if((std::fabs(cell->face(f)->center()(2) + WIDTH_BFS / 2.0) < 1.e-12))
        cell->face(f)->set_all_boundary_ids(3 + 10);
    }
  }

  if(use_grid_stretching_in_y_direction == true)
  {
    // manifold
    unsigned int manifold_id = 1;
    for(auto cell : triangulation->active_cell_iterators())
    {
      cell->set_all_manifold_ids(manifold_id);
    }

    // apply mesh stretching towards no-slip boundaries in y-direction
    static const MyManifold<dim> manifold;
    triangulation->set_manifold(manifold_id, manifold);
  }

  // periodicity in z-direction
  auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 2 + 10, 3 + 10, 2, periodic_faces);
  triangulation->add_periodicity(periodic_faces);

  // perform global refinements
  triangulation->refine_global(n_refine_space);
}

template<int dim>
void
create_grid_precursor(
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
  unsigned int const                                n_refine_space,
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
    periodic_faces)
{
  AssertThrow(dim == 3, ExcMessage("NotImplemented"));

  Tensor<1, dim> dimensions;
  dimensions[0] = LENGTH_CHANNEL;
  dimensions[1] = HEIGHT_CHANNEL;
  dimensions[2] = WIDTH_CHANNEL;

  Tensor<1, dim> center;
  center[0] = -(LENGTH_BFS_UP + GAP_CHANNEL_BFS + LENGTH_CHANNEL / 2.0);
  center[1] = HEIGHT_CHANNEL / 2.0;

  GridGenerator::subdivided_hyper_rectangle(*triangulation,
                                            std::vector<unsigned int>({2, 1, 1}), // refinements
                                            Point<dim>(center - dimensions / 2.0),
                                            Point<dim>(center + dimensions / 2.0));

  if(use_grid_stretching_in_y_direction == true)
  {
    // manifold
    unsigned int manifold_id = 1;
    for(auto cell : triangulation->active_cell_iterators())
    {
      cell->set_all_manifold_ids(manifold_id);
    }

    // apply mesh stretching towards no-slip boundaries in y-direction
    static const MyManifold<dim> manifold;
    triangulation->set_manifold(manifold_id, manifold);
  }

  // set boundary ID's: periodicity
  for(auto cell : triangulation->active_cell_iterators())
  {
    for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
    {
      // periodicity in x-direction
      if(std::fabs(cell->face(f)->center()(0) - (center[0] - dimensions[0] / 2.0)) < 1.e-12)
        cell->face(f)->set_all_boundary_ids(0 + 10);
      if(std::fabs(cell->face(f)->center()(0) - (center[0] + dimensions[0] / 2.0)) < 1.e-12)
        cell->face(f)->set_all_boundary_ids(1 + 10);

      // periodicity in z-direction
      if(std::fabs(cell->face(f)->center()(2) - (center[2] - dimensions[2] / 2.0)) < 1.e-12)
        cell->face(f)->set_all_boundary_ids(2 + 10);
      if(std::fabs(cell->face(f)->center()(2) - (center[2] + dimensions[2] / 2.0)) < 1.e-12)
        cell->face(f)->set_all_boundary_ids(3 + 10);
    }
  }

  auto tria = dynamic_cast<Triangulation<dim> *>(&*triangulation);
  GridTools::collect_periodic_faces(*tria, 0 + 10, 1 + 10, 0, periodic_faces);
  GridTools::collect_periodic_faces(*tria, 2 + 10, 3 + 10, 2, periodic_faces);
  triangulation->add_periodicity(periodic_faces);

  // perform global refinements: use one level finer for the channel
  triangulation->refine_global(n_refine_space);
}

} // namespace Geometry
} // namespace BackwardFacingStep
} // namespace IncNS


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_BACKWARD_FACING_STEP_GEOMETRY_H_ */
