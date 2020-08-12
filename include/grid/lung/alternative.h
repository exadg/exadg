#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_reordering.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/grid/tria.h>
#include <deal.II/numerics/data_out.h>

using namespace dealii;

template<int dim>
class EllipseManifold : public Manifold<dim, dim>
{
public:
  EllipseManifold(const Point<dim> & center,
                  const Point<dim> & axis,
                  const double       main_radius,
                  const double       cut_radius)
    : center(center), direction(axis), main_radius(main_radius), cut_radius(cut_radius)
  {
    // currently we assume to be in the plane x=0.
    AssertThrow(axis[0] == 1 && axis[1] == 0 && axis[2] == 0, ExcNotImplemented());
  }

  virtual std::unique_ptr<Manifold<dim>>
  clone() const override
  {
    return std_cxx14::make_unique<EllipseManifold<dim>>(center, direction, main_radius, cut_radius);
  }

  virtual Point<dim>
  get_intermediate_point(const Point<dim> & p1, const Point<dim> & p2, const double w) const
  {
    AssertThrow(std::abs(p1[0]) < 1e-10 && std::abs(p2[0]) < 1e-10,
                ExcMessage("Other directions than alignment at x=0 currently not implemented."));
    const double a1 =
      std::atan2((p1[2] - center[2]) / cut_radius, (p1[1] - center[1]) / main_radius);
    const double a2 =
      std::atan2((p2[2] - center[2]) / cut_radius, (p2[1] - center[1]) / main_radius);
    const double angle = (1 - w) * a1 + w * a2;
    Point<dim>   p     = center;
    p[1] += main_radius * std::cos(angle);
    p[2] += cut_radius * std::sin(angle);
    return p;
  }

private:
  const Point<dim> center;
  const Point<dim> direction;
  const double     main_radius;
  const double     cut_radius;
};


template<int dim>
void
create_lung(dealii::parallel::distributed::Triangulation<dim> & triat,
            int                                                 n_refine_space,
            int                                                 use_manifolds)
{
  Triangulation<dim> tria(Triangulation<dim>::limit_level_difference_at_vertices);

  const double radius = 1.;
  const double length = 3.;

  // position of auxiliary point to achieve an angle of 120 degrees in corner
  // of inner cell

  // 2d mesh topology of inflow part, involving 8 elements (3 elements on half
  // circle and 1 in center for each half of the cylinder)
  std::vector<CellData<2>> cell_data(8);
  cell_data[0].vertices[0] = 2;
  cell_data[0].vertices[1] = 3;
  cell_data[0].vertices[2] = 0;
  cell_data[0].vertices[3] = 1;
  cell_data[1].vertices[0] = 9;
  cell_data[1].vertices[1] = 2;
  cell_data[1].vertices[2] = 8;
  cell_data[1].vertices[3] = 0;
  cell_data[2].vertices[0] = 3;
  cell_data[2].vertices[1] = 7;
  cell_data[2].vertices[2] = 1;
  cell_data[2].vertices[3] = 5;
  cell_data[3].vertices[0] = 1;
  cell_data[3].vertices[1] = 5;
  cell_data[3].vertices[2] = 0;
  cell_data[3].vertices[3] = 4;
  cell_data[4].vertices[0] = 0;
  cell_data[4].vertices[1] = 4;
  cell_data[4].vertices[2] = 8;
  cell_data[4].vertices[3] = 10;
  cell_data[5].vertices[0] = 8;
  cell_data[5].vertices[1] = 10;
  cell_data[5].vertices[2] = 9;
  cell_data[5].vertices[3] = 11;
  cell_data[6].vertices[0] = 9;
  cell_data[6].vertices[1] = 11;
  cell_data[6].vertices[2] = 2;
  cell_data[6].vertices[3] = 6;
  cell_data[7].vertices[0] = 2;
  cell_data[7].vertices[1] = 6;
  cell_data[7].vertices[2] = 3;
  cell_data[7].vertices[3] = 7;

  // vertices on inflow part, using hexagon-shape
  std::vector<Point<2>> vertices_low{
    {0, 0.55 * radius},
    {0.55 * std::sqrt(3. / 4), 0.55 * 0.5},
    {0, -0.55 * radius},
    {0.55 * std::sqrt(3. / 4), -0.55 * 0.5},
    {0, radius},
    {radius * std::sqrt(3. / 4), radius * 0.5},
    {0, -radius},
    {radius * std::sqrt(3. / 4), -radius * 0.5},
    {-0.55 * std::sqrt(3. / 4), 0.55 * 0.5},
    {-0.55 * std::sqrt(3. / 4), -0.55 * 0.5},
    {-radius * std::sqrt(3. / 4), radius * 0.5},
    {-radius * std::sqrt(3. / 4), -radius * 0.5},
  };

  // vertices on bifurcation side, slightly modified hexagon shape to match
  // with outgoing cyclinder
  const double ycord = 0.55 * radius * std::cos(numbers::PI / 12) /
                       (std::sin(numbers::PI / 12) + std::cos(numbers::PI / 12));
  std::vector<Point<2>> vertices_up{
    {0, 0.55 * radius},
    {ycord, ycord},
    {0, -0.55 * radius},
    {ycord, -ycord},
    {0, radius},
    {radius * std::sqrt(1. / 2), radius * std::sqrt(0.5)},
    {0, -radius},
    {radius * std::sqrt(1. / 2), -radius * std::sqrt(0.5)},
    {-ycord, ycord},
    {-ycord, -ycord},
    {-radius * std::sqrt(1. / 2), radius * std::sqrt(0.5)},
    {-radius * std::sqrt(1. / 2), -radius * std::sqrt(0.5)},
  };
  SubCellData subcell_data;
  GridReordering<2>::reorder_cells(cell_data, true);

  // test output of 2d mesh at outflow
  Triangulation<2> tria_2d;
  tria_2d.create_triangulation(vertices_low, cell_data, subcell_data);
  DataOut<2> data_out;
  data_out.attach_triangulation(tria_2d);
  data_out.build_patches();
  std::ofstream file("grid_2d.vtk");
  data_out.write_vtk(file);

  // create 3d vertex data structures: We are going to have two times the
  // vertices of the inflow part plus 8 vertices for each outlet
  std::vector<Point<dim>> vertices_3d(2 * vertices_up.size() + 16);

  // lower inflow part, z coordinate 0
  for(unsigned int i = 0; i < vertices_up.size(); ++i)
    for(unsigned int d = 0; d < 2; ++d)
      vertices_3d[i][d] = vertices_low[i][d];

  // outflow part, set z coordinate to length minus some adjustment for the
  // outgoing bifurcation
  for(unsigned int i = 0; i < vertices_up.size(); ++i)
  {
    for(unsigned int d = 0; d < 2; ++d)
      vertices_3d[vertices_low.size() + i][d] = vertices_up[i][d];
    vertices_3d[vertices_low.size() + i][2] =
      length - std::tan(numbers::PI / 4) * std::abs(tria_2d.get_vertices()[i][0]) / radius;
  }

  // adjust middle vertices at x=0 to ensure the quad where the two outlets
  // meet is valid
  // vertices_3d[16][1] -= radius*(1-std::cos(numbers::PI/8));
  // vertices_3d[16][2] += std::sin(numbers::PI/8);
  // vertices_3d[18][1] += radius*(1-std::cos(numbers::PI/8));
  // vertices_3d[18][2] += std::sin(numbers::PI/8);
  vertices_3d[12][2] -= 0.15 * length;
  vertices_3d[14][2] -= 0.15 * length;

  // define position at outlet, currently located at cylinder with coordinates
  // (0,0,length)
  std::vector<Point<3>> auxiliary_points{
    {radius * std::sqrt(0.5), radius * std::sqrt(0.5), length},
    {-radius * std::sqrt(0.5), radius * std::sqrt(0.5), length},
    {-radius * std::sqrt(0.5), -radius * std::sqrt(0.5), length},
    {radius * std::sqrt(0.5), -radius * std::sqrt(0.5), length},
    {0.4 * radius * std::sqrt(0.5), 0.4 * radius * std::sqrt(0.5), length},
    {-0.4 * radius * std::sqrt(0.5), 0.4 * radius * std::sqrt(0.5), length},
    {-0.4 * radius * std::sqrt(0.5), -0.4 * radius * std::sqrt(0.5), length},
    {0.4 * radius * std::sqrt(0.5), -0.4 * radius * std::sqrt(0.5), length}};

  // rotate the outlet points by 45 degrees to the right and shift by another
  // length unit
  Tensor<2, 3> transform;
  transform[0][0] = std::sqrt(0.5);
  transform[0][2] = -std::sqrt(0.5);
  transform[2][0] = +std::sqrt(0.5);
  transform[2][2] = std::sqrt(0.5);
  transform[1][1] = 1;
  Point<3> offset(0, 0, length);
  for(unsigned int i = 0; i < 8; ++i)
    vertices_3d[2 * vertices_low.size() + i] = Point<dim>(offset + transform * auxiliary_points[i]);

  // rotate the outlet points by 45 degrees to the left and shift by another
  // length unit
  transform[0][0] = std::sqrt(0.5);
  transform[0][2] = std::sqrt(0.5);
  transform[2][0] = -std::sqrt(0.5);
  transform[2][2] = std::sqrt(0.5);
  for(unsigned int i = 0; i < 8; ++i)
    vertices_3d[2 * vertices_low.size() + 8 + i] =
      Point<dim>(offset + transform * auxiliary_points[i]);

  // create vector of connectivities in 3D. start with the first layer which
  // is simply the extrusion of the hexagon from the base
  std::vector<CellData<dim>> cell_data_3d(18);
  for(unsigned int i = 0; i < 8; ++i)
  {
    for(unsigned int v = 0; v < 4; ++v)
      cell_data_3d[i].vertices[v] = cell_data[i].vertices[v];
    for(unsigned int v = 0; v < 4; ++v)
      cell_data_3d[i].vertices[4 + v] = vertices_low.size() + cell_data[i].vertices[v];
  }
  // upper right part -> identify the base indices from the circle-like mesh,
  // 3 elements on the rim (after some renumbering, 2d cell_data)
  for(unsigned int i = 0; i < 3; ++i)
    for(unsigned int v = 0; v < 4; ++v)
      cell_data_3d[8 + i].vertices[v] = cell_data[4 + i].vertices[v] + 12;

  // manually connect to the points at the outlet
  cell_data_3d[8].vertices[4]  = 24;
  cell_data_3d[8].vertices[5]  = 25;
  cell_data_3d[8].vertices[6]  = 28;
  cell_data_3d[8].vertices[7]  = 29;
  cell_data_3d[9].vertices[4]  = 25;
  cell_data_3d[9].vertices[5]  = 26;
  cell_data_3d[9].vertices[6]  = 29;
  cell_data_3d[9].vertices[7]  = 30;
  cell_data_3d[10].vertices[4] = 27;
  cell_data_3d[10].vertices[5] = 31;
  cell_data_3d[10].vertices[6] = 26;
  cell_data_3d[10].vertices[7] = 30;

  // upper left part -> identify the base indices from the other half of 3
  // elements of the circle-like mesh (after some renumbering, 2d cell_data)
  for(unsigned int i = 0; i < 2; ++i)
    for(unsigned int v = 0; v < 4; ++v)
      cell_data_3d[11 + i].vertices[v] = cell_data[2 + i].vertices[v] + 12;
  for(unsigned int v = 0; v < 4; ++v)
    cell_data_3d[13].vertices[v] = cell_data[7].vertices[v] + 12;

  // manually connect to the points at the outlet
  cell_data_3d[11].vertices[4] = 32;
  cell_data_3d[11].vertices[5] = 36;
  cell_data_3d[11].vertices[6] = 35;
  cell_data_3d[11].vertices[7] = 39;
  cell_data_3d[12].vertices[4] = 32;
  cell_data_3d[12].vertices[5] = 33;
  cell_data_3d[12].vertices[6] = 36;
  cell_data_3d[12].vertices[7] = 37;
  cell_data_3d[13].vertices[4] = 35;
  cell_data_3d[13].vertices[5] = 39;
  cell_data_3d[13].vertices[6] = 34;
  cell_data_3d[13].vertices[7] = 38;

  // middle elements of two outlets, connect with center part on the lower
  // cylinder
  for(unsigned int i = 0; i < 2; ++i)
    for(unsigned int v = 0; v < 4; ++v)
      cell_data_3d[14 + i].vertices[v] = cell_data[i].vertices[v] + 12;
  cell_data_3d[14].vertices[4] = 36;
  cell_data_3d[14].vertices[5] = 37;
  cell_data_3d[14].vertices[6] = 39;
  cell_data_3d[14].vertices[7] = 38;
  cell_data_3d[15].vertices[4] = 28;
  cell_data_3d[15].vertices[5] = 29;
  cell_data_3d[15].vertices[6] = 31;
  cell_data_3d[15].vertices[7] = 30;

  // second to last rim element, right outlet
  cell_data_3d[16].vertices[0] = 12;
  cell_data_3d[16].vertices[1] = 14;
  cell_data_3d[16].vertices[2] = 16;
  cell_data_3d[16].vertices[3] = 18;
  cell_data_3d[16].vertices[4] = 28;
  cell_data_3d[16].vertices[5] = 31;
  cell_data_3d[16].vertices[6] = 24;
  cell_data_3d[16].vertices[7] = 27;

  // last rim element, left otlet
  cell_data_3d[17].vertices[0] = 14;
  cell_data_3d[17].vertices[1] = 12;
  cell_data_3d[17].vertices[2] = 18;
  cell_data_3d[17].vertices[3] = 16;
  cell_data_3d[17].vertices[4] = 38;
  cell_data_3d[17].vertices[5] = 37;
  cell_data_3d[17].vertices[6] = 34;
  cell_data_3d[17].vertices[7] = 33;

  // set manifold id on inlet cylinder -> do it via subcell_data
  subcell_data.boundary_quads.resize(6);
  unsigned int count = 0;
  for(auto & cell : tria_2d.active_cell_iterators())
    for(unsigned int f = 0; f < 4; ++f)
      if(cell->at_boundary(f))
      {
        subcell_data.boundary_quads[count].vertices[0] = cell->face(f)->vertex_index(0);
        subcell_data.boundary_quads[count].vertices[1] = cell->face(f)->vertex_index(1);
        subcell_data.boundary_quads[count].vertices[2] =
          cell->face(f)->vertex_index(0) + tria_2d.n_vertices();
        subcell_data.boundary_quads[count].vertices[3] =
          cell->face(f)->vertex_index(1) + tria_2d.n_vertices();
        subcell_data.boundary_quads[count].manifold_id = 15;
        ++count;
      }
  AssertDimension(count, 6);
  GridReordering<3>::reorder_cells(cell_data_3d, true);
  tria.create_triangulation(vertices_3d, cell_data_3d, subcell_data);

  // set manifold ids on faces of outlet cylinders, negative x coordinate gets
  // manifold id 17 (left part) and positive x coordinate gets manifold id 16
  // (right part)
  for(auto & cell : tria.active_cell_iterators())
    for(unsigned int f = 0; f < 6; ++f)
      if(cell->face(f)->manifold_id() == 15)
        cell->face(f)->set_all_manifold_ids(15);
      else if(cell->at_boundary(f) && f < 4)
      {
        if(cell->center()[0] < 0)
          cell->face(f)->set_manifold_id(17);
        else
          cell->face(f)->set_manifold_id(16);
      }

  // set manifolds for lines of outlet cylinders
  for(auto & cell : tria.active_cell_iterators())
    for(unsigned int f = 0; f < 6; ++f)
    {
      const types::manifold_id manifold = cell->face(f)->manifold_id();
      // identify ridge between manifolds 16 and 17
      for(unsigned int l = 0; l < GeometryInfo<2>::lines_per_cell; ++l)
        if(manifold != numbers::flat_manifold_id &&
           cell->face(f)->line(l)->manifold_id() != numbers::flat_manifold_id &&
           cell->face(f)->line(l)->manifold_id() != manifold &&
           (cell->face(f)->line(l)->manifold_id() == 16 ||
            cell->face(f)->line(l)->manifold_id() == 17))
          cell->face(f)->line(l)->set_manifold_id(21);
        else if(manifold != numbers::flat_manifold_id)
          cell->face(f)->line(l)->set_manifold_id(manifold);
    }

  // attach 3 cylindrical manifolds to mesh

  CylindricalManifold<dim> cm1(Point<dim>{0, 0, 1}, Point<dim>{0, 0, 0});
  CylindricalManifold<dim> cm2(Point<dim>{std::sqrt(0.5), 0, std::sqrt(0.5)},
                               Point<dim>{0, 0, length});
  CylindricalManifold<dim> cm3(Point<dim>{-std::sqrt(0.5), 0, std::sqrt(0.5)},
                               Point<dim>{0, 0, length});
  EllipseManifold<dim>     ellipse1(Point<dim>(0, 0, length),
                                Point<dim>(1, 0, 0),
                                radius,
                                radius / std::sin(numbers::PI / 3.));

  if(use_manifolds)
  {
    tria.set_manifold(15, cm1);
    tria.set_manifold(16, cm2);
    tria.set_manifold(17, cm3);

    // long ellipse axis depends on angle, guess 60 degrees
    tria.set_manifold(21, ellipse1);
  }

  triat.copy_triangulation(tria);
  triat.refine_global(n_refine_space);
}
