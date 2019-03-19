#ifndef LUNG_LUNG_TRIA
#define LUNG_LUNG_TRIA

#include "algebra_util.h"
#include "lung_util.h"

#define LUNG_NUMBER_OF_VERTICES_2D 17

void
create_cylinder(double       radius1,
                double       radius2,
                double       length,
                Tensor<2, 3> transform,
                Tensor<2, 3> transform_parent,
                Point<3>     offset,
                dealii::Tensor<1, 3> /*direction*/,
                std::vector<Point<3>> &    vertices_3d,
                std::vector<CellData<3>> & cell_data_3d,
                double                     deg0,
                double                     deg1,
                double                     deg2,
                double                     degree_seperation,
                bool                       is_left,
                bool                       do_rotate,
                bool                       has_no_children,
                bool                       do_twist,
                unsigned int               n_sections = 1)
{
  bool is_right      = !is_left;
  bool has_children  = !has_no_children;
  bool do_not_rotate = !do_rotate;
  // bool do_twist = false;

  // printf("%7.4f %7.4f %7.4f %7.4f  \n",
  //         deg0 / numbers::PI / 2 *360,
  //         deg1 / numbers::PI / 2 *360,
  //         deg2 / numbers::PI / 2 *360,
  //         degree_seperation / numbers::PI / 2 *360
  //         );

  //        std::cout << "aaa " << radius << " " << length << std::endl;
  //        radius = 1;
  //        length = 3;

  // position of auxiliary point to achieve an angle of 120 degrees in corner
  // of inner cell
  const double radius = 1;
  const double ycord  = 0.55 * radius * std::cos(numbers::PI / 12) /
                       (std::sin(numbers::PI / 12) + std::cos(numbers::PI / 12));
  // vertices for quarter of circle
  std::vector<Point<2>> vertices{{0, 0},
                                 {0.55 * radius, 0},
                                 {ycord, ycord},
                                 {radius, 0},
                                 {radius * std::sqrt(0.5), radius * std::sqrt(0.5)}};

  // create additional vertices for other three quarters of circle -> gives 17
  // vertices in total
  for(unsigned int a = 1; a < 4; ++a)
  {
    Tensor<2, 2> transform;
    transform[0][0] = a == 2 ? -1. : 0;
    transform[1][0] = a == 2 ? 0 : (a == 1 ? 1 : -1);
    transform[0][1] = -transform[1][0];
    transform[1][1] = transform[0][0];
    for(unsigned int i = 1; i < 5; ++i)
      vertices.push_back(Point<2>(transform * vertices[i]));
  }

  // create 12 cells for 2d mesh on base; the first four elements are at the
  // center of the circle
  std::vector<CellData<2>> cell_data(12);
  cell_data[0].vertices[0] = 0;
  cell_data[0].vertices[1] = 1;
  cell_data[0].vertices[2] = 5;
  cell_data[0].vertices[3] = 2;
  cell_data[1].vertices[0] = 9;
  cell_data[1].vertices[1] = 0;
  cell_data[1].vertices[2] = 6;
  cell_data[1].vertices[3] = 5;
  cell_data[2].vertices[0] = 10;
  cell_data[2].vertices[1] = 13;
  cell_data[2].vertices[2] = 9;
  cell_data[2].vertices[3] = 0;
  cell_data[3].vertices[0] = 13;
  cell_data[3].vertices[1] = 14;
  cell_data[3].vertices[2] = 0;
  cell_data[3].vertices[3] = 1;

  // the next 8 elements describe the rim; we take one quarter of the circle
  // in each loop iteration
  for(unsigned int a = 0; a < 4; ++a)
  {
    cell_data[4 + a * 2].vertices[0] = 1 + a * 4;
    cell_data[4 + a * 2].vertices[1] = 3 + a * 4;
    cell_data[4 + a * 2].vertices[2] = 2 + a * 4;
    cell_data[4 + a * 2].vertices[3] = 4 + a * 4;
    cell_data[5 + a * 2].vertices[0] = 2 + a * 4;
    cell_data[5 + a * 2].vertices[1] = 4 + a * 4;
    AssertIndexRange(4 + a * 4, vertices.size());
    cell_data[5 + a * 2].vertices[2] = a == 3 ? 1 : 5 + a * 4;
    cell_data[5 + a * 2].vertices[3] = a == 3 ? 3 : 7 + a * 4;
  }
  SubCellData subcell_data;
  // must reorder cells to get valid 2d triangulation
  GridReordering<2>::reorder_cells(cell_data, true);

  Triangulation<2> tria_2d;
  tria_2d.create_triangulation(vertices, cell_data, subcell_data);

  // create 3d vertices, first two layers of the 2d mesh with the usual
  // extrusion.
  // std::vector<Point<dim>> vertices_3d(2*tria_2d.n_vertices());
  vertices_3d.clear();
  vertices_3d.resize((n_sections + 1) * tria_2d.n_vertices());
  cell_data_3d.clear();
  cell_data_3d.resize(n_sections * cell_data.size());

  AssertThrow(LUNG_NUMBER_OF_VERTICES_2D == tria_2d.n_vertices(),
              ExcMessage("Number of vertices in 2D does not match with the expectation!"));

  std::vector<Point<3>> vertices_3d_temp(2 * tria_2d.n_vertices());
  for(auto & v : vertices_3d_temp)
    v = {0, 0, 0};
  for(unsigned int s = 0; s <= 1; s++)
  {
    const double       beta  = (1.0 * s) / 1;
    const double       alpha = 1.0 - beta;
    const unsigned int shift = s * tria_2d.n_vertices();
    for(unsigned int i = 0; i < tria_2d.n_vertices(); ++i)
    {
      if(do_twist)
      {
        // top part
        if(has_no_children && is_left)
        {
          vertices_3d_temp[shift + i][0] += alpha * (+tria_2d.get_vertices()[i][1] * radius2);
          vertices_3d_temp[shift + i][1] += alpha * (-tria_2d.get_vertices()[i][0] * radius2);
        }
        if(has_no_children && is_right)
        {
          vertices_3d_temp[shift + i][0] += alpha * (-tria_2d.get_vertices()[i][1] * radius2);
          vertices_3d_temp[shift + i][1] += alpha * (+tria_2d.get_vertices()[i][0] * radius2);
        }
        if(has_children)
        {
          vertices_3d_temp[shift + i][0] += alpha * (tria_2d.get_vertices()[i][0] * radius2);
          vertices_3d_temp[shift + i][1] += alpha * (tria_2d.get_vertices()[i][1] * radius2);
        }

        if(tria_2d.get_vertices()[i][0] > 0)
          vertices_3d_temp[shift + i][2] +=
            alpha *
            (-length + std::tan(deg2 / 2) * std::abs(tria_2d.get_vertices()[i][0]) * radius2);
        else
          vertices_3d_temp[shift + i][2] +=
            alpha *
            (-length + std::tan(deg1 / 2) * std::abs(tria_2d.get_vertices()[i][0]) * radius2);
      }
      else
      {
        // top part
        if(has_no_children && is_left)
        {
          vertices_3d_temp[shift + i][0] += alpha * (+tria_2d.get_vertices()[i][1] * radius2);
          vertices_3d_temp[shift + i][1] += alpha * (-tria_2d.get_vertices()[i][0] * radius2);
        }
        if(has_no_children && is_right)
        {
          vertices_3d_temp[shift + i][0] += alpha * (-tria_2d.get_vertices()[i][1] * radius2);
          vertices_3d_temp[shift + i][1] += alpha * (+tria_2d.get_vertices()[i][0] * radius2);
        }
        if(has_children)
        {
          vertices_3d_temp[shift + i][0] += alpha * (+tria_2d.get_vertices()[i][1] * radius2);
          vertices_3d_temp[shift + i][1] += alpha * (-tria_2d.get_vertices()[i][0] * radius2);
        }

        if(tria_2d.get_vertices()[i][1] > 0)
          vertices_3d_temp[shift + i][2] +=
            alpha *
            (-length + std::tan(deg2 / 2) * std::abs(tria_2d.get_vertices()[i][1]) * radius2);
        else
          vertices_3d_temp[shift + i][2] +=
            alpha *
            (-length + std::tan(deg1 / 2) * std::abs(tria_2d.get_vertices()[i][1]) * radius2);
      }

      // bottom part
      if(do_rotate && is_left)
      {
        vertices_3d_temp[shift + i][1] += beta * (-tria_2d.get_vertices()[i][0] * radius1);
        if(tria_2d.get_vertices()[i][1] > 0)
        {
          // auto deg4 = 22.5/360.0*2*numbers::PI;
          auto deg4 = degree_seperation;
          // vertices_3d_temp[shift + i][2] += beta * (-tria_2d.get_vertices()[i][1] * radius1);
          vertices_3d_temp[shift + i][0] -=
            beta * +std::sin(deg4) * (-tria_2d.get_vertices()[i][1] * radius1);
          vertices_3d_temp[shift + i][2] +=
            beta * (-tria_2d.get_vertices()[i][1] * radius1) -
            beta * (std::sin(std::abs(deg4)) * std::abs(tria_2d.get_vertices()[i][1]) * radius1);
        }
        else
        {
          vertices_3d_temp[shift + i][0] += beta * (tria_2d.get_vertices()[i][1] * radius1);
          vertices_3d_temp[shift + i][2] +=
            beta * (std::tan(deg0 / 1) * std::abs(tria_2d.get_vertices()[i][1]) * radius1);
        }
      }
      if(do_rotate && is_right)
      {
        vertices_3d_temp[shift + i][1] += beta * (+tria_2d.get_vertices()[i][0] * radius1);
        if(tria_2d.get_vertices()[i][1] > 0)
        {
          // vertices_3d_temp[shift + i][0] += beta * (+std::tan(deg0 / 1)) *
          // (-tria_2d.get_vertices()[i][1] * radius1); auto deg4 = 22.5/360.0*2*numbers::PI;
          auto deg4 = degree_seperation;
          vertices_3d_temp[shift + i][0] -=
            1.0 * beta * (+std::sin(deg4)) * (-tria_2d.get_vertices()[i][1] * radius1);
          vertices_3d_temp[shift + i][2] +=
            beta * (-tria_2d.get_vertices()[i][1] * radius1) -
            beta * (+std::sin(std::abs(deg4)) * std::abs(tria_2d.get_vertices()[i][1]) * radius1);
        }
        else
        {
          vertices_3d_temp[shift + i][0] += beta * (-tria_2d.get_vertices()[i][1] * radius1);
          vertices_3d_temp[shift + i][2] +=
            beta * (+std::tan(deg0 / 1) * std::abs(tria_2d.get_vertices()[i][1]) * radius1);
        }
      }
      if(do_not_rotate)
      {
        for(unsigned int d = 0; d < 2; ++d)
          vertices_3d_temp[shift + i][d] += beta * (tria_2d.get_vertices()[i][d] * radius1);
      }
    }
    auto tr = transform * alpha + transform_parent * beta;
    for(unsigned int i = 0; i < tria_2d.n_vertices(); ++i)
    {
      vertices_3d_temp[shift + i] = Point<3>(offset + tr * vertices_3d_temp[shift + i]);
    }
  }

  for(unsigned int s = 0; s <= n_sections; s++)
  {
    const double       beta  = (1.0 * s) / n_sections;
    const double       alpha = 1.0 - beta;
    const unsigned int shift = s * tria_2d.n_vertices();
    for(unsigned int i = 0; i < tria_2d.n_vertices(); ++i)
    {
      vertices_3d[shift + i] =
        alpha * vertices_3d_temp[i] + beta * vertices_3d_temp[tria_2d.n_vertices() + i];
    }
  }

  for(unsigned int s = 0; s < n_sections; s++)
    for(unsigned int i = 0; i < cell_data.size(); ++i)
    {
      for(unsigned int v = 0; v < 4; ++v)
        cell_data_3d[cell_data.size() * s + i].vertices[v + 0] =
          (s + 0) * vertices.size() + cell_data[i].vertices[v];
      for(unsigned int v = 0; v < 4; ++v)
        cell_data_3d[cell_data.size() * s + i].vertices[4 + v] =
          (s + 1) * vertices.size() + cell_data[i].vertices[v];
    }
}

void
process_node(Node *                     node,
             std::vector<CellData<3>> & cell_data_3d_global,
             std::vector<Point<3>> &    vertices_3d_global,
             const int                  parent_os         = 0,
             Tensor<2, 3>               transform_parent  = Tensor<2, 3>(),
             double                     degree_parent     = 0.0,
             double                     degree_seperation = 0.0)
{
  // normal and tangential vector in the reference system
  dealii::Tensor<1, 3> src_n({0, 1, 0});
  dealii::Tensor<1, 3> src_t({0, 0, 1});

  unsigned int os = vertices_3d_global.size();

  // get tangential-vector
  auto dst_t = node->get_tangential_vector();
  dst_t /= dst_t.norm();

  // compute rotation matrix based on tangential and normal vector
  auto transform = transform_parent;
  {
    auto dst_n =
      node->has_children() ? node->get_normal_vector() : node->get_parent()->get_normal_vector();
    dst_n /= dst_n.norm();
    transform = compute_rotation_matrix(src_n, src_t, dst_n, dst_t);
  }

  double degree_1 = node->has_children() ? node->get_degree_1() : 0.0;
  double degree_2 = node->has_children() ? node->get_degree_2() : 0.0;

  if(!node->is_dummy())
  {
    // root cannot access parent's rotation matrix -> simply use its own
    if(node->is_root())
      transform_parent = transform;

    // extract some parameters of this branch
    auto source = node->get_source();
    auto target = node->get_target();
    auto vec    = target - source;

    // compute vertices and cells in reference system
    std::vector<CellData<3>> cell_data_3d;
    std::vector<Point<3>>    vertices_3d;
    create_cylinder(node->is_root() ? node->get_radius() : node->get_parent()->get_radius(),
                    node->get_radius(),
                    vec.norm(),
                    transform,
                    transform_parent,
                    source,
                    dst_t,
                    vertices_3d,
                    cell_data_3d,
                    degree_parent / 2,
                    degree_1,
                    degree_2,
                    degree_seperation,
                    node->is_left(),
                    !node->is_root(),
                    !node->has_children(),
                    node->do_twist,
                    node->get_intersections());

    // create triangulation
    SubCellData      subcell_data;
    Triangulation<3> tria;

    try
    {
      tria.create_triangulation(vertices_3d, cell_data_3d, subcell_data);
    }
    catch(const std::exception & e)
    {
      std::cout << e.what();

      std::ostringstream stream;
      stream << "Problematic branch:" << std::endl
             << "   " << node->get_source() << std::endl
             << "   " << node->get_target();

      AssertThrow(false, ExcMessage(stream.str()));
    }

    // WARNING: This section is only reached if the creation of the
    // triangulation was successful (i.e. the cells are not too much deformed)

    unsigned int range_local = (node->get_intersections() + 1) * LUNG_NUMBER_OF_VERTICES_2D;
    unsigned int range_global =
      node->is_root() ?
        0 :
        ((node->get_parent()->get_intersections() +
          (node->is_left() ? 0 : node->get_parent()->get_left_child()->get_intersections())) +
         2) *
          LUNG_NUMBER_OF_VERTICES_2D;

    // mark all vertices of local branch with -1
    std::map<unsigned int, unsigned int> map;
    for(unsigned int i = 0; i < range_local; i++)
      map[i] = numbers::invalid_unsigned_int;

    // check if vertex is already available (i.e. already created by parent or left neighbor)
    for(unsigned int i = 0; i < range_local; i++)
      for(unsigned int j = parent_os;
          (j < parent_os + range_global) && (j < vertices_3d_global.size());
          j++)
      {
        auto t = vertices_3d[i];
        t -= vertices_3d_global[j];
        if(t.norm() < 1e-5)
        {
          map[i] = j;
          break;
        }
      }

    // assign actual new vertices new ids and save the position of these vertices
    int cou = os;
    for(unsigned int i = 0; i < range_local; i++)
      if(map[i] == numbers::invalid_unsigned_int)
      {
        vertices_3d_global.push_back(vertices_3d[i]);
        map[i] = cou++;
      }

    // save cell definition
    for(auto c : cell_data_3d)
    {
      for(int i = 0; i < 8; i++)
        c.vertices[i] = map[c.vertices[i]];
      cell_data_3d_global.push_back(c);
    }
  }

  // process children
  if(node->has_children())
  {
    // left child:
    try
    {
      process_node(node->get_left_child(),
                   cell_data_3d_global,
                   vertices_3d_global,
                   os,
                   transform,
                   degree_1,
                   (degree_2 - degree_1) / 2);
    }
    catch(const std::exception & e)
    {
      std::cout << e.what();
    }

    // right child:
    try
    {
      process_node(node->get_right_child(),
                   cell_data_3d_global,
                   vertices_3d_global,
                   os,
                   transform,
                   degree_2,
                   (degree_2 - degree_1) / 2);
    }
    catch(const std::exception & e)
    {
      std::cout << e.what();
    }
  }
}

#endif
