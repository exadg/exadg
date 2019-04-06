#ifndef LUNG_LUNG_TRIA
#define LUNG_LUNG_TRIA

#include "algebra_util.h"
#include "lung_util.h"

#define LUNG_NUMBER_OF_VERTICES_2D 17


void
create_reference_cylinder(const bool                 do_transition,
                          const unsigned int         n_sections,
                          std::vector<Point<3>> &    vertices_3d,
                          std::vector<CellData<3>> & cell_data_3d)
{
  if(do_transition)
    printf("WARNING: Transition has not been implemented yet (TODO)!\n");

  // position of auxiliary point to achieve an angle of 120 degrees in corner
  // of inner cell
  const double radius = 1;
  const double ycord  = 0.55 * std::sqrt(0.5) * radius * std::cos(numbers::PI / 12) /
                       (std::sin(numbers::PI / 12) + std::cos(numbers::PI / 12));
  // vertices for quarter of circle
  std::vector<Point<2>> vertices{
    {0, 0}, {0.5 * radius, 0}, {ycord, ycord}, {radius, 0}, {radius * 0.5, radius * 0.5}};


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
  GridReordering<2>::reorder_cells(cell_data, true);

  Triangulation<2> tria_2d;
  tria_2d.create_triangulation(vertices, cell_data, subcell_data);

  vertices_3d.clear();
  vertices_3d.resize((n_sections + 1) * tria_2d.n_vertices());
  cell_data_3d.clear();
  cell_data_3d.resize(n_sections * cell_data.size());

  for(unsigned int s = 0; s <= n_sections; s++)
  {
    const double       beta  = (1.0 * s) / n_sections;
    const unsigned int shift = s * tria_2d.n_vertices();
    for(unsigned int i = 0; i < tria_2d.n_vertices(); ++i)
    {
      vertices_3d[shift + i][0] = tria_2d.get_vertices()[i][0];
      vertices_3d[shift + i][1] = tria_2d.get_vertices()[i][1];
      vertices_3d[shift + i][2] = beta;
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
create_cylinder(double       radius1,
                double       radius2,
                double       length,
                Tensor<2, 3> transform,
                Tensor<2, 3> transform_parent,
                Point<3>     offset,
                dealii::Tensor<1, 3> /*direction*/,
                std::vector<Point<3>> &    vertices_3d,
                std::vector<Point<3>> &    skeleton,
                std::vector<CellData<3>> & cell_data_3d,
                double                     deg0,
                double                     deg1,
                double                     deg2,
                double                     degree_seperation,
                bool                       is_left,
                bool                       do_rotate,
                bool                       has_no_children,
                bool                       do_twist,
                bool                       do_rot,
                bool                       do_rot_parent,
                unsigned int               n_sections = 1)
{
  // create some short cuts
  bool is_right      = !is_left;
  bool has_children  = !has_no_children;
  bool do_not_rotate = !do_rotate;

  /**************************************************************************
   * Create reference cylinder
   **************************************************************************/
  bool                  do_transition = false;
  std::vector<Point<3>> vertices_3d_temp;
  create_reference_cylinder(do_transition, n_sections, vertices_3d_temp, cell_data_3d);
  vertices_3d.resize(vertices_3d_temp.size());

  /**************************************************************************
   * Loop over all points and transform
   **************************************************************************/
  for(unsigned int i = 0; i < vertices_3d_temp.size(); ++i)
  {
    // get reference to input and output
    auto &point_in = vertices_3d_temp[i], &point_out = vertices_3d[i];

    // transform point in both coordinate system
    Point<3> point_out_alph, point_out_beta;

    // get blending factor
    const double beta = point_in[2];

    /************************************************************************
     * Top part
     ************************************************************************/
    if(do_twist)
    {
      if(has_no_children && is_left)
      {
        point_out_alph[0] += (+point_in[1] * radius2);
        point_out_alph[1] += (-point_in[0] * radius2);
      }
      if(has_no_children && is_right)
      {
        point_out_alph[0] += (-point_in[1] * radius2);
        point_out_alph[1] += (+point_in[0] * radius2);
      }
      if(has_children)
      {
        point_out_alph[0] += (point_in[0] * radius2);
        point_out_alph[1] += (point_in[1] * radius2);
      }

      if(point_in[0] > 0)
        point_out_alph[2] += (-length + std::tan(deg2 / 2) * std::abs(point_in[0]) * radius2);
      else
        point_out_alph[2] += (-length + std::tan(deg1 / 2) * std::abs(point_in[0]) * radius2);
    }
    else if(do_rot)
    {
      point_out_alph[0] += (+point_in[1] * radius2);
      point_out_alph[1] += (-point_in[0] * radius2);

      if(point_in[0] > 0)
        point_out_alph[2] += (-length + std::tan(deg2 / 2) * std::abs(point_in[0]) * radius2);
      else
        point_out_alph[2] += (-length + std::tan(deg1 / 2) * std::abs(point_in[0]) * radius2);
    }
    else
    {
      // top part
      if(has_no_children && is_left)
      {
        point_out_alph[0] += (+point_in[1] * radius2);
        point_out_alph[1] += (-point_in[0] * radius2);
      }
      if(has_no_children && is_right)
      {
        point_out_alph[0] += (-point_in[1] * radius2);
        point_out_alph[1] += (+point_in[0] * radius2);
      }
      if(has_children)
      {
        point_out_alph[0] += (+point_in[1] * radius2);
        point_out_alph[1] += (-point_in[0] * radius2);
      }

      if(point_in[1] > 0)
        point_out_alph[2] += (-length + std::tan(deg2 / 2) * std::abs(point_in[1]) * radius2);
      else
        point_out_alph[2] += (-length + std::tan(deg1 / 2) * std::abs(point_in[1]) * radius2);
    }

    /************************************************************************
     * Bottom part
     ************************************************************************/
    if(do_rot_parent)
    {
      if(do_rotate && is_left)
      {
        if(point_in[0] > 0) // side
        {
          point_out_beta[1] += (-(point_in[0] * 1.0) * radius1);
          point_out_beta[0] += (point_in[1] * radius1);
          point_out_beta[2] -= (point_in[0] * radius1) * 1.0;
        }
        else // main
        {
          point_out_beta[1] += (-point_in[0] * radius1);
          point_out_beta[0] += (point_in[1] * radius1);
          point_out_beta[2] += (std::tan(deg0) * std::abs(point_in[0]) * radius1);
        }
      }
      else
      {
        if(point_in[0] < 0) // side
        {
          point_out_beta[1] += ((-point_in[0] * 0.0) * radius1);
          point_out_beta[0] += (point_in[1] * radius1);
          point_out_beta[2] += (point_in[0] * radius1) * 1.5;
        }
        else // main
        {
          point_out_beta[1] += (-point_in[0] * radius1);
          point_out_beta[0] += (point_in[1] * radius1);
          point_out_beta[2] += (std::tan(deg0) * std::abs(point_in[0]) * radius1);
        }
      }
    }
    else
    {
      if(do_rotate && is_left)
      {
        point_out_beta[1] += (-point_in[0] * radius1);
        if(point_in[1] > 0)
        {
          auto deg4 = degree_seperation;
          point_out_beta[0] -= +std::sin(deg4) * (-point_in[1] * radius1);
          point_out_beta[2] +=
            (-point_in[1] * radius1) - (std::sin(std::abs(deg4)) * std::abs(point_in[1]) * radius1);
        }
        else
        {
          point_out_beta[0] += (point_in[1] * radius1);
          point_out_beta[2] += (std::tan(deg0 / 1) * std::abs(point_in[1]) * radius1);
        }
      }
      if(do_rotate && is_right)
      {
        point_out_beta[1] += (+point_in[0] * radius1);
        if(point_in[1] > 0)
        {
          auto deg4 = degree_seperation;
          point_out_beta[0] -= 1.0 * (+std::sin(deg4)) * (-point_in[1] * radius1);
          point_out_beta[2] += (-point_in[1] * radius1) -
                               (+std::sin(std::abs(deg4)) * std::abs(point_in[1]) * radius1);
        }
        else
        {
          point_out_beta[0] += (-point_in[1] * radius1);
          point_out_beta[2] += (+std::tan(deg0 / 1) * std::abs(point_in[1]) * radius1);
        }
      }
      if(do_not_rotate)
      {
        for(unsigned int d = 0; d < 2; ++d)
          point_out_beta[d] += (point_in[d] * radius1);
      }
    }

    /************************************************************************
     * Combine points and blend
     ************************************************************************/
    point_out = (1 - beta) * Point<3>(offset + transform * point_out_alph) +
                beta * Point<3>(offset + transform_parent * point_out_beta);
    
    if((beta==0.0 || beta==1.0) && (std::abs(point_in[0])==1.0 || std::abs(point_in[1])==1.0))
      skeleton.push_back(point_out);
  }
}

void
process_node(Node *                     node,
             std::vector<CellData<3>> & cell_data_3d_global,
             std::vector<Point<3>> &    vertices_3d_global,
             const int                  id                = LungID::create_root(),
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
                    node->skeleton,
                    cell_data_3d,
                    degree_parent / 2,
                    degree_1,
                    degree_2,
                    degree_seperation,
                    node->is_left(),
                    !node->is_root(),
                    !node->has_children(),
                    node->do_twist,
                    node->do_rot,
                    node->is_root() ? false : node->parent->do_rot,
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
      c.material_id = id;
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
                   LungID::generate(id, true),
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
                   LungID::generate(id, false),
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
