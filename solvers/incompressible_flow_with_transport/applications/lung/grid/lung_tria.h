#ifndef LUNG_LUNG_TRIA
#define LUNG_LUNG_TRIA

// ExaDG
#include "algebra_util.h"
#include "lung_util.h"

#define LUNG_NUMBER_OF_VERTICES_2D 17

#define DEBUG_LUNG_TRIANGULATION 0

namespace ExaDG
{
using namespace dealii;

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
  const double radius  = 1;
  const double y_coord = 0.55 * std::sqrt(0.5) * radius * std::cos(numbers::PI / 12) /
                         (std::sin(numbers::PI / 12) + std::cos(numbers::PI / 12));
  // vertices for quarter of circle
  std::vector<Point<2>> vertices{
    {0, 0}, {0.5 * radius, 0}, {y_coord, y_coord}, {radius, 0}, {radius * 0.5, radius * 0.5}};


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
      vertices.emplace_back(transform * vertices[i]);
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

#if DEBUG_LUNG_TRIANGULATION
  // create triangulation
  Triangulation<3> tria(Triangulation<3>::MeshSmoothing::none, true);

  tria.create_triangulation(vertices_3d, cell_data_3d, subcell_data);
  GridOut gridout;

  std::ofstream out("mesh_reference_cylinder.vtu");

  gridout.write_vtu(tria, out);
#endif
}


void
create_cylinder(double       radius1,
                double       radius2,
                double       length,
                Tensor<2, 3> transform_top,
                Tensor<2, 3> transform_bottom,
                Point<3>     offset,
                std::vector<Point<3>> &    vertices_3d,
                std::vector<Point<3>> &    skeleton,
                std::vector<CellData<3>> & cell_data_3d,
                double                     deg0,
                double                     deg1,
                double                     deg2,
                double                     degree_parent_intersection,
                double                     degree_child_intersection,
                double                     degree_separation,
                bool                       is_left,
                bool                       left_right_mixed_up_bottom,
                bool                       left_right_mixed_up_top,
                bool                       do_rotate_bottom,
                unsigned int               n_sections = 1)
{
  if(left_right_mixed_up_bottom)
    std::swap(deg1, deg2);

  if(left_right_mixed_up_top)
    std::swap(deg0, degree_separation);

  // create some short cuts
  bool is_right      = !is_left;

  /**************************************************************************
   * Create reference cylinder
   **************************************************************************/
  bool                  do_transition = false;
  std::vector<Point<3>> vertices_3d_temp;
  create_reference_cylinder(do_transition, n_sections, vertices_3d_temp, cell_data_3d);
  vertices_3d.resize(vertices_3d_temp.size());

  skeleton.clear();
  skeleton.resize(8);

  /**************************************************************************
   * Loop over all points and transform
   **************************************************************************/
  for(unsigned int i = 0; i < vertices_3d_temp.size(); ++i)
  {
    // get reference to input and output
    auto &point_in = vertices_3d_temp[i], &point_out = vertices_3d[i];

    // transform point in both coordinate systems
    Point<3> point_out_alpha, point_out_beta;

    // get blending factor
    const double beta = point_in[2];

    /************************************************************************
     * Top part
     ************************************************************************/
    point_out_alpha[0] = point_in[0] * radius1;
    point_out_alpha[1] = point_in[1] * radius1;

    auto   deg4       = degree_separation;

    if(is_right)
    {
      if(point_in[0] > 0)
        point_out_alpha[2] = length * beta +
                             std::tan((numbers::PI_2 - deg4)) * std::abs(point_in[0]) * radius1 +
                             std::tan(degree_child_intersection) * point_in[1] * radius1;
      else
        point_out_alpha[2] = length * beta +
                             std::tan(numbers::PI_2 - deg0) * std::abs(point_in[0]) * radius1 +
                             std::tan(degree_child_intersection) * point_in[1] * radius1;
    }
    else
    {
      if(point_in[0] > 0)
        point_out_alpha[2] = length * beta +
                             std::tan((numbers::PI_2 - deg0)) * std::abs(point_in[0]) * radius1 +
                             std::tan(degree_child_intersection) * point_in[1] * radius1;
      else
        point_out_alpha[2] = length * beta +
                             std::tan(numbers::PI_2 - deg4) * std::abs(point_in[0]) * radius1 +
                             std::tan(degree_child_intersection) * point_in[1] * radius1;
    }


    /************************************************************************
     * Bottom part
     ************************************************************************/

    point_out_beta[0] = point_in[0] * radius2;
    point_out_beta[1] = point_in[1] * radius2;


    if(!do_rotate_bottom)
    {
      if(point_in[0] > 0)
        point_out_beta[2] = length * beta -
                            std::tan(numbers::PI_2 - deg1) * std::abs(point_in[0]) * radius2 -
                            std::tan(degree_parent_intersection) * point_in[1] * radius2;
      else
        point_out_beta[2] = length * beta -
                            std::tan(numbers::PI_2 - deg2) * std::abs(point_in[0]) * radius2 -
                            std::tan(degree_parent_intersection) * point_in[1] * radius2;
    }
    else
    {
      if(point_in[1] < 0)
        point_out_beta[2] = length * beta -
                            std::tan(numbers::PI_2 - deg1) * std::abs(point_in[1]) * radius2 -
                            std::tan(degree_parent_intersection) * point_in[0] * radius2;
      else
        point_out_beta[2] = length * beta -
                            std::tan(numbers::PI_2 - deg2) * std::abs(point_in[1]) * radius2 -
                            std::tan(degree_parent_intersection) * point_in[0] * radius2;
    }

    /************************************************************************
     * Combine points and blend
     ************************************************************************/
    point_out = (1 - beta) * Point<3>(offset + transform_top * point_out_alpha) +
                beta * Point<3>(offset + transform_bottom * point_out_beta);

    /************************************************************************
    * Fill skeleton vector with corner nodes
    ************************************************************************/

    if((beta == 0.0 || beta == 1.0) && (std::abs(std::abs(point_in[0]) - 1.0) < 1e-8 ||
                                        std::abs(std::abs(point_in[1]) - 1.0) < 1e-8))
    {
      const unsigned int idz            = beta == 0.0 ? 1 : 0;
      const unsigned int idy            = (point_in[1] == -1 || point_in[0] == -1) ? 1 : 0;
      const unsigned int idx            = (point_in[0] == +1 || point_in[1] == -1) ? 1 : 0;
      skeleton[idz * 4 + idy * 2 + idx] = point_out;
    }
  }
}

void
process_node(Node *                                   node,
             std::vector<CellData<3>> &               cell_data_3d_global,
             std::vector<Point<3>> &                  vertices_3d_global,
             const unsigned int                       id,
             const unsigned int                       parent_os                  = 0,
             double                                   degree_parent              = numbers::PI_2,
             double                                   degree_separation          = numbers::PI_2,
             double                                   degree_child_intersection  = 0.0,
             dealii::Tensor<1, 3>                     normal_rotation_child      = Tensor<1, 3>(),
             bool                                     left_right_mixed_up_parent = false)
{
  // normal and tangential vector in the reference system
  dealii::Tensor<1, 3> src_n({0, 1, 0});
  dealii::Tensor<1, 3> src_t({0, 0, 1});

  unsigned int os = vertices_3d_global.size();

  // get tangential-vector
  auto dst_t = node->get_tangential_vector();
  dst_t /= dst_t.norm();

  dealii::Tensor<1, 3> normal_rotation_parent;
  dealii::Tensor<1, 3> normal_rotation_left_child;
  dealii::Tensor<1, 3> normal_rotation_right_child;

  double degree_parent_intersection      = 0.0;
  double degree_left_child_intersection  = 0.0;
  double degree_right_child_intersection = 0.0;

  bool left_right_mixed_up = false;

  double degree_parent_left_child      = numbers::PI_2;
  double degree_parent_right_child     = numbers::PI_2;
  double degree_left_child_right_child = numbers::PI_2;


  if(node->has_children())
  {
    // calculate plane normals
    auto normal_children     = node->get_normal_vector_children();
    auto normal_parent_left  = node->get_normal_vector_parent_left();
    auto normal_parent_right = node->get_normal_vector_parent_right();

#if DEBUG_LUNG_TRIANGULATION
    std::cout << "normal children plane: " << normal_children / normal_children.norm() << std::endl;
    std::cout << "normal parent left plane: " << normal_parent_left / normal_parent_left.norm()
              << std::endl;
    std::cout << "normal parent right plane: " << normal_parent_right / normal_parent_right.norm()
              << std::endl
              << std::endl;
#endif

    // tangents of vectors
    auto tangent_parent      = -node->get_tangential_vector();
    auto tangent_right_child = node->right_child->get_tangential_vector();
    auto tangent_left_child  = node->left_child->get_tangential_vector();

    // check if planar
    bool is_bifurcation_planar =
      Node::check_if_planar(tangent_parent, tangent_left_child, tangent_right_child);

    dealii::Tensor<1, 3> normal_intersection_plane;
    if(is_bifurcation_planar)
      normal_intersection_plane = normal_children;
    else
    {
      // calculate normal intersection plane

      std::vector<dealii::Tensor<1, 3>> normal_intersection_planes(8);
      normal_intersection_planes[0] = normal_children + normal_parent_left + normal_parent_right;
      normal_intersection_planes[1] = normal_children + normal_parent_left - normal_parent_right;
      normal_intersection_planes[2] = normal_children - normal_parent_left + normal_parent_right;
      normal_intersection_planes[3] = normal_children - normal_parent_left - normal_parent_right;
      normal_intersection_planes[4] = -normal_children + normal_parent_left + normal_parent_right;
      normal_intersection_planes[5] = -normal_children + normal_parent_left - normal_parent_right;
      normal_intersection_planes[6] = -normal_children - normal_parent_left + normal_parent_right;
      normal_intersection_planes[7] = -normal_children - normal_parent_left - normal_parent_right;

#if DEBUG_LUNG_TRIANGULATION
      std::cout << "normal_intersection_planes:" << std::endl;

      for(auto i : normal_intersection_planes)
        std::cout << i << std::endl;

      std::cout << std::endl;
#endif

      std::vector<dealii::Tensor<1, 3>> normals_rotation_parent(8);
      for(int i = 0; i < 8; i++)
        normals_rotation_parent[i] =
          normal_intersection_planes[i] - normal_intersection_planes[i] * tangent_parent /
                                            tangent_parent.norm_square() * tangent_parent;

      std::vector<double> degree_intersection_normal(8);

      for(int i = 0; i < 8; i++)
      {
        degree_intersection_normal[i] =
          Node::get_degree(normal_intersection_planes[i], normals_rotation_parent[i]);
      }

#if DEBUG_LUNG_TRIANGULATION
      std::cout << "degree_intersection_normal:" << std::endl;

      for(auto i : degree_intersection_normal)
        std::cout << i << std::endl;

      std::cout << std::endl;
#endif

      std::vector<double> sum_degree_intersection_normal_tangents(8);

      for(int i = 0; i < 8; i++)
        sum_degree_intersection_normal_tangents[i] =
          Node::get_degree(normal_intersection_planes[i], tangent_parent) +
          Node::get_degree(normal_intersection_planes[i], tangent_left_child) +
          Node::get_degree(normal_intersection_planes[i], tangent_right_child);

#if DEBUG_LUNG_TRIANGULATION
      std::cout << "sum_degree_intersection_normal_tangents:" << std::endl;

      for(auto i : sum_degree_intersection_normal_tangents)
        std::cout << i << std::endl;

      std::cout << std::endl;
#endif

      std::vector<double> product_degrees(8);

      for(int i = 0; i < 8; i++)
        product_degrees[i] =
          degree_intersection_normal[i] * sum_degree_intersection_normal_tangents[i];

      auto minimum_element = std::min_element(product_degrees.begin(), product_degrees.end());

      int minimum_element_at = std::distance(product_degrees.begin(), minimum_element);

      normal_intersection_plane = normal_intersection_planes[minimum_element_at];

#if DEBUG_LUNG_TRIANGULATION
      std::cout << "normal_intersection_plane_chosen: " << normal_intersection_plane << std::endl
                << std::endl;
#endif
    }

    normal_intersection_plane = normal_intersection_plane / normal_intersection_plane.norm();

#if DEBUG_LUNG_TRIANGULATION
    std::cout << "normal intersection plane: " << normal_intersection_plane << std::endl
              << std::endl;
#endif

    // calculate rotation normals
    normal_rotation_parent =
      normal_intersection_plane -
      normal_intersection_plane * tangent_parent / tangent_parent.norm_square() * tangent_parent;

    normal_rotation_left_child =
      normal_intersection_plane - normal_intersection_plane * tangent_left_child /
                                    tangent_left_child.norm_square() * tangent_left_child;

    normal_rotation_right_child =
      normal_intersection_plane - normal_intersection_plane * tangent_right_child /
                                    tangent_right_child.norm_square() * tangent_right_child;

#if DEBUG_LUNG_TRIANGULATION
    std::cout << "normal_rotation_parent : "
              << normal_rotation_parent / normal_rotation_parent.norm() << std::endl;
    std::cout << "normal_rotation_left_child : "
              << normal_rotation_left_child / normal_rotation_left_child.norm() << std::endl;
    std::cout << "normal_rotation_right_child : "
              << normal_rotation_right_child / normal_rotation_right_child.norm() << std::endl
              << std::endl;
#endif

    // calculate degrees between rotation normals and intersection normal

    degree_parent_intersection =
      Node::get_degree(normal_intersection_plane, normal_rotation_parent);
    degree_left_child_intersection =
      Node::get_degree(normal_intersection_plane, normal_rotation_left_child);
    degree_right_child_intersection =
      Node::get_degree(normal_intersection_plane, normal_rotation_right_child);

#if DEBUG_LUNG_TRIANGULATION
    std::cout << "degree_parent_intersection: " << degree_parent_intersection << std::endl;
    std::cout << "degree_left_child_intersection: " << degree_left_child_intersection << std::endl;
    std::cout << "degree_right_child_intersection: " << degree_right_child_intersection << std::endl
              << std::endl;
#endif

    // calculate direction of vector

    auto direction_parent_left_child =
      cross_product_3d(normal_rotation_parent, normal_rotation_left_child);
    auto direction_parent_right_child =
      cross_product_3d(normal_rotation_parent, normal_rotation_right_child);
    auto direction_left_child_right_child =
      cross_product_3d(normal_rotation_left_child, normal_rotation_right_child);

    if(direction_parent_left_child * tangent_parent > 0.0 &&
       direction_parent_left_child * tangent_left_child > 0)
      direction_parent_left_child =
        direction_parent_left_child / direction_parent_left_child.norm();
    else
      direction_parent_left_child =
        -direction_parent_left_child / direction_parent_left_child.norm();

    if(direction_parent_right_child * tangent_parent > 0.0 &&
       direction_parent_right_child * tangent_right_child > 0)
      direction_parent_right_child =
        direction_parent_right_child / direction_parent_right_child.norm();
    else
      direction_parent_right_child =
        -direction_parent_right_child / direction_parent_right_child.norm();

    if(direction_left_child_right_child * tangent_right_child > 0.0 &&
       direction_left_child_right_child * tangent_left_child > 0)
      direction_left_child_right_child =
        direction_left_child_right_child / direction_left_child_right_child.norm();
    else
      direction_left_child_right_child =
        -direction_left_child_right_child / direction_left_child_right_child.norm();

    if(!is_bifurcation_planar)
    {
      degree_parent_left_child  = Node::get_degree(tangent_parent, direction_parent_left_child);
      degree_parent_right_child = Node::get_degree(tangent_parent, direction_parent_right_child);
      degree_left_child_right_child =
        Node::get_degree(tangent_left_child, direction_left_child_right_child);
    }
    else
    {
      degree_parent_left_child  = Node::get_degree(tangent_parent, tangent_left_child) / 2.0;
      degree_parent_right_child = Node::get_degree(tangent_parent, tangent_right_child) / 2.0;
      degree_left_child_right_child =
        Node::get_degree(tangent_left_child, tangent_right_child) / 2.0;
    }

#if DEBUG_LUNG_TRIANGULATION
    std::cout << "degree_parent_left_child: " << degree_parent_left_child << std::endl;
    std::cout << "degree_parent_right_child: " << degree_parent_right_child << std::endl;
    std::cout << "degree_left_child_right_child: " << degree_left_child_right_child << std::endl
              << std::endl;
#endif

    // check if right_children is right and left_children is left
    auto normal_right = cross_product_3d(normal_rotation_parent, tangent_parent);
    auto normal_left  = -normal_right;

    bool left_child_is_left   = normal_left * tangent_left_child >= 0.0;
    bool right_child_is_right = normal_right * tangent_right_child >= 0.0;

#if DEBUG_LUNG_TRIANGULATION
    if(left_child_is_left)
      std::cout << "left child is on left side" << std::endl;
    else
      std::cout << "left child is on right side" << std::endl;

    if(right_child_is_right)
      std::cout << "right child is on right side" << std::endl << std::endl;
    else
      std::cout << "right child is on left side" << std::endl << std::endl;
#endif

    // switch parent children degrees if children tangents are mixed up
    if(!left_child_is_left && !right_child_is_right)
    {
      left_right_mixed_up = true;
    }
  }
  else if(node->has_child()) // left child filled
  {
    auto normal_parent_left  = -node->get_normal_vector_parent_left();

    // tangents of vectors
    auto tangent_parent      = -node->get_tangential_vector();
    auto tangent_left_child  = node->left_child->get_tangential_vector();

    normal_rotation_left_child = normal_parent_left;
    normal_rotation_parent = normal_parent_left;

    degree_parent_left_child  = Node::get_degree(tangent_parent, tangent_left_child) / 2.0;
    degree_parent_right_child = -degree_parent_left_child;
    degree_left_child_right_child = degree_parent_left_child;
  }

  dealii::Tensor<2, 3> transform_top;
  dealii::Tensor<2, 3> transform_bottom;

  // compute top rotation matrix

  if(!node->is_root())
  {
    auto dst_n_top = normal_rotation_child;

    dst_n_top /= dst_n_top.norm();
#if DEBUG_LUNG_TRIANGULATION
    std::cout << "normal vector top: " << dst_n_top << std::endl << std::endl;
#endif
    transform_top = compute_rotation_matrix(src_n, src_t, dst_n_top, dst_t);
  }

  // compute bottom rotation matrix

  if(node->has_children() or node->has_child())
  {
    auto dst_n_bottom = normal_rotation_parent;

    dst_n_bottom /= dst_n_bottom.norm();
#if DEBUG_LUNG_TRIANGULATION
    std::cout << "normal vector bottom: " << dst_n_bottom << std::endl << std::endl;
#endif
    transform_bottom = compute_rotation_matrix(src_n, src_t, dst_n_bottom, dst_t);
  }
  else
  {
    transform_bottom       = transform_top;
    normal_rotation_parent = normal_rotation_child;
  }

  if(node->is_root())
    normal_rotation_child = normal_rotation_parent;

  // calculate degree between rotation normals and choose rotation

  double degree_rotation_normals = Node::get_degree(normal_rotation_parent, normal_rotation_child);
#if DEBUG_LUNG_TRIANGULATION
  std::cout << "degree_rotation_normals: " << degree_rotation_normals << std::endl << std::endl;
#endif
  bool do_rotate_bottom = false;

  // switch between connection along x- or y-direction

  // case x-direction
  if(degree_rotation_normals > numbers::PI_4 && degree_rotation_normals < 3.0 * numbers::PI_4)
  {
    auto direction_rotation = cross_product_3d(normal_rotation_parent, normal_rotation_child);

    src_n[0] = 1;
    src_n[1] = 0;
    src_n[2] = 0;

    if(direction_rotation * node->get_tangential_vector() < 0)
    {
      normal_rotation_parent          = -normal_rotation_parent;
      normal_rotation_left_child      = -normal_rotation_left_child;
      normal_rotation_right_child     = -normal_rotation_right_child;
      degree_parent_intersection      = -degree_parent_intersection;
      degree_left_child_intersection  = -degree_left_child_intersection;
      degree_right_child_intersection = -degree_right_child_intersection;
      left_right_mixed_up             = !left_right_mixed_up;
    }

    auto dst_n_bottom = normal_rotation_parent;

    dst_n_bottom /= dst_n_bottom.norm();

    do_rotate_bottom = true;

    node->do_rot = true;

    transform_bottom = compute_rotation_matrix(src_n, src_t, dst_n_bottom, dst_t);
#if DEBUG_LUNG_TRIANGULATION
    std::cout << "transform matrices:  " << transform_bottom << std::endl
              << transform_top << std::endl;
#endif
  }
  else if(degree_rotation_normals > 3.0 * numbers::PI_4) // case y-direction
  {
    normal_rotation_parent      = -normal_rotation_parent;
    normal_rotation_left_child  = -normal_rotation_left_child;
    normal_rotation_right_child = -normal_rotation_right_child;
    auto dst_n_bottom           = normal_rotation_parent;

    dst_n_bottom /= dst_n_bottom.norm();

#if DEBUG_LUNG_TRIANGULATION
    std::cout << "normal vector bottom (adjusted): " << dst_n_bottom << std::endl << std::endl;
#endif

    transform_bottom                = compute_rotation_matrix(src_n, src_t, dst_n_bottom, dst_t);
    degree_parent_intersection      = -degree_parent_intersection;
    degree_left_child_intersection  = -degree_left_child_intersection;
    degree_right_child_intersection = -degree_right_child_intersection;
    left_right_mixed_up             = !left_right_mixed_up;
  }

  // define degrees
  double degree_1                   = degree_parent_left_child;
  double degree_2                   = degree_parent_right_child;
  double degree_separation_children = degree_left_child_right_child;

  // root does not have a top rotation matrix -> use bottom one
  if(node->is_root())
  {
    transform_top             = transform_bottom;
    degree_child_intersection = 0.0;
  }

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
                  transform_top,
                  transform_bottom,
                  source,
                  vertices_3d,
                  node->skeleton,
                  cell_data_3d,
                  degree_parent,
                  degree_1,
                  degree_2,
                  degree_parent_intersection,
                  degree_child_intersection,
                  degree_separation,
                  node->is_left(),
                  left_right_mixed_up,
                  left_right_mixed_up_parent,
                  do_rotate_bottom,
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
  unsigned int cou = os;
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

  // process children
  if(node->has_children())
  {
    // left child:
    try
    {
      process_node(node->get_left_child(),
                   cell_data_3d_global,
                   vertices_3d_global,
                   node->left_child->id,
                   os,
                   degree_1,
                   degree_separation_children,
                   degree_left_child_intersection,
                   normal_rotation_left_child,
                   left_right_mixed_up);
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
                   node->right_child->id,
                   os,
                   degree_2,
                   degree_separation_children,
                   degree_right_child_intersection,
                   normal_rotation_right_child,
                   left_right_mixed_up);
    }
    catch(const std::exception & e)
    {
      std::cout << e.what();
    }
  }
  else if(node->has_child())
  {
    // left child (only child):
    try
    {
      process_node(node->get_left_child(),
                   cell_data_3d_global,
                   vertices_3d_global,
                   node->left_child->id,
                   os,
                   degree_1,
                   -degree_1,
                   degree_left_child_intersection,
                   normal_rotation_left_child,
                   left_right_mixed_up);
    }
    catch(const std::exception & e)
    {
      std::cout << e.what();
    }
  }
}

} // namespace ExaDG

#endif
