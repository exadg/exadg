#ifndef LUNG_GRID
#define LUNG_GRID

// merge every second root such that they have a flat starting points
//#define USE_FLAT_ROOT

// create a dummy root for every second root such that there is only a single inlet
//#define DUMMY_ROOT

//#define DEBUG_INFO
#ifdef DEAL_II_WITH_METIS
#  include <metis.h>
#endif

#include <deal.II/grid/grid_reordering.h>
#include <vector>

#include "deform_via_splines.h"
#include "lung_tria.h"
#include "lung_util.h"
#include "process_file.h"
#include "triangulation_util.h"

#define USE_CHILD_GEOMETRY
//#define USE_ADULT_GEOMETRY
//#define USE_ADULT_GEOMETRY_OLD

namespace dealii
{
namespace GridGenerator
{
void
lung_to_node(int                               generations,
             std::vector<Point<3>> &           points,
             std::vector<CellData<1>> &        cells,
             std::vector<CellAdditionalInfo> & cells_additional_data,
             std::vector<Node *> &             roots)
{
  // create dual graph with Metis s.t. we know which element is
  // connected to which one
  int *xadj_vertex, *adjncy_vertex;
  create_dual_graph(points, cells, xadj_vertex, adjncy_vertex);

  // get minimum generation number -> nodes with this number are treated
  // as roots
  int min_generation = std::numeric_limits<int>::max();
  for(unsigned int i = 0; i < cells_additional_data.size(); i++)
    min_generation = std::min(cells_additional_data[i].generation, min_generation);

  // setup forest of trees
  for(unsigned int i = 0; i < cells_additional_data.size(); i++)
    if(cells_additional_data[i].generation == min_generation)
      roots.push_back(Node::create_root(
        i, xadj_vertex, adjncy_vertex, cells_additional_data, cells, points, generations));
}

std::function<void(std::vector<Node *> & roots, unsigned int)>
lung_files_to_node(std::vector<std::string> files)
{
  return [files](std::vector<Node *> & roots, unsigned int generations) {
    for(auto file : files)
    {
      // process files
      // get coordinates of points, connectivity of mesh, and info on
      // branches (generation, radius)
      std::vector<Point<3>>           points;
      std::vector<CellData<1>>        cells;
      std::vector<CellAdditionalInfo> cells_additional_data;
      load_files({file}, points, cells, cells_additional_data);

      int n_bifurcations = generations; /* TODO */

      if(file.find("leftbot") != std::string::npos || file.find("lefttop") != std::string::npos ||
         file.find("righttop") != std::string::npos)
      {
        n_bifurcations = generations - 4;
      }
      else if(file.find("rightbot") != std::string::npos ||
              file.find("rightmid") != std::string::npos)
      {
        n_bifurcations = generations - 5;
      }
      else
      {
        AssertThrow(false, ExcMessage("Filename specified for generation of lung mesh is wrong."));
      }

      lung_to_node(n_bifurcations, points, cells, cells_additional_data, roots);
    }

#ifdef DEBUG
    // check the read data
    for(auto & root : roots)
      Assert(root->check_if_planar(), ExcMessage("Bifurcation is not planar!"));
    std::cout << "Check:   All bifurcations are planar!" << std::endl;
#endif

#ifdef USE_FLAT_ROOT
    std::vector<Node *> roots_temp = roots;
    roots.clear();
    for(unsigned int i = 0; i < roots_temp.size(); i += 2)
      roots.push_back(new DummyNode(roots_temp[i + 1], roots_temp[i]));
#endif

#ifdef DUMMY_ROOT
    std::vector<Node *> roots_temp = roots;
    roots.clear();
    for(unsigned int i = 0; i < roots_temp.size(); i += 2)
    {
      Point<3> dst  = roots_temp[i]->from;
      Point<3> norm = (roots_temp[i]->to + roots_temp[i + 1]->to);
      norm          = norm / 2;
      norm          = Point<3>(norm - dst);
      Point<3> src  = Point<3>(dst - norm);
      roots.push_back(new Node(roots_temp[i + 1], roots_temp[i], src, true));
    }
#endif

    if(roots.size() != 10)
      return;

    std::vector<Node *> roots_temp = roots;
    roots.clear();

#ifdef USE_CHILD_GEOMETRY

    // clang-format off
    {
    auto temp                            = roots_temp[4]->right_child;
    roots_temp[4]->right_child           = roots_temp[4]->left_child;
    roots_temp[4]->left_child            = temp;
    roots_temp[4]->left_child->_is_left  = true;
    roots_temp[4]->right_child->_is_left = false;
    }
    
    {
    auto temp                            = roots_temp[5]->right_child;
    roots_temp[5]->right_child           = roots_temp[5]->left_child;
    roots_temp[5]->left_child            = temp;
    roots_temp[5]->left_child->_is_left  = true;
    roots_temp[5]->right_child->_is_left = false;
    }
    
    //roots[0]->left_child->right_child->right_child->left_child

    roots.push_back(new Node(
      new Node(
        new Node(roots_temp[9], roots_temp[8], {-0.012978481772800358, 0.03523408779189564, 0.007048238472570871}, true,false),
        new Node(
          new Node(roots_temp[6], roots_temp[7], Point<3>({-0.01820662231978939, 0.03544419220009717, -0.001789115751719632}), false),
          new Node(roots_temp[5], roots_temp[4], Point<3>({-0.01820662231978939, 0.03544419220009717, -0.001789115751719632}), false,false, true),
          {-0.012978481772800358, 0.03523408779189564, 0.007048238472570871}, true),
        {-0.00876064681897349, 0.03269315981334528, 0.010104936601351125}, false, false),
      new Node(
        new Node(roots_temp[2], roots_temp[3], Point<3>({0.004594151748164817, 0.03753150327901492, 0.0011504181300876433}), false, false),
        new Node(roots_temp[1], roots_temp[0], Point<3>({0.004594151748164817, 0.03753150327901492, 0.0011504181300876433}), false),
        {-0.00876064681897349, 0.03269315981334528, 0.010104936601351125}, false,false),
      {-0.004691444584156128, 0.012111498522501644, 0.040161793504990745}, true));
    // clang-format on

#endif

#ifdef USE_ADULT_GEOMETRY_OLD
    // with twist in generation 1

    // clang-format off
    roots.push_back(new Node(
      new Node(
        new Node(new Node(roots_temp[6], roots_temp[7], Point<3>({-33.9827e-3, 155.9265e-3, -208.6529e-3}), true),
                 new Node(roots_temp[4], roots_temp[5], Point<3>({-33.9827e-3, 155.9265e-3, -208.6529e-3}), false),
                 {-24.3016e-3, 156.6774e-3, -201.6689e-3}, true),
        new Node(roots_temp[8], roots_temp[9], {-24.3016e-3, 156.6774e-3, -201.6689e-3}, false),
        {8.826887618228566e-3, 157.61678106896196e-3, -187.4708043895141e-3},false),
      new Node(
        new Node(roots_temp[3], roots_temp[2], Point<3>({47.4151e-3, 147.2595e-3, -201.9566e-3}), false),
        new Node(roots_temp[1], roots_temp[0], Point<3>({47.4151e-3, 147.2595e-3, -201.9566e-3}), false),
        {8.826887618228566e-3, 157.61678106896196e-3, -187.4708043895141e-3}, false),
      {8.864201963148962e-3, 200.1647444329594e-3, -69.43970578881185e-3},
      true));
    // clang-format on
#endif

#ifdef USE_ADULT_GEOMETRY
    // without twist in generation 1

    // clang-format off
    roots.push_back(new Node(
      new Node(
        new Node(roots_temp[8], roots_temp[9], {-24.3016e-3, 156.6774e-3, -201.6689e-3}, true, false),
        new Node(new Node(roots_temp[6], roots_temp[7], Point<3>({-33.9827e-3, 155.9265e-3, -208.6529e-3}), false),
                 new Node(roots_temp[4], roots_temp[5], Point<3>({-33.9827e-3, 155.9265e-3, -208.6529e-3}), false,false),
                 {-24.3016e-3, 156.6774e-3, -201.6689e-3}, true),
        {8.826887618228566e-3, 157.61678106896196e-3, -187.4708043895141e-3},false,false),
      new Node(
        new Node(roots_temp[3], roots_temp[2], Point<3>({47.4151e-3, 147.2595e-3, -201.9566e-3}), false),
        new Node(roots_temp[1], roots_temp[0], Point<3>({47.4151e-3, 147.2595e-3, -201.9566e-3}), false),
        {8.826887618228566e-3, 157.61678106896196e-3, -187.4708043895141e-3}, false),
      {8.864201963148962e-3, 200.1647444329594e-3, -69.43970578881185e-3},
      true));
    // clang-format on
#endif
  };
}

template<typename T>
std::pair<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>>
face_vertices(T face)
{

  std::pair<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>> result;

  std::vector<unsigned int> v(4);

  for(unsigned int d = 0; d < 4; d++)
    v[d] = face->vertex_index(d);

  std::sort(v.begin(), v.end());

  result.first.first   = v[0];
  result.first.second  = v[1];
  result.second.first  = v[2];
  result.second.second = v[3];

  return result;
}

template<typename T, typename Map>
bool
mark(T cell, const int number, Map & map)
{
  // is not at boundary
  if(!cell->at_boundary(4))
    return false;

  // already visited
  if(cell->face(4)->boundary_id() > 0)
    return false;

  // set boundary id
  cell->face(4)->set_all_boundary_ids(number);
  map[face_vertices(cell->face(4))] = number;

  // mark all neighbors
  for(unsigned int d = 0; d < 6; d++)
  {
    // face is at boundary: there is no neighbor to mark
    if(cell->at_boundary(d))
      continue;

    // mark neighbor
    mark(cell->neighbor(d), number, map);
  }

  // cell has been marked
  return true;
}

void lung_unrefined(dealii::Triangulation<3> &                                     tria,
                    std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
                    std::map<std::string, double> &                                timings,
                    unsigned int const &                                           outlet_id_first,
                    unsigned int &                                                 outlet_id_last,
                    const std::string &                                            bspline_file,
                    std::map<types::material_id, DeformTransfinitelyViaSplines<3>> & deform,
                    std::shared_ptr<LungID::Checker>                                 branch_filter)
{
  Timer timer;

  timer.restart();
  std::vector<Node *> roots;
  create_tree(roots, branch_filter->get_generations());

  timings["create_triangulation_1_load_data"] = timer.wall_time();

  timer.restart();
  // ... by processing each tree
  std::vector<CellData<3>> cell_data_3d;
  std::vector<Point<3>>    vertices_3d;
  SubCellData              subcell_data;
  for(unsigned int i = 0; i < roots.size(); i++)
  {
    process_node(roots[i], cell_data_3d, vertices_3d, branch_filter, vertices_3d.size());
    // break;
  }

#ifdef DEBUG
  for(unsigned int i = 0; i < roots.size(); i++)
    for(auto v : roots[i]->skeleton)
      printf("%+10.6f, %+10.6f, %+10.6f\n", v[0], v[1], v[2]);
  printf("\n");

  for(unsigned int i = 0; i < roots.size(); i++)
    for(auto v : roots[i]->right_child->skeleton)
      printf("%+10.6f, %+10.6f, %+10.6f\n", v[0], v[1], v[2]);
  printf("\n");

  for(unsigned int i = 0; i < roots.size(); i++)
    for(auto v : roots[i]->left_child->skeleton)
      printf("%+10.6f, %+10.6f, %+10.6f\n", v[0], v[1], v[2]);
  printf("\n");
#endif

  timings["create_triangulation_2_mesh"] = timer.wall_time();

  std::vector<BSpline2D<3, 3>> splines;
  {
    std::ifstream file(bspline_file.c_str());
    AssertThrow(file.good(), ExcMessage("BSpline does not exist!"));

    unsigned int n_splines;
    file.read(reinterpret_cast<char *>(&n_splines), sizeof(unsigned int));
    splines.resize(n_splines);
    for(unsigned int s = 0; s < n_splines; ++s)
      splines[s].read_from_file(file);
  }


  timer.restart();

  // collect faces and their ids for the non-reordered triangulation
  std::map<std::pair<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>>, unsigned int> map;

  {
    dealii::Triangulation<3> tria;
    tria.create_triangulation(vertices_3d, cell_data_3d, subcell_data);

    // set boundary ids
    unsigned int counter = outlet_id_first; // counter for outlets
    for(auto cell : tria.active_cell_iterators())
    {
    // the mesh is generated in a way that inlet/outlets are one faces with normal vector
    // in positive or negative z-direction (faces 4/5)
      if(cell->at_boundary(5) && cell->material_id() == (unsigned int) LungID::create_root()) // inlet
        map[face_vertices(cell->face(5))] = 1;

      if(cell->at_boundary(4)) // outlets (>1)
        if(mark(cell, counter, map))
          counter++;
    }
    // set outlet_id_last which is needed by the application setting the boundary conditions
    outlet_id_last = counter;
  }

  GridReordering<3>::reorder_cells(cell_data_3d, true);
  tria.create_triangulation(vertices_3d, cell_data_3d, subcell_data);



  // actually set boundary ids
  for(auto cell : tria.active_cell_iterators())
    for(unsigned int d = 0; d < 6; d++)
      if(cell->at_boundary(d) && map.find(face_vertices(cell->face(d))) != map.end())
        cell->face(d)->set_all_boundary_ids(map[face_vertices(cell->face(d))]);

  timings["create_triangulation_4_serial_triangulation"] = timer.wall_time();

  if(roots[0]->skeleton.size() > 0)
  {
    deform.insert({(unsigned int)LungID::create_root(),
                   DeformTransfinitelyViaSplines<3>(splines, 0, roots[0]->skeleton, {0, 0})});
  }



  if(roots[0]->right_child->skeleton.size() > 0)
  {
    deform.insert(
      {(unsigned int)LungID::generate(LungID::create_root(), false),
       (DeformTransfinitelyViaSplines<3>(splines, 4, roots[0]->right_child->skeleton, {0, 0}))});
  }

  if(roots[0]->left_child->skeleton.size() > 0)
  {
    deform.insert(
      {(unsigned int)LungID::generate(LungID::create_root(), true),
       (DeformTransfinitelyViaSplines<3>(splines, 8, roots[0]->left_child->skeleton, {0, 0}))});
  }



  if(roots[0]->right_child->right_child->skeleton.size() > 0)
  {
    auto & temp   = roots[0]->right_child->right_child->skeleton;
    auto   temp_c = temp;
    temp_c[3]     = temp[0];
    temp_c[2]     = temp[1];
    temp_c[1]     = temp[2];
    temp_c[0]     = temp[3];
    temp_c[7]     = temp[4];
    temp_c[6]     = temp[5];
    temp_c[5]     = temp[6];
    temp_c[4]     = temp[7];
    temp          = temp_c;
    deform.insert(
      {(unsigned int)LungID::generate(LungID::generate(LungID::create_root(), false), false),
       (DeformTransfinitelyViaSplines<3>(
         splines, 16, roots[0]->right_child->right_child->skeleton, {0, 1}))});
  }

  if(roots[0]->right_child->left_child->skeleton.size() > 0)
  {
    auto & temp   = roots[0]->right_child->left_child->skeleton;
    auto   temp_c = temp;
    temp_c[3]     = temp[0];
    temp_c[2]     = temp[1];
    temp_c[1]     = temp[2];
    temp_c[0]     = temp[3];
    temp_c[7]     = temp[4];
    temp_c[6]     = temp[5];
    temp_c[5]     = temp[6];
    temp_c[4]     = temp[7];
    temp          = temp_c;
    deform.insert(
      {(unsigned int)LungID::generate(LungID::generate(LungID::create_root(), false), true),
       (DeformTransfinitelyViaSplines<3>(
         splines, 12, roots[0]->right_child->left_child->skeleton, {0, 0}))});
  }

  if(roots[0]->left_child->right_child->skeleton.size() > 0)
  {
    deform.insert(
      {(unsigned int)LungID::generate(LungID::generate(LungID::create_root(), true), false),
       (DeformTransfinitelyViaSplines<3>(
         splines, 20, roots[0]->left_child->right_child->skeleton, {0, 1}))});
  }

  if(roots[0]->left_child->left_child->skeleton.size() > 0)
  {
    deform.insert(
      {(unsigned int)LungID::generate(LungID::generate(LungID::create_root(), true), true),
       (DeformTransfinitelyViaSplines<3>(
         splines, 24, roots[0]->left_child->left_child->skeleton, {0, 0}))});
  }



  if(roots[0]->right_child->right_child->right_child->skeleton.size() > 0)
  {
    auto & temp   = roots[0]->right_child->right_child->right_child->skeleton;
    auto   temp_c = temp;
    temp_c[3]     = temp[0];
    temp_c[2]     = temp[1];
    temp_c[1]     = temp[2];
    temp_c[0]     = temp[3];
    temp_c[7]     = temp[4];
    temp_c[6]     = temp[5];
    temp_c[5]     = temp[6];
    temp_c[4]     = temp[7];
    temp          = temp_c;
    deform.insert(
      {(unsigned int)LungID::generate(
         LungID::generate(LungID::generate(LungID::create_root(), false), false), false),
       (DeformTransfinitelyViaSplines<3>(
         splines, 40, roots[0]->right_child->right_child->right_child->skeleton, {0, 1}, true))});
  }

  if(roots[0]->right_child->right_child->left_child->skeleton.size() > 0)
  {
    auto & temp   = roots[0]->right_child->right_child->left_child->skeleton;
    auto   temp_c = temp;
    temp_c[3]     = temp[0];
    temp_c[2]     = temp[1];
    temp_c[1]     = temp[2];
    temp_c[0]     = temp[3];
    temp_c[7]     = temp[4];
    temp_c[6]     = temp[5];
    temp_c[5]     = temp[6];
    temp_c[4]     = temp[7];
    temp          = temp_c;
    deform.insert(
      {(unsigned int)LungID::generate(
         LungID::generate(LungID::generate(LungID::create_root(), false), false), true),
       (DeformTransfinitelyViaSplines<3>(
         splines, 36, roots[0]->right_child->right_child->left_child->skeleton, {0, 1}, true))});
  }

  if(roots[0]->right_child->left_child->left_child->skeleton.size() > 0)
  {
    auto & temp   = roots[0]->right_child->left_child->left_child->skeleton;
    auto   temp_c = temp;


    temp_c[3] = temp[0];
    temp_c[2] = temp[1];
    temp_c[1] = temp[2];
    temp_c[0] = temp[3];

    temp_c[7] = temp[4];
    temp_c[6] = temp[5];
    temp_c[5] = temp[6];
    temp_c[4] = temp[7];

    temp = temp_c;
    deform.insert(
      {(unsigned int)LungID::generate(
         LungID::generate(LungID::generate(LungID::create_root(), false), true), true),
       (DeformTransfinitelyViaSplines<3>(
         splines, 28, roots[0]->right_child->left_child->left_child->skeleton, {0, 1}, true))});
  }

  if(roots[0]->right_child->left_child->right_child->skeleton.size() > 0)
  {
    auto & temp   = roots[0]->right_child->left_child->right_child->skeleton;
    auto   temp_c = temp;

    temp_c[3] = temp[0];
    temp_c[2] = temp[1];
    temp_c[1] = temp[2];
    temp_c[0] = temp[3];

    temp_c[7] = temp[4];
    temp_c[6] = temp[5];
    temp_c[5] = temp[6];
    temp_c[4] = temp[7];
    temp      = temp_c;
    deform.insert(
      {(unsigned int)LungID::generate(
         LungID::generate(LungID::generate(LungID::create_root(), false), true), false),
       (DeformTransfinitelyViaSplines<3>(
         splines, 32, roots[0]->right_child->left_child->right_child->skeleton, {0, 1}, true))});
  }



  if(roots[0]->left_child->right_child->right_child->skeleton.size() > 0)
  {
    deform.insert(
      {(unsigned int)LungID::generate(
         LungID::generate(LungID::generate(LungID::create_root(), true), false), false),
       (DeformTransfinitelyViaSplines<3>(
         splines, 44, roots[0]->left_child->right_child->right_child->skeleton, {0, 1}, false))});
  }

  if(roots[0]->left_child->right_child->left_child->skeleton.size() > 0)
  {
    deform.insert(
      {(unsigned int)LungID::generate(
         LungID::generate(LungID::generate(LungID::create_root(), true), false), true),
       (DeformTransfinitelyViaSplines<3>(
         splines, 48, roots[0]->left_child->right_child->left_child->skeleton, {0, 1}, true))});
  }



  if(roots[0]->left_child->right_child->right_child->right_child->skeleton.size() > 0)
  {
    auto & temp   = roots[0]->left_child->right_child->right_child->right_child->skeleton;
    auto   temp_c = temp;

    temp_c[2] = temp[0];
    temp_c[0] = temp[1];
    temp_c[3] = temp[2];
    temp_c[1] = temp[3];

    temp_c[6] = temp[4];
    temp_c[4] = temp[5];
    temp_c[7] = temp[6];
    temp_c[5] = temp[7];


    temp = temp_c;
    deform.insert(
      {(unsigned int)LungID::generate(
         LungID::generate(LungID::generate(LungID::generate(LungID::create_root(), true), false),
                          false),
         false),
       (DeformTransfinitelyViaSplines<3>(
         splines,
         56,
         roots[0]->left_child->right_child->right_child->right_child->skeleton,
         {0, 0},
         true))});
  }

  if(roots[0]->left_child->right_child->right_child->left_child->skeleton.size() > 0)
  {
    auto & temp   = roots[0]->left_child->right_child->right_child->left_child->skeleton;
    auto   temp_c = temp;

    temp_c[1] = temp[0];
    temp_c[3] = temp[1];
    temp_c[0] = temp[2];
    temp_c[2] = temp[3];

    temp_c[5] = temp[4];
    temp_c[7] = temp[5];
    temp_c[4] = temp[6];
    temp_c[6] = temp[7];


    temp = temp_c;
    deform.insert(
      {(unsigned int)LungID::generate(
         LungID::generate(LungID::generate(LungID::generate(LungID::create_root(), true), false),
                          false),
         true),
       (DeformTransfinitelyViaSplines<3>(
         splines,
         52,
         roots[0]->left_child->right_child->right_child->left_child->skeleton,
         {0, 0},
         true))});
  }

  if(roots[0]->left_child->left_child->right_child->skeleton.size() > 0)
  {
    deform.insert(
      {(unsigned int)LungID::generate(
         LungID::generate(LungID::generate(LungID::create_root(), true), true), false),
       (DeformTransfinitelyViaSplines<3>(
         splines, 68, roots[0]->left_child->left_child->right_child->skeleton, {0, 1}, true))});
  }

  if(roots[0]->left_child->left_child->left_child->skeleton.size() > 0)
  {
    deform.insert(
      {(unsigned int)LungID::generate(
         LungID::generate(LungID::generate(LungID::create_root(), true), true), true),
       (DeformTransfinitelyViaSplines<3>(
         splines, 72, roots[0]->left_child->left_child->left_child->skeleton, {0, 1}, true))});
  }

  // clean up
  for(unsigned int i = 0; i < roots.size(); i++)
    delete roots[i];
}

void update_mapping(dealii::Triangulation<3> &                                       tria,
                    std::map<types::material_id, DeformTransfinitelyViaSplines<3>> & deform)
{
  if(deform.size() == 0)
    return;

#ifdef DEBUG
  std::cout << deform.size() << std::endl;
#endif

  //  //std::vector<Point<3>> & tria_points =
  //  const_cast<std::vector<Point<3>>&>(tria.get_vertices());
  //  //for (Point<3> &p : tria_points)
  //  //  p = deform.transform_to_deformed(p);
  //  std::map<types::material_id,unsigned int> map_to_splines;
  //  map_to_splines[LungID::create_root()] = 0;
  //  map_to_splines[LungID::generate(LungID::create_root(), false)] = 1;
  //  map_to_splines[LungID::generate(LungID::create_root(), true)] = 2;
  //  map_to_splines[LungID::generate(LungID::generate(LungID::create_root(), false),false)] = 3;
  //  map_to_splines[LungID::generate(LungID::generate(LungID::create_root(), false),true)] = 4;
  //  map_to_splines[LungID::generate(LungID::generate(LungID::create_root(), true),false)] = 5;
  //  map_to_splines[LungID::generate(LungID::generate(LungID::create_root(), true),true)] = 6;
  //
  //  map_to_splines[LungID::generate(LungID::generate(LungID::generate(LungID::create_root(),
  //  false), false), false)] = 7;
  //  map_to_splines[LungID::generate(LungID::generate(LungID::generate(LungID::create_root(),
  //  false), false), true)] = 8;
  //  //map_to_splines[LungID::generate(LungID::generate(LungID::generate(LungID::create_root(),
  //  false), true), true)] = 9;
  //  //map_to_splines[LungID::generate(LungID::generate(LungID::generate(LungID::create_root(),
  //  false), true), false)]  = 10;
  //
  //  map_to_splines[LungID::generate(LungID::generate(LungID::generate(LungID::create_root(),
  //  true), false), false)] = 11;
  //  map_to_splines[LungID::generate(LungID::generate(LungID::generate(LungID::create_root(),
  //  true), false), true)] = 12;
  //
  //  map_to_splines[LungID::generate(LungID::generate(LungID::generate(LungID::generate(LungID::create_root(),
  //  true), false), false), false)] = 13;
  //  map_to_splines[LungID::generate(LungID::generate(LungID::generate(LungID::generate(LungID::create_root(),
  //  true), false), false), true)] = 14;
  //
  //  map_to_splines[LungID::generate(LungID::generate(LungID::generate(LungID::create_root(),
  //  true),true),false)] = 15;
  //  map_to_splines[LungID::generate(LungID::generate(LungID::generate(LungID::create_root(),
  //  true),true),true)] = 16;

  GridOut       grid_out;
  std::ofstream file("mesh-b.vtu");
  grid_out.write_vtu(tria, file);

  std::vector<bool> touched(tria.n_vertices(), false);
  for(auto cell : tria.active_cell_iterators())
    if(deform.find(cell->material_id()) != deform.end())
      for(unsigned int v = 0; v < GeometryInfo<3>::vertices_per_cell; ++v)
        if(touched[cell->vertex_index(v)] == false)
        {
          // std::cout << cell->material_id() << " " << LungID::to_string(cell->material_id()) <<
          // std::endl;
          try
          {
            cell->vertex(v) = deform[cell->material_id()].transform_to_deformed(cell->vertex(v));
          }
          catch(typename Mapping<3, 3>::ExcTransformationFailed & exc)
          {
            std::cout << "Failed for material id " << LungID::to_string(cell->material_id())
                      << std::endl;
            // throw exc;
          }
          touched[cell->vertex_index(v)] = true;
        }


  // TODO only print if desired
  bool print = false;
  if(print)
    print_tria_statistics(tria);
}

void lung(dealii::Triangulation<2> &                                     tria,
          int                                                            refinements,
          std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
          std::map<std::string, double> &                                timings,
          unsigned int const &                                           outlet_id_first,
          unsigned int &                                                 outlet_id_last,
          const std::string &                                            bspline_file = "",
          std::shared_ptr<LungID::Checker>                               branch_filter =
            std::shared_ptr<LungID::Checker>(new LungID::NoneChecker()))
{
  (void)tria;
  (void)refinements;
  (void)create_tree;
  (void)timings;
  (void)outlet_id_first;
  (void)outlet_id_last;
  (void)bspline_file;
  (void)branch_filter;

  AssertThrow(false, ExcMessage("Not implemented for dim = 2."));
}

void lung(dealii::Triangulation<3> &                                     tria,
          int                                                            refinements,
          std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
          std::map<std::string, double> &                                timings,
          unsigned int const &                                           outlet_id_first,
          unsigned int &                                                 outlet_id_last,
          const std::string &                                            bspline_file = "",
          std::shared_ptr<LungID::Checker>                               branch_filter =
            std::shared_ptr<LungID::Checker>(new LungID::NoneChecker()))
{
  std::map<types::material_id, DeformTransfinitelyViaSplines<3>> deform;
  lung_unrefined(tria,
                 create_tree,
                 timings,
                 outlet_id_first,
                 outlet_id_last,
                 bspline_file,
                 deform,
                 branch_filter);
  tria.refine_global(refinements);
  update_mapping(tria, deform);
}

void lung(dealii::parallel::distributed::Triangulation<2> &              tria,
          int                                                            refinements,
          std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
          std::map<std::string, double> &                                timings,
          unsigned int const &                                           outlet_id_first,
          unsigned int &                                                 outlet_id_last,
          const std::string &                                            bspline_file = "",
          std::shared_ptr<LungID::Checker>                               branch_filter =
            std::shared_ptr<LungID::Checker>(new LungID::NoneChecker()))
{
  (void)tria;
  (void)refinements;
  (void)create_tree;
  (void)timings;
  (void)outlet_id_first;
  (void)outlet_id_last;
  (void)bspline_file;
  (void)branch_filter;

  AssertThrow(false, ExcMessage("Not implemented for dim = 2."));
}

void lung(dealii::parallel::distributed::Triangulation<3> &              tria,
          int                                                            refinements,
          std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
          std::map<std::string, double> &                                timings,
          unsigned int const &                                           outlet_id_first,
          unsigned int &                                                 outlet_id_last,
          const std::string &                                            bspline_file = "",
          std::shared_ptr<LungID::Checker>                               branch_filter =
            std::shared_ptr<LungID::Checker>(new LungID::NoneChecker()))
{
  // create sequential coarse grid (no refinements)
  dealii::Triangulation<3>                                       tria_seq;
  std::map<types::material_id, DeformTransfinitelyViaSplines<3>> deform;
  lung_unrefined(tria_seq,
                 create_tree,
                 timings,
                 outlet_id_first,
                 outlet_id_last,
                 bspline_file,
                 deform,
                 branch_filter);
  // copy coarse grid to distributed triangulation and ...
  tria.copy_triangulation(tria_seq);
  // ... refine
  tria.refine_global(refinements);
  update_mapping(tria, deform);

  outlet_id_last = Utilities::MPI::max(outlet_id_last, MPI_COMM_WORLD);
}

void lung(dealii::parallel::fullydistributed::Triangulation<2> &         tria,
          int                                                            refinements1,
          int                                                            refinements2,
          std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
          std::map<std::string, double> &                                timings,
          unsigned int const &                                           outlet_id_first,
          unsigned int &                                                 outlet_id_last,
          const std::string &                                            bspline_file = "",
          std::shared_ptr<LungID::Checker>                               branch_filter =
            std::shared_ptr<LungID::Checker>(new LungID::NoneChecker()))
{
  (void)tria;
  (void)refinements1;
  (void)refinements2;
  (void)create_tree;
  (void)timings;
  (void)outlet_id_first;
  (void)outlet_id_last;
  (void)bspline_file;
  (void)branch_filter;

  AssertThrow(false, ExcMessage("Not implemented for dim = 2."));
}

void lung(dealii::parallel::fullydistributed::Triangulation<3> &         tria,
          int                                                            refinements1,
          int                                                            refinements2,
          std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
          std::map<std::string, double> &                                timings,
          unsigned int const &                                           outlet_id_first,
          unsigned int &                                                 outlet_id_last,
          const std::string &                                            bspline_file = "",
          std::shared_ptr<LungID::Checker>                               branch_filter =
            std::shared_ptr<LungID::Checker>(new LungID::NoneChecker()))
{
  Timer timer;
  timer.restart();

  parallel::fullydistributed::AdditionalData ad;
  ad.partition_group_size = 1;
  ad.partition_group      = parallel::fullydistributed::single;

  std::map<types::material_id, DeformTransfinitelyViaSplines<3>> deform;
  // create partitioned triangulation ...
  tria.reinit(refinements2,
              [&](auto & tria) mutable {
                // ... by creating a refined sequential triangulation and partition it
                lung_unrefined(tria,
                               create_tree,
                               timings,
                               outlet_id_first,
                               outlet_id_last,
                               bspline_file,
                               deform,
                               branch_filter);
                tria.refine_global(refinements1);
                // update_mapping(tria, deform);
              },
              ad);
  update_mapping(tria, deform);

  outlet_id_last = Utilities::MPI::max(outlet_id_last, MPI_COMM_WORLD);

  timings["create_triangulation_0_overall"] = timer.wall_time();
}
} // namespace GridGenerator

} // namespace dealii

#endif
