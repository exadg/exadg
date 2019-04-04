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

#include "lung_tria.h"
#include "lung_util.h"
#include "process_file.h"
#include "triangulation_util.h"

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

      int n_bifurcations = generations;

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

    //        roots.push_back(new Node(
    //            new Node(roots_temp[1], roots_temp[0],Point<3>({47.4151e-3,  147.2595e-3,
    //            -201.9566e-3}),false), new Node(roots_temp[3], roots_temp[2],Point<3>({47.4151e-3,
    //            147.2595e-3, -201.9566e-3}),false),{8.826887618228566e-3, 157.61678106896196e-3,
    //            -187.4708043895141e-3},false)
    //        );
    ////
    //
    ////        roots.push_back(new Node(roots_temp[9], roots_temp[8],{-24.3016e-3,  156.6774e-3,
    ///-198.6689e-3},false)); /        roots.push_back(new Node( /                new
    /// Node(roots_temp[4], roots_temp[5],Point<3>({-33.9827e-3,  155.9265e-3,
    /// -208.6529e-3}),false), /                new Node(roots_temp[6],
    ///roots_temp[7],Point<3>({-33.9827e-3,  155.9265e-3, -208.6529e-3}),true), / {-24.3016e-3,
    ///156.6774e-3, -198.6689e-3}, true));
    //
    //        roots.push_back(new Node(
    //            new Node(
    //                new Node(roots_temp[4], roots_temp[5],Point<3>({-33.9827e-3,  155.9265e-3,
    //                -208.6529e-3}),false), new Node(roots_temp[6],
    //                roots_temp[7],Point<3>({-33.9827e-3,  155.9265e-3, -208.6529e-3}),true),
    //                {-24.3016e-3,  156.6774e-3, -198.6689e-3}, true),
    //            new Node(roots_temp[9], roots_temp[8],{-24.3016e-3,  156.6774e-3,
    //            -198.6689e-3},false), { 8.826887618228566e-3, 157.61678106896196e-3,
    //            -187.4708043895141e-3},false
    //        ));



    //        roots.push_back(new Node(
    //            new Node(
    //                new Node(
    //                    new Node(roots_temp[6], roots_temp[7],Point<3>({-33.9827e-3,  155.9265e-3,
    //                    -208.6529e-3}),true), new Node(roots_temp[4],
    //                    roots_temp[5],Point<3>({-33.9827e-3,  155.9265e-3, -208.6529e-3}),false),
    //                    {-24.3016e-3,  156.6774e-3, -198.6689e-3}, true),
    //                new Node(roots_temp[8], roots_temp[9],{-24.3016e-3,  156.6774e-3,
    //                -198.6689e-3},false), { 8.826887618228566e-3, 157.61678106896196e-3,
    //                -187.4708043895141e-3},false),
    //            new Node(
    //                new Node(roots_temp[3], roots_temp[2],Point<3>({47.4151e-3,  147.2595e-3,
    //                -201.9566e-3}),false), new Node(roots_temp[1],
    //                roots_temp[0],Point<3>({47.4151e-3,  147.2595e-3, -201.9566e-3}),false),
    //                {8.826887618228566e-3, 157.61678106896196e-3, -187.4708043895141e-3},false),
    //            { 8.864201963148962e-3, 200.1647444329594e-3,  -69.43970578881185e-3},true));

    roots.push_back(new Node(
      new Node(
        new Node(new Node(roots_temp[6],
                          roots_temp[7],
                          Point<3>({-33.9827e-3, 155.9265e-3, -208.6529e-3}),
                          true),
                 new Node(roots_temp[4],
                          roots_temp[5],
                          Point<3>({-33.9827e-3, 155.9265e-3, -208.6529e-3}),
                          false),
                 {-24.3016e-3, 156.6774e-3, -201.6689e-3},
                 true),
        new Node(roots_temp[8], roots_temp[9], {-24.3016e-3, 156.6774e-3, -201.6689e-3}, false),
        {8.826887618228566e-3, 157.61678106896196e-3, -187.4708043895141e-3},
        false),
      new Node(
        new Node(
          roots_temp[3], roots_temp[2], Point<3>({47.4151e-3, 147.2595e-3, -201.9566e-3}), false),
        new Node(
          roots_temp[1], roots_temp[0], Point<3>({47.4151e-3, 147.2595e-3, -201.9566e-3}), false),
        {8.826887618228566e-3, 157.61678106896196e-3, -187.4708043895141e-3},
        false),
      {8.864201963148962e-3, 200.1647444329594e-3, -69.43970578881185e-3},
      true));
  };
}

template<typename T>
bool
mark(T cell, const int number)
{
  // is not at boundary
  if(!cell->at_boundary(4))
    return false;

  // already visited
  if(cell->face(4)->boundary_id() > 0)
    return false;

  // set boundary id
  cell->face(4)->set_all_boundary_ids(number);

  // mark all neighbors
  for(unsigned int d = 0; d < 6; d++)
  {
    // face is at boundary: there is no neighbor to mark
    if(cell->at_boundary(d))
      continue;

    // mark neighbor
    mark(cell->neighbor(d), number);
  }

  // cell has been marked
  return true;
}

void lung(dealii::Triangulation<3> &                                     tria,
          int                                                            generations,
          int                                                            refinements,
          std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
          std::map<std::string, double> &                                timings,
          unsigned int const &                                           outlet_id_first,
          unsigned int &                                                 outlet_id_last)
{
  Timer timer;

  timer.restart();
  std::vector<Node *> roots;
  create_tree(roots, generations);

  timings["create_triangulation_1_load_data"] = timer.wall_time();

  parallel::fullydistributed::AdditionalData ad;
  ad.partition_group_size = 1;
  ad.partition_group      = parallel::fullydistributed::single;

  timer.restart();
  // ... by processing each tree
  std::vector<CellData<3>> cell_data_3d;
  std::vector<Point<3>>    vertices_3d;
  SubCellData              subcell_data;
  for(unsigned int i = 0; i < roots.size(); i++)
  {
    process_node(roots[i], cell_data_3d, vertices_3d, vertices_3d.size());
    // break;
  }

  timings["create_triangulation_2_mesh"] = timer.wall_time();

  // clean up
  for(unsigned int i = 0; i < roots.size(); i++)
    delete roots[i];

  timer.restart();
  // GridReordering<3>::reorder_cells(cell_data_3d, true);
  tria.create_triangulation(vertices_3d, cell_data_3d, subcell_data);
  timings["create_triangulation_4_serial_triangulation"] = timer.wall_time();

  // set boundary ids
  unsigned int counter = outlet_id_first; // counter for outlets
  for(auto cell : tria.active_cell_iterators())
  {
    // the mesh is generated in a way that inlet/outlets are one faces with normal vector
    // in positive or negative z-direction (faces 4/5)
    if(cell->at_boundary(5)) // inlet
      cell->face(5)->set_all_boundary_ids(1);
    if(cell->at_boundary(4)) // outlets (>1)
      if(mark(cell, counter))
        counter++;
  }

  // set outlet_id_last which is needed by the application setting the boundary conditions
  outlet_id_last = counter;

  timer.restart();
  tria.refine_global(refinements);
  timings["create_triangulation_5_serial_refinement"] = timer.wall_time();

  // TODO only print if desired
  bool print = false;
  if(print)
    print_tria_statistics(tria);
}

void lung(dealii::parallel::distributed::Triangulation<3> &              tria,
          int                                                            generations,
          int                                                            refinements,
          std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
          std::map<std::string, double> &                                timings,
          unsigned int const &                                           outlet_id_first,
          unsigned int &                                                 outlet_id_last)
{
  // create sequential coarse grid (no refinements)
  dealii::Triangulation<3> tria_seq;
  lung(tria_seq, generations, 0, create_tree, timings, outlet_id_first, outlet_id_last);
  // copy coarse grid to distributed triangulation and ...
  tria.copy_triangulation(tria_seq);
  // ... refine
  tria.refine_global(refinements);

  outlet_id_last = Utilities::MPI::max(outlet_id_last, MPI_COMM_WORLD);
}

void lung(dealii::parallel::fullydistributed::Triangulation<3> &         tria,
          int                                                            generations,
          int                                                            refinements1,
          int                                                            refinements2,
          std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
          std::map<std::string, double> &                                timings,
          unsigned int const &                                           outlet_id_first,
          unsigned int &                                                 outlet_id_last)
{
  Timer timer;
  timer.restart();

  // create partitioned triangulation ...
  tria.reinit(refinements2, [&](auto & tria) mutable {
    // ... by creating a refined sequential triangulation and partition it
    lung(tria, generations, refinements1, create_tree, timings, outlet_id_first, outlet_id_last);
  });

  outlet_id_last = Utilities::MPI::max(outlet_id_last, MPI_COMM_WORLD);

  timings["create_triangulation_0_overall"] = timer.wall_time();
}
} // namespace GridGenerator

} // namespace dealii

#endif
