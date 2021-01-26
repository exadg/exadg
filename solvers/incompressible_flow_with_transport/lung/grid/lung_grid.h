#ifndef LUNG_GRID
#define LUNG_GRID

#ifdef DEAL_II_WITH_METIS
#  include <metis.h>
#endif

// C/C++
#include <vector>

// deal.II
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q_cache.h>
#include <deal.II/grid/grid_reordering.h>

// ExaDG
#include "deform_via_splines.h"
#include "lung_tria.h"
#include "lung_util.h"
#include "process_file.h"
#include "triangulation_util.h"

namespace ExaDG::GridGen
{
using namespace dealii;

// TODO: adjust local refinement
const int MAX_REFINED_GENERATION = -1;

void
lung_to_node(unsigned int                      generations,
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
  for(auto & i : cells_additional_data)
    min_generation = std::min(i.generation, min_generation);

  // setup forest of trees
  for(unsigned int i = 0; i < cells_additional_data.size(); i++)
    if(cells_additional_data[i].generation == min_generation)
      roots.push_back(Node::create_root(
        i, xadj_vertex, adjncy_vertex, cells_additional_data, cells, points, generations));
}

std::function<void(std::vector<Node *> & roots, unsigned int)>
lung_files_to_node(const std::vector<std::string> & files)
{
  return [files](std::vector<Node *> & roots, unsigned int generations) {
    for(const auto & file : files)
    {
      // process files
      // get coordinates of points, connectivity of mesh, and info on
      // branches (generation, radius)

      unsigned int number_of_points = 1;

      for(unsigned int i = 0; i <= generations; i++)
        number_of_points += (unsigned int)std::pow(2, i);

      std::vector<Point<3>>           points(number_of_points);
      std::vector<CellData<1>>        cells;
      std::vector<CellAdditionalInfo> cells_additional_data;

      // TODO: remove old file structure when baby is compatible with new one
      // load_files({file}, points, cells, cells_additional_data);

      load_new_files({file}, points, cells, cells_additional_data, generations);

      unsigned int n_bifurcations = generations;

      lung_to_node(n_bifurcations, points, cells, cells_additional_data, roots);
    }
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
mark(T cell, const unsigned int number, Map & map)
{
  // is not at boundary
  if(!cell->at_boundary(5))
    return false;

  // already visited
  if(cell->face(5)->boundary_id() > 0)
    return false;

  // set boundary id
  cell->face(5)->set_all_boundary_ids(number);
  map[face_vertices(cell->face(5))] = number;

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

void
fill_deform(std::map<types::material_id, DeformTransfinitelyViaCylinder<3>> & deform,
            Node *                                                            node_to_deform)
{
  const unsigned int index_in = node_to_deform->do_rot ? 1 : 0;

  deform.insert({node_to_deform->id,
                 DeformTransfinitelyViaCylinder<3>(node_to_deform->skeleton, {{index_in, 0}})});

  if(node_to_deform->left_child != nullptr)
    fill_deform(deform, node_to_deform->left_child);

  if(node_to_deform->right_child != nullptr)
    fill_deform(deform, node_to_deform->right_child);
}

void
lung_unrefined(dealii::Triangulation<3> &                                             tria,
               const std::function<void(std::vector<Node *> & roots, unsigned int)> & create_tree,
               std::map<std::string, double> &                                        timings,
               unsigned int const &                                              outlet_id_first,
               unsigned int &                                                    outlet_id_last,
               const std::string &                                               bspline_file,
               std::map<types::material_id, DeformTransfinitelyViaCylinder<3>> & deform,
               const unsigned int max_resolved_generation)
{
  Timer timer;

  timer.restart();
  std::vector<Node *> roots;
  create_tree(roots, max_resolved_generation);

  timings["create_triangulation_1_load_data"] = timer.wall_time();

  timer.restart();
  // ... by processing each tree
  std::vector<CellData<3>> cell_data_3d;
  std::vector<Point<3>>    vertices_3d;
  SubCellData              subcell_data;
  for(auto & root : roots)
  {
    process_node(root, cell_data_3d, vertices_3d, root->id, vertices_3d.size());
    // break;
  }

#ifdef DEBUG
  for(auto & root : roots)
    for(auto v : root->skeleton)
      printf("%+10.6f, %+10.6f, %+10.6f\n", v[0], v[1], v[2]);
  printf("\n");

  for(auto & root : roots)
    for(auto v : root->right_child->skeleton)
      printf("%+10.6f, %+10.6f, %+10.6f\n", v[0], v[1], v[2]);
  printf("\n");

  for(auto & root : roots)
    for(auto v : root->left_child->skeleton)
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
  std::map<std::pair<std::pair<unsigned int, unsigned int>, std::pair<unsigned int, unsigned int>>,
           unsigned int>
    map;

  {
    dealii::Triangulation<3> tria_tmp(Triangulation<3>::MeshSmoothing::none, true);
    tria_tmp.create_triangulation(vertices_3d, cell_data_3d, subcell_data);

    // set boundary ids
    unsigned int counter = outlet_id_first; // counter for outlets
    for(auto cell : tria_tmp.active_cell_iterators())
    {
      // the mesh is generated in a way that inlet/outlets are one faces with normal vector
      // in positive or negative z-direction (faces 4/5)
      if(cell->at_boundary(4) && cell->material_id() == roots[0]->id) // inlet
        map[face_vertices(cell->face(4))] = 1;

      if(cell->at_boundary(5)) // outlets (>1)
        if(mark(cell, counter, map))
          counter++;
    }
    // set outlet_id_last which is needed by the application setting the boundary conditions
    outlet_id_last = counter;
  }

  GridReordering<3>::reorder_cells(cell_data_3d, true);
  tria.create_triangulation(vertices_3d, cell_data_3d, subcell_data);

#ifdef DEBUG
  GridOut       grid_out;
  std::ofstream file("mesh-tria.vtu");
  grid_out.write_vtu(tria, file);
#endif

  // actually set boundary ids
  for(auto cell : tria.active_cell_iterators())
    for(unsigned int d = 0; d < 6; d++)
      if(cell->at_boundary(d) && map.find(face_vertices(cell->face(d))) != map.end())
        cell->face(d)->set_all_boundary_ids(map[face_vertices(cell->face(d))]);

  timings["create_triangulation_4_serial_triangulation"] = timer.wall_time();

  bool use_spline_deform = false;
  if(use_spline_deform)
  {
    if(!roots[0]->skeleton.empty())
    {
      deform.insert(
        {roots[0]->id, DeformTransfinitelyViaSplines<3>(splines, 0, roots[0]->skeleton, {{0, 0}})});
    }


    if(!roots[0]->right_child->skeleton.empty())
    {
      deform.insert({roots[0]->right_child->id,
                     (DeformTransfinitelyViaSplines<3>(
                       splines, 4, roots[0]->right_child->skeleton, {{0, 0}}))});
    }

    if(!roots[0]->left_child->skeleton.empty())
    {
      deform.insert(
        {roots[0]->left_child->id,
         (DeformTransfinitelyViaSplines<3>(splines, 8, roots[0]->left_child->skeleton, {{0, 0}}))});
    }
  }
  else
  {
    // fill deform map for all children of root
    Node * node_to_deform = roots[0];

    fill_deform(deform, node_to_deform);
  }

  // clean up
  for(auto & root : roots)
    delete root;
}

void
local_refinement(dealii::Triangulation<3> & tria, const int max_refined_generation)
{
  // set refine flag until specified generation
  unsigned int max_material_id = 0;

  for(int i = 0; i <= max_refined_generation; i++)
    max_material_id += (unsigned int)std::pow(2, i);

  for(auto & cell : tria.active_cell_iterators())
    if(cell->material_id() <= max_material_id)
      cell->set_refine_flag();

  // execute local refinement
  tria.execute_coarsening_and_refinement();
}

void
update_mapping(dealii::Triangulation<3> &                                        tria,
               std::map<types::material_id, DeformTransfinitelyViaCylinder<3>> & deform)
{
  if(deform.empty())
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
        if(!touched[cell->vertex_index(v)])
        {
          // std::cout << cell->material_id() << " " << LungID::to_string(cell->material_id()) <<
          // std::endl;
          try
          {
            cell->vertex(v) = deform[cell->material_id()].transform_to_deformed(cell->vertex(v));
          }
          catch(typename Mapping<3, 3>::ExcTransformationFailed & exc)
          {
            std::cout << "Failed for material id " << cell->material_id() << std::endl;
            // throw exc;
          }
          touched[cell->vertex_index(v)] = true;
        }

  // TODO only print if desired
  bool print = false;
  if(print)
    print_tria_statistics(tria);
}



template<int dim>
class MeshByDeformation : public Mesh<dim>
{
public:
  MeshByDeformation(
    const dealii::Triangulation<dim> &                                        tria,
    const unsigned int                                                        mapping_degree,
    const std::map<types::material_id, DeformTransfinitelyViaCylinder<dim>> & deform)
    : Mesh<dim>(mapping_degree), mapping(mapping_degree)
  {
    dealii::FE_Q<dim> fe_q(mapping_degree);
    FEValues<dim>     fe_values(fe_q,
                            Quadrature<dim>(fe_q.get_unit_support_points()),
                            update_quadrature_points);
    mapping.initialize(tria, [&](const typename Triangulation<dim>::cell_iterator & cell) {
      fe_values.reinit(cell);
      std::vector<Point<dim>> support_points = fe_values.get_quadrature_points();
      const auto              my_deform      = deform.find(cell->material_id());
      Assert(my_deform != deform.end(),
             ExcMessage("Could not find the given material id " +
                        std::to_string(cell->material_id())));
      for(auto & p : support_points)
        p = my_deform->second.transform_to_deformed(p);
      return support_points;
    });
  }

  virtual Mapping<dim> const &
  get_mapping() const
  {
    return mapping;
  }

private:
  MappingQCache<dim> mapping;
};



void lung(dealii::Triangulation<2> &                                             tria,
     int                                                                    refinements,
     const std::function<void(std::vector<Node *> & roots, unsigned int)> & create_tree,
     std::shared_ptr<Mesh<2>> &                                             mesh,
     std::map<std::string, double> &                                        timings,
     unsigned int const &                                                   outlet_id_first,
     unsigned int &                                                         outlet_id_last,
     const std::string &                                                    bspline_file = "",
     const unsigned int max_resolved_generation                                          = 0)
{
  (void)tria;
  (void)refinements;
  (void)create_tree;
  (void)mesh;
  (void)timings;
  (void)outlet_id_first;
  (void)outlet_id_last;
  (void)bspline_file;
  (void)max_resolved_generation;

  AssertThrow(false, ExcMessage("Not implemented for dim = 2."));
}

void lung(dealii::Triangulation<3> &                                             tria,
     int                                                                    refinements,
     const std::function<void(std::vector<Node *> & roots, unsigned int)> & create_tree,
     std::shared_ptr<Mesh<3>> &                                             mesh,
     std::map<std::string, double> &                                        timings,
     unsigned int const &                                                   outlet_id_first,
     unsigned int &                                                         outlet_id_last,
     const std::string &                                                    bspline_file = "",
     const unsigned int max_resolved_generation                                          = 0)
{
  std::map<types::material_id, DeformTransfinitelyViaCylinder<3>> deform;
  lung_unrefined(tria,
                 create_tree,
                 timings,
                 outlet_id_first,
                 outlet_id_last,
                 bspline_file,
                 deform,
                 max_resolved_generation);
  local_refinement(tria, MAX_REFINED_GENERATION);
  tria.refine_global(refinements);
  mesh = std::make_shared<MeshByDeformation<3>>(tria, 5, deform);
}

void lung(dealii::parallel::distributed::Triangulation<2> &                      tria,
     int                                                                    refinements,
     const std::function<void(std::vector<Node *> & roots, unsigned int)> & create_tree,
     std::shared_ptr<Mesh<2>> &                                             mesh,
     std::map<std::string, double> &                                        timings,
     unsigned int const &                                                   outlet_id_first,
     unsigned int &                                                         outlet_id_last,
     const std::string &                                                    bspline_file = "",
     const unsigned int max_resolved_generation                                          = 0)
{
  (void)tria;
  (void)refinements;
  (void)create_tree;
  (void)mesh;
  (void)timings;
  (void)outlet_id_first;
  (void)outlet_id_last;
  (void)bspline_file;
  (void)max_resolved_generation;

  AssertThrow(false, ExcMessage("Not implemented for dim = 2."));
}

void lung(dealii::parallel::distributed::Triangulation<3> &                      tria,
     int                                                                    refinements,
     const std::function<void(std::vector<Node *> & roots, unsigned int)> & create_tree,
     std::shared_ptr<Mesh<3>> &                                             mesh,
     std::map<std::string, double> &                                        timings,
     unsigned int const &                                                   outlet_id_first,
     unsigned int &                                                         outlet_id_last,
     const std::string &                                                    bspline_file = "",
     const unsigned int max_resolved_generation                                          = 0)
{
  // create sequential coarse grid (no refinements)
  dealii::Triangulation<3> tria_seq;
  tria_seq.set_mesh_smoothing(Triangulation<3>::limit_level_difference_at_vertices);
  std::map<types::material_id, DeformTransfinitelyViaCylinder<3>> deform;
  lung_unrefined(tria_seq,
                 create_tree,
                 timings,
                 outlet_id_first,
                 outlet_id_last,
                 bspline_file,
                 deform,
                 max_resolved_generation);
  // copy coarse grid to distributed triangulation and ...
  tria.copy_triangulation(tria_seq);
  // ... refine
  local_refinement(tria, MAX_REFINED_GENERATION);
  tria.refine_global(refinements);
  mesh = std::make_shared<MeshByDeformation<3>>(tria, 5, deform);

  outlet_id_last = Utilities::MPI::max(outlet_id_last, tria.get_communicator());
}

void lung(dealii::parallel::fullydistributed::Triangulation<2> &                 tria,
     int                                                                    refinements1,
     int                                                                    refinements2,
     const std::function<void(std::vector<Node *> & roots, unsigned int)> & create_tree,
     std::shared_ptr<Mesh<2>> &                                             mesh,
     std::map<std::string, double> &                                        timings,
     unsigned int const &                                                   outlet_id_first,
     unsigned int &                                                         outlet_id_last,
     const std::string &                                                    bspline_file = "",
     const unsigned int max_resolved_generation                                          = 0)
{
  (void)tria;
  (void)refinements1;
  (void)refinements2;
  (void)create_tree;
  (void)mesh;
  (void)timings;
  (void)outlet_id_first;
  (void)outlet_id_last;
  (void)bspline_file;
  (void)max_resolved_generation;

  AssertThrow(false, ExcMessage("Not implemented for dim = 2."));
}

void lung(dealii::parallel::fullydistributed::Triangulation<3> & tria,
     int                                                    refinements1,
     int /*refinements2*/,
     std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
     std::shared_ptr<Mesh<3>> &                                     mesh,
     std::map<std::string, double> &                                timings,
     unsigned int const &                                           outlet_id_first,
     unsigned int &                                                 outlet_id_last,
     const std::string &                                            bspline_file            = "",
     const unsigned int                                             max_resolved_generation = 0)
{
  Timer timer;
  timer.restart();

  std::map<types::material_id, DeformTransfinitelyViaCylinder<3>> deform;

  // create partitioned triangulation ...
  const auto construction_data =
    TriangulationDescription::Utilities::create_description_from_triangulation_in_groups<3, 3>(
      [&](dealii::Triangulation<3, 3> & tria) mutable {
        // ... by creating a refined sequential triangulation and partition it
        lung_unrefined(tria,
                       create_tree,
                       timings,
                       outlet_id_first,
                       outlet_id_last,
                       bspline_file,
                       deform,
                       max_resolved_generation);
        local_refinement(tria, MAX_REFINED_GENERATION);
        tria.refine_global(refinements1);
      },
      [](dealii::Triangulation<3, 3> & tria,
         const MPI_Comm                comm,
         const unsigned int /*group_size*/) {
        GridTools::partition_triangulation_zorder(Utilities::MPI::n_mpi_processes(comm), tria);
      },
      tria.get_communicator(),
      1 /*group size*/);
  tria.create_triangulation(construction_data);
  mesh = std::make_shared<MeshByDeformation<3>>(tria, 5, deform);

  outlet_id_last = Utilities::MPI::max(outlet_id_last, tria.get_communicator());

  timings["create_triangulation_0_overall"] = timer.wall_time();
}
} // namespace ExaDG::GridGen

#endif
