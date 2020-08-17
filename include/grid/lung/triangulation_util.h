#ifndef LUNG_TRIA_UTIL
#define LUNG_TRIA_UTIL

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/data_out.h>

namespace ExaDG
{
using namespace dealii;

void print_triangulation_3_1(std::vector<Point<3>> & points, std::vector<CellData<1>> & cells)
{
  std::vector<Point<3>>    points2;
  std::vector<CellData<1>> cells2;

  int i = 0;
  for(auto & celli : cells)
  {
    CellData<1> cell;
    cell.vertices[0] = i++;
    cell.vertices[1] = i++;
    cells2.push_back(cell);

    points2.push_back(points[celli.vertices[0]]);
    points2.push_back(points[celli.vertices[1]]);
  }

  Triangulation<1, 3> tria;
  tria.create_triangulation(points2, cells2, SubCellData());

  std::cout << "AA" << std::endl;
  DoFHandler<1, 3> dofhanlder(tria);
  FE_DGQ<1, 3>     fe2(0);
  dofhanlder.distribute_dofs(fe2);

  std::cout << "BB" << std::endl;

  DataOut<1, DoFHandler<1, 3>> data_out;
  data_out.attach_dof_handler(dofhanlder);

  data_out.build_patches(1);

  std::filebuf fb;
  fb.open("test.vtu", std::ios::out);
  std::ostream os(&fb);
  data_out.write_vtu(os);
  fb.close();
}

void print_tria_statistics(Triangulation<3> & tria)
{
  std::cout << "Statistics:" << std::endl;
  printf("        cells:       %7d\n", tria.n_active_cells());
  printf("        faces:       %7d\n", tria.n_active_faces());
  printf("        vertices:    %7d\n", tria.n_vertices());

  std::vector<int> face_counter{0, 0, 0, 0, 0, 0};
  for(auto cell : tria.active_cell_iterators())
    for(int i = 0; i < 6; i++)
      if(cell->at_boundary(i))
        face_counter[i]++;

  printf("        boundaries:  ");
  for(auto f : face_counter)
    printf("%7d", f);
  printf("\n\n");



  std::map<int, int> id_map;
  for(auto cell : tria.active_cell_iterators())
    for(int i = 0; i < 6; i++)
      id_map[cell->face(i)->boundary_id()] = 0;

  for(auto cell : tria.active_cell_iterators())
    for(int i = 0; i < 6; i++)
      id_map[cell->face(i)->boundary_id()]++;

  for(auto i : id_map)
    std::cout << i.first << " " << i.second << std::endl;
}

} // namespace ExaDG

#endif
