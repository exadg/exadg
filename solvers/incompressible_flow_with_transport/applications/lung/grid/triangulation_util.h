#ifndef LUNG_TRIA_UTIL
#define LUNG_TRIA_UTIL

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/numerics/data_out.h>

namespace ExaDG
{
using namespace dealii;

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
