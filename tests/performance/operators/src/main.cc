#include <deal.II/grid/grid_generator.h>

#include "operator_wrappers/comp_navier_stokes.h"


const int      dim             = 3;
const int      degree          = 3;
const int      n_q_points_conv = degree + 1;
const int      n_q_points_vis  = degree + 1;
typedef double Number;
const MPI_Comm comm = MPI_COMM_WORLD;

using namespace dealii;

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  double                                    left = -1, right = +1;
  parallel::distributed::Triangulation<dim> triangulation(comm);

  GridGenerator::hyper_cube(triangulation, left, right);

  typename Triangulation<dim>::cell_iterator cell = triangulation.begin(),
                                             endc = triangulation.end();
  for(; cell != endc; ++cell)
  {
    for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
        ++face_number)
    {
      // x-direction
      if((std::fabs(cell->face(face_number)->center()(0) - left) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(0);
      else if((std::fabs(cell->face(face_number)->center()(0) - right) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(1);
      // y-direction
      else if((std::fabs(cell->face(face_number)->center()(1) - left) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(2);
      else if((std::fabs(cell->face(face_number)->center()(1) - right) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(3);
      // z-direction
      else if((std::fabs(cell->face(face_number)->center()(2) - left) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(4);
      else if((std::fabs(cell->face(face_number)->center()(2) - right) < 1e-12))
        cell->face(face_number)->set_all_boundary_ids(5);
    }
  }

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;
  GridTools::collect_periodic_faces(triangulation, 0, 1, 0 /*x-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(triangulation, 2, 3, 1 /*y-direction*/, periodic_faces);
  GridTools::collect_periodic_faces(triangulation, 4, 5, 2 /*z-direction*/, periodic_faces);
  triangulation.add_periodicity(periodic_faces);
  triangulation.refine_global(2);

  OperatorWrapperCompNS<dim, degree, n_q_points_conv, n_q_points_vis, Number> ns(triangulation);
  ns.run();
}