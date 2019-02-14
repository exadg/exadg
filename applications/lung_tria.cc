//#include <deal.II/grid/tria.h>

#include <deal.II/base/mpi.h>
#include <stdio.h>
#include "mpi.h"

#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>

#include <deal.II/distributed/tria.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <algorithm>
#include <initializer_list>
#include <iostream>
#include <iterator>
#include <vector>

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/manifold.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/data_out_dof_data.h>
#include <fstream>
#include <iostream>

#include "poisson_test_cases/lung/lung_environment.h"

#include <limits>

using namespace dealii;

//#define TEST_LUNG_MANUAL

#include "poisson_test_cases/lung/lung_grid.h"


void
run(int generations, int refinements1, int refinements2, std::vector<std::string> & files)
{
  Timer                         timer;
  std::map<std::string, double> timings;

#ifdef TEST_LUNG_MANUAL
  auto tree_factory = [](std::vector<Node *> & roots, unsigned int generations) {
    std::vector<Point<3>>           points(4);
    std::vector<CellData<1>>        cells(3);
    std::vector<CellAdditionalInfo> cells_additional_data(3);

    points[0] = {+0.0, +0.0, +0.0};
    points[1] = {+0.0, +0.0, +1.0};
    points[2] = {+0.0, +0.5, +1.0};
    points[3] = {+0.0, -0.5, +0.7};

    cells[0].vertices[0] = 0;
    cells[0].vertices[1] = 1;
    cells[1].vertices[0] = 1;
    cells[1].vertices[1] = 2;
    cells[2].vertices[0] = 1;
    cells[2].vertices[1] = 3;

    cells_additional_data[0] = {0.2, 0};
    cells_additional_data[1] = {0.1, 1};
    cells_additional_data[2] = {0.1, 1};

    try
    {
      dealii::GridGenerator::lung_to_node(generations, points, cells, cells_additional_data, roots);
    }
    catch(const std::exception & e)
    {
      std::cout << e.what();
    }
  };
#else
  auto tree_factory = dealii::GridGenerator::lung_files_to_node(files);
#endif

  // parallel::distributed::Triangulation<3> tria_dist(MPI_COMM_WORLD);
  // dealii::GridGenerator::lung(tria_dist, generations, refinements2, tree_factory, timings);

  parallel::fullydistributed::Triangulation<3> tria(MPI_COMM_WORLD);
  dealii::GridGenerator::lung(tria, generations, refinements1, refinements2, tree_factory, timings);


  {
    timer.restart();
    DoFHandler<3> dofhanlder2(tria);
    FE_DGQ<3>     fe2(0);
    dofhanlder2.distribute_dofs(fe2);
    timings["dofhandler"] = timer.wall_time();

    timer.restart();
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    DataOut<3> data_out;
    data_out.attach_dof_handler(dofhanlder2);

    LinearAlgebra::distributed::Vector<double> ranks(dofhanlder2.locally_owned_dofs(),
                                                     MPI_COMM_WORLD);

    ranks = rank;

    data_out.add_data_vector(ranks, "ranks", DataOut<3>::DataVectorType::type_dof_data);

    data_out.build_patches(1);

    data_out.write_vtu_in_parallel("mesh-b.vtu", MPI_COMM_WORLD);
    timings["vtk"] = timer.wall_time();
  }

  print_tria_statistics(tria);

  printf("### Generations: %d; refs_1: %d; refs_2: %d; cells: %d\n",
         generations,
         refinements1,
         refinements2,
         tria.n_active_cells());
  std::cout << "| Region | Timings [s] |" << std::endl;
  std::cout << "| :----- | ----------: |" << std::endl;
  for(auto value : timings)
    printf("| %s | %7.4f |\n", value.first.c_str(), value.second);
  std::cout << std::endl;
}

int
main(int argc, char ** argv)
{
  using namespace dealii;
  Utilities::MPI::MPI_InitFinalize mpi_init(argc, argv, 1);

  std::vector<std::string> files;

  int generations  = atoi(argv[1]);
  int refinements1 = atoi(argv[2]);
  int refinements2 = atoi(argv[3]);

  for(int i = 4; i < argc; ++i)
    files.push_back(argv[i]);

  if(files.size() == 0)
    get_lung_files_from_environment(files);

  run(generations, refinements1, refinements2, files);

  return 0;
}