#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_generator.h>

#include "operator_wrappers/laplace_wrapper.h"
#include "operator_wrappers/comp_navier_stokes.h"

//#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"

//#define CORE
//#define SELF

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

const int      best_of = 10;
typedef double Number;

#ifdef CORE
    const MPI_Comm comm = MPI_COMM_SELF;
#else
    #ifdef SELF
        const MPI_Comm comm = MPI_COMM_SELF;
    #else
        const MPI_Comm comm = MPI_COMM_WORLD;
    #endif
#endif
using namespace dealii;


template<int dim, int fe_degree, typename Function>
void
repeat(ConvergenceTable & convergence_table, std::string label, bool curv, Function f)
{
  Timer  time;
  double min_time = std::numeric_limits<double>::max();
#ifdef LIKWID_PERFMON
  std::string likwid_label = label + "-" + std::to_string(dim) +  (curv ? std::string("-curv-") : std::string("-cart-")) + std::to_string(fe_degree);
#endif
  for(int i = 0; i < best_of; i++)
  {
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_START(likwid_label.c_str());
#endif
    MPI_Barrier(MPI_COMM_WORLD);
    time.restart();
    f();
    double temp = time.wall_time();
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP(likwid_label.c_str());
#endif
    min_time = std::min(min_time, temp);
  }
  convergence_table.add_value(label, min_time);
  convergence_table.set_scientific(label, true);
}


template<int dim, int fe_degree>
class Run
{
public:
  static void
  run(ConvergenceTable & convergence_table, bool curv, MPI_Comm comm)
  {
    double                                    left = -1, right = +1;
    parallel::distributed::Triangulation<dim> triangulation(comm);

    GridGenerator::hyper_cube(triangulation, left, right);

    for(auto & cell : triangulation)
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        // x-direction
        if((std::fabs(cell.face(face_number)->center()(0) - left) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(0);
        else if((std::fabs(cell.face(face_number)->center()(0) - right) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(1);
        // y-direction
        else if((std::fabs(cell.face(face_number)->center()(1) - left) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(2);
        else if((std::fabs(cell.face(face_number)->center()(1) - right) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(3);
        // z-direction
        else if((std::fabs(cell.face(face_number)->center()(2) - left) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(4);
        else if((std::fabs(cell.face(face_number)->center()(2) - right) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(5);
      }

    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_faces;
    GridTools::collect_periodic_faces(triangulation, 0, 1, 0 /*x-direction*/, periodic_faces);
    GridTools::collect_periodic_faces(triangulation, 2, 3, 1 /*y-direction*/, periodic_faces);
    if(dim==3)
      GridTools::collect_periodic_faces(triangulation, 4, 5, 2 /*z-direction*/, periodic_faces);
    triangulation.add_periodicity(periodic_faces);
    
//     
//   const double deformation = +0.1;
//    const double frequnency  = +2.0;
//    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequnency);
//    if(curv){
//        triangulation.set_all_manifold_ids(1);
//        triangulation.set_manifold(1, manifold);
//    }
        

    int procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    int dofs = dim==2 ? 5e7 : 5e7;
    if (comm==MPI_COMM_SELF)
        dofs /= procs;
    int ref = log(std::pow(dofs, 1.0 / dim) / (fe_degree + 1)) / log(2);
    
#ifdef CORE
    ref = 1;
#endif
    
    triangulation.refine_global(ref);

    convergence_table.add_value("procs", procs);
    convergence_table.add_value("self", MPI_COMM_SELF==comm);
    convergence_table.add_value("dim", dim);
    convergence_table.add_value("curv", curv);
    convergence_table.add_value("deg", fe_degree);
    convergence_table.add_value("refs", triangulation.n_global_levels());
    convergence_table.add_value("dofs",
                                (int)std::pow(fe_degree + 1, dim) *
                                  triangulation.n_global_active_cells());

    /*
    {
      Poisson::OperatorWrapper<dim, fe_degree, Number,1> ns(triangulation,false,false);
      repeat<dim, fe_degree>(convergence_table, "poisson-1-cell", curv, [&]() mutable { ns.run(); });
    }
     */

    {
      Poisson::OperatorWrapper<dim, fe_degree, Number, 1> ns(triangulation,true,false);
      repeat<dim, fe_degree>(convergence_table, "poisson-1-face-based", curv, [&]() mutable { ns.run(); });
    }

    {
      Poisson::OperatorWrapper<dim, fe_degree, Number, 1> ns(triangulation,true,true);
      repeat<dim, fe_degree>(convergence_table, "poisson-1-cell-based", curv, [&]() mutable { ns.run(); });
    }

    {
      Poisson::OperatorWrapper<dim, fe_degree, Number, dim> ns(triangulation,true,false);
      repeat<dim, fe_degree>(convergence_table, "poisson-d-face-based", curv, [&]() mutable { ns.run(); });
    }

    {
      Poisson::OperatorWrapper<dim, fe_degree, Number, dim> ns(triangulation,true,true);
      repeat<dim, fe_degree>(convergence_table, "poisson-d-cell-based", curv, [&]() mutable { ns.run(); });
    }
    {
      CompNS::CombinedWrapper<dim, fe_degree, fe_degree+1, fe_degree+1, Number> ns(triangulation,false);
      repeat<dim, fe_degree>(convergence_table, "comp-face", curv, [&]() mutable { ns.run(); });
    }
    {
      CompNS::CombinedWrapper<dim, fe_degree, fe_degree+1, fe_degree+1, Number> ns(triangulation,true);
      repeat<dim, fe_degree>(convergence_table, "comp-cell-based", curv, [&]() mutable { ns.run(); });
    }

  }
};

template<int dim>
void
run(bool curv, MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ConvergenceTable convergence_table;
  Run<dim,  1>::run(convergence_table, curv, comm);
  Run<dim,  2>::run(convergence_table, curv, comm);
  Run<dim,  3>::run(convergence_table, curv, comm);
  Run<dim,  4>::run(convergence_table, curv, comm);
  Run<dim,  5>::run(convergence_table, curv, comm);
  Run<dim,  6>::run(convergence_table, curv, comm);
  Run<dim,  7>::run(convergence_table, curv, comm);
  Run<dim,  8>::run(convergence_table, curv, comm);
  Run<dim,  9>::run(convergence_table, curv, comm);
  Run<dim, 10>::run(convergence_table, curv, comm);
  Run<dim, 11>::run(convergence_table, curv, comm);
  Run<dim, 12>::run(convergence_table, curv, comm);
  Run<dim, 13>::run(convergence_table, curv, comm);
  Run<dim, 14>::run(convergence_table, curv, comm);
  Run<dim, 15>::run(convergence_table, curv, comm);
/*
*/

  if(!rank)
  {
     std::string   file_name = "out." + (curv ? std::string("curv.") : std::string("cart.")) + std::to_string(dim) + ".csv";
     std::ofstream outfile;
     outfile.open(file_name.c_str());
    convergence_table.write_text(std::cout);
     convergence_table.write_text(outfile);
     outfile.close();
  }
}

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#  pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
  }
#else
  std::cout << "WARNING: Not compiled with LIKWID!" << std::endl;
#endif

  //run<2>(false, comm);
  //run<2>(true , comm);
  run<3>(false, comm);
  //run<3>(true , comm);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}