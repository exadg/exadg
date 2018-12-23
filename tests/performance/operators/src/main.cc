#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/grid/grid_generator.h>

#include "operator_wrappers/comp_navier_stokes.h"
#include "operator_wrappers/conv_diff_convective_wrapper.h"
#include "operator_wrappers/conv_diff_diffusive_wrapper.h"
#include "operator_wrappers/conv_diff_mass_wrapper.h"
#include "operator_wrappers/icomp_wrapper.h"
#include "operator_wrappers/laplace_wrapper.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

const int      best_of = 10;
typedef double Number;

const MPI_Comm comm = MPI_COMM_WORLD;

using namespace dealii;


template<int dim, int fe_degree, typename Function>
void
repeat(ConvergenceTable & convergence_table, std::string label, Function f)
{
  Timer  time;
  double min_time = std::numeric_limits<double>::max();
#ifdef LIKWID_PERFMON
  std::string likwid_label = label + "-" + std::to_string(dim) + "-" + std::to_string(fe_degree);
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
  run(ConvergenceTable & convergence_table)
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
    GridTools::collect_periodic_faces(triangulation, 4, 5, 2 /*z-direction*/, periodic_faces);
    triangulation.add_periodicity(periodic_faces);
    triangulation.refine_global(log(std::pow(5e7, 1.0 / dim) / (fe_degree + 1)) / log(2));

    int procs;
    MPI_Comm_size(comm, &procs);

    convergence_table.add_value("procs", procs);
    convergence_table.add_value("dim", dim);
    convergence_table.add_value("deg", fe_degree);
    convergence_table.add_value("refs", triangulation.n_global_levels());
    convergence_table.add_value("dofs",
                                (int)std::pow(fe_degree + 1, dim) *
                                  triangulation.n_global_active_cells());

    {
      Poisson::OperatorWrapper<dim, fe_degree, Number> ns(triangulation);
      repeat<dim, fe_degree>(convergence_table, "poisson", [&]() mutable { ns.run(); });
    }

    {
      OperatorWrapperMassMatrix<dim, fe_degree, Number> ns(triangulation);
      repeat<dim, fe_degree>(convergence_table, "cd-mass", [&]() mutable { ns.run(); });
    }

    {
      OperatorWrapperDiffusiveOperator<dim, fe_degree, Number> ns(triangulation);
      repeat<dim, fe_degree>(convergence_table, "cd-diff", [&]() mutable { ns.run(); });
    }

    {
      OperatorWrapperConvectiveOperator<dim, fe_degree, fe_degree, Number> ns(
        triangulation, ConvDiff::TypeVelocityField::Analytical);
      repeat<dim, fe_degree>(convergence_table, "cd-conv-1", [&]() mutable { ns.run(); });
    }

    {
      OperatorWrapperConvectiveOperator<dim, fe_degree, fe_degree, Number> ns(
        triangulation, ConvDiff::TypeVelocityField::Numerical);
      repeat<dim, fe_degree>(convergence_table, "cd-conv-2", [&]() mutable { ns.run(); });
    }

    {
      IncNS::ProjectionWrapper<dim, fe_degree, fe_degree, Number> ns(triangulation);
      repeat<dim, fe_degree>(convergence_table, "ns-icomp-proj", [&]() mutable { ns.run(); });
    }

    {
      IncNS::MassMatrixWrapper<dim, fe_degree, fe_degree, Number> ns(triangulation);
      repeat<dim, fe_degree>(convergence_table, "ns-icomp-mass", [&]() mutable { ns.run(); });
    }

    {
      IncNS::ConvectiveWrapper<dim, fe_degree, fe_degree, Number> ns(triangulation);
      repeat<dim, fe_degree>(convergence_table, "ns-icomp-conv", [&]() mutable { ns.run(); });
    }

    {
      IncNS::ViscousWrapper<dim, fe_degree, fe_degree, Number> ns(triangulation);
      repeat<dim, fe_degree>(convergence_table, "ns-icomp-visc", [&]() mutable { ns.run(); });
    }

    {
      CompNS::OperatorWrapper<dim, fe_degree, fe_degree + 1, fe_degree + 1, Number> ns(
        triangulation);
      repeat<dim, fe_degree>(convergence_table, "ns-comp", [&]() mutable { ns.run(); });
    }
  }
};

template<int dim>
void
run()
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ConvergenceTable convergence_table;
  Run<dim, 3>::run(convergence_table);
  //  Run<dim,  4>::run(convergence_table);
  //  Run<dim,  5>::run(convergence_table);
  //  Run<dim,  6>::run(convergence_table);
  //  Run<dim,  7>::run(convergence_table);
  //  Run<dim,  8>::run(convergence_table);
  //  Run<dim,  9>::run(convergence_table);
  //  Run<dim, 10>::run(convergence_table);

  if(!rank)
  {
    // std::string   file_name = "out" + std::to_string(dim) + ".csv";
    // std::ofstream outfile;
    // outfile.open(file_name.c_str());
    convergence_table.write_text(std::cout);
    // convergence_table.write_text(outfile);
    // outfile.close();
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

  run<2>();
  run<3>();

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}