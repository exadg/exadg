#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/constraint_matrix.h>
#include <deal.II/lac/lapack_full_matrix.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/point_value_history.h>
#include <deal.II/numerics/solution_transfer.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/base/mg_level_object.h>

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_q.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_in.h>
#include <deal.II/grid/grid_out.h>

#include <deal.II/base/convergence_table.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/multigrid/mg_base.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include "../../../../include/poisson/spatial_discretization/laplace_operator.h"
#include "../../operation-base-util/l2_norm.h"
#include "../../operation-base-util/sparse_matrix_util.h"

#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

//#define DETAIL_OUTPUT
const int PATCHES = 10;
const int best_of = 3;

typedef double value_type;

using namespace dealii;
using namespace Poisson;

template<int dim, int fe_degree, typename Function>
void
repeat(ConvergenceTable & convergence_table, std::string label, Function f)
{
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Timer  time;
  double min_time = std::numeric_limits<double>::max();
  for(int i = 0; i < best_of; i++)
  {
    MPI_Barrier(MPI_COMM_WORLD);
    if(!rank)
      printf("  %10s#%d#%d#%d: ", label.c_str(), dim, fe_degree, i);
#ifdef LIKWID_PERFMON
    std::string likwid_label = label + "#" + std::to_string(dim) + "#" + std::to_string(fe_degree);
    LIKWID_MARKER_START(likwid_label.c_str());
#endif
    time.restart();
    f();
    double temp = time.wall_time();
#ifdef LIKWID_PERFMON
    LIKWID_MARKER_STOP(likwid_label.c_str());
#endif
    if(!rank)
      printf("%10.6f\n", temp);
    min_time = std::min(min_time, temp);
  }
  convergence_table.add_value(label, min_time);
  convergence_table.set_scientific(label, true);
}


template<int dim, int fe_degree, typename FE_TYPE>
class Runner
{
public:
  Runner(ConvergenceTable & convergence_table, const int approx_system_size)
    : comm(MPI_COMM_WORLD),
      rank(get_rank(comm)),
      triangulation(comm,
                    dealii::Triangulation<dim>::none,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe_dgq(fe_degree),
      dof_handler_dg(triangulation),
      convergence_table(convergence_table),
      mapping(fe_degree),
      quadrature(fe_degree + 1),
      global_refinements(log(std::pow(approx_system_size, 1.0 / dim) / (fe_degree + 1)) / log(2))
  {
  }

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

private:
  MPI_Comm                                  comm;
  int                                       rank;
  parallel::distributed::Triangulation<dim> triangulation;
  FE_TYPE                                   fe_dgq;
  DoFHandler<dim>                           dof_handler_dg;
  ConvergenceTable &                        convergence_table;
  MappingQGeneric<dim>                      mapping;
  QGauss<1>                                 quadrature;
  const unsigned int                        global_refinements;
  MatrixFree<dim, value_type>               data;
  std::shared_ptr<BoundaryDescriptor<dim>>  bc;
  MGConstrainedDoFs                         mg_constrained_dofs;
  ConstraintMatrix                          dummy;

  static int
  get_rank(MPI_Comm comm)
  {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
  }

  void
  init_triangulation_and_dof_handler()
  {
    const double left        = -1.0;
    const double right       = +1.0;
    const double deformation = +0.1;
    const double frequnency  = +2.0;

    GridGenerator::hyper_cube(triangulation, left, right);
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequnency);
    triangulation.set_all_manifold_ids(1);
    triangulation.set_manifold(1, manifold);
    triangulation.refine_global(global_refinements);

    for(auto cell : triangulation)
      for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
        if(cell.face(face)->at_boundary())
          if(std::abs(cell.face(face)->center()(0) - 1.0) < 1e-12)
            cell.face(face)->set_boundary_id(1);

    dof_handler_dg.distribute_dofs(fe_dgq);
    dof_handler_dg.distribute_mg_dofs();
  }

  void
  init_boundary_conditions()
  {
    bc.reset(new BoundaryDescriptor<dim>());
    bc->dirichlet_bc[0] = std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());
    bc->neumann_bc[1]   = std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());
  }

  void
  init_matrixfree_and_constraint_matrix()
  {
    typename MatrixFree<dim, value_type>::AdditionalData additional_data;

    if(fe_dgq.dofs_per_vertex == 0)
      additional_data.build_face_info = true;

    additional_data.level_mg_handler = global_refinements;

    // set boundary conditions: Dirichlet BC
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
    dirichlet_bc[0] = std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());

    // ...: Neumann BC: nothing to do

    // ...: Periodic BC: TODO

    // Setup constraints: for MG
    mg_constrained_dofs.clear();
    std::set<types::boundary_id> dirichlet_boundary;
    for(auto it : this->bc->dirichlet_bc)
      dirichlet_boundary.insert(it.first);
    mg_constrained_dofs.initialize(dof_handler_dg);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_dg, dirichlet_boundary);

    if(fe_dgq.dofs_per_vertex > 0)
      dummy.add_lines(mg_constrained_dofs.get_boundary_indices(global_refinements));

    dummy.close();
#ifdef DETAIL_OUTPUT
    dummy.print(std::cout);
#endif
    data.reinit(mapping, dof_handler_dg, dummy, quadrature, additional_data);
  }

  void
  run(LaplaceOperator<dim, fe_degree, value_type> & laplace,
      unsigned int                                  mg_level = numbers::invalid_unsigned_int)
  {
    Timer time;

    // determine level: -1 and globarl_refinements map to the same level
    int level = mg_level == numbers::invalid_unsigned_int ? -1 : mg_level;

    LinearAlgebra::distributed::Vector<value_type> vec_src;
    LinearAlgebra::distributed::Vector<value_type> vec_dst_1;
    LinearAlgebra::distributed::Vector<value_type> vec_dst_2;
    LinearAlgebra::distributed::Vector<value_type> vec_diag;
    LinearAlgebra::distributed::Vector<value_type> vec_dst_4;

    auto & data = laplace.get_data();
    data.initialize_dof_vector(vec_src);
    data.initialize_dof_vector(vec_dst_1);
    data.initialize_dof_vector(vec_dst_2);
    data.initialize_dof_vector(vec_dst_4);

    int procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    {
      convergence_table.add_value("procs", procs);
      convergence_table.add_value("dim", dim);
      convergence_table.add_value("deg", fe_degree);
      convergence_table.add_value("lev", level);
      convergence_table.add_value("dofs", vec_src.size());
    }

    // ... matrix-free VMult:
    repeat<dim, fe_degree>(convergence_table, "vmult", [&]() mutable { laplace.vmult(vec_dst_1, vec_src); });

    repeat<dim, fe_degree>(convergence_table, "d-init", [&]() mutable {
      laplace.calculate_diagonal(vec_diag);
    });

    repeat<dim, fe_degree>(convergence_table, "d-scale", [&]() mutable { vec_dst_4.scale(vec_diag); });

    const double entries_per_block = std::pow(fe_degree + 1, dim * 2);
    const double n_cells_glob      = vec_src.size() / std::pow(fe_degree + 1, dim);

    convergence_table.add_value("m-nnz-est", (entries_per_block * (dim * 2 + 1) * n_cells_glob));
    const double max_entries = 3.0e9;
    if(entries_per_block * (dim * 2 + 1) * n_cells_glob < max_entries)
    {
      std::cout << "  -- " << entries_per_block * (dim * 2 + 1) * n_cells_glob << std::endl;
      time.restart();
      TrilinosWrappers::SparseMatrix system_matrix;
      laplace.init_system_matrix(system_matrix);
      convergence_table.add_value("m-init", time.wall_time());
      convergence_table.set_scientific("m-init", true);

      convergence_table.add_value("m-nnz", system_matrix.n_nonzero_elements());

      time.restart();
      laplace.calculate_system_matrix(system_matrix);
      convergence_table.add_value("m-assembly", time.wall_time());
      convergence_table.set_scientific("m-assembly", true);

      repeat<dim, fe_degree>(convergence_table, "m-vmult", [&]() mutable {
        system_matrix.vmult(vec_dst_2, vec_src);
      });
    }
    else
    {
      convergence_table.add_value("m-init", 0.0);
      convergence_table.set_scientific("m-init", true);
      convergence_table.add_value("m-nnz", 0);
      convergence_table.add_value("m-assembly", 0.0);
      convergence_table.set_scientific("m-assembly", true);
      convergence_table.add_value("m-vmult", 0.0);
      convergence_table.set_scientific("m-vmult", true);
    }
  }

public:
  void
  run()
  {
    // initialize the system
    init_triangulation_and_dof_handler();
    init_boundary_conditions();
    init_matrixfree_and_constraint_matrix();

    // initialize the operator and ...
    LaplaceOperator<dim, fe_degree, value_type> laplace;
    // ... its additional data
    LaplaceOperatorData<dim> laplace_additional_data;
    laplace_additional_data.bc = this->bc;

    // run through all multigrid level
    for(unsigned int level = 0; level <= global_refinements; level++)
    {
      laplace.reinit(dof_handler_dg, mapping, (void *)&laplace_additional_data, mg_constrained_dofs, level);
      run(laplace, level);
    }

    // run on fine grid without multigrid
    {
      laplace.initialize(mapping, data, dummy, laplace_additional_data);
      run(laplace);
    }
  }
};

template<int dim, int fe_degree>
class Run
{
public:
  static void
  run(ConvergenceTable & convergence_table, const int approx_system_size)
  {
    Runner<dim, fe_degree, FE_DGQ<dim>> run_cg(convergence_table, approx_system_size);
    run_cg.run();
  }
};

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream               pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  int                              rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#  pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
  }
#endif
  if(false)
  {
    const int        approx_system_size = 1e8;
    ConvergenceTable convergence_table;
    Run<2, 1>::run(convergence_table, approx_system_size);
    Run<2, 2>::run(convergence_table, approx_system_size);
    Run<2, 3>::run(convergence_table, approx_system_size);
    Run<2, 4>::run(convergence_table, approx_system_size);
    Run<2, 5>::run(convergence_table, approx_system_size);
    Run<2, 6>::run(convergence_table, approx_system_size);
    Run<2, 7>::run(convergence_table, approx_system_size);
    Run<2, 8>::run(convergence_table, approx_system_size);
    Run<2, 9>::run(convergence_table, approx_system_size);
    if(!rank)
    {
      convergence_table.write_text(std::cout);
    }
  }

  if(true)
  {
    const int        approx_system_size = 1.0e7;
    ConvergenceTable convergence_table;
    Run<2, 1>::run(convergence_table, approx_system_size);
    Run<2, 2>::run(convergence_table, approx_system_size);
    Run<2, 3>::run(convergence_table, approx_system_size);
    Run<2, 4>::run(convergence_table, approx_system_size);
    Run<2, 5>::run(convergence_table, approx_system_size);
    Run<2, 6>::run(convergence_table, approx_system_size);
    Run<2, 7>::run(convergence_table, approx_system_size);
    Run<2, 8>::run(convergence_table, approx_system_size);
    Run<2, 9>::run(convergence_table, approx_system_size);
    if(!rank)
    {
      convergence_table.write_text(std::cout);
    }
  }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}