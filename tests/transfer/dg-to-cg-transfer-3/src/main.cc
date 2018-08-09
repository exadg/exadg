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

#include "../../../../include/laplace/spatial_discretization/laplace_operator.h"
#include "../../../operators/operation-base-util/l2_norm.h"

#include "../../../operators/operation-base-1/src/include/rhs_operator.h"

#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"
#include "../../../../include/solvers_and_preconditioners/transfer/dg_to_cg_transfer.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

//#define DETAIL_OUTPUT
const int PATCHES = 10;

typedef double value_type;

using namespace dealii;
using namespace Laplace;

const int best_of = 10;


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

template<int dim, int fe_degree>
class Runner
{
public:
  Runner(ConvergenceTable & convergence_table)
    : convergence_table(convergence_table),
      comm(MPI_COMM_WORLD),
      rank(get_rank(comm)),
      triangulation(comm,
                    dealii::Triangulation<dim>::none,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe_dgq(fe_degree),
      fe_q(fe_degree),
      dof_handler_dg(triangulation),
      dof_handler_cg(triangulation),
      mapping(fe_degree),
      quadrature(fe_degree + 1),
      global_refinements(log(std::pow(1e8, 1.0 / dim) / (fe_degree + 1)) / log(2))
  {
  }

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

private:
  ConvergenceTable &                        convergence_table;
  MPI_Comm                                  comm;
  int                                       rank;
  parallel::distributed::Triangulation<dim> triangulation;
  FE_DGQ<dim>                               fe_dgq;
  FE_Q<dim>                                 fe_q;
  DoFHandler<dim>                           dof_handler_dg;
  DoFHandler<dim>                           dof_handler_cg;
  MappingQGeneric<dim>                      mapping;
  QGauss<1>                                 quadrature;
  const unsigned int                        global_refinements;
  MatrixFree<dim, value_type>               data_dg;
  MatrixFree<dim, value_type>               data_cg;
  std::shared_ptr<BoundaryDescriptor<dim>>  bc;
  MGConstrainedDoFs                         mg_constrained_dofs_cg;
  MGConstrainedDoFs                         mg_constrained_dofs_dg;
  ConstraintMatrix                          dummy_dg;
  ConstraintMatrix                          dummy_cg;

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

    dof_handler_cg.distribute_dofs(fe_q);
    dof_handler_cg.distribute_mg_dofs();
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
    typename MatrixFree<dim, value_type>::AdditionalData additional_data_dg;
    additional_data_dg.build_face_info = true;
    //    additional_data_dg.level_mg_handler = global_refinements;
    typename MatrixFree<dim, value_type>::AdditionalData additional_data_cg;
    //    additional_data_cg.level_mg_handler = global_refinements;

    // set boundary conditions: Dirichlet BC
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
    dirichlet_bc[0] = std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());

    // ...: Neumann BC: nothing to do

    // ...: Periodic BC: TODO

    // Setup constraints: for MG
    mg_constrained_dofs_cg.clear();
    std::set<types::boundary_id> dirichlet_boundary;
    for(auto it : this->bc->dirichlet_bc)
      dirichlet_boundary.insert(it.first);
    mg_constrained_dofs_cg.initialize(dof_handler_cg);
    mg_constrained_dofs_cg.make_zero_boundary_constraints(dof_handler_cg, dirichlet_boundary);

    //    if (fe_dgq.dofs_per_vertex > 0)
    dummy_cg.add_lines(mg_constrained_dofs_cg.get_boundary_indices(global_refinements));

    dummy_cg.close();
#ifdef DETAIL_OUTPUT
    dummy_cg.print(std::cout);
#endif
    mg_constrained_dofs_dg.clear();
    dummy_dg.clear();

    data_dg.reinit(mapping, dof_handler_dg, dummy_dg, quadrature, additional_data_dg);
    data_cg.reinit(mapping, dof_handler_cg, dummy_cg, quadrature, additional_data_cg);
  }

  void
  run(LaplaceOperator<dim, fe_degree, value_type> & laplace_dg,
      LaplaceOperator<dim, fe_degree, value_type> & laplace_cg,
      unsigned int                                  mg_level = numbers::invalid_unsigned_int)
  {
    // determine level: -1 and globarl_refinements map to the same level
    unsigned int level = std::min(global_refinements, mg_level);

    int procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    // Right hand side
    VectorType vec_rhs_dg, vec_rhs_cg;
    laplace_dg.initialize_dof_vector(vec_rhs_dg);
    laplace_cg.initialize_dof_vector(vec_rhs_cg);

    convergence_table.add_value("procs", procs);
    convergence_table.add_value("dim", dim);
    convergence_table.add_value("deg", fe_degree);
    convergence_table.add_value("dofs_dg", vec_rhs_dg.size());
    convergence_table.add_value("dofs_cg", vec_rhs_cg.size());
    convergence_table.add_value("lev", level);

    // setup implicitly RHS for CG via DG-CG-Transfer
    CGToDGTransfer<dim, value_type> transfer(laplace_dg.get_data(),
                                             laplace_cg.get_data(),
                                             mg_level,
                                             fe_degree);



    repeat<dim, fe_degree>(convergence_table, "toCG", [&]() mutable {
      transfer.toCG(vec_rhs_cg, vec_rhs_dg);
    });

    repeat<dim, fe_degree>(convergence_table, "toDG", [&]() mutable {
      transfer.toDG(vec_rhs_dg, vec_rhs_cg);
    });
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
    LaplaceOperator<dim, fe_degree, value_type> laplace_dg;
    LaplaceOperator<dim, fe_degree, value_type> laplace_cg;
    // ... its additional data
    LaplaceOperatorData<dim> laplace_additional_data;
    laplace_additional_data.bc = this->bc;

    // run through all multigrid level
    for(unsigned int level = 0; level <= global_refinements; level++)
    {
      laplace_dg.reinit(
        dof_handler_dg, mapping, (void *)&laplace_additional_data, mg_constrained_dofs_dg, level);
      laplace_cg.reinit(
        dof_handler_cg, mapping, (void *)&laplace_additional_data, mg_constrained_dofs_cg, level);
      run(laplace_dg, laplace_cg, level);
    }

    // run on fine grid without multigrid
    {
      laplace_dg.initialize(mapping, data_dg, dummy_dg, laplace_additional_data);
      laplace_cg.initialize(mapping, data_cg, dummy_cg, laplace_additional_data);
      run(laplace_dg, laplace_cg);
    }
  }
};

template<int dim, int fe_degree_1>
class Run
{
public:
  static void
  run(ConvergenceTable & convergence_table)
  {
    Runner<dim, fe_degree_1> run_cg(convergence_table);
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
  ConvergenceTable convergence_table;

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#  pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
  }
#endif

  Run<2, 1>::run(convergence_table);
  Run<2, 2>::run(convergence_table);
  Run<2, 3>::run(convergence_table);
  Run<2, 4>::run(convergence_table);
  Run<2, 5>::run(convergence_table);
  Run<2, 6>::run(convergence_table);
  Run<2, 7>::run(convergence_table);
  Run<2, 8>::run(convergence_table);
  Run<2, 9>::run(convergence_table);

  if(!rank)
  {
    std::ofstream outfile;
    outfile.open("dg-to-cg-transfer.csv");
    convergence_table.write_text(std::cout);
    convergence_table.write_text(outfile);
    outfile.close();
  }
  pcout << std::endl;

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
