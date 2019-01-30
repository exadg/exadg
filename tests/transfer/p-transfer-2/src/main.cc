#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/polynomial.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/utilities.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria.h>
#include <deal.II/lac/affine_constraints.h>
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
#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#endif
#include <deal.II/multigrid/mg_base.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include "../../../../include/poisson/spatial_discretization/laplace_operator.h"
#include "../../../operators/operation-base-util/l2_norm.h"

#include "../../../operators/operation-base-1/src/include/rhs_operator.h"

#include "../../../operators/operation-base-util/interpolate.h"

#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"
#include "../../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_p.h"

#include "../../../operators/operation-base-util/operator_reinit_multigrid.h"

#ifdef LIKWID_PERFMON
#  include <likwid.h>
#endif

//#define DETAIL_OUTPUT
const int PATCHES = 10;

typedef double value_type;

using namespace dealii;
using namespace Poisson;

const int best_of = 10;


template<int dim, int fe_degree_1, int fe_degree_2, typename Function>
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
      printf("  %10s#%d#%d#%d#%d: ", label.c_str(), dim, fe_degree_1, fe_degree_2, i);
#ifdef LIKWID_PERFMON
    std::string likwid_label = label + "#" + std::to_string(dim) + "#" +
                               std::to_string(fe_degree_1) + "#" + std::to_string(fe_degree_2);
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


template<int dim, int fe_degree_1, int fe_degree_2, typename FE_TYPE>
class Runner
{
public:
  Runner(ConvergenceTable & convergence_table)
    : comm(MPI_COMM_WORLD),
      rank(get_rank(comm)),
      convergence_table(convergence_table),
      triangulation(comm,
                    dealii::Triangulation<dim>::none,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe_1(fe_degree_1),
      fe_2(fe_degree_2),
      dof_handler_1(triangulation),
      dof_handler_2(triangulation),
      mapping_1(fe_degree_1),
      mapping_2(fe_degree_2),
      quadrature_1(fe_degree_1 + 1),
      quadrature_2(fe_degree_2 + 1),
      global_refinements(log(std::pow(5e7, 1.0 / dim) / (fe_degree_1 + 1)) / log(2) +
                         (fe_degree_1 == 5 && dim == 3))
  {
  }

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

private:
  MPI_Comm comm;
  int      rank;

  ConvergenceTable & convergence_table;

  parallel::distributed::Triangulation<dim> triangulation;
  FE_TYPE                                   fe_1;
  FE_TYPE                                   fe_2;
  DoFHandler<dim>                           dof_handler_1;
  DoFHandler<dim>                           dof_handler_2;
  MappingQGeneric<dim>                      mapping_1;
  MappingQGeneric<dim>                      mapping_2;

  AffineConstraints<double> dummy_1;
  AffineConstraints<double> dummy_2;

  QGauss<1>                                quadrature_1;
  QGauss<1>                                quadrature_2;
  const unsigned int                       global_refinements;
  MatrixFree<dim, value_type>              data_1;
  MatrixFree<dim, value_type>              data_2;
  std::shared_ptr<BoundaryDescriptor<dim>> bc;

  MGConstrainedDoFs mg_constrained_dofs_1;
  MGConstrainedDoFs mg_constrained_dofs_2;

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

    dof_handler_1.distribute_dofs(fe_1);
    dof_handler_1.distribute_mg_dofs();

    dof_handler_2.distribute_dofs(fe_2);
    dof_handler_2.distribute_mg_dofs();
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
    LaplaceOperatorData<dim>                             laplace_additional_data;
    typename MatrixFree<dim, value_type>::AdditionalData additional_data_1;
    typename MatrixFree<dim, value_type>::AdditionalData additional_data_2;

    if(fe_1.dofs_per_vertex == 0)
    {
      additional_data_1.mapping_update_flags = laplace_additional_data.mapping_update_flags;
      additional_data_1.mapping_update_flags_inner_faces =
        laplace_additional_data.mapping_update_flags_inner_faces;
      additional_data_1.mapping_update_flags_boundary_faces =
        laplace_additional_data.mapping_update_flags_boundary_faces;
      additional_data_2.mapping_update_flags = laplace_additional_data.mapping_update_flags;
      additional_data_2.mapping_update_flags_inner_faces =
        laplace_additional_data.mapping_update_flags_inner_faces;
      additional_data_2.mapping_update_flags_boundary_faces =
        laplace_additional_data.mapping_update_flags_boundary_faces;
    }

    dummy_1.clear();
    dummy_2.clear();

    if(fe_1.dofs_per_vertex != 0)
    {
      std::set<types::boundary_id> dirichlet_boundary;
      for(auto it : this->bc->dirichlet_bc)
        dirichlet_boundary.insert(it.first);
      mg_constrained_dofs_1.initialize(dof_handler_1);
      mg_constrained_dofs_1.make_zero_boundary_constraints(dof_handler_1, dirichlet_boundary);
      dummy_1.add_lines(mg_constrained_dofs_1.get_boundary_indices(global_refinements));

      mg_constrained_dofs_2.initialize(dof_handler_2);
      mg_constrained_dofs_2.make_zero_boundary_constraints(dof_handler_2, dirichlet_boundary);
      dummy_2.add_lines(mg_constrained_dofs_2.get_boundary_indices(global_refinements));
    }

    data_1.reinit(mapping_1, dof_handler_1, dummy_1, quadrature_1, additional_data_1);
    data_2.reinit(mapping_2, dof_handler_2, dummy_2, quadrature_2, additional_data_2);
  }

  void
  run(LaplaceOperator<dim, fe_degree_1, value_type> & laplace_1,
      LaplaceOperator<dim, fe_degree_2, value_type> & laplace_2,
      unsigned int                                    mg_level = numbers::invalid_unsigned_int)
  {
    // determine level: -1 and globarl_refinements map to the same level
    int level = mg_level == numbers::invalid_unsigned_int ? -1 : mg_level;

    int procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    LinearAlgebra::distributed::Vector<value_type> vec_1, vec_2, vec_3, vec_4, vec_5, vec_6, vec_7,
      vec_8, vec_9, vec_10, vec_11, vec_12, vec_13, vec_14, vec_15, vec_16, vec_17, vec_18, vec_19,
      vec_20, vec_21, vec_22;

    laplace_1.get_data().initialize_dof_vector(vec_1);
    laplace_1.get_data().initialize_dof_vector(vec_3);
    laplace_1.get_data().initialize_dof_vector(vec_5);
    laplace_1.get_data().initialize_dof_vector(vec_7);
    laplace_1.get_data().initialize_dof_vector(vec_9);
    laplace_1.get_data().initialize_dof_vector(vec_11);
    laplace_1.get_data().initialize_dof_vector(vec_13);
    laplace_1.get_data().initialize_dof_vector(vec_15);
    laplace_1.get_data().initialize_dof_vector(vec_17);
    laplace_1.get_data().initialize_dof_vector(vec_19);
    laplace_1.get_data().initialize_dof_vector(vec_21);

    laplace_2.get_data().initialize_dof_vector(vec_2);
    laplace_2.get_data().initialize_dof_vector(vec_4);
    laplace_2.get_data().initialize_dof_vector(vec_6);
    laplace_2.get_data().initialize_dof_vector(vec_8);
    laplace_2.get_data().initialize_dof_vector(vec_10);
    laplace_2.get_data().initialize_dof_vector(vec_12);
    laplace_2.get_data().initialize_dof_vector(vec_14);
    laplace_2.get_data().initialize_dof_vector(vec_16);
    laplace_2.get_data().initialize_dof_vector(vec_18);
    laplace_2.get_data().initialize_dof_vector(vec_20);
    laplace_2.get_data().initialize_dof_vector(vec_22);

    laplace_1.vmult(vec_21, vec_21);
    laplace_2.vmult(vec_22, vec_22);

    //    auto vec_3 = vec_1;
    //    auto vec_5 = vec_1;
    //    auto vec_7 = vec_1;
    //    auto vec_9 = vec_1;
    //    auto vec_11 = vec_1;
    //    auto vec_13 = vec_1;
    //    auto vec_15 = vec_1;
    //    auto vec_17 = vec_1;
    //    auto vec_19 = vec_1;
    //
    //    auto vec_4 = vec_2;
    //    auto vec_6 = vec_2;
    //    auto vec_8 = vec_2;
    //    auto vec_10 = vec_2;
    //    auto vec_12 = vec_2;
    //    auto vec_14 = vec_2;
    //    auto vec_16 = vec_2;
    //    auto vec_18 = vec_2;
    //    auto vec_20 = vec_2;

    convergence_table.add_value("procs", procs);
    convergence_table.add_value("dim", dim);
    convergence_table.add_value("deg1", fe_degree_1);
    convergence_table.add_value("dofs1", vec_1.size());
    convergence_table.add_value("deg2", fe_degree_2);
    convergence_table.add_value("dofs2", vec_2.size());
    convergence_table.add_value("lev", level);

    MGTransferMFP<dim, value_type, VectorType> transfer(&laplace_1.get_data(),
                                                        &laplace_2.get_data(),
                                                        fe_degree_1,
                                                        fe_degree_2);

    int i;
    i = 0;
    repeat<dim, fe_degree_1, fe_degree_2>(convergence_table, "restrict", [&]() mutable {
      if((i % 10) == 0)
        transfer.restrict_and_add(0, vec_2, vec_1);
      else if((i % 10) == 1)
        transfer.restrict_and_add(0, vec_4, vec_3);
      else if((i % 10) == 2)
        transfer.restrict_and_add(0, vec_6, vec_5);
      else if((i % 10) == 3)
        transfer.restrict_and_add(0, vec_8, vec_7);
      else if((i % 10) == 4)
        transfer.restrict_and_add(0, vec_10, vec_9);
      else if((i % 10) == 5)
        transfer.restrict_and_add(0, vec_12, vec_11);
      else if((i % 10) == 6)
        transfer.restrict_and_add(0, vec_14, vec_13);
      else if((i % 10) == 7)
        transfer.restrict_and_add(0, vec_16, vec_15);
      else if((i % 10) == 8)
        transfer.restrict_and_add(0, vec_18, vec_17);
      else if((i % 10) == 9)
        transfer.restrict_and_add(0, vec_20, vec_19);
      i++;
    });

    laplace_1.vmult(vec_21, vec_21);
    laplace_2.vmult(vec_22, vec_22);

    i = 0;
    repeat<dim, fe_degree_1, fe_degree_2>(convergence_table, "prolongate", [&]() mutable {
      if((i % 10) == 0)
        transfer.prolongate(0, vec_1, vec_2);
      else if((i % 10) == 1)
        transfer.prolongate(0, vec_3, vec_4);
      else if((i % 10) == 2)
        transfer.prolongate(0, vec_5, vec_6);
      else if((i % 10) == 3)
        transfer.prolongate(0, vec_7, vec_8);
      else if((i % 10) == 4)
        transfer.prolongate(0, vec_9, vec_10);
      else if((i % 10) == 5)
        transfer.prolongate(0, vec_11, vec_12);
      else if((i % 10) == 6)
        transfer.prolongate(0, vec_13, vec_14);
      else if((i % 10) == 7)
        transfer.prolongate(0, vec_15, vec_16);
      else if((i % 10) == 8)
        transfer.prolongate(0, vec_17, vec_18);
      else if((i % 10) == 9)
        transfer.prolongate(0, vec_19, vec_20);
      i++;
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
    LaplaceOperator<dim, fe_degree_1, value_type> laplace_1;
    LaplaceOperator<dim, fe_degree_2, value_type> laplace_2;
    // ... its additional data
    LaplaceOperatorData<dim> laplace_additional_data_1;
    laplace_additional_data_1.bc             = this->bc;
    laplace_additional_data_1.degree_mapping = fe_degree_1;
    LaplaceOperatorData<dim> laplace_additional_data_2;
    laplace_additional_data_2.bc             = this->bc;
    laplace_additional_data_2.degree_mapping = fe_degree_2;
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_face_pairs;

    // run through all multigrid level
    for(unsigned int level = 0; level <= global_refinements; level++)
    {
      MatrixFree<dim, value_type> matrixfree_1;
      AffineConstraints<double>   contraint_matrix_1;
      do_reinit_multigrid(dof_handler_1,
                          mapping_1,
                          laplace_additional_data_1,
                          mg_constrained_dofs_1,
                          periodic_face_pairs,
                          level,
                          matrixfree_1,
                          contraint_matrix_1);
      laplace_1.reinit(matrixfree_1, contraint_matrix_1, laplace_additional_data_1);

      MatrixFree<dim, value_type> matrixfree_2;
      AffineConstraints<double>   contraint_matrix_2;
      do_reinit_multigrid(dof_handler_2,
                          mapping_2,
                          laplace_additional_data_2,
                          mg_constrained_dofs_2,
                          periodic_face_pairs,
                          level,
                          matrixfree_2,
                          contraint_matrix_2);
      laplace_2.reinit(matrixfree_2, contraint_matrix_2, laplace_additional_data_2);

      run(laplace_1, laplace_2, level);
    }

    // run on fine grid without multigrid
    {
      laplace_1.reinit(data_1, dummy_1, laplace_additional_data_1);
      laplace_2.reinit(data_2, dummy_2, laplace_additional_data_2);
      run(laplace_1, laplace_2);
    }
  }
};

template<int dim, int fe_degree_1, typename FE_TYPE>
class Run
{
public:
  static void
  run(ConvergenceTable & convergence_table)
  {
    Runner<dim, fe_degree_1, fe_degree_1 / 2, FE_TYPE> run_cg(convergence_table);
    run_cg.run();
  }
};

template<int dim, typename FE_TYPE>
void
run(std::string label)
{
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  int                rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ConvergenceTable convergence_table;
  Run<dim, 2 /*to: 1*/, FE_TYPE>::run(convergence_table);
  Run<dim, 3 /*to: 1*/, FE_TYPE>::run(convergence_table);
  Run<dim, 4 /*to: 2*/, FE_TYPE>::run(convergence_table);
  Run<dim, 5 /*to: 2*/, FE_TYPE>::run(convergence_table);
  Run<dim, 6 /*to: 3*/, FE_TYPE>::run(convergence_table);
  Run<dim, 7 /*to: 3*/, FE_TYPE>::run(convergence_table);
  Run<dim, 8 /*to: 4*/, FE_TYPE>::run(convergence_table);
  Run<dim, 9 /*to: 4*/, FE_TYPE>::run(convergence_table);

  if(!rank)
  {
    std::string   file_name = "p-transfer-" + label + std::to_string(dim) + ".csv";
    std::ofstream outfile;
    outfile.open(file_name.c_str());
    convergence_table.write_text(std::cout);
    convergence_table.write_text(outfile);
    outfile.close();
  }
  pcout << std::endl;
}

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#  pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
  }
#endif

  run<2, FE_Q<2>>("cg");
  run<3, FE_Q<3>>("cg");

  run<2, FE_DGQ<2>>("dg");
  run<3, FE_DGQ<3>>("dg");

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}
