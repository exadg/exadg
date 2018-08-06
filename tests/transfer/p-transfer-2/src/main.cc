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

#include "../../../operators/operation-base-util/interpolate.h"

#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"
#include "../../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_p.h"

#ifdef LIKWID_PERFMON
    #include <likwid.h>
#endif

//#define DETAIL_OUTPUT
const int PATCHES = 10;

typedef double value_type;

using namespace dealii;
using namespace Laplace;

const int best_of = 10;


template <int dim, int fe_degree_1, int fe_degree_2, typename Function>
void repeat(ConvergenceTable &convergence_table, std::string label,
            Function f) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Timer time;
  double min_time = std::numeric_limits<double>::max();
  for (int i = 0; i < best_of; i++) {
  MPI_Barrier(MPI_COMM_WORLD);
    if(!rank) printf("  %10s#%d#%d#%d#%d: ", label.c_str(), dim, fe_degree_1, fe_degree_2, i);  
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
    if(!rank) printf("%10.6f\n", temp);  
    min_time = std::min(min_time, temp);
  }
  convergence_table.add_value(label, min_time);
  convergence_table.set_scientific(label, true);
}


template <int dim, int fe_degree_1, int fe_degree_2> class Runner {

public:
  Runner(ConvergenceTable& convergence_table)
      : comm(MPI_COMM_WORLD), rank(get_rank(comm)),convergence_table(convergence_table),
        triangulation(comm, dealii::Triangulation<dim>::none,
                      parallel::distributed::Triangulation<
                          dim>::construct_multigrid_hierarchy),
        fe_1(fe_degree_1), fe_2(fe_degree_2), dof_handler_1(triangulation), 
        dof_handler_2(triangulation), mapping_1(fe_degree_1), mapping_2(fe_degree_2),
        quadrature_1(fe_degree_1 + 1), quadrature_2(fe_degree_2 + 1), 
              global_refinements( log(std::pow(1e8,1.0/dim)/(fe_degree_1+1))/log(2)) {}

  typedef LinearAlgebra::distributed::Vector<value_type> VNumber;

private:
  MPI_Comm comm;
  int rank;
  
  ConvergenceTable& convergence_table;
  
  parallel::distributed::Triangulation<dim> triangulation;
  FE_DGQ<dim> fe_1;
  FE_DGQ<dim> fe_2;
  DoFHandler<dim> dof_handler_1;
  DoFHandler<dim> dof_handler_2;
  MappingQGeneric<dim> mapping_1;
  MappingQGeneric<dim> mapping_2;
  
  ConstraintMatrix dummy_1;
  ConstraintMatrix dummy_2;
  
  QGauss<1> quadrature_1;
  QGauss<1> quadrature_2;
  const unsigned int global_refinements;
  MatrixFree<dim, value_type> data_1;
  MatrixFree<dim, value_type> data_2;
  std::shared_ptr<BoundaryDescriptor<dim>> bc;

  static int get_rank(MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);
    return rank;
  }

  void init_triangulation_and_dof_handler() {
      
    const double left = -1.0;
    const double right = +1.0;
    const double deformation = +0.1;
    const double frequnency = +2.0;

    GridGenerator::hyper_cube(triangulation, left, right);
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequnency);
    triangulation.set_all_manifold_ids(1);
    triangulation.set_manifold(1, manifold);
    triangulation.refine_global(global_refinements);

    for (auto cell : triangulation)
      for (unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell;
           ++face)
        if (cell.face(face)->at_boundary())
          if (std::abs(cell.face(face)->center()(0) - 1.0) < 1e-12)
            cell.face(face)->set_boundary_id(1);

    dof_handler_1.distribute_dofs(fe_1);
    dof_handler_1.distribute_mg_dofs();

    dof_handler_2.distribute_dofs(fe_2);
    dof_handler_2.distribute_mg_dofs();
  }

  void init_boundary_conditions() {
    bc.reset(new BoundaryDescriptor<dim>());
    bc->dirichlet_bc[0] =
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());
    bc->neumann_bc[1] =
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());
  }

  void init_matrixfree_and_constraint_matrix() {
    typename MatrixFree<dim, value_type>::AdditionalData additional_data_1;
    additional_data_1.build_face_info = true;
    typename MatrixFree<dim, value_type>::AdditionalData additional_data_2;
    additional_data_2.build_face_info = true;

    dummy_1.clear();
    dummy_2.clear();
    
    data_1.reinit(mapping_1, dof_handler_1, dummy_1, quadrature_1, additional_data_1);
    data_2.reinit(mapping_2, dof_handler_2, dummy_2, quadrature_2, additional_data_2);
  }

  void run(LaplaceOperator<dim, fe_degree_1, value_type> &laplace_1,
           LaplaceOperator<dim, fe_degree_2, value_type> &laplace_2,
           unsigned int mg_level = numbers::invalid_unsigned_int) {

    // determine level: -1 and globarl_refinements map to the same level
    unsigned int level = std::min(global_refinements, mg_level);
    
    int procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);

    LinearAlgebra::distributed::Vector<value_type> vec_1, vec_2;
    
    laplace_1.get_data().initialize_dof_vector(vec_1);
    laplace_2.get_data().initialize_dof_vector(vec_2);
    
    convergence_table.add_value("procs", procs);
    convergence_table.add_value("dim", dim);
    convergence_table.add_value("deg1", fe_degree_1);
    convergence_table.add_value("dofs1", vec_1.size());
    convergence_table.add_value("deg2", fe_degree_2);
    convergence_table.add_value("dofs2", vec_2.size());
    convergence_table.add_value("lev", level);
    
    MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, value_type, VNumber> 
        transfer(dof_handler_1, dof_handler_2, level);
    
    repeat<dim, fe_degree_1, fe_degree_2>(convergence_table, "restrict",
               [&]() mutable { transfer.restrict_and_add(0, vec_2, vec_1); });
    
    repeat<dim, fe_degree_1, fe_degree_2>(convergence_table, "prolongate",
               [&]() mutable { transfer.prolongate(0, vec_1, vec_2); });
    

  }

public:
  void run() {

    // initialize the system
    init_triangulation_and_dof_handler();
    init_boundary_conditions();
    init_matrixfree_and_constraint_matrix();

    // initialize the operator and ...
    LaplaceOperator<dim, fe_degree_1, value_type> laplace_1;
    LaplaceOperator<dim, fe_degree_2, value_type> laplace_2;
    // ... its additional data
    LaplaceOperatorData<dim> laplace_additional_data;
    laplace_additional_data.bc = this->bc;
    
    MGConstrainedDoFs mg_constrained_dofs;
    mg_constrained_dofs.clear();

    // run through all multigrid level
    for (unsigned int level = 0; level <= global_refinements; level++) {
      laplace_1.reinit(dof_handler_1, mapping_1, (void *)&laplace_additional_data, 
                     mg_constrained_dofs, level);
      laplace_2.reinit(dof_handler_2, mapping_2, (void *)&laplace_additional_data, 
                     mg_constrained_dofs, level);
      run(laplace_1, laplace_2, level);
    }

    // run on fine grid without multigrid
    {
      laplace_1.initialize(mapping_1, data_1, dummy_1, laplace_additional_data);
      laplace_2.initialize(mapping_2, data_2, dummy_2, laplace_additional_data);
      run(laplace_1, laplace_2);
    }

  }
};

template <int dim, int fe_degree_1>
class Run{
public:
    static void run(ConvergenceTable& convergence_table){
        Runner<dim, fe_degree_1, fe_degree_1/2> run_cg(convergence_table); run_cg.run();
    }    
};

int main(int argc, char **argv) {
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  ConvergenceTable convergence_table;
  
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
  }
#endif

  Run<2, 2 /*to: 1*/>::run(convergence_table);
  Run<2, 3 /*to: 1*/>::run(convergence_table);
  Run<2, 4 /*to: 2*/>::run(convergence_table);
  Run<2, 5 /*to: 2*/>::run(convergence_table);
  Run<2, 6 /*to: 3*/>::run(convergence_table);
  Run<2, 7 /*to: 3*/>::run(convergence_table);
  Run<2, 8 /*to: 4*/>::run(convergence_table);
  Run<2, 9 /*to: 4*/>::run(convergence_table);

    if (!rank){
      std::ofstream outfile;
      outfile.open("p-transfer.csv");
      convergence_table.write_text(std::cout);
      convergence_table.write_text(outfile);
      outfile.close();
    }
  pcout << std::endl;

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}