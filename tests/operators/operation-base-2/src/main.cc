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

#include "../../operation-base-util/laplace_operator.h"
#include "include/tests.h"

#ifdef LIKWID_PERFMON
    #include <likwid.h>
#endif

//#define DETAIL_OUTPUT
const int PATCHES = 10;

const unsigned int global_refinements = 3;
//const int dim = 2;
//const int fe_degree = 2;
typedef double value_type;

using namespace dealii;


const int dofs = 0.01e6;
const int dim = 2;
const int fe_degree_min = 2;
const int fe_degree_max = 2;
const int best_of = 1;
typedef double value_type;

using namespace dealii;

template <int dim> class ExactSolution : public Function<dim> {
public:
  ExactSolution(const double time = 0.)
      : Function<dim>(1, time), wave_number(1.) {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const {
    double result = std::sin(wave_number * p[0] * numbers::PI);
    for (unsigned int d = 1; d < dim; ++d)
      result *= std::sin((d + 1) * wave_number * p[d] * numbers::PI);
    return result;
  }

private:
  const double wave_number;
};

template <int dim, int fe_degree, typename Function>
void repeat(ConvergenceTable &convergence_table, std::string label,
            Function f) {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  Timer time;
  double min_time = std::numeric_limits<double>::max();
  for (int i = 0; i < best_of; i++) {
  MPI_Barrier(MPI_COMM_WORLD);
    if(!rank) printf("  %10s#%d#%d#%d: ", label.c_str(), dim, fe_degree, i);  
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
    if(!rank) printf("%10.6f\n", temp);  
    min_time = std::min(min_time, temp);
  }
  convergence_table.add_value(label, min_time);
  convergence_table.set_scientific(label, true);
}

template <int fe_degree> class Runner {
public:
  static void run(ConvergenceTable &convergence_table) {
    Timer time;
    MPI_Comm comm = MPI_COMM_SELF;

    parallel::distributed::Triangulation<dim> triangulation(comm);

    const int cells_x =
        (dim == 2 ? std::sqrt(dofs) : std::cbrt(dofs)) / (fe_degree + 1);
    GridGenerator::subdivided_hyper_cube(
        triangulation, cells_x, -0.5 * numbers::PI, +0.5 * numbers::PI);
    GridTools::distort_random(0.4,triangulation);
    FE_DGQ<dim> fe_dgq(fe_degree);

    DoFHandler<dim> dof_handler_dg(triangulation);
    dof_handler_dg.distribute_dofs(fe_dgq);
    
    // compute dofs on all processes
    int comm_size = 1;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_size);
    int dofs_actual = dof_handler_dg.n_dofs() * comm_size;

    convergence_table.add_value("dim", dim);
    convergence_table.add_value("degree", fe_degree);
    convergence_table.add_value("dofs", dofs_actual);

    MatrixFree<dim, value_type> data;

    LinearAlgebra::distributed::Vector<value_type> vec_src;
    LinearAlgebra::distributed::Vector<value_type> vec_dst_1;
    LinearAlgebra::distributed::Vector<value_type> vec_dst_2;
    LinearAlgebra::distributed::Vector<value_type> vec_diag;
    LinearAlgebra::distributed::Vector<value_type> vec_dst_3;
    LinearAlgebra::distributed::Vector<value_type> vec_dst_4;

    QGauss<1> quadrature(fe_degree + 1);
    typename MatrixFree<dim, value_type>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_values);
    additional_data.mapping_update_flags_inner_faces =
        (update_JxW_values | update_normal_vectors | update_values);
    additional_data.mapping_update_flags_boundary_faces =
        (update_JxW_values | update_normal_vectors | update_quadrature_points |
         update_values);

    ConstraintMatrix dummy;
    dummy.close();
    data.reinit(dof_handler_dg, dummy, quadrature, additional_data);
    data.initialize_dof_vector(vec_src);
    data.initialize_dof_vector(vec_dst_1);
    data.initialize_dof_vector(vec_dst_2);
    data.initialize_dof_vector(vec_dst_3);
    data.initialize_dof_vector(vec_dst_4);
    
    VectorTools::interpolate(dof_handler_dg, ExactSolution<dim>(0), vec_src);

    LaplaceOperator<dim, fe_degree, value_type> laplace;
    LaplaceOperatorData<dim> laplace_additional_data;
    std::shared_ptr<BoundaryDescriptor<dim>> bc(new BoundaryDescriptor<dim>());
    bc->dirichlet_bc[0] =
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());
    
    laplace_additional_data.bc = bc;
    
    laplace.reinit(data, dummy, laplace_additional_data);

    repeat<dim,fe_degree>(convergence_table, "vmult",
           [&]() mutable { laplace.vmult(vec_dst_1, vec_src); });

    repeat<dim,fe_degree>(convergence_table, "d-init",
           [&]() mutable { laplace.calculate_diagonal(vec_diag); });

    repeat<dim,fe_degree>(convergence_table, "d-scale",
           [&]() mutable { vec_dst_4.scale(vec_diag); });

    if(comm_size==1){
      repeat<dim,fe_degree>(convergence_table, "bd-init",
             [&]() mutable { laplace.update_block_jacobi(false /*TODO*/); });
      repeat<dim,fe_degree>(convergence_table, "bd-vmult",
             [&]() mutable { laplace.apply_block_diagonal(vec_dst_3, vec_src); });
    }
           
    if (best_of == 1) {
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

        repeat<dim,fe_degree>(convergence_table, "m-vmult",
               [&]() mutable { system_matrix.vmult(vec_dst_2, vec_src); });
        // perform tests...
        Tests<value_type> tests;
        
        // ... SparseMatrix vs. MatrixFree
        tests.test_sm_vs_mf(vec_dst_1, vec_dst_2);
        
        // ... SparseMatrix vs. DiagonalMatrix
        tests.test_sm_vs_diag(system_matrix, vec_diag);
        
        // ... SparseMatrix vs. BlockDiagonalMatrix
        tests.test_block_diag(vec_src, vec_dst_3, laplace, dim, fe_degree);
        
    }

    Runner<fe_degree + 1>::run(convergence_table);
  }
};

template <> class Runner<fe_degree_max+1> {
public:
  static void run(ConvergenceTable & /*convergence_table*/) {}
};

int main(int argc, char **argv) {
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_INIT;
#pragma omp parallel
  {
    LIKWID_MARKER_THREADINIT;
  }
#endif


  ConvergenceTable convergence_table;
  Runner<fe_degree_min>::run(convergence_table);
  if(!rank){
      std::ofstream outfile;
      outfile.open("ctable.csv");
      convergence_table.write_text(outfile);
      outfile.close();
  }

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_CLOSE;
#endif
}