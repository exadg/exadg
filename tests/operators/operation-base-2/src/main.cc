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
#include "include/tests.h"
#include "include/operator_base_test.h"

#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"
#include "../../operation-base-util/categorization.h"

using namespace dealii;
using namespace Laplace;

const unsigned int global_refinements = 3;
typedef double value_type;
const int fe_degree_min = 1;
const int fe_degree_max = 3;

typedef double value_type;

using namespace dealii;

template <int dim, int fe_degree, bool CATEGORIZE, typename FE_TYPE> class Runner {
public:
  static void run(ConvergenceTable &convergence_table) {
      
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
      
    // setup triangulation
    parallel::distributed::Triangulation<dim> triangulation(MPI_COMM_WORLD, dealii::Triangulation<dim>::none,
                      parallel::distributed::Triangulation<
                          dim>::construct_multigrid_hierarchy);
      
    const double left = -1.0;
    const double right = +1.0;
    const double deformation = +0.1;
    const double frequnency = +2.0;
    
    GridGenerator::hyper_cube(triangulation, left, right);
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequnency);
    triangulation.set_all_manifold_ids(1);
    triangulation.set_manifold(1, manifold);
    triangulation.refine_global(global_refinements);
    
    // setup dofhandler
    FE_TYPE fe_dgq(fe_degree);
    DoFHandler<dim> dof_handler_dg(triangulation);
    dof_handler_dg.distribute_dofs(fe_dgq);
    dof_handler_dg.distribute_mg_dofs();
    bool is_dg = (fe_dgq.dofs_per_vertex == 0);
    
    MappingQGeneric<dim> mapping(fe_degree);
    
    // setup matrixfree
    MatrixFree<dim, value_type> data;
    
    QGauss<1> quadrature(fe_degree + 1);
    typename MatrixFree<dim, value_type>::AdditionalData additional_data;
//    additional_data.mapping_update_flags =
//        (update_gradients | update_JxW_values | update_values);
//    additional_data.mapping_update_flags_inner_faces =
//        (update_JxW_values | update_normal_vectors | update_values);
//    additional_data.mapping_update_flags_boundary_faces =
//        (update_JxW_values | update_normal_vectors | update_quadrature_points |
//         update_values);
    
    if (fe_dgq.dofs_per_vertex == 0)
      additional_data.build_face_info = true;
    
    ConstraintMatrix dummy;
    
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
    dirichlet_bc[0] =
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());

    // ...: Neumann BC: nothing to do

    // ...: Periodic BC: TODO

    // Setup constraints: for MG
    MGConstrainedDoFs mg_constrained_dofs;
    mg_constrained_dofs.clear();
    std::set<types::boundary_id> dirichlet_boundary;
    for (auto it : dirichlet_bc)
      dirichlet_boundary.insert(it.first);
    mg_constrained_dofs.initialize(dof_handler_dg);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_dg,
                                                       dirichlet_boundary);
    if (fe_dgq.dofs_per_vertex > 0)
      dummy.add_lines(
          mg_constrained_dofs.get_boundary_indices(global_refinements));
    dummy.close();
    
if(CATEGORIZE){
    additional_data.build_face_info = true;
    Categorization::do_cell_based_loops(triangulation, additional_data);
}
    
    data.reinit(mapping, dof_handler_dg, dummy, quadrature, additional_data);

    // setup operator
    LaplaceOperator<dim, fe_degree, value_type> laplace;
    LaplaceOperatorData<dim> laplace_additional_data;
    std::shared_ptr<BoundaryDescriptor<dim>> bc(new BoundaryDescriptor<dim>());
    bc->dirichlet_bc[0] =
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());
    
if(CATEGORIZE){
    laplace_additional_data.use_cell_based_loops = true;
}
    
    laplace_additional_data.bc = bc;
    laplace.initialize(mapping, data, dummy, laplace_additional_data);

    // run tests
    convergence_table.add_value("procs", size);
    convergence_table.add_value("cell", CATEGORIZE);
    convergence_table.add_value("vers", is_dg);
    OperatorBaseTest::test(laplace, convergence_table,true,true,true, 
            (CATEGORIZE || size==1)&&is_dg);
    if((!CATEGORIZE && size!=1) || !is_dg){
        convergence_table.add_value("(B*v)_L2", 0);
        convergence_table.add_value("(B*v-B(S)*v)_L2", 0);
    }

    // go to next parameter
    Runner<dim, fe_degree + 1,CATEGORIZE, FE_TYPE>::run(convergence_table);
  }
};

template <int dim, bool categorize, typename FE_TYPE> 
class Runner<dim, fe_degree_max+1,categorize, FE_TYPE> {
public:
  static void run(ConvergenceTable & /*convergence_table*/) {}
};

int main(int argc, char **argv) {
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if (true)
  {
    ConvergenceTable convergence_table;
    // run for 2-d
    Runner<2, fe_degree_min,true, FE_DGQ<2>>::run(convergence_table);
    Runner<2, fe_degree_min,false, FE_DGQ<2>>::run(convergence_table);
    // run for 3-d
    Runner<3, fe_degree_min,true, FE_DGQ<3>>::run(convergence_table);
    Runner<3, fe_degree_min,false, FE_DGQ<3>>::run(convergence_table);
    if(!rank)
      convergence_table.write_text(std::cout);
  }

  if(true)
  {
    ConvergenceTable convergence_table;
    // run for 2-d
    //Runner<2, fe_degree_min,true, FE_Q<2>>::run(convergence_table);
    Runner<2, fe_degree_min,false, FE_Q<2>>::run(convergence_table);
    // run for 3-d
    //Runner<3, fe_degree_min,true, FE_Q<3>>::run(convergence_table);
    Runner<3, fe_degree_min,false, FE_Q<3>>::run(convergence_table);
    if(!rank)
      convergence_table.write_text(std::cout);
  }
  
}