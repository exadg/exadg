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

//#define DETAIL_OUTPUT
const int PATCHES = 10;

typedef double value_type;

using namespace dealii;
using namespace Laplace;


template <int dim> class ExactSolution : public Function<dim> {
public:
  ExactSolution(const double time = 0.) : Function<dim>(1, time) {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const {
    double result = std::sin(p[0] * numbers::PI);
    for (unsigned int d = 1; d < dim; ++d)
      result *= std::sin((p[d] + d) * numbers::PI);
    return std::abs(result + 1);
    //        return p[0];
  }
};



template <int dim, int fe_degree_1, int fe_degree_2> class Runner {

public:
  Runner(ConvergenceTable& convergence_table)
      : comm(MPI_COMM_WORLD), rank(get_rank(comm)),convergence_table(convergence_table),
        triangulation(comm, dealii::Triangulation<dim>::none,
                      parallel::distributed::Triangulation<
                          dim>::construct_multigrid_hierarchy),
        fe_1(fe_degree_1), fe_2(fe_degree_2), dof_handler_1(triangulation), 
        dof_handler_2(triangulation), mapping_1(fe_degree_1), mapping_2(fe_degree_2),
        quadrature_1(fe_degree_1 + 1), quadrature_2(fe_degree_2 + 1), global_refinements(dim==2?4:3) {}

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
    
    for(int v = 0; v <= 1; v++){
    
    convergence_table.add_value("vers", v);
    convergence_table.add_value("procs", procs);
    convergence_table.add_value("dim", dim);
    convergence_table.add_value("deg1", fe_degree_1);
    convergence_table.add_value("deg2", fe_degree_2);
    convergence_table.add_value("lev", level);

    LinearAlgebra::distributed::Vector<value_type>
        vec_sol_fine_delta, vec_sol_fine_delta2, vector_rhs, vec_sol_fine_1, vec_sol_fine_2, 
        vec_sol_fine_3, deflect, vec_rhs_coarse, vec_sol_coarse;
    
    data_1.initialize_dof_vector(vector_rhs);
    data_1.initialize_dof_vector(vec_sol_fine_1);
    data_1.initialize_dof_vector(vec_sol_fine_2);
    data_1.initialize_dof_vector(vec_sol_fine_3);
    data_1.initialize_dof_vector(vec_sol_fine_delta);
    data_1.initialize_dof_vector(vec_sol_fine_delta2);
    data_1.initialize_dof_vector(deflect);
    
    data_2.initialize_dof_vector(vec_rhs_coarse);
    data_2.initialize_dof_vector(vec_sol_coarse);
    
    MGTransferMatrixFreeP<dim, fe_degree_1, fe_degree_2, value_type, VNumber> 
        transfer(dof_handler_1, dof_handler_2, level);
    
    SolverControl solver_control(1000, 1e-12);
    SolverCG<VNumber> solver(solver_control);
    
    RHSOperator<dim, fe_degree_1, value_type> rhs(laplace_1.get_data());
    rhs.evaluate(vector_rhs);
    
    if(v==0){
        
    try {
      solver.solve(laplace_1, vec_sol_fine_1, vector_rhs, PreconditionIdentity());
    } catch (SolverControl::NoConvergence &) {
      //std::cout << "DG: not converved!" << std::endl;
    }
    convergence_table.add_value("it1", solver_control.last_step());
    } else {
        vec_sol_fine_1 = 0;
    convergence_table.add_value("it1", 0);
    }
    
    
    laplace_1.vmult(deflect, vec_sol_fine_1);
    deflect *= -1;
    deflect += vector_rhs;
    
    transfer.restrict_and_add(0, vec_rhs_coarse, deflect);
    
    try {
      solver.solve(laplace_2, vec_sol_coarse, vec_rhs_coarse, PreconditionIdentity());
    } catch (SolverControl::NoConvergence &) {
      //std::cout << "DG: not converved!" << std::endl;
    }
    
    convergence_table.add_value("it2", solver_control.last_step());
    
    transfer.prolongate(0, vec_sol_fine_delta, vec_sol_coarse);
    vec_sol_fine_2 = vec_sol_fine_1;
    vec_sol_fine_2 += vec_sol_fine_delta;
    
    laplace_1.vmult(deflect, vec_sol_fine_2);
    deflect *= -1;
    deflect += vector_rhs;
    
    try {
      solver.solve(laplace_1, vec_sol_fine_delta2, deflect, PreconditionIdentity());
    } catch (SolverControl::NoConvergence &) {
      //std::cout << "DG: not converved!" << std::endl;
    }
    
    convergence_table.add_value("it3", solver_control.last_step());
    
    vec_sol_fine_3 = vec_sol_fine_2;
    vec_sol_fine_3 += vec_sol_fine_delta2;
    
    {
      L2Norm<dim, fe_degree_1, value_type> l2norm(laplace_1.get_data());
      convergence_table.add_value("fine1", l2norm.run(vec_sol_fine_1));
      convergence_table.set_scientific("fine1", true); 
    }
    {
      L2Norm<dim, fe_degree_2, value_type> l2norm(laplace_2.get_data());
      convergence_table.add_value("coarse", l2norm.run(vec_sol_coarse));
      convergence_table.set_scientific("coarse", true); 
    }
    {  
      L2Norm<dim, fe_degree_1, value_type> l2norm(laplace_1.get_data());
      convergence_table.add_value("fine2", l2norm.run(vec_sol_fine_2));
      convergence_table.set_scientific("fine2", true); 
    }
    {  
      L2Norm<dim, fe_degree_1, value_type> l2norm(laplace_1.get_data());
      convergence_table.add_value("fine3", l2norm.run(vec_sol_fine_3));
      convergence_table.set_scientific("fine3", true); 
    }
        
    }

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

    laplace_1.initialize(mapping_1, data_1, dummy_1, laplace_additional_data);
    laplace_2.initialize(mapping_2, data_2, dummy_2, laplace_additional_data);
    run(laplace_1, laplace_2);

  }
};

int main(int argc, char **argv) {
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0);
  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  {
      
    ConvergenceTable convergence_table;
      
    {
      Runner<2, 7, 3> run_cg(convergence_table); run_cg.run();
    }
    {
      Runner<2, 3, 1> run_cg(convergence_table); run_cg.run();
    }
    if (!rank)
      convergence_table.write_text(std::cout);
    pcout << std::endl;
  }
}