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
#include <deal.II/lac/trilinos_sparse_matrix.h>
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

#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"
#include "../../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_c.h"

#include "../../../operators/operation-base-util/operator_reinit_multigrid.h"

//#define DETAIL_OUTPUT
const int PATCHES = 10;

typedef double value_type;

using namespace dealii;
using namespace Poisson;


template<int dim, int fe_degree>
class Runner
{
public:
  Runner()
    : comm(MPI_COMM_WORLD),
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
      global_refinements(dim == 2 ? 4 : 3)
  {
  }

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

private:
  MPI_Comm                                  comm;
  int                                       rank;
  parallel::distributed::Triangulation<dim> triangulation;
  FE_DGQ<dim>                               fe_dgq;
  FE_Q<dim>                                 fe_q;
  DoFHandler<dim>                           dof_handler_dg;
  DoFHandler<dim>                           dof_handler_cg;
  ConvergenceTable                          convergence_table;
  MappingQGeneric<dim>                      mapping;
  QGauss<1>                                 quadrature;
  const unsigned int                        global_refinements;
  MatrixFree<dim, value_type>               data_dg;
  MatrixFree<dim, value_type>               data_cg;
  std::shared_ptr<BoundaryDescriptor<dim>>  bc;
  MGConstrainedDoFs                         mg_constrained_dofs_cg;
  AffineConstraints<double>                 dummy_dg;
  AffineConstraints<double>                 dummy_cg;

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
    LaplaceOperatorData<dim> laplace_additional_data;

    typename MatrixFree<dim, value_type>::AdditionalData additional_data_dg;
    additional_data_dg.mapping_update_flags = laplace_additional_data.mapping_update_flags;
    additional_data_dg.mapping_update_flags_inner_faces =
      laplace_additional_data.mapping_update_flags_inner_faces;
    additional_data_dg.mapping_update_flags_boundary_faces =
      laplace_additional_data.mapping_update_flags_boundary_faces;
    typename MatrixFree<dim, value_type>::AdditionalData additional_data_cg;
    additional_data_cg.mapping_update_flags = laplace_additional_data.mapping_update_flags;

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

    convergence_table.add_value("procs", procs);
    convergence_table.add_value("dim", dim);
    convergence_table.add_value("deg", fe_degree);
    convergence_table.add_value("lev", level);

    // Right hand side
    VectorType vec_rhs_dg, vec_rhs_cg;
    laplace_dg.initialize_dof_vector(vec_rhs_dg);
    laplace_cg.initialize_dof_vector(vec_rhs_cg);

    // setup explicitly RHS for DG
    RHSOperator<dim, fe_degree, value_type> rhs(laplace_dg.get_data());
    rhs.evaluate(vec_rhs_dg);

    // setup implicitly RHS for CG via DG-CG-Transfer
    MGTransferMFC<dim, value_type> transfer(
      data_dg, data_cg, dummy_dg, dummy_cg, mg_level, fe_degree);
    transfer.restrict_and_add(0, vec_rhs_cg, vec_rhs_dg);

    auto bs = mg_constrained_dofs_cg.get_boundary_indices(level);
    auto ls = vec_rhs_cg.locally_owned_elements();
    auto gs = bs;
    gs.subtract_set(ls);
    bs.subtract_set(gs);
    for(auto i : bs)
      vec_rhs_cg(i) = 0.0;

    {
      VectorType vec_rhs_cg_;
      laplace_cg.initialize_dof_vector(vec_rhs_cg_);
      RHSOperator<dim, fe_degree, value_type> rhs_cg(laplace_cg.get_data());
      rhs_cg.evaluate(vec_rhs_cg_);
    }

    // Solve linear equation system: setup solution vectors
    VectorType vec_sol_dg, vec_sol_cg, vec_sol_cg_dg;

    // ... fill with zeroes
    laplace_dg.initialize_dof_vector(vec_sol_dg);
    laplace_cg.initialize_dof_vector(vec_sol_cg);
    laplace_dg.initialize_dof_vector(vec_sol_cg_dg);

    // ... fill ghost values with zeroes
    vec_sol_dg.update_ghost_values();
    vec_sol_cg.update_ghost_values();

    // .. setup conjugate-gradient-solver
    SolverControl        solver_control(1000, 1e-12);
    SolverCG<VectorType> solver(solver_control);

    // ... solve with sparse matrix
    try
    {
      solver.solve(laplace_dg, vec_sol_dg, vec_rhs_dg, PreconditionIdentity());
    }
    catch(SolverControl::NoConvergence &)
    {
      std::cout << "DG: not converved!" << std::endl;
    }
    convergence_table.add_value("steps-dg", solver_control.last_step());

    // ... solve matrix-free
    try
    {
      solver.solve(laplace_cg, vec_sol_cg, vec_rhs_cg, PreconditionIdentity());
    }
    catch(SolverControl::NoConvergence &)
    {
      std::cout << "CG: not converved!" << std::endl;
    }
    convergence_table.add_value("steps-cg", solver_control.last_step());

    // ... output result to paraview
    if(mg_level == numbers::invalid_unsigned_int)
    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler_dg);

      data_out.add_data_vector(vec_sol_dg, "solution");
      data_out.build_patches(PATCHES);

      const std::string filename = "output/solution.1.vtu";
      data_out.write_vtu_in_parallel(filename.c_str(), comm);
    }

    if(mg_level == numbers::invalid_unsigned_int)
    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler_cg);

      data_out.add_data_vector(vec_sol_cg, "solution");
      data_out.build_patches(PATCHES);

      const std::string filename = "output/solution.2.vtu";
      data_out.write_vtu_in_parallel(filename.c_str(), comm);
    }

    transfer.prolongate(0, vec_sol_cg_dg, vec_sol_cg);

    if(mg_level == numbers::invalid_unsigned_int)
    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler_dg);

      data_out.add_data_vector(vec_sol_cg_dg, "solution");
      data_out.build_patches(PATCHES);

      const std::string filename = "output/solution.3.vtu";
      data_out.write_vtu_in_parallel(filename.c_str(), comm);
    }

    {
      L2Norm<dim, fe_degree, value_type> l2norm(laplace_dg.get_data());

      // l2-norm of dg result
      auto temp1 = vec_sol_dg;
      convergence_table.add_value("sol_dg_l2", l2norm.run(temp1));
      convergence_table.set_scientific("sol_dg_l2", true);

      // l2-norm of cg result
      L2Norm<dim, fe_degree, value_type> l2norm_cg(laplace_cg.get_data());
      auto                               temp2 = vec_sol_cg;
      convergence_table.add_value("sol_cg_l2", l2norm_cg.run(temp2));
      convergence_table.set_scientific("sol_cg_l2", true);

      // l2-norm of cg result
      auto temp3 = vec_sol_cg_dg;
      convergence_table.add_value("sol_cg_dg_l2", l2norm.run(temp3));
      convergence_table.set_scientific("sol_cg_dg_l2", true);

      // l2-norm of error
      auto temp_dg = vec_sol_dg;
      auto temp_cg = vec_sol_cg_dg;
      temp_dg -= temp_cg;

      convergence_table.add_value("err_l2", l2norm.run(temp_dg));
      convergence_table.set_scientific("err_l2", true);
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
    LaplaceOperator<dim, fe_degree, value_type> laplace_dg;
    LaplaceOperator<dim, fe_degree, value_type> laplace_cg;
    // ... its additional data
    LaplaceOperatorData<dim> laplace_additional_data;
    laplace_additional_data.bc = this->bc;
    laplace_additional_data.degree_mapping = fe_degree;
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_face_pairs;

    // run through all multigrid level
    for(unsigned int level = 0; level <= global_refinements; level++)
    {
      MatrixFree<dim, value_type> matrixfree_dg;
      AffineConstraints<double>   contraint_matrix_dg;
      do_reinit_multigrid(dof_handler_dg,
                          mapping,
                          laplace_additional_data,
                          mg_constrained_dofs_cg,
                          periodic_face_pairs,
                          level,
                          matrixfree_dg,
                          contraint_matrix_dg);
      laplace_dg.reinit(matrixfree_dg, contraint_matrix_dg, laplace_additional_data);

      MatrixFree<dim, value_type> matrixfree_cg;
      AffineConstraints<double>   contraint_matrix_cg;
      do_reinit_multigrid(dof_handler_cg,
                          mapping,
                          laplace_additional_data,
                          mg_constrained_dofs_cg,
                          periodic_face_pairs,
                          level,
                          matrixfree_cg,
                          contraint_matrix_cg);
      laplace_cg.reinit(matrixfree_cg, contraint_matrix_cg, laplace_additional_data);
      run(laplace_dg, laplace_cg, level);
    }

    // run on fine grid without multigrid
    {
      laplace_dg.reinit(data_dg, dummy_dg, laplace_additional_data);
      laplace_cg.reinit(data_cg, dummy_cg, laplace_additional_data);
      run(laplace_dg, laplace_cg);
    }

    // output convergence table
    if(!rank)
      convergence_table.write_text(std::cout);
  }
};

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  {
    pcout << "2D 1st-order:" << std::endl;
    Runner<2, 1> run_cg;
    run_cg.run();
    pcout << std::endl;
  }
  {
    pcout << "2D 2nd-order:" << std::endl;
    Runner<2, 2> run_cg;
    run_cg.run();
    pcout << std::endl;
  }
  {
    pcout << "2D 3rd-order:" << std::endl;
    Runner<2, 3> run_cg;
    run_cg.run();
    pcout << std::endl;
  }
}
