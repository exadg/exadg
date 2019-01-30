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
#ifdef DEAL_II_WITH_TRILINOS
#  include <deal.II/lac/trilinos_sparse_matrix.h>
#endif
#include <deal.II/multigrid/mg_base.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <deal.II/grid/manifold.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

// from navier
#include "include/poisson/spatial_discretization/laplace_operator.h"

#include "applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"

//
#include "../../operation-base-util/l2_norm.h"
#include "../../operation-base-util/sparse_matrix_util.h"
#include "include/rhs_operator.h"

#include "../../operation-base-util/operator_reinit_multigrid.h"

//#define DETAIL_OUTPUT
const int PATCHES = 10;

typedef double value_type;

using namespace dealii;
using namespace Poisson;


template<int dim, int fe_degree, typename FE_TYPE>
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
      dof_handler_dg(triangulation),
      mapping(fe_degree),
      quadrature(fe_degree + 1),
      global_refinements(dim == 2 ? 5 : 3)
  {
  }

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

private:
  MPI_Comm                                  comm;
  int                                       rank;
  parallel::distributed::Triangulation<dim> triangulation;
  FE_TYPE                                   fe_dgq;
  DoFHandler<dim>                           dof_handler_dg;
  ConvergenceTable                          convergence_table;
  MappingQGeneric<dim>                      mapping;
  QGauss<1>                                 quadrature;
  const unsigned int                        global_refinements;
  MatrixFree<dim, value_type>               data;
  std::shared_ptr<BoundaryDescriptor<dim>>  bc;
  MGConstrainedDoFs                         mg_constrained_dofs;
  AffineConstraints<double>                 dummy;

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

    LaplaceOperatorData<dim> laplace_additional_data;
    additional_data.mapping_update_flags = laplace_additional_data.mapping_update_flags;
    if(fe_dgq.dofs_per_vertex == 0)
    {
      additional_data.mapping_update_flags_inner_faces =
        laplace_additional_data.mapping_update_flags_inner_faces;
      additional_data.mapping_update_flags_boundary_faces =
        laplace_additional_data.mapping_update_flags_boundary_faces;
    }

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
    // determine level: -1 and globarl_refinements map to the same level
    unsigned int level = std::min(global_refinements, mg_level);

    // System matrix

#ifdef DEAL_II_WITH_TRILINOS
    TrilinosWrappers::SparseMatrix system_matrix;
    laplace.init_system_matrix(system_matrix);
    laplace.calculate_system_matrix(system_matrix);

#  ifdef DETAIL_OUTPUT
    print_ascii(system_matrix);
    print_matlab(system_matrix);
#  endif
#endif

    // Right hand side
    VectorType vec_rhs;
    laplace.initialize_dof_vector(vec_rhs);
    RHSOperator<dim, fe_degree, value_type> rhs(laplace.get_data());
    rhs.evaluate(vec_rhs);

#ifdef DETAIL_OUTPUT
    std::cout << "RHS: ";
    vec_rhs.print(std::cout);
#endif

    // Solve linear equation system: setup solution vectors
    VectorType vec_sol_sm;
    VectorType vec_sol_mf;

    // ... fill with zeroes
    laplace.initialize_dof_vector(vec_sol_sm);
    laplace.initialize_dof_vector(vec_sol_mf);

    // ... fill ghost values with zeroes
    vec_sol_sm.update_ghost_values();
    vec_sol_mf.update_ghost_values();

    // .. setup conjugate-gradient-solver
    SolverControl        solver_control(1000, 1e-12);
    SolverCG<VectorType> solver(solver_control);

    // ... solve with sparse matrix
    try
    {
#ifdef DEAL_II_WITH_TRILINOS
      solver.solve(system_matrix, vec_sol_sm, vec_rhs, PreconditionIdentity());
#endif
    }
    catch(SolverControl::NoConvergence &)
    {
      std::cout << "MB: not converved!" << std::endl;
    }
    const int last_step_sm = solver_control.last_step();

    // ... solve matrix-free
    try
    {
      solver.solve(laplace, vec_sol_mf, vec_rhs, PreconditionIdentity());
    }
    catch(SolverControl::NoConvergence &)
    {
      std::cout << "MF: not converved!" << std::endl;
    }
    const int last_step_mf = solver_control.last_step();

#ifdef DETAIL_OUTPUT
    std::cout << "SOL-MB: ";
    vec_sol_sm.print(std::cout);
    std::cout << "SOL-MF: ";
    vec_sol_mf.print(std::cout);
#endif

    // Perform vmult: setup vectors...
    VectorType vec_dst_sm;
    VectorType vec_dst_mf;
    VectorType vec_src;
    laplace.initialize_dof_vector(vec_dst_sm);
    laplace.initialize_dof_vector(vec_dst_mf);
    laplace.initialize_dof_vector(vec_src);

    // ... make source vector unique vector
    vec_src = 1.0;

    // ... zero out constrained entries in source vector
    auto bs = mg_constrained_dofs.get_boundary_indices(level);
    auto ls = vec_src.locally_owned_elements();
    auto gs = bs;
    gs.subtract_set(ls);
    bs.subtract_set(gs);
    for(auto i : bs)
      vec_src(i) = 0.0;

#ifdef DETAIL_OUTPUT
    std::cout << "X: ";
    vec_src.print(std::cout);
#endif

    // ... vmult with sparse matrix
#ifdef DEAL_II_WITH_TRILINOS
    system_matrix.vmult(vec_dst_sm, vec_src);
#endif
    // ... vmult matrix free
    laplace.vmult(vec_dst_mf, vec_src);

#ifdef DETAIL_OUTPUT
    std::cout << "Y-MB: ";
    vec_dst_sm.print(std::cout);
    std::cout << "X-MF: ";
    vec_dst_mf.print(std::cout);
#endif

    // Postprocessing: compare vmults, ...
    {
      vec_dst_mf -= vec_dst_sm;
      convergence_table.add_value("dim", dim);
      convergence_table.add_value("deg", fe_degree);
      convergence_table.add_value("lev", level);
      convergence_table.add_value("dofs", vec_rhs.size());
      convergence_table.add_value("vers", fe_dgq.dofs_per_vertex > 0);
      double n = vec_dst_mf.l2_norm();
      convergence_table.add_value("vmult-diff", n);
      convergence_table.set_scientific("vmult-diff", true);
    }

    // ... CG-solution: sparse matrix
    {
      L2Norm<dim, fe_degree, value_type> integrator(laplace.get_data());
      auto                               t = vec_sol_sm;
      t.update_ghost_values();
      double n = integrator.run(t);
      convergence_table.add_value("steps-sm", last_step_sm);
      convergence_table.add_value("int-sm", n);
      convergence_table.set_scientific("int-sm", true);
    }

    // ... CG-solution: matrix-free, and
    {
      L2Norm<dim, fe_degree, value_type> integrator(laplace.get_data());
      double                             n = integrator.run(vec_sol_mf);
      convergence_table.add_value("steps-mf", last_step_mf);
      convergence_table.add_value("int-mf", n);
      convergence_table.set_scientific("int-mf", true);
    }

    // ... output result to paraview
    if(mg_level == numbers::invalid_unsigned_int)
    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler_dg);

      data_out.add_data_vector(vec_sol_sm, "solution");

      auto vec_rank = vec_sol_sm;
      vec_rank      = rank;
      vec_rank.update_ghost_values();
      data_out.add_data_vector(vec_rank, "rank");
      data_out.build_patches(PATCHES);

      const std::string filename = "output/solution.vtu";
      data_out.write_vtu_in_parallel(filename.c_str(), comm);
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
    laplace_additional_data.bc             = this->bc;
    laplace_additional_data.degree_mapping = fe_degree;
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_face_pairs;

    // run through all multigrid level
    for(unsigned int level = 0; level <= global_refinements; level++)
    {
      MatrixFree<dim, value_type> matrixfree;
      AffineConstraints<double>   contraint_matrix;
      do_reinit_multigrid(dof_handler_dg,
                          mapping,
                          laplace_additional_data,
                          mg_constrained_dofs,
                          periodic_face_pairs,
                          level,
                          matrixfree,
                          contraint_matrix);

      laplace.reinit(matrixfree, contraint_matrix, laplace_additional_data);
      run(laplace, level);
    }

    // run on fine grid without multigrid
    {
      laplace.reinit(data, dummy, laplace_additional_data);
      run(laplace);
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
    Runner<2, 1, FE_Q<2>> run_cg;
    run_cg.run();
    Runner<2, 1, FE_DGQ<2>> run_dg;
    run_dg.run();
    pcout << std::endl;
  }

  {
    pcout << "2D 2nd-order:" << std::endl;
    Runner<2, 2, FE_Q<2>> run_cg;
    run_cg.run();
    Runner<2, 2, FE_DGQ<2>> run_dg;
    run_dg.run();
    pcout << std::endl;
  }

  {
    pcout << "3D 1st-order:" << std::endl;
    Runner<3, 1, FE_Q<3>> run_cg;
    run_cg.run();
    Runner<3, 1, FE_DGQ<3>> run_dg;
    run_dg.run();
    pcout << std::endl;
  }

  {
    pcout << "3D 2nd-order:" << std::endl;
    Runner<3, 2, FE_Q<3>> run_cg;
    run_cg.run();
    Runner<3, 2, FE_DGQ<3>> run_dg;
    run_dg.run();
    pcout << std::endl;
  }
}
