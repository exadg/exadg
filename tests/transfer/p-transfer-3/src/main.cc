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

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>
#include <vector>

using namespace dealii;

#include "../../../../include/poisson/spatial_discretization/laplace_operator.h"
#include "../../../operators/operation-base-util/l2_norm.h"

#include "../../../operators/operation-base-1/src/include/rhs_operator.h"

#include "../../../operators/operation-base-util/interpolate.h"

#include "../../../../applications/grid_tools/deformed_cube_manifold.h"
#include "../../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_c.h"
#include "../../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_p.h"

#include "../../../../include/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../../../../include/solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"

//#define DETAIL_OUTPUT
const int PATCHES = 10;

typedef double value_type;

using namespace Poisson;

template<int dim>
class ExactSolution : public Function<dim>
{
public:
  ExactSolution(const double time = 0.) : Function<dim>(1, time)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const
  {
    double result = std::sin(p[0] * numbers::PI);
    for(unsigned int d = 1; d < dim; ++d)
      result *= std::sin((p[d] + d) * numbers::PI);
    return std::abs(result + 1);
    //        return p[0];
  }
};



template<int dim, int fe_degree, typename FE_TYPE>
class Runner
{
public:
  Runner(ConvergenceTable & convergence_table, MultigridType mg_type)
    : comm(MPI_COMM_WORLD),
      rank(get_rank(comm)),
      convergence_table(convergence_table),
      mg_type(mg_type),
      triangulation(comm,
                    dealii::Triangulation<dim>::none,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe_1(fe_degree),
      dof_handler_1(triangulation),
      mapping_1(fe_degree),
      quadrature_1(fe_degree + 1),
      global_refinements(dim == 2 ? 5 : 3)
  // global_refinements(dim == 2 ? 10 : 5)
  {
  }

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

public:
  MPI_Comm comm;
  int      rank;

  ConvergenceTable & convergence_table;
  MultigridType      mg_type;

  parallel::distributed::Triangulation<dim> triangulation;
  FE_TYPE                                   fe_1;
  DoFHandler<dim>                           dof_handler_1;
  MappingQGeneric<dim>                      mapping_1;

  AffineConstraints<double> dummy_1;

  QGauss<1>                                quadrature_1;
  const unsigned int                       global_refinements;
  MatrixFree<dim, value_type>              data_1;
  std::shared_ptr<BoundaryDescriptor<dim>> bc;

  MGConstrainedDoFs mg_constrained_dofs_1;
  MGConstrainedDoFs mg_constrained_dofs_2;

private:
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
    typename MatrixFree<dim, value_type>::AdditionalData additional_data_1;
    LaplaceOperatorData<dim>                             laplace_additional_data;
    additional_data_1.mapping_update_flags = laplace_additional_data.mapping_update_flags;
    if(fe_1.dofs_per_vertex == 0)
    {
      additional_data_1.mapping_update_flags_inner_faces =
        laplace_additional_data.mapping_update_flags_inner_faces;
      additional_data_1.mapping_update_flags_boundary_faces =
        laplace_additional_data.mapping_update_flags_boundary_faces;
    }

    dummy_1.clear();

    if(fe_1.dofs_per_vertex != 0)
    {
      std::set<types::boundary_id> dirichlet_boundary;
      for(auto it : this->bc->dirichlet_bc)
        dirichlet_boundary.insert(it.first);
      mg_constrained_dofs_1.initialize(dof_handler_1);
      mg_constrained_dofs_1.make_zero_boundary_constraints(dof_handler_1, dirichlet_boundary);
      dummy_1.add_lines(mg_constrained_dofs_1.get_boundary_indices(global_refinements));
    }

    data_1.reinit(mapping_1, dof_handler_1, dummy_1, quadrature_1, additional_data_1);
  }

  LinearAlgebra::distributed::Vector<value_type>
  run(LaplaceOperator<dim, fe_degree, value_type> & laplace_1, VectorType v_in)
  {
    int procs, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    LinearAlgebra::distributed::Vector<value_type> vec_rhs, vec_sol;

    data_1.initialize_dof_vector(vec_rhs);
    data_1.initialize_dof_vector(vec_sol);

    CGSolverData solver_data;
    solver_data.solver_tolerance_abs = 1e-9;
    solver_data.solver_tolerance_rel = 1e-8;
    solver_data.max_iter             = 1000;
    solver_data.use_preconditioner   = true;

    MultigridData multigrid_data;
    multigrid_data.smoother = MultigridSmoother::Chebyshev;
    // MG smoother data
    multigrid_data.gmres_smoother_data.preconditioner       = PreconditionerGMRESSmoother::None;
    multigrid_data.gmres_smoother_data.number_of_iterations = 5;
    // MG coarse grid solver
    if(mg_type == MultigridType::pMG)
    {
      multigrid_data.coarse_solver = MultigridCoarseGridSolver::AMG_ML; // GMRES_PointJacobi;
      multigrid_data.type          = MultigridType::pMG;
    }
    else
    {
      multigrid_data.coarse_solver = MultigridCoarseGridSolver::PCG_PointJacobi;
      multigrid_data.type          = MultigridType::hMG;
    }

    multigrid_data.coarse_ml_data.use_conjugate_gradient_solver = true;

    multigrid_data.c_transfer_back  = true;
    multigrid_data.c_transfer_front = false;
    multigrid_data.p_sequence = PSequenceType::BISECTION; // GO_TO_ONE, DECREASE_BY_ONE, BISECTION

    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_face_pairs;

    typedef float                                            Number;
    typedef Poisson::LaplaceOperator<dim, fe_degree, Number> MG_OPERATOR;

    parallel::Triangulation<dim> const * tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler_1.get_triangulation());
    const FiniteElement<dim> & fe = dof_handler_1.get_fe();

    MultigridPreconditionerBase<dim, value_type, Number> preconditioner(
      std::shared_ptr<MG_OPERATOR>(new MG_OPERATOR()));
    preconditioner.initialize(multigrid_data,
                              tria,
                              fe,
                              mapping_1,
                              laplace_1.get_operator_data(),
                              &laplace_1.get_operator_data().bc->dirichlet_bc,
                              &periodic_face_pairs);

    CGSolver<Poisson::LaplaceOperator<dim, fe_degree, value_type>,
             PreconditionerBase<value_type>,
             VectorType>
      solver(laplace_1, preconditioner, solver_data);

    RHSOperator<dim, fe_degree, value_type> rhs(laplace_1.get_data());
    rhs.evaluate(vec_rhs);

    if(v_in.size() != 0)
    {
      auto temp = v_in;
      temp      = 0.0;
      laplace_1.vmult(temp, v_in);
      vec_rhs -= temp;

      convergence_table.add_value("vers", 2);
    }
    else
    {
      convergence_table.add_value("vers", fe_1.dofs_per_vertex != 0);
    }


    Timer time;
    time.restart();
    auto   n    = solver.solve(vec_sol, vec_rhs, false);
    double temp = time.wall_time();

    if(!rank)
      printf("%10.5f\n", temp);

    auto t = vec_sol;
    {
      convergence_table.add_value("procs", procs);
      convergence_table.add_value("dim", dim);
      convergence_table.add_value("deg1", fe_degree);
      convergence_table.add_value("lev", global_refinements);
      convergence_table.add_value("dofs", vec_sol.size());
      L2Norm<dim, fe_degree, value_type> l2norm(laplace_1.get_data());
      convergence_table.add_value("norm", vec_sol.l2_norm());
      convergence_table.set_scientific("norm", true);
      convergence_table.add_value("fine1", l2norm.run(vec_sol));
      convergence_table.set_scientific("fine1", true);
      convergence_table.add_value("n", n);
      convergence_table.add_value("time", temp);
      convergence_table.set_scientific("time", true);
    }

    return t;
  }

public:
  LinearAlgebra::distributed::Vector<value_type>
  run(VectorType vt = VectorType())
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

    laplace.reinit(data_1, dummy_1, laplace_additional_data);
    return run(laplace, vt);
  }
};

template<int dim>
void
run(ConvergenceTable & convergence_table)
{
  std::vector<MultigridType> types = {MultigridType::hMG, MultigridType::pMG};

  for(auto t : types)
  {
    if(true)
    {
      Runner<dim, 7, FE_DGQ<dim>> run_cg(convergence_table, t);
      run_cg.run();
    }

    if(true)
    {
      Runner<dim, 7, FE_Q<dim>> run_cg(convergence_table, t);
      run_cg.run();
    }

    if(true)
    {
      ConvergenceTable                               convergence_table_dummy;
      Runner<dim, 7, FE_Q<dim>>                      run_cg(convergence_table_dummy, t);
      Runner<dim, 7, FE_DGQ<dim>>                    run_dg_1(convergence_table_dummy, t);
      LinearAlgebra::distributed::Vector<value_type> result_cg = run_cg.run();
      run_dg_1.run();

      MGTransferMFC<dim, value_type> transfer(run_dg_1.data_1,
                                              run_cg.data_1,
                                              run_dg_1.dummy_1,
                                              run_cg.dummy_1,
                                              run_cg.global_refinements,
                                              7);

      LinearAlgebra::distributed::Vector<value_type> result_dg;
      result_dg = 0.0;
      run_dg_1.data_1.initialize_dof_vector(result_dg);
      transfer.prolongate(0, result_dg, result_cg);

      Runner<dim, 7, FE_DGQ<dim>> run_dg_2(convergence_table, t);
      run_dg_2.run(result_dg);
    }
  }
}

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  int                rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  {
    ConvergenceTable convergence_table;

    run<2>(convergence_table);
    run<3>(convergence_table);

    if(!rank)
      convergence_table.write_text(std::cout);
    pcout << std::endl;
  }
}