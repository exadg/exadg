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
#include <vector>

#include "../../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_c.h"
#include "../../../operators/operation-base-util/l2_norm.h"

#include "../../../operators/operation-base-util/interpolate.h"

#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"

//#define DETAIL_OUTPUT
const int          PATCHES              = 10;
const unsigned int n_global_refinements = 4;

typedef double Number;

using namespace dealii;

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

template<int dim, int fe_degree>
class Runner
{
public:
  Runner(ConditionalOStream & pcout, ConvergenceTable & convergence_table)
    : pcout(pcout),
      convergence_table(convergence_table),
      triangulation(MPI_COMM_WORLD,
                    dealii::Triangulation<dim>::none,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe_q(fe_degree),
      fe_dgq(fe_degree),
      mapping(fe_degree),
      quadrature(fe_degree + 1),
      dof_handler_cg(triangulation),
      dof_handler_dg(triangulation)
  {
  }

  void
  run()
  {
    setup_grid_and_dofs();

    std::vector<unsigned int> levels;
    for(unsigned int i = 0; i <= n_global_refinements; i++)
      levels.push_back(i);
    levels.push_back(numbers::invalid_unsigned_int);

    for(auto level : levels)
    {
      setup_mf(level);
      run_1(level);
      run_2(level);
    }
  }

private:
  void
  run_1(unsigned int level)
  {
    MGTransferMFC<dim, Number> transfer(data_dg, data_cg, cm, cm, level, fe_degree);

    LinearAlgebra::distributed::Vector<Number> vector_cg;
    LinearAlgebra::distributed::Vector<Number> vector_dg;

    data_cg.initialize_dof_vector(vector_cg);
    data_dg.initialize_dof_vector(vector_dg);

    MGTools::interpolate(dof_handler_dg, ExactSolution<dim>(), vector_dg, level);

    transfer.restrict_and_add(0, vector_cg, vector_dg);

    auto norm_cg = vector_cg.mean_value() * vector_cg.size();
    auto norm_dg = vector_dg.mean_value() * vector_dg.size();

    convergence_table.add_value("dofs_cg", vector_cg.size());
    convergence_table.add_value("dofs_dg", vector_dg.size());
    convergence_table.add_value("test1_norm", norm_dg);
    convergence_table.set_scientific("test1_norm", true);
    convergence_table.add_value("test1_err", std::abs(norm_dg - norm_cg));
    convergence_table.set_scientific("test1_err", true);

    if(level == n_global_refinements || level == numbers::invalid_unsigned_int)
    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler_cg);
      vector_cg.update_ghost_values();
      data_out.add_data_vector(vector_cg, "solution_1");
      data_out.build_patches(PATCHES);

      std::ofstream output_pressure("solution_1.1.vtu");
      data_out.write_vtu(output_pressure);
    }

    if(level == n_global_refinements || level == numbers::invalid_unsigned_int)
    {
      DataOut<dim> data_out;
      data_out.attach_dof_handler(dof_handler_dg);
      vector_dg.update_ghost_values();
      vector_dg.update_ghost_values();
      data_out.add_data_vector(vector_dg, "solution_1");
      data_out.build_patches(PATCHES);

      std::ofstream output_pressure("solution_1.2.vtu");
      data_out.write_vtu(output_pressure);
    }
  }

  void
  run_2(unsigned int level)
  {
    MGTransferMFC<dim, Number> transfer(data_dg, data_cg, cm, cm, level, fe_degree);

    LinearAlgebra::distributed::Vector<Number> vector_cg;
    LinearAlgebra::distributed::Vector<Number> vector_dg;

    data_cg.initialize_dof_vector(vector_cg);
    vector_cg = 0.0;
    data_dg.initialize_dof_vector(vector_dg);
    vector_dg = 0.0;

    MGTools::interpolate(dof_handler_cg, ExactSolution<dim>(), vector_cg, level);

    transfer.prolongate(0, vector_dg, vector_cg);

    double norm_cg, norm_dg;
    {
      auto t = vector_cg;
      t.update_ghost_values();
      L2Norm<dim, fe_degree, Number> integrator(data_cg);
      norm_cg = integrator.run(t);

      if(level == n_global_refinements || level == numbers::invalid_unsigned_int)
      {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler_cg);
        vector_cg.update_ghost_values();
        data_out.add_data_vector(vector_cg, "solution");
        data_out.build_patches(PATCHES);

        std::ofstream output_pressure("solution.1.vtu");
        data_out.write_vtu(output_pressure);
      }
    }

    {
      auto t = vector_dg;
      t.update_ghost_values();
      L2Norm<dim, fe_degree, Number> integrator(data_dg);
      norm_dg = integrator.run(t);

      if(level == n_global_refinements || level == numbers::invalid_unsigned_int)
      {
        DataOut<dim> data_out;
        data_out.attach_dof_handler(dof_handler_dg);
        data_out.add_data_vector(vector_dg, "solution");
        data_out.build_patches(PATCHES);

        std::ofstream output_pressure("solution.2.vtu");
        data_out.write_vtu(output_pressure);
      }
    }

    convergence_table.add_value("test2_norm", norm_cg);
    convergence_table.set_scientific("test2_norm", true);
    convergence_table.add_value("test2_err", std::abs(norm_dg - norm_cg));
    convergence_table.set_scientific("test2_err", true);
  }

  void
  setup_grid_and_dofs()
  {
    // create triangulation
    const double left        = -1.0;
    const double right       = +1.0;
    const double deformation = +0.1;
    const double frequnency  = +2.0;
    GridGenerator::hyper_cube(triangulation, left, right);
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequnency);
    triangulation.set_all_manifold_ids(1);
    triangulation.set_manifold(1, manifold);
    triangulation.refine_global(n_global_refinements);

    // create dof_handler
    this->dof_handler_cg.distribute_dofs(fe_q);
    this->dof_handler_cg.distribute_mg_dofs();
    this->dof_handler_dg.distribute_dofs(fe_dgq);
    this->dof_handler_dg.distribute_mg_dofs();
  }

  void
  setup_mf(unsigned int level)
  {
    // create matrix-free
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.level_mg_handler = level;
    data_cg.reinit(mapping, dof_handler_cg, cm, quadrature, additional_data);
    data_dg.reinit(mapping, dof_handler_dg, cm, quadrature, additional_data);

    convergence_table.add_value("dim", dim);
    convergence_table.add_value("deg", fe_degree);
    convergence_table.add_value("refs", n_global_refinements);
    convergence_table.add_value("lev", std::min(level, n_global_refinements));
  }

  ConditionalOStream &                      pcout;
  ConvergenceTable &                        convergence_table;
  parallel::distributed::Triangulation<dim> triangulation;
  FE_Q<dim>                                 fe_q;
  FE_DGQ<dim>                               fe_dgq;
  MappingQGeneric<dim>                      mapping;
  QGauss<1>                                 quadrature;

  DoFHandler<dim> dof_handler_cg, dof_handler_dg;

  MatrixFree<dim, Number> data_cg;
  MatrixFree<dim, Number> data_dg;

  AffineConstraints<double> cm;
};

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  int                rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  ConvergenceTable convergence_table;

  {
    Runner<2, 1> r(pcout, convergence_table);
    r.run();
  }
  {
    Runner<2, 2> r(pcout, convergence_table);
    r.run();
  }
  {
    Runner<2, 3> r(pcout, convergence_table);
    r.run();
  }

  if(!rank)
    convergence_table.write_text(std::cout);
}
