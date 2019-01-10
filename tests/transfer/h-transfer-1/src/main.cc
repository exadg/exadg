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
#include <deal.II/lac/trilinos_sparse_matrix.h>
#include <deal.II/multigrid/mg_base.h>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include <deal.II/base/mg_level_object.h>
#include <bits/stl_vector.h>

//#define DETAIL_OUTPUT
const int PATCHES = 10;

typedef double value_type;

using namespace dealii;

#include "../../../operators/operation-base-util/interpolate.h"
#include "../../../operators/operation-base-util/l2_norm.h"

#include "../../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_mg_level_object.h"

template<int dim, typename DoFHandlerType = DoFHandler<dim>>
class MGDataOut : public DataOut<dim, DoFHandlerType>
{
public:
  MGDataOut(unsigned int level) : level(level)
  {
  }

private:
  typename DataOut<dim, DoFHandlerType>::cell_iterator
  first_cell()
  {
    // return this->triangulation->begin_active();

    return this->triangulation->begin(level);
    // typename DoFHandler<dim>::cell_iterator endc=dof_handler.end(level-1);
  }

  typename DataOut<dim, DoFHandlerType>::cell_iterator
  next_cell(const typename DataOut<dim, DoFHandlerType>::cell_iterator & cell)
  {
    //     // convert the iterator to an active_iterator and advance this to the next
    //     // active cell
    //     typename Triangulation<DoFHandlerType::dimension,
    //                            DoFHandlerType::space_dimension>::active_cell_iterator
    //       active_cell = cell;
    //     ++active_cell;
    //     return active_cell;
    typename Triangulation<DoFHandlerType::dimension,
                           DoFHandlerType::space_dimension>::cell_iterator active_cell = cell;

    if(cell == this->triangulation->end(level))
      return this->dofs->end();

    ++active_cell;
    return active_cell;
  }

  unsigned int level;
};


template<int dim>
class TestSolution : public Function<dim>
{
public:
  TestSolution(unsigned int dir, const double time = 0.) : Function<dim>(1, time), dir(dir)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const
  {
    return p[dir];
    // double result = std::sin(wave_number * p[0] * numbers::PI);
    // for(unsigned int d = 1; d < dim; ++d)
    //  result *= std::sin(wave_number * p[d] * numbers::PI);
    // return result;
  }

  const unsigned int dir;
};


template<int dim, int fe_degree_1>
class Runner
{
public:
  Runner(bool use_dg, unsigned int dir, ConvergenceTable & convergence_table)
    : use_dg(use_dg),
      dir(dir),
      convergence_table(convergence_table),
      triangulation(MPI_COMM_WORLD,
                    dealii::Triangulation<dim>::none,
                    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy),
      fe(use_dg ? new FESystem<dim>(FE_DGQ<dim>(fe_degree_1), 1) :
                  new FESystem<dim>(FE_Q<dim>(fe_degree_1), 1)),
      dof_handler(new DoFHandler<dim>(triangulation)),
      mapping_1(fe_degree_1),
      quadrature_1(fe_degree_1 + 1),
      global_refinements(dim == 2 ? 4 : 3)
  {
  }

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

private:
  bool                                      use_dg;
  unsigned int                              dir;
  ConvergenceTable &                        convergence_table;
  parallel::distributed::Triangulation<dim> triangulation;
  std::shared_ptr<FESystem<dim>>            fe;
  std::shared_ptr<DoFHandler<dim>>          dof_handler;
  MappingQGeneric<dim>                      mapping_1;

  MGLevelObject<std::shared_ptr<AffineConstraints<double>>> dummy_1;

  QGauss<1>          quadrature_1;
  const unsigned int global_refinements;

  MGLevelObject<std::shared_ptr<MatrixFree<dim, value_type>>> data_1;
  MGLevelObject<VectorType>                                   vectors;


  void
  init_triangulation_and_dof_handler()
  {
    const double left  = -1.0;
    const double right = +1.0;

    GridGenerator::hyper_cube(triangulation, left, right);
    triangulation.refine_global(global_refinements);

    dof_handler->distribute_dofs(*fe);
    dof_handler->distribute_mg_dofs();
  }

  void
  init_boundary_conditions()
  {
    // TODO
  }

  void
  init_matrixfree_and_constraint_matrix()
  {
    dummy_1.resize(0, global_refinements);
    data_1.resize(0, global_refinements);

    for(unsigned int level = 0; level <= global_refinements; level++)
    {
      dummy_1[level].reset(new AffineConstraints<double>);
      dummy_1[level]->clear();

      data_1[level].reset(new MatrixFree<dim, value_type>);
      typename MatrixFree<dim, value_type>::AdditionalData additional_data_1;
      additional_data_1.mapping_update_flags =
        update_gradients | update_JxW_values | update_quadrature_points;
      if(use_dg)
      {
        additional_data_1.mapping_update_flags_inner_faces =
          additional_data_1.mapping_update_flags | update_values | update_normal_vectors;
        additional_data_1.mapping_update_flags_boundary_faces =
          additional_data_1.mapping_update_flags_inner_faces | update_quadrature_points;
      }
      additional_data_1.level_mg_handler = level;
      data_1[level]->reinit(
        mapping_1, *dof_handler, *dummy_1[level], quadrature_1, additional_data_1);
    }
  }

  void
  init_vectors(unsigned int                                                  min_level,
               unsigned int                                                  max_level,
               MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> &       mg_dofhandler,
               MGLevelObject<std::shared_ptr<MatrixFree<dim, value_type>>> & data_1)
  {
    vectors.resize(min_level, max_level);
    for(unsigned int level = min_level; level <= max_level; level++)
      data_1[level]->initialize_dof_vector(vectors[level]);

    MGTools::interpolate(*mg_dofhandler[max_level],
                         TestSolution<dim>(dir, 0),
                         vectors[max_level],
                         numbers::invalid_unsigned_int);
  }

  void
  setup_sequence(std::vector<MGLevelIdentifier> &      global_levels,
                 std::vector<MGDofHandlerIdentifier> & p_levels)
  {
    for(unsigned int i = 0; i <= global_refinements; i++)
      global_levels.push_back({i, fe_degree_1, true});

    for(auto i : global_levels)
      p_levels.push_back(i.id);

    sort(p_levels.begin(), p_levels.end());
    p_levels.erase(unique(p_levels.begin(), p_levels.end()), p_levels.end());
    std::reverse(std::begin(p_levels), std::end(p_levels));

    //  for(unsigned int i = 1; i < global_levels.size(); i++)
    //  {
    //      if(p_levels.back().degree!=global_levels[i].degree ||
    //      p_levels.back().degree!=global_levels[i].is_dg)
    //    p_levels.push_back({fe_degree_1, true});
    //  }
  }

public:
  void
  run()
  {
    std::vector<MGLevelIdentifier>      global_levels;
    std::vector<MGDofHandlerIdentifier> p_levels;

    setup_sequence(global_levels, p_levels);
    unsigned int min_level = 0;
    unsigned int max_level = global_levels.size() - 1;

    // initialize the system
    init_triangulation_and_dof_handler();
    init_boundary_conditions();
    init_matrixfree_and_constraint_matrix();

    MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> mg_dofhandler;
    MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>     mg_constrained_dofs;

    mg_dofhandler.resize(0, global_refinements);
    for(unsigned int i = 0; i <= global_refinements; i++)
      mg_dofhandler[i] = dof_handler;

    std::shared_ptr<MGConstrainedDoFs> constrained_dofs(new MGConstrainedDoFs);
    constrained_dofs->initialize(*dof_handler);
    mg_constrained_dofs.resize(0, global_refinements);
    for(unsigned int i = 0; i <= global_refinements; i++)
      mg_constrained_dofs[i] = constrained_dofs;

    init_vectors(min_level, max_level, mg_dofhandler, data_1);

    // create transfer-operator
    MGTransferMF_MGLevelObject<dim, VectorType> transfer;
    transfer.template reinit<value_type>(
      1, 0, global_levels, p_levels, data_1, dummy_1, mg_dofhandler, mg_constrained_dofs);

    // interpolate solution on the fines grid onto coarse grids
    for(unsigned int level = max_level; level >= 1 + min_level; level--)
      transfer.interpolate(level, vectors[level - 1], vectors[level]);

    for(unsigned int level = min_level; level <= max_level; level++)
    {
#ifdef PRINT
      L2Norm<dim, fe_degree_1, value_type> norm(*data_1[level]);
      std::cout << level << ": " << norm.run(vectors[level]) << " " << vectors[level].l2_norm()
                << std::endl;
#endif

      value_type accumulator = 0;

      FEEvaluation<dim, fe_degree_1, fe_degree_1 + 1, 1, value_type> fe_eval(*data_1[level], 0);

      for(unsigned int cell = 0; cell < data_1[level]->n_macro_cells(); ++cell)
      {
        fe_eval.reinit(cell);
        fe_eval.gather_evaluate(vectors[level], true, false);

        for(unsigned int i = 0; i < fe_eval.static_dofs_per_cell; i++)
        {
          auto point_real = fe_eval.quadrature_point(i)[dir];
          auto point_ref  = fe_eval.begin_values()[i];

          unsigned int const n_filled_lanes = data_1[level]->n_active_entries_per_cell_batch(cell);
#ifdef PRINT
          for(unsigned int v = 0; v < n_filled_lanes; v++)
            printf("%10.5f", point_real[v]);
          for(unsigned int v = n_filled_lanes; v < VectorizedArray<value_type>::n_array_elements;
              v++)
            printf("          ");
          printf("     ");
          for(unsigned int v = 0; v < n_filled_lanes; v++)
            printf("%10.5f", point_ref[v]);
          for(unsigned int v = n_filled_lanes; v < VectorizedArray<value_type>::n_array_elements;
              v++)
            printf("          ");
          printf("\n");
#else
          auto diff = point_ref - point_real;
          diff *= diff;
          for(unsigned int v = 0; v < n_filled_lanes; v++)
            accumulator += diff[v];

#endif
        }
      }

      std::string label = std::string(use_dg ? "_dg_" : "_cg_") + std::to_string(dir);
      convergence_table.add_value(std::string("level") + label, level);
      convergence_table.add_value(std::string("dofs_1") + label, vectors[level].size());
      convergence_table.add_value(std::string("err") + label, sqrt(accumulator));
      convergence_table.set_scientific(std::string("err") + label, true);
      // printf("%10.5e\n", sqrt(accumulator));

#ifdef PRINT
      printf("\n\n");
#endif

      /*
      MGDataOut<dim> data_out(level);
      data_out.attach_dof_handler(dof_handler);

      data_out.add_data_vector(vectors[global_refinements], "solution");
      data_out.build_patches(fe_degree_1);

      const std::string filename = "output/solution." + std::to_string(level) + ".vtu";
      data_out.write_vtu_in_parallel(filename.c_str(), MPI_COMM_WORLD);
       */
    }
  }
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
    Runner<2, 3> run_dg(true, 0, convergence_table);
    run_dg.run();
  }
  {
    Runner<2, 3> run_dg(true, 1, convergence_table);
    run_dg.run();
  }
  {
    Runner<2, 3> run_cg(false, 0, convergence_table);
    run_cg.run();
  }
  {
    Runner<2, 3> run_cg(false, 1, convergence_table);
    run_cg.run();
  }

  if(!rank)
    convergence_table.write_text(std::cout);
}
