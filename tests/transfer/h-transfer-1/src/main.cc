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

#include <deal.II/base/mg_level_object.h>
#include <bits/stl_vector.h>

//#define DETAIL_OUTPUT
const int          PATCHES   = 10;
const unsigned int fe_degree = 7;

//#define PRINT

typedef double value_type;

using namespace dealii;

#include "../../../operators/operation-base-util/interpolate.h"
#include "../../../operators/operation-base-util/l2_norm.h"

#include "../../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_mg_level_object.h"

enum RunConfiguration
{
  h_coarsening,
  p_coarsening,
  c_coarsening
};

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
  Runner(bool               use_dg,
         unsigned int       dir,
         ConvergenceTable & convergence_table,
         RunConfiguration   run_configuration)
    : use_dg(use_dg),
      dir(dir),
      convergence_table(convergence_table),
      global_refinements(dim == 2 ? 4 : 3),
      run_configuration(run_configuration)
  {
  }

  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

private:
  bool               use_dg;
  unsigned int       dir;
  ConvergenceTable & convergence_table;
  const unsigned int global_refinements;
  RunConfiguration   run_configuration;

  MGLevelObject<VectorType> vectors;


  void
  init_triangulation_and_dof_handler(
    MGLevelInfo                                                  id,
    std::shared_ptr<parallel::distributed::Triangulation<dim>> & triangulation,
    std::shared_ptr<FESystem<dim>> &                             fe,
    std::shared_ptr<const DoFHandler<dim>> &                     dof_handler,
    std::shared_ptr<MappingQGeneric<dim>> &                      mapping,
    std::shared_ptr<QGauss<1>> &                                 quadrature)
  {
    triangulation.reset(new parallel::distributed::Triangulation<dim>(
      MPI_COMM_WORLD,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));

    fe.reset(id.is_dg ? new FESystem<dim>(FE_DGQ<dim>(id.degree), 1) :
                        new FESystem<dim>(FE_Q<dim>(id.degree), 1));
    mapping.reset(new MappingQGeneric<dim>(id.degree));
    quadrature.reset(new QGauss<1>(id.degree + 1));

    const double left  = -1.0;
    const double right = +1.0;

    GridGenerator::hyper_cube(*triangulation, left, right);
    triangulation->refine_global(global_refinements);

    auto temp = new DoFHandler<dim>(*triangulation);
    temp->distribute_dofs(*fe);
    temp->distribute_mg_dofs();
    dof_handler.reset(temp);
  }

  void
  init_boundary_conditions()
  {
    // TODO
  }

  void
  init_matrixfree_and_constraint_matrix(MGLevelInfo                                    id,
                                        std::shared_ptr<AffineConstraints<double>> &   dummy,
                                        std::shared_ptr<MatrixFree<dim, value_type>> & data,
                                        std::shared_ptr<const DoFHandler<dim>> &       dof_handler,
                                        std::shared_ptr<MappingQGeneric<dim>> &        mapping,
                                        std::shared_ptr<QGauss<1>> &                   quadrature)
  {
    dummy.reset(new AffineConstraints<double>);
    dummy->clear();

    data.reset(new MatrixFree<dim, value_type>);
    typename MatrixFree<dim, value_type>::AdditionalData additional_data_1;
    additional_data_1.mapping_update_flags =
      update_gradients | update_JxW_values | update_quadrature_points;
    if(id.is_dg)
    {
      additional_data_1.mapping_update_flags_inner_faces =
        additional_data_1.mapping_update_flags | update_values | update_normal_vectors;
      additional_data_1.mapping_update_flags_boundary_faces =
        additional_data_1.mapping_update_flags_inner_faces | update_quadrature_points;
    }
    additional_data_1.level_mg_handler = id.level;
    data->reinit(*mapping, *dof_handler, *dummy, *quadrature, additional_data_1);
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
  setup_sequence(std::vector<MGLevelInfo> & global_levels)
  {
    if(run_configuration == RunConfiguration::h_coarsening)
    {
      for(unsigned int i = 0; i <= global_refinements; i++)
        global_levels.push_back({i, fe_degree_1, use_dg});
    }
    else if(run_configuration == RunConfiguration::p_coarsening)
    {
      unsigned int temp_degree = fe_degree_1;
      while(temp_degree != 0)
      {
        global_levels.insert(global_levels.begin(), {global_refinements, temp_degree, use_dg});
        temp_degree /= 2;
      }
    }
    else if((run_configuration == RunConfiguration::c_coarsening))
    {
      global_levels.push_back({global_refinements, fe_degree_1, false});
      global_levels.push_back({global_refinements, fe_degree_1, true});
    }
    else
      AssertThrow(false, ExcMessage("This run configuration is not implemented!"));
  }

public:
  void
  run()
  {
    std::vector<MGLevelInfo> global_levels;

    setup_sequence(global_levels);
    unsigned int min_level = 0;
    unsigned int max_level = global_levels.size() - 1;

    MGLevelObject<std::shared_ptr<parallel::distributed::Triangulation<dim>>> mg_triangulation;
    MGLevelObject<std::shared_ptr<FESystem<dim>>>                             mg_fe;
    MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>                     mg_dofhandler;
    MGLevelObject<std::shared_ptr<MappingQGeneric<dim>>>                      mg_mapping;
    MGLevelObject<std::shared_ptr<QGauss<1>>>                                 mg_quadrature;
    MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>                         mg_constrained_dofs;
    MGLevelObject<std::shared_ptr<MatrixFree<dim, value_type>>>               mg_data;
    MGLevelObject<std::shared_ptr<AffineConstraints<double>>>                 mg_dummy;

    mg_triangulation.resize(min_level, max_level);
    mg_fe.resize(min_level, max_level);
    mg_dofhandler.resize(min_level, max_level);
    mg_mapping.resize(min_level, max_level);
    mg_quadrature.resize(min_level, max_level);
    mg_constrained_dofs.resize(min_level, max_level);
    mg_data.resize(min_level, max_level);
    mg_dummy.resize(min_level, max_level);

    for(unsigned int level = min_level; level <= max_level; level++)
    {
      init_triangulation_and_dof_handler(global_levels[level],
                                         mg_triangulation[level],
                                         mg_fe[level],
                                         mg_dofhandler[level],
                                         mg_mapping[level],
                                         mg_quadrature[level]);

      init_boundary_conditions();

      init_matrixfree_and_constraint_matrix(global_levels[level],
                                            mg_dummy[level],
                                            mg_data[level],
                                            mg_dofhandler[level],
                                            mg_mapping[level],
                                            mg_quadrature[level]);

      std::shared_ptr<MGConstrainedDoFs> constrained_dofs(new MGConstrainedDoFs);
      constrained_dofs->initialize(*mg_dofhandler[level]);
      mg_constrained_dofs[level] = constrained_dofs;
    }

    init_vectors(min_level, max_level, mg_dofhandler, mg_data);

    // create transfer-operator
    MGTransferMF_MGLevelObject<dim, VectorType> transfer;
    transfer.template reinit<value_type>(mg_data, mg_dummy, mg_constrained_dofs);

    // interpolate solution on the fines grid onto coarse grids
    for(unsigned int level = max_level; level >= 1 + min_level; level--)
      transfer.interpolate(level, vectors[level - 1], vectors[level]);

    for(unsigned int level = min_level; level <= max_level; level++)
    {
#ifdef PRINT
      L2Norm<dim, fe_degree_1, value_type> norm(*mg_data[level]);
      std::cout << level << ": " << norm.run(vectors[level]) << " " << vectors[level].l2_norm()
                << std::endl;
#endif

      value_type accumulator = 0;

      switch(global_levels[level].degree)
      {
        case 1:
          accumulator = check<1>(*mg_data[level], vectors[level]);
          break;
        case 3:
          accumulator = check<3>(*mg_data[level], vectors[level]);
          break;
        case 7:
          accumulator = check<7>(*mg_data[level], vectors[level]);
          break;
        // error:
        default:
          AssertThrow(false, ExcMessage("Not implemented! Just extend jump table!"));
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

  template<int fe_degree>
  value_type
  check(MatrixFree<dim, value_type> & data, VectorType & vector)
  {
    value_type accumulator = 0;


    FEEvaluation<dim, fe_degree, fe_degree + 1, 1, value_type> fe_eval(data, 0);
    vector.update_ghost_values();

    for(unsigned int cell = 0; cell < data.n_macro_cells(); ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.gather_evaluate(vector, true, false);

      for(unsigned int i = 0; i < fe_eval.static_dofs_per_cell; i++)
      {
        auto point_real = fe_eval.quadrature_point(i)[dir];
        auto point_ref  = fe_eval.begin_values()[i];

        unsigned int const n_filled_lanes = data.n_active_entries_per_cell_batch(cell);
#ifdef PRINT
        for(unsigned int v = 0; v < n_filled_lanes; v++)
          printf("%10.5f", point_real[v]);
        for(unsigned int v = n_filled_lanes; v < VectorizedArray<value_type>::n_array_elements; v++)
          printf("          ");
        printf("     ");
        for(unsigned int v = 0; v < n_filled_lanes; v++)
          printf("%10.5f", point_ref[v]);
        for(unsigned int v = n_filled_lanes; v < VectorizedArray<value_type>::n_array_elements; v++)
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
    return accumulator;
  }
};

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  int                rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  ConvergenceTable convergence_table_h, convergence_table_p, convergence_table_c;

  if(true) // h-coarsening
  {
    { // CG + dir=0
      Runner<2, fe_degree> run_cg(false, 0, convergence_table_h, RunConfiguration::h_coarsening);
      run_cg.run();
    }
    { // CG + dir=1
      Runner<2, fe_degree> run_cg(false, 1, convergence_table_h, RunConfiguration::h_coarsening);
      run_cg.run();
    }
    { // DG + dir=0
      Runner<2, fe_degree> run_cg(true, 0, convergence_table_h, RunConfiguration::h_coarsening);
      run_cg.run();
    }
    { // DG + dir=1
      Runner<2, fe_degree> run_cg(true, 1, convergence_table_h, RunConfiguration::h_coarsening);
      run_cg.run();
    }
  }

  if(true) // p-coarsening
  {
    { // CG + dir=0
      Runner<2, fe_degree> run_cg(false, 0, convergence_table_p, RunConfiguration::p_coarsening);
      run_cg.run();
    }
    { // CG + dir=1
      Runner<2, fe_degree> run_cg(false, 1, convergence_table_p, RunConfiguration::p_coarsening);
      run_cg.run();
    }
    { // DG + dir=0
      Runner<2, fe_degree> run_cg(true, 0, convergence_table_p, RunConfiguration::p_coarsening);
      run_cg.run();
    }
    { // DG + dir=1
      Runner<2, fe_degree> run_cg(true, 1, convergence_table_p, RunConfiguration::p_coarsening);
      run_cg.run();
    }
  }

  if(true) // c-coarsening
  {
    { // dir=0
      Runner<2, fe_degree> run_cg(true, 0, convergence_table_c, RunConfiguration::c_coarsening);
      run_cg.run();
    }
    { // dir=1
      Runner<2, fe_degree> run_cg(true, 1, convergence_table_c, RunConfiguration::c_coarsening);
      run_cg.run();
    }
  }

  // if(!rank)
  {
    std::cout << "h-coasening:" << std::endl;
    convergence_table_h.write_text(std::cout);
    std::cout << std::endl;

    std::cout << "p-coasening:" << std::endl;
    convergence_table_p.write_text(std::cout);
    std::cout << std::endl;

    std::cout << "c-coasening:" << std::endl;
    convergence_table_c.write_text(std::cout);
    std::cout << std::endl;
  }
}
