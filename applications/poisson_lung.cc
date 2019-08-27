/*
 * poisson_lung.cc
 *
 * program to test different solver configurations on lung triangulation
 *
 *  Created on: 2018
 *      Author: münch
 */

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/revision.h>
#include <deal.II/base/timer.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/data_out.h>
#include <vector>

#include "../include/poisson/spatial_discretization/laplace_operator.h"
#include "../include/poisson/user_interface/analytical_solution.h"
#include "../include/poisson/user_interface/boundary_descriptor.h"
#include "../include/poisson/user_interface/field_functions.h"
#include "../include/poisson/user_interface/input_parameters.h"

#include "../include/functionalities/dynamic_convergence_table.h"
#include "../include/functionalities/measure_minimum_time.h"
#include "../include/poisson/spatial_discretization/operator.h"
#include "../include/solvers_and_preconditioners/multigrid/smoothers/chebyshev_smoother.h"
#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"

// SPECIFY THE TEST CASE THAT HAS TO BE SOLVED

#include "grid_tools/lung/alternative.h"
#include "poisson_test_cases/lung.h"

#include "grid_tools/lung/lung_environment.h"

#define MANUAL
#define VERSION 4

#define LUNG_GENERATIONS 6

using namespace dealii;
using namespace Poisson;

template<int dim, int fe_degree, typename Number = double>
class PoissonProblem
{
public:
  typedef double value_type;
  PoissonProblem(bool use_cg, bool use_pmg, bool use_amg, bool use_block);

  void
  solve_problem(ConvergenceTable &                                             convergence_table,
                DynamicConvergenceTable &                                      dct,
                std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
                int                                                            n_refine_space);

private:
  void
  print_header();

  void
  print_grid_data();

  template<typename Vec>
  void
  output_data(std::string filename, Vec & solution)
  {
    DataOut<dim> data_out;
    data_out.attach_dof_handler(poisson_operation->get_dof_handler());
    solution.update_ghost_values();
    data_out.add_data_vector(solution, "solution");

    {
      // output reference solution
      auto ref_solution = solution;
      VectorTools::interpolate(poisson_operation->get_dof_handler(),
                               *analytical_solution->solution,
                               ref_solution);
      data_out.add_data_vector(ref_solution, "reference");
    }

    data_out.build_patches(fe_degree);

    data_out.write_vtu_in_parallel(filename.c_str(), MPI_COMM_WORLD);
  }

  ConditionalOStream                            pcout;
  bool                                          use_cg;
  bool                                          use_pmg;
  bool                                          use_amg;
  bool                                          use_block;
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;
  Poisson::InputParameters                      param;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  std::shared_ptr<Poisson::FieldFunctions<dim>>     field_functions;
  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<Poisson::AnalyticalSolution<dim>> analytical_solution;

  std::shared_ptr<Poisson::DGOperator<dim, value_type>> poisson_operation;
};

template<int dim, int fe_degree, typename Number>
PoissonProblem<dim, fe_degree, Number>::PoissonProblem(bool use_cg,
                                                       bool use_pmg,
                                                       bool use_amg,
                                                       bool use_block)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    use_cg(use_cg),
    use_pmg(use_pmg),
    use_amg(use_amg),
    use_block(use_block)
{
  print_header();
  print_MPI_info(pcout);

  // create triangulation
#if VERSION == 0 || VERSION == 1
  triangulation.reset(new parallel::fullydistributed::Triangulation<dim>(MPI_COMM_WORLD));
#elif VERSION == 2 || VERSION == 3 || VERSION == 4
  triangulation.reset(new parallel::distributed::Triangulation<dim>(
    MPI_COMM_WORLD,
    dealii::Triangulation<dim>::none,
    parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
#else
  // TODO: assert
#endif

  // get default input parameters
  set_input_parameters(param);

  // override parameters
  param.degree_mapping = fe_degree;

  if(use_cg)
    param.spatial_discretization = SpatialDiscretization::CG;
  else
    param.spatial_discretization = SpatialDiscretization::DG;

  if(use_pmg)
    param.multigrid_data.type = MultigridType::pMG;
  else
    param.multigrid_data.type = MultigridType::hMG;

  if(use_amg)
  {
    // TODO
    // param.multigrid_data.c_transfer_back                              = true;
    // TODO
    // param.multigrid_data.coarse_ml_data.use_conjugate_gradient_solver = true;
    // TODO
    // param.multigrid_data.coarse_solver = MultigridCoarseGridSolver::AMG_ML; // GMRES_PointJacobi;
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
    param.multigrid_data.dg_to_cg_transfer             = DG_To_CG_Transfer::Coarse;
    param.multigrid_data.p_sequence                    = PSequenceType::Bisect;
  }
  else
  {
    // TODO
    // param.multigrid_data.coarse_solver   = MultigridCoarseGridSolver::Chebyshev;
    // TODO
    // param.multigrid_data.c_transfer_back = false;
    param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::Chebyshev;
    param.multigrid_data.dg_to_cg_transfer     = DG_To_CG_Transfer::None;
  }

  if(use_block)
  {
    // TODO
    // param.multigrid_data.smoother = MultigridSmoother::Jacobi;
    // TODO
    param.multigrid_data.smoother_data.smoother       = MultigridSmoother::Jacobi;
    param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
    // param.enable_cell_based_face_loops = true;
  }
  else
  {
    // TODO
    param.multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev;
    // param.multigrid_data.smoother = MultigridSmoother::Chebyshev;
  }

  param.check_input_parameters();
  param.print(pcout, "List of input parameters:");

  field_functions.reset(new Poisson::FieldFunctions<dim>());
  set_field_functions(field_functions);

  analytical_solution.reset(new Poisson::AnalyticalSolution<dim>());
  set_analytical_solution(analytical_solution);

  boundary_descriptor.reset(new Poisson::BoundaryDescriptor<dim>());

  poisson_operation.reset(new Poisson::DGOperator<dim, Number>(*triangulation, param));
}

template<int dim, int fe_degree, typename Number>
void
PoissonProblem<dim, fe_degree, Number>::print_header()
{
  // clang-format off
  pcout << std::endl << std::endl << std::endl
        << "_________________________________________________________________________________" << std::endl
        << "                                                                                 " << std::endl
        << "                High-order discontinuous Galerkin solver for the                 " << std::endl
        << "                                Poisson equation                                 " << std::endl
        << "_________________________________________________________________________________" << std::endl
        << std::endl;
  // clang-format on
}

template<int dim, int fe_degree, typename Number>
void
PoissonProblem<dim, fe_degree, Number>::print_grid_data()
{
  pcout << std::endl
        << "Generating grid for " << dim << "-dimensional problem:" << std::endl
        << std::endl;

  print_parameter(pcout, "Number of refinements", triangulation->n_global_levels());
  print_parameter(pcout, "Number of cells", triangulation->n_global_active_cells());
  print_parameter(pcout, "Number of faces", triangulation->n_active_faces());
  print_parameter(pcout, "Number of vertices", triangulation->n_vertices());
}

template<int dim, int fe_degree, typename Number>
void
PoissonProblem<dim, fe_degree, Number>::solve_problem(
  ConvergenceTable &                                             convergence_table,
  DynamicConvergenceTable &                                      dct,
  std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
  int                                                            n_refine_space)
{
  Timer timer;
  // create grid and set bc
  timer.restart();

  int generations = LUNG_GENERATIONS;
  // int n_refine_space = 2;

  std::map<std::string, double> timings;

  unsigned int outlet_id_first = 2, outlet_id_last = 2;
#if VERSION == 0 || VERSION == 1 || VERSION == 4
  // create triangulation
  if(auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation))
  {
    std::shared_ptr<LungID::Checker> generation_limiter(new LungID::GenerationChecker(generations));
    std::string                      spline_file = get_lung_spline_file_from_environment();
    dealii::GridGenerator::lung(*tria,
                                n_refine_space,
                                create_tree,
                                timings,
                                outlet_id_first,
                                outlet_id_last,
                                spline_file,
                                generation_limiter);
  }
  else
    AssertThrow(false, ExcMessage("Unknown triangulation!"));
#elif VERSION == 2 || VERSION == 3
  // create triangulation
  if(auto triat = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation))
    create_lung(*triat, n_refine_space, VERSION == 2);
  else
    AssertThrow(false, ExcMessage("Unknown triangulation!"));
#else
  // TODO: assert
#endif

  // set boundary conditions
  std::shared_ptr<Function<dim>> zero_function_scalar;
  zero_function_scalar.reset(new Functions::ZeroFunction<dim>(1));
  // boundary_descriptor->neumann_bc.insert({0, zero_function_scalar});
  boundary_descriptor->dirichlet_bc.insert({0, zero_function_scalar});
  boundary_descriptor->dirichlet_bc.insert({1, zero_function_scalar});

  for(unsigned int i = outlet_id_first; i <= outlet_id_last; i++)
    boundary_descriptor->dirichlet_bc.insert({i, zero_function_scalar});

  print_grid_data();
  dct.put("_grid", timer.wall_time());

  timer.restart();
  // setup poisson operation
  poisson_operation->setup(periodic_faces, boundary_descriptor, field_functions);

  poisson_operation->setup_solver();
  double time_setup = timer.wall_time();

  // allocate vectors
  LinearAlgebra::distributed::Vector<Number> rhs;
  LinearAlgebra::distributed::Vector<Number> solution;
  poisson_operation->initialize_dof_vector(rhs);
  poisson_operation->initialize_dof_vector(solution);

  // solve problem
  int procs;
  MPI_Comm_size(MPI_COMM_WORLD, &procs);
  convergence_table.add_value("procs", procs);
  convergence_table.add_value("dim", dim);
  convergence_table.add_value("degree", fe_degree);
  convergence_table.add_value("refs", n_refine_space);
  convergence_table.add_value("dofs", solution.size());

  convergence_table.add_value("feq", use_cg);
  convergence_table.add_value("pmg", use_pmg);
  convergence_table.add_value("amg", use_amg);
  convergence_table.add_value("block", use_block);

  convergence_table.add_value("setup", time_setup);
  convergence_table.set_scientific("setup", true);

  dct.put("procs", procs);
  dct.put("_dim", dim);
  dct.put("_degree", fe_degree);
  dct.put("_refs", n_refine_space);
  dct.put("_dofs", solution.size());

  if(false && param.output_data.write_output)
    this->output_data(param.output_data.output_folder + param.output_data.output_name + "0.vtu",
                      solution);

  poisson_operation->rhs(rhs);

  int cycles = 0;
  solution   = 0;
  timer.restart();
  cycles = poisson_operation->solve(solution, rhs);
  std::cout << ">>>>>>>>> " << cycles << std::endl;
  dct.put("_solve", timer.wall_time());

  convergence_table.add_value("solve", timer.wall_time());
  convergence_table.add_value("cycles", cycles);

  if(false && param.output_data.write_output)
    this->output_data(param.output_data.output_folder + param.output_data.output_name + "1.vtu",
                      solution);



  LinearAlgebra::distributed::Vector<Number> check1, check2, tmp, check3, check4, check5, check6,
    t7, t8;
  poisson_operation->initialize_dof_vector(check1);
  poisson_operation->initialize_dof_vector(check2);
  poisson_operation->initialize_dof_vector(check3);
  poisson_operation->initialize_dof_vector(check4);
  poisson_operation->initialize_dof_vector(check5);
  poisson_operation->initialize_dof_vector(check6);
  poisson_operation->initialize_dof_vector(tmp);
  // TODO
  //  for(unsigned int i = 0; i < check1.local_size(); ++i)
  //    if(!poisson_operation->constraint_matrix.is_constrained(
  //         check1.get_partitioner()->local_to_global(i)))
  //      check1.local_element(i) = (double)rand() / RAND_MAX;
  //
  //  poisson_operation->laplace_operator.apply(tmp, check1);
  //  tmp *= -1.0;
  //  poisson_operation->preconditioner.vmult(check2, tmp);
  //  check2 += check1;


  //  LinearAlgebra::distributed::Vector<float>  tmp_float, check3_float;
  //  tmp_float = tmp;
  //  check3_float = check3;
  //  if(dynamic_cast<MultigridPreconditionerBase<3,double,float>*>(&*(poisson_operation->preconditioner))
  //  == nullptr)
  //      std::cout << "AAA" << std::endl;
  //  auto & mg_smoothers =
  //  (dynamic_cast<MultigridPreconditionerBase<3,double,float>*>(&*(poisson_operation->preconditioner)))->mg_smoother;
  //  mg_smoothers[mg_smoothers.max_level()]->vmult(check3_float, tmp_float);
  //  check3 = check3_float;
  //  check3 += check1;


  typedef ChebyshevSmoother<Poisson::LaplaceOperator<dim, Number>,
                            LinearAlgebra::distributed::Vector<Number>>
                                              CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  // TODO
//  poisson_operation->laplace_operator.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
//  poisson_operation->laplace_operator.calculate_inverse_diagonal(
//    smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  /*
  std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[level],
  smoother_data.matrix_diagonal_inverse);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Eigenvalues on level l = " << level << std::endl;
    std::cout << std::scientific << std::setprecision(3)
              <<"Max EV = " << eigenvalues.second << " : Min EV = " <<
  eigenvalues.first << std::endl;
  }
  */

  // TODO
  // smoother_data.smoothing_range     =
  // param.multigrid_data.chebyshev_smoother_data.smoother_smoothing_range;
  smoother_data.degree = 30; // param.multigrid_data.chebyshev_smoother_data.smoother_poly_degree;
  // TODO
  // smoother_data.eig_cg_n_iterations =
  // param.multigrid_data.chebyshev_smoother_data.eig_cg_n_iterations;

  CHEBYSHEV_SMOOTHER smoother;
  // TODO
  //  smoother.initialize(poisson_operation->laplace_operator, smoother_data);
  smoother.vmult(check3, tmp);
  check3 += check1;


  auto &         dof_handler = poisson_operation->get_dof_handler();
  DataOut<dim>   data_out;
  Vector<double> owner(triangulation->n_active_cells());
  owner = (double)Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);
  check1.update_ghost_values();
  data_out.add_data_vector(dof_handler, check1, "initial_field");
  check2.update_ghost_values();
  data_out.add_data_vector(dof_handler, check2, "mg_cycle");
  check3.update_ghost_values();
  data_out.add_data_vector(dof_handler, check3, "chebyshev");

  data_out.add_data_vector(owner, "owner");
  MappingQGeneric<dim> mapping(dof_handler.get_fe().degree);
  data_out.build_patches(mapping, dof_handler.get_fe().degree, DataOut<dim>::curved_inner_cells);
  std::ofstream out(("dg_sol_" + Utilities::to_string(fe_degree) + "." +
                     Utilities::to_string(n_refine_space) + ".vtk")
                      .c_str());
  data_out.write_vtk(out);

  // LinearAlgebra::distributed::Vector<Number> vec_diag;
  // poisson_operation->laplace_operator.calculate_diagonal(vec_diag);
  //
  //  for(unsigned int i = 0; i < check1.local_size(); i++){
  //    LinearAlgebra::distributed::Vector<Number> base_in, base_out;
  //    poisson_operation->initialize_dof_vector(base_out);
  //    poisson_operation->initialize_dof_vector(base_in);
  //    base_in[i] = 1.0;
  //    poisson_operation->laplace_operator.apply(base_out, base_in);
  //
  //    if(abs(base_out[i]-vec_diag[i]) > 1e-10)
  //        std::cout << "@@@@@@@@@@@ " << base_out[i] << " " << base_in[i] << " " << vec_diag[i] <<
  //        std::endl;
  //  }


  dct.add_new_row();
}

template<int fe_degeee>
void
run_single(ConvergenceTable &                                             convergence_table,
           DynamicConvergenceTable &                                      dct,
           std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree,
           int                                                            n_refinements)
{
  { // CG + PMG+AMG + Chebyshev
    PoissonProblem<3, fe_degeee> poisson_problem(true, true, true, false);
    poisson_problem.solve_problem(convergence_table, dct, create_tree, n_refinements);
  }
#if false
    { // CG + HMG + Chebyshev
        PoissonProblem<3, fe_degeee> poisson_problem(true, false, false, false); 
        poisson_problem.solve_problem(convergence_table,dct, create_tree, n_refinements);
    }
    { // DG + PMG+AMG + Chebyshev
        PoissonProblem<3, fe_degeee> poisson_problem(false, true, true, false); 
        poisson_problem.solve_problem(convergence_table,dct, create_tree, n_refinements);
    }
    { // DG + HMG + Chebyshev
        PoissonProblem<3, fe_degeee> poisson_problem(false, false, false, false); 
        poisson_problem.solve_problem(convergence_table,dct, create_tree, n_refinements);
    }
    { // DG + PMG+AMG + Block Jacobi
        PoissonProblem<3, fe_degeee> poisson_problem(false, true, true, true); 
        poisson_problem.solve_problem(convergence_table,dct, create_tree, n_refinements);
    }
    { // DG + HMG + Block Jacobi
        PoissonProblem<3, fe_degeee> poisson_problem(false, false, false, true); 
        poisson_problem.solve_problem(convergence_table,dct, create_tree, n_refinements);
    }
#endif
}

void
run(ConvergenceTable &                                             convergence_table,
    DynamicConvergenceTable &                                      dct,
    std::function<void(std::vector<Node *> & roots, unsigned int)> create_tree)
{
  for(unsigned int n_refinements = 0; n_refinements < 3; n_refinements++)
    for(unsigned int fe_degree = 0; fe_degree < 5; fe_degree++)
    {
      switch(fe_degree)
      {
        case 1:
          run_single<1>(convergence_table, dct, create_tree, n_refinements);
          break;
        case 2:
          run_single<2>(convergence_table, dct, create_tree, n_refinements);
          break;
        case 3:
          run_single<3>(convergence_table, dct, create_tree, n_refinements);
          break;
        case 4:
          run_single<4>(convergence_table, dct, create_tree, n_refinements);
          break;
      }
    }
}

int
main(int argc, char ** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    int rank = Utilities::MPI::this_mpi_process(MPI_COMM_WORLD);

    if(!rank)
    {
      std::cout << "deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch "
                << DEAL_II_GIT_BRANCH << std::endl
                << std::endl;
    }

    deallog.depth_console(0);


    ConvergenceTable        convergence_table;
    DynamicConvergenceTable dct;

#if VERSION == 4
    std::vector<std::string> files;
    get_lung_files_from_environment(files);
    auto tree_factory = dealii::GridGenerator::lung_files_to_node(files);
#else
    auto tree_factory = [](std::vector<Node *> & roots, unsigned int generations) {
      std::vector<Point<3>>           points(4);
      std::vector<CellData<1>>        cells(3);
      std::vector<CellAdditionalInfo> cells_additional_data(3);

#  if VERSION == 0
      double                          phi = numbers::PI / 8;

      points[0] = {+0.0, +0.0, +0.0};
      points[1] = {+0.0, +0.0, +1.0};
      points[2] = {+0.0, +1.0 * cos(phi), +1.0 + 1.0 * sin(phi)};
      points[3] = {+0.0, -1.0 * cos(phi), +1.0 - 1.0 * sin(phi)};

      cells[0].vertices[0] = 0;
      cells[0].vertices[1] = 1;
      cells[1].vertices[0] = 1;
      cells[1].vertices[1] = 2;
      cells[2].vertices[0] = 1;
      cells[2].vertices[1] = 3;

      cells_additional_data[0] = {0.2, 0};
      cells_additional_data[1] = {0.2, 1};
      cells_additional_data[2] = {0.2, 1};
#  elif VERSION == 1
      double phi    = numbers::PI / 4;
      double radius = 1.0;
      double length = 1.5;

      points[0] = {+0.0, +0.0, +0.0};
      points[1] = {+0.0, +0.0, +length};
      points[2] = {+0.0, +length * cos(phi), +length + length * sin(phi)};
      points[3] = {+0.0, -length * cos(phi), +length + length * sin(phi)};

      cells[0].vertices[0] = 0;
      cells[0].vertices[1] = 1;
      cells[1].vertices[0] = 1;
      cells[1].vertices[1] = 2;
      cells[2].vertices[0] = 1;
      cells[2].vertices[1] = 3;

      cells_additional_data[0] = {radius, 0};
      cells_additional_data[1] = {radius, 1};
      cells_additional_data[2] = {radius, 1};
#  else
      // TODO: assert
#  endif

      try
      {
        dealii::GridGenerator::lung_to_node(
          generations, points, cells, cells_additional_data, roots);
      }
      catch(const std::exception & e)
      {
        std::cout << e.what();
      }
    };
#endif

    run(convergence_table, dct, tree_factory);

    if(!rank)
    {
      std::ofstream outfile;
      outfile.open("lung-table1.csv");
      convergence_table.write_text(outfile);
      outfile.close();
      dct.print("lung-table2.csv");
    }
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }
  return 0;
}
