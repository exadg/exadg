/*
 * poisson_dual.cc
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
#include "../include/solvers_and_preconditioners/multigrid/transfer/mg_transfer_mf_c.h"
#include "functionalities/print_functions.h"
#include "functionalities/print_general_infos.h"

// SPECIFY THE TEST CASE THAT HAS TO BE SOLVED

// Laplace problems
//#include "poisson_test_cases/cosinus.h"
//#include "poisson_test_cases/cosinus_dual.h"
#include "poisson_test_cases/gaussian.h"
//#include "poisson_test_cases/hyper_l.h"
//#include "poisson_test_cases/torus.h"

using namespace dealii;
using namespace Poisson;

const int BEST_OF = 10;

template<int dim, int fe_degree, typename Number = double>
class PoissonProblemInstance
{
public:
  typedef double value_type;
  PoissonProblemInstance(const unsigned int n_refine_space_in,
                         PSequenceType      psqeuence = PSequenceType::Manual,
                         bool               use_amg   = true,
                         bool               use_dg    = true,
                         MultigridType      mg_type   = MultigridType::Undefined,
                         bool               use_aux   = true,
                         bool /*use_pcg*/             = true)
    : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
      n_refine_space(n_refine_space_in)
  {
    print_MPI_info(pcout);

    set_input_parameters(param);

    triangulation.reset(new parallel::distributed::Triangulation<dim>(
      MPI_COMM_WORLD,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));

    param.degree_mapping = fe_degree;

    if(use_dg)
    {
      param.spatial_discretization = SpatialDiscretization::DG;
      param.solver_data.rel_tol    = 1e-1;
    }
    else
      param.spatial_discretization = SpatialDiscretization::CG;

    if(mg_type != MultigridType::Undefined)
      param.multigrid_data.type = mg_type;

    if(use_aux)
      param.multigrid_data.dg_to_cg_transfer = DG_To_CG_Transfer::Coarse;
    // TODO
    //    param.multigrid_data.coarse_ml_data.use_conjugate_gradient_solver = use_pcg;

    if(use_amg)
      param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::AMG;
    else
    {
      param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
      param.multigrid_data.dg_to_cg_transfer     = DG_To_CG_Transfer::None;
    }
    if(!(psqeuence == PSequenceType::Manual))
      param.multigrid_data.p_sequence = psqeuence;

    param.check_input_parameters();
    param.print(pcout, "List of input parameters:");

    field_functions.reset(new Poisson::FieldFunctions<dim>());
    set_field_functions(field_functions);

    analytical_solution.reset(new Poisson::AnalyticalSolution<dim>());
    set_analytical_solution(analytical_solution);

    boundary_descriptor.reset(new Poisson::BoundaryDescriptor<dim>());

    poisson_operation.reset(new Poisson::DGOperator<dim, Number>(*triangulation, param));
  }

  void
  setup()
  {
    create_grid_and_set_boundary_conditions(triangulation,
                                            n_refine_space,
                                            boundary_descriptor,
                                            periodic_faces);
    poisson_operation->setup(periodic_faces, boundary_descriptor, field_functions);
  }

  void
  setup_solver()
  {
    poisson_operation->setup_solver();
  }

public:
  ConditionalOStream                            pcout;
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation;
  const unsigned int                            n_refine_space;
  Poisson::InputParameters                      param;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_faces;

  std::shared_ptr<Poisson::FieldFunctions<dim>>     field_functions;
  std::shared_ptr<Poisson::BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<Poisson::AnalyticalSolution<dim>> analytical_solution;

  std::shared_ptr<Poisson::DGOperator<dim, value_type>> poisson_operation;
};

template<int dim, int fe_degree, typename Number = double>
class PoissonProblem
{
public:
  typedef double value_type;
  PoissonProblem(const unsigned int n_refine_space,
                 PSequenceType      psqeuence = PSequenceType::Manual,
                 bool               use_amg   = true,
                 bool               use_dg    = true,
                 MultigridType      mg_type   = MultigridType::Undefined,
                 bool               use_aux   = true,
                 bool               use_pcg   = true);

  void
  solve_problem(bool is_not_convergence_study = true)
  {
    ConvergenceTable        convergence_table_dummy;
    DynamicConvergenceTable dynamic_convergence_table;
    this->solve_problem(is_not_convergence_study,
                        convergence_table_dummy,
                        dynamic_convergence_table,
                        1);
  }

  void
  solve_problem(bool                      is_not_convergence_study,
                ConvergenceTable &        convergence_table,
                DynamicConvergenceTable & dct,
                unsigned int              best_of);

private:
  void
  print_header();

  void
  print_grid_data();

  ConditionalOStream                             pcout;
  unsigned int                                   n_refine_space_in;
  PoissonProblemInstance<dim, fe_degree, Number> ppi_cg;
  PoissonProblemInstance<dim, fe_degree, Number> ppi_dg;
};

template<int dim, int fe_degree, typename Number>
PoissonProblem<dim, fe_degree, Number>::PoissonProblem(const unsigned int n_refine_space_in,
                                                       PSequenceType      psqeuence,
                                                       bool               use_amg,
                                                       bool /*use_dg*/,
                                                       MultigridType mg_type,
                                                       bool          use_aux,
                                                       bool          use_pcg)
  : pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0),
    n_refine_space_in(n_refine_space_in),
    ppi_cg(n_refine_space_in, psqeuence, use_amg, false, mg_type, use_aux, use_pcg),
    ppi_dg(n_refine_space_in, psqeuence, use_amg, true, mg_type, use_aux, use_pcg)
{
  print_header();
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
}

struct Results
{
  Results()
    : procs(0),
      deg(0),
      ref(0),
      dofs_cg(0),
      dofs_dg(0),
      cycles_cg(0),
      cycles_dg(0),
      time_total(0),
      time_cg(0),
      time_dg(0)
  {
  }

  int    procs;
  int    deg;
  int    ref;
  int    dofs_cg;
  int    dofs_dg;
  int    cycles_cg;
  int    cycles_dg;
  double time_total;
  double time_cg;
  double time_dg;
};

template<int dim, int fe_degree, typename Number>
void
PoissonProblem<dim, fe_degree, Number>::solve_problem(bool /*is_not_convergence_study*/,
                                                      ConvergenceTable & convergence_table,
                                                      DynamicConvergenceTable & /*dct*/,
                                                      unsigned int /*best_of*/)
{
  ppi_cg.setup();
  ppi_cg.setup_solver();
  ppi_dg.setup();

  // allocate vectors
  LinearAlgebra::distributed::Vector<Number> rhs_dg, rhs_cg, temp, temp2, rhs_cg_temp;
  LinearAlgebra::distributed::Vector<Number> solution_dg, solution_cg;
  ppi_dg.poisson_operation->initialize_dof_vector(temp);
  ppi_dg.poisson_operation->initialize_dof_vector(temp2);
  ppi_dg.poisson_operation->initialize_dof_vector(rhs_dg);
  ppi_dg.poisson_operation->initialize_dof_vector(solution_dg);
  ppi_cg.poisson_operation->initialize_dof_vector(rhs_cg);
  ppi_cg.poisson_operation->initialize_dof_vector(rhs_cg_temp);
  ppi_cg.poisson_operation->initialize_dof_vector(solution_cg);

  Timer time;

  // TODO
  AffineConstraints<double>      constraint_matrix;
  MGTransferMFC<dim, value_type> transfer(ppi_dg.poisson_operation->get_data(),
                                          ppi_cg.poisson_operation->get_data(),
                                          constraint_matrix,
                                          constraint_matrix,
                                          n_refine_space_in,
                                          fe_degree);
  //  MGTransferMFC<dim, value_type> transfer(ppi_dg.poisson_operation->matrix_free,
  //                                          ppi_cg.poisson_operation->matrix_free,
  //                                          ppi_dg.poisson_operation->constraint_matrix,
  //                                          ppi_cg.poisson_operation->constraint_matrix,
  //                                          n_refine_space_in,
  //                                          fe_degree);

  ppi_dg.poisson_operation->rhs(rhs_dg);
  ppi_cg.poisson_operation->rhs(rhs_cg);
  ppi_cg.poisson_operation->rhs(rhs_cg_temp);
  rhs_cg = 0.0;
  transfer.restrict_and_add(0, rhs_cg, rhs_dg);

  auto    rhs_dg_temp = rhs_dg;
  Results rr;

  // rhs_cg_temp.print(std::cout);
  // rhs_cg.print(std::cout);
  // rhs_dg.print(std::cout);

  double norm_0 = rhs_dg.l2_norm();

  for(int i = 0; i < BEST_OF; i++)
  {
    rhs_dg      = rhs_dg_temp;
    solution_cg = 0.0;
    solution_dg = 0.0;
    temp2       = 0.0;
    time.restart();
    auto   cycles_cg = ppi_cg.poisson_operation->solve(solution_cg, rhs_cg);
    double time_cg   = time.wall_time();

    time.restart();
    transfer.prolongate(0, solution_dg, solution_cg);

    temp = 0.0;
    // TODO
    //    ppi_dg.poisson_operation->laplace_operator.vmult(temp, solution_dg);
    rhs_dg -= temp;
    double time_0 = time.wall_time();

    if(i == 0)
    {
      double norm_n = rhs_dg.l2_norm();
      ppi_dg.param.solver_data.rel_tol *= norm_0 / norm_n;
      ppi_dg.setup_solver();
    }

    solution_dg = 0.0;

    time.restart();
    auto   cycles_dg = ppi_dg.poisson_operation->solve(temp2, rhs_dg);
    double time_dg   = time.wall_time();

    time.restart();
    solution_dg += temp2;
    double time_2 = time.wall_time();

    int procs;
    MPI_Comm_size(MPI_COMM_WORLD, &procs);
    Results r;
    r.procs      = procs;
    r.deg        = fe_degree;
    r.ref        = n_refine_space_in;
    r.dofs_cg    = solution_cg.size();
    r.dofs_dg    = solution_dg.size();
    r.cycles_cg  = cycles_cg;
    r.cycles_dg  = cycles_dg;
    r.time_total = time_0 + time_2 + time_cg + time_dg;
    ;
    r.time_cg = time_cg;
    r.time_dg = time_dg;

    if(i == 0 || r.time_total < rr.time_total)
      rr = r;
  }
  double time_dg_pure = 10000;
  int    cycles_dg_pure;
  ppi_dg.param.solver_data.rel_tol = ppi_cg.param.solver_data.rel_tol;
  ppi_dg.setup_solver();
  for(int i = 0; i < BEST_OF; i++)
  {
    temp2  = 0.0;
    rhs_dg = rhs_dg_temp;
    time.restart();
    cycles_dg_pure = ppi_dg.poisson_operation->solve(temp2, rhs_dg);
    time_dg_pure   = std::min(time.wall_time(), time_dg_pure);
  }
  {
    convergence_table.add_value("procs", rr.procs);
    convergence_table.add_value("deg", rr.deg);
    convergence_table.add_value("ref", rr.ref);
    convergence_table.add_value("dofs-cg", rr.dofs_cg);
    convergence_table.add_value("dofs-dg", rr.dofs_dg);
    convergence_table.add_value("cycles-cg", rr.cycles_cg);
    convergence_table.add_value("cycles-dg", rr.cycles_dg);
    convergence_table.add_value("time-total", rr.time_total);
    convergence_table.set_scientific("time-total", true);
    convergence_table.add_value("time-cg", rr.time_cg);
    convergence_table.set_scientific("time-cg", true);
    convergence_table.add_value("time-dg", rr.time_dg);
    convergence_table.set_scientific("time-dg", true);
    convergence_table.add_value("time-dg-pure", time_dg_pure);
    convergence_table.set_scientific("time-dg-pure", true);
    convergence_table.add_value("cycles-dg-pure", cycles_dg_pure);
  }
}


struct DataC
{
  DataC()
    : fe_degree_min(FE_DEGREE_MIN),
      fe_degree_max(FE_DEGREE_MAX),
      size_min(0),
      size_max(std::numeric_limits<unsigned int>::max()),
      best_of(BEST_OF),
      use_dg(true),
      norm_dg(false),
      use_amg(true),
      refinements_provided(false),
      sequence(PSequenceType::Manual),
      fe_degree_eff(FE_DEGREE_MIN), // TODO
      mg_type(MultigridType::Undefined),
      use_aux(true),
      use_pcg(true)
  {
  }

  unsigned int  fe_degree_min;
  unsigned int  fe_degree_max;
  unsigned int  size_min;
  unsigned int  size_max;
  unsigned int  best_of;
  unsigned int  use_dg;
  unsigned int  norm_dg;
  unsigned int  use_amg;
  bool          refinements_provided;
  PSequenceType sequence;
  unsigned int  fe_degree_eff;
  MultigridType mg_type;
  bool          use_aux;
  bool          use_pcg;
};

template<int dim, int fe_degree>
class Run
{
public:
  static void
  run(ConvergenceTable & convergence_table, DynamicConvergenceTable & dct, DataC d)
  {
    if(d.refinements_provided)
    {
      for(unsigned int refinement = d.size_min; refinement <= d.size_max; refinement++)
      {
        PoissonProblem<dim, fe_degree> conv_diff_problem(
          refinement, d.sequence, d.use_amg, d.use_dg, d.mg_type, d.use_aux, d.use_pcg);
        conv_diff_problem.solve_problem(false, convergence_table, dct, d.best_of);
      }
    }
    else
    {
      std::vector<unsigned int> sizes;
      if(d.size_min == d.size_max)
        sizes = {d.size_min};
      else
        sizes = {1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192};

      for(auto size : sizes)
      {
        if(size < d.fe_degree_eff || size < d.size_min || d.size_max < size)
          continue;
        int refinement = std::log(size / d.fe_degree_eff) / std::log(2.0);

        PoissonProblem<dim, fe_degree> conv_diff_problem(
          refinement, d.sequence, d.use_amg, d.use_dg, d.mg_type, d.use_aux, d.use_pcg);
        conv_diff_problem.solve_problem(false, convergence_table, dct, d.best_of);
      }
    }
  }
};

template<int dim>
void
do_run(ConvergenceTable & convergence_table, DynamicConvergenceTable & dct, DataC & d)
{
  for(unsigned int fe_degree = d.fe_degree_min; fe_degree <= d.fe_degree_max; fe_degree++)
  {
    d.fe_degree_eff = fe_degree + d.use_amg;
    switch(fe_degree)
    {
        // clang-format off
//#if DEGREE_1
          case  1: Run<dim,  1>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_2
          case  2: Run<dim,  2>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_3
          case  3: Run<dim,  3>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_4
          case  4: Run<dim,  4>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_5
          case  5: Run<dim,  5>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_6
          case  6: Run<dim,  6>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_7
          case  7: Run<dim,  7>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_8
          case  8: Run<dim,  8>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_9
          case  9: Run<dim,  9>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_10
          case 10: Run<dim, 10>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_11
          case 11: Run<dim, 11>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_12
          case 12: Run<dim, 12>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_13
          case 13: Run<dim, 13>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_14
          case 14: Run<dim, 14>::run(convergence_table,dct,d); break;
//#endif
//#if DEGREE_15
          case 15: Run<dim, 15>::run(convergence_table,dct,d); break;
//#endif
        // clang-format on
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

    AssertThrow(argc > 1, ExcMessage("Type of the run has to be specified!"));

    std::string type{argv[1]};

    if(type == "-h" || type == "--help")
    {
      if(rank == 0)
      {
        printf("Usage: ./poisson [MODUS] [OPTION]...\n");
        printf("Run Poisson application in different modi.\n");
        printf("\nOptional arguments:\n");
        printf("   -h --help           print this help page\n");
        printf("   -s --simple         simple run\n");
        printf("   -c --convergence    perform parameter study\n");

        printf("\nOptional arguments/flags (-s): none\n\n");

        printf("\nOptional flags (-c):\n");
        printf("  --cg                     normalize size by fe_degree \n");
        printf("  --dg                     normalize size by fe_degree+1 \n");

        printf("\nOptional arguments (-c):\n");
        printf("  --best-of n              run measurements n times \n");
        printf("  --degree d               use singe degree d \n");
        printf("  --degree-range d1 d2     use degree range [d1:1:d2]\n");
        printf("  --size s                 use single size s \n");
        printf("  --size-range s1 s2       use size range [s1:1:s2] \n");
        printf("  --refinement r           use single refinement r \n");
        printf("  --refinement-range r1 r2 use refinement range [r1:1:r2] \n");

        printf("\nExample:\n");
        printf("  ./poisson -s\n");
        printf("  ./poisson -c --size 256 --best-of 10 --cg\n");
        printf("  ./poisson -c --refinement-range 1 5 --best-of 10\n");
      }
    }
    else if(type == "-s" || type == "--simple")
    {
      PoissonProblem<DIMENSION, FE_DEGREE> poisson_problem(REFINE_STEPS);
      poisson_problem.solve_problem();
    }
    else if(type == "-c" || type == "--convergence")
    {
      ConvergenceTable        convergence_table;
      DynamicConvergenceTable dct;

      DataC d;
      int   argp = 2;
      int   dim  = 2;

      while(argc > argp)
      {
        std::string type{argv[argp++]};
        if(type == "--degree")
        {
          unsigned int degree = atoi(argv[argp++]);
          d.fe_degree_min     = std::max(degree, d.fe_degree_min);
          d.fe_degree_max     = std::min(degree, d.fe_degree_max);
        }
        if(type == "--degree-range")
        {
          d.fe_degree_min = std::max((unsigned int)atoi(argv[argp++]), d.fe_degree_min);
          d.fe_degree_max = std::min((unsigned int)atoi(argv[argp++]), d.fe_degree_max);
        }
        if(type == "--dim")
        {
          dim = atoi(argv[argp++]);
        }
        if(type == "--size")
        {
          unsigned int size      = atoi(argv[argp++]);
          d.size_min             = size;
          d.size_max             = size;
          d.refinements_provided = false;
        }
        if(type == "--size-range")
        {
          d.size_min             = atoi(argv[argp++]);
          d.size_max             = atoi(argv[argp++]);
          d.refinements_provided = false;
        }
        if(type == "--refinement")
        {
          unsigned int size      = atoi(argv[argp++]);
          d.size_min             = size;
          d.size_max             = size;
          d.refinements_provided = true;
        }
        if(type == "--refinement-range")
        {
          d.size_min             = atoi(argv[argp++]);
          d.size_max             = atoi(argv[argp++]);
          d.refinements_provided = true;
        }
        if(type == "--best-of")
        {
          d.best_of = atoi(argv[argp++]);
        }
        if(type == "--sequence")
        {
          unsigned int temp = atoi(argv[argp++]);
          if(temp == 1)
            d.sequence = PSequenceType::GoToOne;
          else if(temp == 2)
            d.sequence = PSequenceType::DecreaseByOne;
          else if(temp == 3)
            d.sequence = PSequenceType::Bisect;
        }
        if(type == "--norm-dg")
        {
          d.norm_dg = true;
        }
        if(type == "--norm-cg")
        {
          d.norm_dg = false;
        }
        if(type == "--dg")
        {
          d.use_dg = true;
        }
        if(type == "--cg")
        {
          d.use_dg = false;
        }
        if(type == "--amg")
        {
          d.use_amg = true;
        }
        if(type == "--dcg")
        {
          d.use_amg = false;
        }
        if(type == "--hmg")
        {
          d.mg_type = MultigridType::hMG;
        }
        if(type == "--pmg")
        {
          d.mg_type = MultigridType::pMG;
        }
        if(type == "--hpmg")
        {
          d.mg_type = MultigridType::hMG;
        }
        if(type == "--phmg")
        {
          d.mg_type = MultigridType::pMG;
        }
        if(type == "--noaux")
        {
          d.use_aux = false;
        }
        if(type == "--nopcg")
        {
          d.use_pcg = false;
        }
      }

      //#if DIM_2 == 1
      if(dim == 2)
        do_run<2>(convergence_table, dct, d);
      //#endif
      //#if DIM_3 == 1
      if(dim == 3)
        do_run<3>(convergence_table, dct, d);
      //#endif

      if(!rank)
      {
        std::ofstream outfile;
        outfile.open("poisson-dual-table1.csv");
        convergence_table.write_text(outfile);
        outfile.close();
        dct.print("poisson-dual-table2.csv");
      }
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
