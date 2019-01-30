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

#include <deal.II/lac/precondition.h>
#include <deal.II/lac/solver_cg.h>

#include "../../../../include/convection_diffusion/spatial_discretization/operators/convection_diffusion_operator.h"
#include "../../../../include/convection_diffusion/spatial_discretization/operators/convective_operator.h"
#include "../../../../include/convection_diffusion/spatial_discretization/operators/diffusive_operator.h"
#include "../../../../include/convection_diffusion/spatial_discretization/operators/mass_operator.h"
#include "../../../../include/poisson/spatial_discretization/laplace_operator.h"

#include "include/operator_base_test.h"

#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"

#include "../../operation-base-util/operator_reinit_multigrid.h"

using namespace dealii;
using namespace Poisson;

const unsigned int global_refinements = 2;
typedef double     value_type;
const int          fe_degree_min = 1;
const int          fe_degree_max = 3;

typedef double value_type;

using namespace dealii;


template<int dim>
class VelocityField : public Function<dim>
{
public:
  VelocityField(const unsigned int n_components = dim, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  virtual ~VelocityField(){};

  virtual double
  value(const Point<dim> & point, const unsigned int component) const
  {
    double value = 0.0;

    if((std::abs(std::abs(point[0]) - 1.0) < 1e-11) ||
       (std::abs(std::abs(point[1]) - 1.0) < 1e-11) ||
       ((dim == 3 && std::abs(std::abs(point[2]) - 1.0) < 1e-11)))
      return 0.0;

    if(component == 0)
      value = 1.0;

    return value;
  }
};

template<int dim, int fe_degree, bool CATEGORIZE, typename FE_TYPE>
class Runner
{
public:
  static void
  run(ConvergenceTable & convergence_table)
  {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // setup triangulation
    parallel::distributed::Triangulation<dim> triangulation(
      MPI_COMM_WORLD,
      dealii::Triangulation<dim>::none,
      parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);

    const double left        = -1.0;
    const double right       = +1.0;
    const double deformation = +0.1;
    const double frequnency  = +2.0;

    GridGenerator::hyper_cube(triangulation, left, right);
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequnency);
    triangulation.set_all_manifold_ids(1);
    triangulation.set_manifold(1, manifold);
    triangulation.refine_global(global_refinements);

    // setup dofhandler
    FE_TYPE         fe_dgq(fe_degree);
    DoFHandler<dim> dof_handler_dg(triangulation);
    dof_handler_dg.distribute_dofs(fe_dgq);
    dof_handler_dg.distribute_mg_dofs();
    bool is_dg = (fe_dgq.dofs_per_vertex == 0);

    MappingQGeneric<dim> mapping(fe_degree);

    // setup matrixfree
    MatrixFree<dim, value_type> data;

    QGauss<1>                                            quadrature(fe_degree + 1);
    typename MatrixFree<dim, value_type>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    if(fe_dgq.dofs_per_vertex == 0)
    {
      // additional_data.build_face_info = true;
      additional_data.mapping_update_flags_inner_faces =
        (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);

      additional_data.mapping_update_flags_boundary_faces =
        (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);
    }

    AffineConstraints<double> dummy;

    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc;
    dirichlet_bc[0] = std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());

    // ...: Neumann BC: nothing to do

    // ...: Periodic BC: TODO

    // Setup constraints: for MG
    MGConstrainedDoFs mg_constrained_dofs;
    mg_constrained_dofs.clear();
    std::set<types::boundary_id> dirichlet_boundary;
    for(auto it : dirichlet_bc)
      dirichlet_boundary.insert(it.first);
    mg_constrained_dofs.initialize(dof_handler_dg);
    mg_constrained_dofs.make_zero_boundary_constraints(dof_handler_dg, dirichlet_boundary);
    if(fe_dgq.dofs_per_vertex > 0)
      dummy.add_lines(mg_constrained_dofs.get_boundary_indices(global_refinements));
    dummy.close();

    if(CATEGORIZE)
    {
      // TODO
      // additional_data.build_face_info = true;
      Categorization::do_cell_based_loops(triangulation, additional_data);
    }

    data.reinit(mapping, dof_handler_dg, dummy, quadrature, additional_data);


    if(true)
    {
      // Test operators on Poisson namespace
      std::shared_ptr<Poisson::BoundaryDescriptor<dim>> bc_poisson(
        new Poisson::BoundaryDescriptor<dim>());
      bc_poisson->dirichlet_bc[0] =
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());
      // Laplace operator
      Poisson::LaplaceOperator<dim, fe_degree, value_type> laplace;
      Poisson::LaplaceOperatorData<dim>                    laplace_additional_data;
      laplace_additional_data.bc             = bc_poisson;
      laplace_additional_data.degree_mapping = fe_degree;
      if(CATEGORIZE)
        laplace_additional_data.use_cell_based_loops = true;
      std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
        periodic_face_pairs;

      // run through all multigrid level
      if(!CATEGORIZE || true)
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
          process(laplace, laplace_additional_data, size, is_dg, 0, convergence_table, level);
        }

      // run on fine grid without multigrid
      {
        laplace.reinit(data, dummy, laplace_additional_data);
        process(laplace, laplace_additional_data, size, is_dg, 0, convergence_table);
      }
    }
    if(true)
    {
      // Test operators on ConvDiff namespace
      std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> bc_convdiff(
        new ConvDiff::BoundaryDescriptor<dim>());
      bc_convdiff->dirichlet_bc[0] =
        std::shared_ptr<Function<dim>>(new Functions::ZeroFunction<dim>());
      // Mass matrix operator
      ConvDiff::MassMatrixOperator<dim, fe_degree, value_type> mass_matrix_operator;
      ConvDiff::MassMatrixOperatorData<dim>                    mass_data;
      if(CATEGORIZE)
        mass_data.use_cell_based_loops = true;
      mass_matrix_operator.reinit(data, dummy, mass_data);

      ConvDiff::DiffusiveOperator<dim, fe_degree, value_type> diffusive_operator;
      ConvDiff::DiffusiveOperatorData<dim>                    diffusive_data;
      diffusive_data.bc             = bc_convdiff;
      diffusive_data.degree_mapping = fe_degree;
      if(CATEGORIZE)
        diffusive_data.use_cell_based_loops = true;
      diffusive_operator.reinit(data, dummy, diffusive_data);

      // Convective operator
      ConvDiff::ConvectiveOperator<dim, fe_degree, fe_degree, value_type> convective_operator;
      ConvDiff::ConvectiveOperatorData<dim>                               convective_data;
      convective_data.bc = bc_convdiff;
      if(CATEGORIZE)
        convective_data.use_cell_based_loops = true;
      convective_data.numerical_flux_formulation =
        ConvDiff::NumericalFluxConvectiveOperator::LaxFriedrichsFlux; // LaxFriedrichsFlux,
                                                                      // CentralFlux
      convective_data.type_velocity_field = ConvDiff::TypeVelocityField::Analytical;

      convective_data.velocity = std::shared_ptr<Function<dim>>(new VelocityField<dim>());
      convective_operator.reinit(data, dummy, convective_data);

      // Convection diffusion operator
      ConvDiff::ConvectionDiffusionOperatorData<dim> conv_diff_operator_data;
      conv_diff_operator_data.mass_matrix_operator_data = mass_matrix_operator.get_operator_data();
      conv_diff_operator_data.convective_operator_data  = convective_operator.get_operator_data();
      conv_diff_operator_data.diffusive_operator_data   = diffusive_operator.get_operator_data();
      conv_diff_operator_data.update_mapping_update_flags();
      conv_diff_operator_data.scaling_factor_time_derivative_term = 1.0;
      // TODO
      // conv_diff_operator_data.bc                                  = bc_convdiff;
      conv_diff_operator_data.unsteady_problem   = true;
      conv_diff_operator_data.diffusive_problem  = true;
      conv_diff_operator_data.convective_problem = true;
      conv_diff_operator_data.mg_operator_type =
        ConvDiff::MultigridOperatorType::ReactionConvectionDiffusion;
      if(CATEGORIZE)
        conv_diff_operator_data.use_cell_based_loops = true;

      process(mass_matrix_operator, mass_data, size, is_dg, 1, convergence_table);
      process(diffusive_operator, diffusive_data, size, is_dg, 2, convergence_table);
      process(convective_operator, convective_data, size, is_dg, 3, convergence_table);

      ConvDiff::ConvectionDiffusionOperator<dim, fe_degree, value_type> conv_diff_operator;
      std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
        periodic_face_pairs;

      // run through all multigrid level
      if(!CATEGORIZE || true)
        for(unsigned int level = 0; level <= global_refinements; level++)
        {
          MatrixFree<dim, value_type> matrixfree;
          AffineConstraints<double>   contraint_matrix;
          do_reinit_multigrid(dof_handler_dg,
                              mapping,
                              conv_diff_operator_data,
                              mg_constrained_dofs,
                              periodic_face_pairs,
                              level,
                              matrixfree,
                              contraint_matrix);
          conv_diff_operator.reinit(matrixfree, contraint_matrix, conv_diff_operator_data);

          process(
            conv_diff_operator, conv_diff_operator_data, size, is_dg, 4, convergence_table, level);
        }

      // run on fine grid without multigrid

      {
        conv_diff_operator.reinit(data,
                                  dummy,
                                  conv_diff_operator_data,
                                  mass_matrix_operator,
                                  convective_operator,
                                  diffusive_operator);
        process(conv_diff_operator, conv_diff_operator_data, size, is_dg, 4, convergence_table);
      }
    }
    // go to next parameter
    Runner<dim, fe_degree + 1, CATEGORIZE, FE_TYPE>::run(convergence_table);
  }

  template<typename OP, typename OPData>
  static void
  process(OP & laplace,
          OPData &,
          int                size,
          bool               is_dg,
          int                op,
          ConvergenceTable & convergence_table,
          unsigned int       mg_level = numbers::invalid_unsigned_int)
  {
    int level = std::min(global_refinements, mg_level);
    // run tests
    convergence_table.add_value("procs", size);
    convergence_table.add_value("level", level);
    convergence_table.add_value("cell", CATEGORIZE);
    convergence_table.add_value("vers", is_dg);
    convergence_table.add_value("op", op);
    OperatorBaseTest::test(
      laplace, convergence_table, true, true, true, (CATEGORIZE || size == 1) && is_dg);
    if((!CATEGORIZE && size != 1) || !is_dg)
    {
      convergence_table.add_value("(B*v)_L2", 0);
      convergence_table.add_value("(B*v-B(S)*v)_L2", 0);
    }
  }
};

template<int dim, bool categorize, typename FE_TYPE>
class Runner<dim, fe_degree_max + 1, categorize, FE_TYPE>
{
public:
  static void
  run(ConvergenceTable & /*convergence_table*/)
  {
  }
};

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  int                              rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  if(true)
  {
    ConvergenceTable convergence_table;
    // run for 2-d
    Runner<2, fe_degree_min, true, FE_DGQ<2>>::run(convergence_table);
    Runner<2, fe_degree_min, false, FE_DGQ<2>>::run(convergence_table);
    // run for 3-d
    Runner<3, fe_degree_min, true, FE_DGQ<3>>::run(convergence_table);
    Runner<3, fe_degree_min, false, FE_DGQ<3>>::run(convergence_table);
    if(!rank)
      convergence_table.write_text(std::cout);
  }

  if(true)
  {
    ConvergenceTable convergence_table;
    // run for 2-d
    // Runner<2, fe_degree_min,true, FE_Q<2>>::run(convergence_table);
    Runner<2, fe_degree_min, false, FE_Q<2>>::run(convergence_table);
    // run for 3-d
    // Runner<3, fe_degree_min,true, FE_Q<3>>::run(convergence_table);
    Runner<3, fe_degree_min, false, FE_Q<3>>::run(convergence_table);
    if(!rank)
      convergence_table.write_text(std::cout);
  }
}
