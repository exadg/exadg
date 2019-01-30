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

using namespace dealii;
using namespace Poisson;

const unsigned int                                     global_refinements = 2;
typedef double                                         value_type;
const int                                              fe_degree_min = 1;
const int                                              fe_degree_max = 3;
typedef LinearAlgebra::distributed::Vector<value_type> VectorType;


typedef double value_type;

using namespace dealii;

template<int dim, int fe_degree, typename Number>
void
calculate_penalty_parameter(AlignedVector<VectorizedArray<Number>> & array_penalty_parameter,
                            MatrixFree<dim, Number> const &          data)
{
  unsigned int n_cells = data.n_cell_batches() + data.n_ghost_cell_batches();
  array_penalty_parameter.resize(n_cells);

  for(unsigned int i = 0; i < n_cells; ++i)
  {
    for(unsigned int v = 0; v < data.n_components_filled(i); ++v)
    {
      auto s                        = data.get_cell_iterator(i, v)->id().to_string();
      auto ss                       = s.substr(s.length() - 2);
      array_penalty_parameter[i][v] = (ss.at(0) - 48) * 4 + (ss.at(1) - 48);
      // std::cout << data.get_cell_iterator(i, v)->id() <<  " -> " << (ss.at(0)-48)*4+(ss.at(1)-48)
      // << std::endl;
    }
  }
}

template<int dim, int fe_degree, int n_q_points_1d = fe_degree + 1, typename number = double>
class CellBasedOperator : public Subscriptor
{
public:
  typedef number                                value_type;
  typedef MatrixFree<dim, number>               MF;
  typedef std::pair<unsigned int, unsigned int> Range;
  typedef CellBasedOperator                     This;
  const unsigned int                            nr_faces = GeometryInfo<dim>::faces_per_cell;

  CellBasedOperator(MatrixFree<dim, number> & data) : data(data)
  {
    calculate_penalty_parameter<dim, fe_degree, number>(ip, this->data);
    data.initialize_dof_vector(src, 0);
    data.initialize_dof_vector(dst, 0);
  };

  void
  apply_loop() const
  {
    data.loop(&This::local_diagonal_cell,
              &This::local_diagonal_face,
              &This::local_diagonal_boundary,
              this,
              src,
              src);
  }

  void
  apply() const
  {
    printf("\n");
    src.update_ghost_values();
    data.cell_loop(&This::local_diagonal_by_cell, this, dst, src);
    printf("\n");
  }

  mutable VectorType src, dst;

private:
  void
  local_diagonal_by_cell(const MF & data,
                         VectorType & /*dst*/,
                         const VectorType & src,
                         const Range &      cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number>     phi(data);
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phif_1(data, true);
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phif_2(data, false);

    for(auto cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      // do stuff
      auto temp = phi.read_cell_data(ip);
      printf("c   ");
      for(int i = 0; i < 4; i++)
        printf("%3d ", (int)temp[i]);
      printf("\n");

      for(unsigned int face = 0; face < nr_faces; ++face)
      {
        phif_1.reinit(cell, face);
        phif_2.reinit(cell, face);
        auto bids = data.get_faces_by_cells_boundary_id(cell, face);
        auto bid  = bids[0];
        if(bid == numbers::internal_face_boundary_id)
        {
          {
            phif_1.read_dof_values(src);
            printf("m-%d ", face);
            for(unsigned int i = 0; i < 1 /*phif_1.static_dofs_per_face*/; i++)
              for(int v = 0; v < 4; v++)
                printf("%7.3f ", phif_1.begin_dof_values()[i][v]);
            printf("\n");
          }
          {
            phif_2.read_dof_values(src);
            printf("p-%d ", face);
            for(unsigned int i = 0; i < 1 /*phif_2.static_dofs_per_face*/; i++)
              for(int v = 0; v < 4; v++)
                printf("%7.3f ", phif_2.begin_dof_values()[i][v]);
            printf("\n");
          }
        }
        //                {
        //                auto temp = phif_2.read_cell_data(ip);
        //                printf("p-%d ", face);
        //                for (int i = 0; i < 4; i++)
        //                    printf("%3d ", (int) temp[i]);
        //                printf("\n");
        //                }
      }
      printf("\n");
    }
  }

  void
  local_diagonal_cell(const MF &   data,
                      VectorType & dst,
                      const VectorType &,
                      const Range & cell_range) const
  {
    FEEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi(data);

    for(auto cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      printf("c ");
      auto temp = phi.read_cell_data(ip);
      for(int i = 0; i < 4; i++)
        printf("%3d ", (int)temp[i]);

      for(unsigned int i = 0; i < phi.static_dofs_per_cell; i++)
        phi.begin_dof_values()[i] = temp;

      phi.distribute_local_to_global(dst);

      printf("\n");
    }
  }

  void
  local_diagonal_face(const MF & data,
                      VectorType &,
                      const VectorType &,
                      const Range & cell_range) const
  {
    FEFaceEvaluation<dim, fe_degree, n_q_points_1d, 1, number> phi(data);

    for(auto cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      phi.reinit(cell);
      auto temp = phi.read_cell_data(ip);
      printf("f ");
      for(int i = 0; i < 4; i++)
        printf("%3d ", (int)temp[i]);
      printf("\n");
    }
  }

  void
  local_diagonal_boundary(const MF &, VectorType &, const VectorType &, const Range &) const
  {
  }

  MatrixFree<dim, number> &              data;
  AlignedVector<VectorizedArray<number>> ip;
};

template<int dim, int fe_degree, bool CATEGORIZE, typename FE_TYPE>
class Runner
{
public:
  static void
  run(ConvergenceTable & /*convergence_table*/)
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
    // bool is_dg = (fe_dgq.dofs_per_vertex == 0);

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

    Categorization::do_cell_based_loops(triangulation, additional_data);

    data.reinit(mapping, dof_handler_dg, dummy, quadrature, additional_data);

    CellBasedOperator<dim, fe_degree> lo(data);
    lo.apply_loop();
    lo.apply();
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
    Runner<2, fe_degree_min, false, FE_DGQ<2>>::run(convergence_table);
  }
}
