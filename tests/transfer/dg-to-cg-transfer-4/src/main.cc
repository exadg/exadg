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

#include "../../../../include/poisson/spatial_discretization/laplace_operator.h"
#include "../../../operators/operation-base-util/l2_norm.h"

#include "../../../operators/operation-base-1/src/include/rhs_operator.h"

#include "../../../operators/operation-base-util/interpolate.h"

#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"
#include "../../../../include/solvers_and_preconditioners/transfer/mg_transfer_mf_c.h"

//#define DETAIL_OUTPUT
const int PATCHES = 10;

typedef double                                         value_type;
typedef double                                         Number;
typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

const MPI_Comm comm = MPI_COMM_WORLD;

using namespace dealii;
using namespace Poisson;

template<int dim, int fe_degree_1>
class Runner
{
  static const int fe_degree_2 = fe_degree_1;

public:
  Runner()
  {
  }

public:
  void
  run()
  {
    double                                    left = -1, right = +1;
    parallel::distributed::Triangulation<dim> triangulation(comm);

    GridGenerator::hyper_cube(triangulation, left, right);

    for(auto & cell : triangulation)
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        // x-direction
        if((std::fabs(cell.face(face_number)->center()(0) - left) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(0);
        else if((std::fabs(cell.face(face_number)->center()(0) - right) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(1);
        // y-direction
        else if((std::fabs(cell.face(face_number)->center()(1) - left) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(2);
        else if((std::fabs(cell.face(face_number)->center()(1) - right) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(3);
        // z-direction
        else if(dim == 3 && (std::fabs(cell.face(face_number)->center()(2) - left) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(4);
        else if(dim == 3 && (std::fabs(cell.face(face_number)->center()(2) - right) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(5);
      }

    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_faces;
    GridTools::collect_periodic_faces(triangulation, 0, 1, 0 /*x-direction*/, periodic_faces);
    GridTools::collect_periodic_faces(triangulation, 2, 3, 1 /*y-direction*/, periodic_faces);
    if(dim == 3)
      GridTools::collect_periodic_faces(triangulation, 4, 5, 2 /*z-direction*/, periodic_faces);
    triangulation.add_periodicity(periodic_faces);


    const double                     deformation = +0.1;
    const double                     frequnency  = +2.0;
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequnency);
    triangulation.set_all_manifold_ids(1);
    triangulation.set_manifold(1, manifold);
    triangulation.refine_global(1);



    std::shared_ptr<FESystem<dim>> fe_u_1;
    fe_u_1.reset(new FESystem<dim>(FE_DGQ<dim>(fe_degree_1), dim));
    MappingQGeneric<dim> mapping_1(fe_degree_1 + 1);
    DoFHandler<dim>      dof_handler_1(triangulation);
    dof_handler_1.distribute_dofs(*fe_u_1);

    typename MatrixFree<dim, Number>::AdditionalData additional_data_1;
    additional_data_1.mapping_update_flags = (update_JxW_values | update_values);

    MatrixFree<dim, Number> data_1;

    QGauss<1>                 quadrature_1(fe_degree_1 + 1);
    AffineConstraints<double> constraint_1;
    constraint_1.close();
    data_1.reinit(mapping_1, dof_handler_1, constraint_1, quadrature_1, additional_data_1);



    std::shared_ptr<FESystem<dim>> fe_u_2;
    fe_u_2.reset(new FESystem<dim>(FE_Q<dim>(fe_degree_2), dim));
    MappingQGeneric<dim> mapping_2(fe_degree_2 + 1);
    DoFHandler<dim>      dof_handler_2(triangulation);
    dof_handler_2.distribute_dofs(*fe_u_2);

    typename MatrixFree<dim, Number>::AdditionalData additional_data_2;
    additional_data_2.mapping_update_flags = (update_JxW_values | update_values);

    MatrixFree<dim, Number> data_2;

    QGauss<1>                 quadrature_2(fe_degree_2 + 1);
    AffineConstraints<double> constraint_2;
    constraint_2.close();
    data_2.reinit(mapping_2, dof_handler_2, constraint_2, quadrature_2, additional_data_2);



    MGTransferMFC<dim, value_type, VectorType, dim> transfer(
      data_1, data_2, constraint_1, constraint_2, -1, fe_degree_1);

    LinearAlgebra::distributed::Vector<value_type> vec_1, vec_2;
    data_1.initialize_dof_vector(vec_1);
    data_2.initialize_dof_vector(vec_2);

    transfer.restrict_and_add(0, vec_2, vec_1);
    transfer.prolongate(0, vec_1, vec_2);
  }
};

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  {
    {
      Runner<2, 7> run_cg;
      run_cg.run();
    }
    {
      Runner<2, 3> run_cg;
      run_cg.run();
    }
  }
}
