#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>
#include <deal.II/base/std_cxx14/memory.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/manifold.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>


#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

#include "../../../../applications/incompressible_navier_stokes_test_cases/deformed_cube_manifold.h"

#include "../../../../include/incompressible_navier_stokes/spatial_discretization/operators/mass_matrix_operator.h"
#include "operators/mass_operator.h"
#include "operators/helmholtz_operator.h"
#include "../../operation-base-util/sparse_matrix_util.h"

const int      best_of = 10;
typedef double Number;

const MPI_Comm comm = MPI_COMM_WORLD;


template<int dim, int fe_degree>
class Run
{
public:
  static void
  run(bool use_dg)
  {
    double                                    left = -1, right = +1;
    parallel::distributed::Triangulation<dim> triangulation(comm);

    GridGenerator::hyper_cube(triangulation, left, right);

#if false
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
        else if(dim==3 && (std::fabs(cell.face(face_number)->center()(2) - left) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(4);
        else if(dim==3 && (std::fabs(cell.face(face_number)->center()(2) - right) < 1e-12))
          cell.face(face_number)->set_all_boundary_ids(5);
      }

    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
      periodic_faces;
    GridTools::collect_periodic_faces(triangulation, 0, 1, 0, periodic_faces);
    GridTools::collect_periodic_faces(triangulation, 2, 3, 1, periodic_faces);
    if (dim==3)
        GridTools::collect_periodic_faces(triangulation, 4, 5, 2, periodic_faces);
    triangulation.add_periodicity(periodic_faces);
#endif
    
    
    const double deformation = +0.1;
    const double frequnency  = +2.0;
    static DeformedCubeManifold<dim> manifold(left, right, deformation, frequnency);
    triangulation.set_all_manifold_ids(1);
    triangulation.set_manifold(1, manifold);
    triangulation.refine_global(1);
    
    std::shared_ptr<FESystem<dim>> fe_u;
    if(use_dg)
        fe_u.reset(new FESystem<dim>(FE_DGQ<dim>(fe_degree), dim));
    else
        fe_u.reset(new FESystem<dim>(FE_Q<dim>(fe_degree), dim));
    MappingQGeneric<dim> mapping(fe_degree+1);
    DoFHandler<dim> dof_handler_u(triangulation);
    dof_handler_u.distribute_dofs(*fe_u);
    
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.mapping_update_flags =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  additional_data.mapping_update_flags_inner_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

  additional_data.mapping_update_flags_boundary_faces =
    (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
     update_values);

    MatrixFree<dim, Number> data;
    
    QGauss<1> quadrature(fe_degree+1);
    AffineConstraints<double> constraint_u; constraint_u.close();
    data.reinit(mapping, dof_handler_u, constraint_u, quadrature, additional_data);
    
    IncNS::MassMatrixOperator<dim,fe_degree,Number> mm1;
    IncNS::MassMatrixOperatorData mmo1;
    mm1.initialize(data, mmo1);
    
    IncNS::MassMatrixOperatorNew<dim,fe_degree,Number> mm2;
    IncNS::MassMatrixOperatorDataNew<dim> mmo2;
    mm2.reinit(data, constraint_u, mmo2);
    
    std::cout << "EXPERIMENTS " <<
            (use_dg ? "DG " : "CG ") <<
            dim << "D " << fe_degree << " degree" << std::endl;

    {
        // test apply/vmult
        LinearAlgebra::distributed::Vector<Number> v1,v2,v3,v4;

        mm2.do_initialize_dof_vector(v1);
        mm2.do_initialize_dof_vector(v2);
        mm2.do_initialize_dof_vector(v3);
        mm2.do_initialize_dof_vector(v4);
        v1 = 1.0;
        v2 = 1.0;

        mm1.apply(v3,v1);
        mm2.apply_add(v4,v2);
        std::cout << v3.l2_norm() << " " << v4.l2_norm() << std::endl;
    }
    
    {
        // create Helmholtz-operator
        IncNS::HelmholtzOperatorNew<dim,fe_degree,Number> ho2;
        IncNS::HelmholtzOperatorDataNew<dim> hoo2;
#if true
      std::shared_ptr<Function<dim> > zero_function_vectorial;
      zero_function_vectorial.reset(new Functions::ZeroFunction<dim>(dim));
      IncNS::BoundaryDescriptorU<dim>* bd = new IncNS::BoundaryDescriptorU<dim>();
      bd->dirichlet_bc.insert(
          std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,zero_function_vectorial));
      hoo2.bc.reset(bd);
#endif
        
        ho2.reinit(mapping, data, constraint_u, hoo2);
        
        // test apply/vmult
        LinearAlgebra::distributed::Vector<Number> v2,v4;

        ho2.do_initialize_dof_vector(v2);
        ho2.do_initialize_dof_vector(v4);
        v2 = 0.0;
        v2 = 1.0;

        ho2.apply_add(v4,v2);
        std::cout << v4.l2_norm() << std::endl;
    }

    {
        // test apply/vmult
        LinearAlgebra::distributed::Vector<Number> v1;
        mm2.do_initialize_dof_vector(v1);
        v1 = 0.0;
        mm1.calculate_diagonal(v1);
        
        LinearAlgebra::distributed::Vector<Number> v2;
        mm2.calculate_diagonal(v2);
        std::cout << v1.l2_norm() << " " << v2.l2_norm() << std::endl;
    }
    
    if(use_dg){
        // test block diagonal (has to be same as apply)
        LinearAlgebra::distributed::Vector<Number> v1, v2;
        mm2.do_initialize_dof_vector(v2);
        mm2.do_initialize_dof_vector(v1);
        v2 = 1.0;
        mm2.calculate_block_diagonal_matrices();
        mm2.apply_block_diagonal_matrix_based(v1, v2);
        
        std::cout << v1.l2_norm() << std::endl;
    }
    
    {
        TrilinosWrappers::SparseMatrix system_matrix;
        mm2.do_init_system_matrix(system_matrix);
        mm2.do_calculate_system_matrix(system_matrix);
        
        LinearAlgebra::distributed::Vector<Number> v1,v2;
        mm2.do_initialize_dof_vector(v1);
        mm2.do_initialize_dof_vector(v2);
        v2 = 1.0;
        system_matrix.vmult(v1,v2);
        
        //print_matlab(system_matrix);
        
        std::cout << v1.l2_norm() << std::endl;
    }
    std::cout << std::endl;
  }
};

template<int dim>
void
run()
{
  Run<dim,  1>::run(true);
  Run<dim,  1>::run(false);
  Run<dim,  2>::run(true);
  Run<dim,  2>::run(false);
  Run<dim,  3>::run(true);
  Run<dim,  3>::run(false);
}

int
main(int argc, char ** argv)
{
  Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);
  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

  run<2>();
  
}