#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include <deal.II/fe/mapping_fe_field.h>

#include "moving_mesh_dat.h"
#include "../../../applications/grid_tools/mesh_movement_functions.h"
#include "../include/time_integration/push_back_vectors.h"
//#include "interface.h"




using namespace dealii;

namespace IncNS
{


template<int dim, typename Number>
class MovingMesh
{
public:
  typedef MappingFEField<dim,dim,LinearAlgebra::distributed::Vector<Number>> MappingField;
  typedef MappingQGeneric<dim> MappingQ;
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  MovingMesh(MovingMeshData const & data_in,
              parallel::Triangulation<dim> const & triangulation_in)
  :moving_mesh_data(data_in),
   fe_grid(new FESystem<dim>(FE_Q<dim>(data_in.degree_u), dim)),
   fe_u_grid(new FESystem<dim>(FE_DGQ<dim>(data_in.degree_u), dim)),
   dof_handler_grid(triangulation_in),
   dof_handler_u_grid(triangulation_in),
   position_grid_new_multigrid(triangulation_in.n_global_levels()),
   d_grid(data_in.order_time_integrator + 1),
   dof_handler_d_grid(triangulation_in)
  {
  }


  void initialize_velocity_high_order_start(double time_in)
  {

        move_mesh(time_in);
//        update();
        get_analytical_grid_velocity(time_in);

      push_back(d_grid);
      fill_d_grid();

}


  void
  initialize(std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel_in,
        std::shared_ptr<MappingQGeneric<dim>>     const &                     mapping_init_in,
        MatrixFree<dim, Number> matrix_free_in,
        std::shared_ptr<FieldFunctions<dim>> const    field_functions_in)
  {
    field_functions = field_functions_in;
    matrix_free = matrix_free_in;
    convective_kernel = convective_kernel_in;
    mapping_init = mapping_init_in;
   // time_operator_base = std::make_shared<Interface::TimeOperatorBase<Number>>();

    dof_handler_grid.distribute_dofs(*fe_grid);
    dof_handler_grid.distribute_mg_dofs();
    dof_handler_u_grid.distribute_dofs(*fe_u_grid);
    dof_handler_u_grid.distribute_mg_dofs();
    dof_handler_d_grid.distribute_dofs(*fe_u_grid);
    dof_handler_d_grid.distribute_mg_dofs();

    initialize_mapping_field();

    initialize_d_grid_and_u_grid_np();



  }

//  void set_dof_handler_vec(std::vector<const DoFHandler<dim> *> dof_handler_vec_in){
//    dof_handler_vec = dof_handler_vec_in;
//    std::cout<<"dof_handler_vec initialized"<<std::endl;
//  }
//  void set_constraint_matrix_vec(std::vector<const AffineConstraints<double> /***/> constraint_matrix_vec_in){
//    constraint_matrix_vec = constraint_matrix_vec_in;
//    std::cout<<"dof_handler_vec initialized"<<std::endl;
//  }

//  void set_test(AffineConstraints<double> constraint_u_in,
//      AffineConstraints<double> constraint_p_in,
//      AffineConstraints<double> constraint_u_scalar_in,
//      const unsigned int dof_index_u_in,
//      const unsigned int dof_index_p_in,
//      const unsigned int dof_index_u_scalar_in)
//  {
//    constraint_u=constraint_u_in;
//    constraint_p=constraint_p_in;
//    constraint_u_scalar=constraint_u_scalar_in;
//    dof_index_u = dof_index_u_in;
//    dof_index_p = dof_index_p_in;
//    dof_index_u_scalar = dof_index_u_scalar_in;
//
//    constraint_u.close();
//    constraint_p.close();
//    constraint_u_scalar.close();
//    constraint_matrix_vec[dof_index_u]        = &constraint_u;
//    constraint_matrix_vec[dof_index_p]        = &constraint_p;
//    constraint_matrix_vec[dof_index_u_scalar] = &constraint_u_scalar;
//  }
//
//  void set_quadratures(std::vector<Quadrature<1>> quadratures_in){
//    quadratures = quadratures_in;
//    std::cout<<"quadratures initialized"<<std::endl;
//  }
//  void set_additional_data(typename MatrixFree<dim, Number>::AdditionalData additional_data_in){
//    additional_data = additional_data_in;
//    additional_data.mapping_update_flags                = ale_update_flags;
//    additional_data.mapping_update_flags_inner_faces    = ale_update_flags;
//    additional_data.mapping_update_flags_boundary_faces = ale_update_flags;
//    std::cout<<"additional data initialized"<<std::endl;
//  }
//
  void
  advance_mesh(double time_in)
  {
    move_mesh(time_in);

//    update();
  }

  void
  set_grid_velocities(double time_in)
  {

    if(moving_mesh_data.u_ana==true)
      get_analytical_grid_velocity(time_in);
    else
     compute_grid_velocity();
//
    convective_kernel->set_velocity_grid_ptr(u_grid_np);
  }

  Mapping<dim> const &
  get_mapping() const
  {
      return *mapping;
  }

private:

  void
  get_analytical_grid_velocity(double const evaluation_time)
  {
    field_functions->analytical_solution_grid_velocity->set_time_velocity(evaluation_time);

    // This is necessary if Number == float
    typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

    VectorTypeDouble grid_velocity_double;
    grid_velocity_double = u_grid_np;

    VectorTools::interpolate(*mapping_init,
                             dof_handler_u_grid,
                             *(field_functions->analytical_solution_grid_velocity),
                             grid_velocity_double);//!!!should be grid_velocity_double

    u_grid_np = grid_velocity_double;
  }


//  void
//  update()
//  {
   // matrix_free.reinit(get_mapping(), dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
//  }

  void
  move_mesh(double time_in)
  {
    field_functions->analytical_solution_grid_velocity->set_time_displacement(time_in);
    interpolate_mg<MappingQ>(*mapping_init);
  }


  void
  initialize_mapping_field()
  {
    field_functions->analytical_solution_grid_velocity->set_time_displacement(0.0);
    interpolate_mg<MappingQ>(*mapping_init);
    mapping.reset(new MappingFEField<dim,dim,LinearAlgebra::distributed::Vector<Number>> (dof_handler_grid, position_grid_new_multigrid));
  }

  /*
   * Two cases can occur:
   * The user wants to advance the mesh from the one of the last timestep:
   *  This is the case if a mesh moving algorithm is applied
   * The user wants to advance the mesh with an analytical mesh movement function and wants to use the initial mesh as a reference
   * Hence:
   * MappingTypeIn can be MappingQ(Case2) or MappingField(Case1)
   */
  template <class MappingTypeIn>
  void interpolate_mg(MappingTypeIn& mapping_in)
  {
          VectorType position_grid_init;
          VectorType displacement_grid;
          VectorType position_grid_new;
          IndexSet relevant_dofs_grid;


          unsigned int nlevel = dof_handler_grid.get_triangulation().n_global_levels();
          for (unsigned int level=0; level<nlevel; ++level)
          {
          DoFTools::extract_locally_relevant_level_dofs(dof_handler_grid, level,
              relevant_dofs_grid);

          position_grid_init.reinit(dof_handler_grid.locally_owned_mg_dofs(level), relevant_dofs_grid, MPI_COMM_WORLD);
          displacement_grid.reinit(dof_handler_grid.locally_owned_mg_dofs(level), relevant_dofs_grid, MPI_COMM_WORLD);
          position_grid_new.reinit(dof_handler_grid.locally_owned_mg_dofs(level), relevant_dofs_grid, MPI_COMM_WORLD);

          FEValues<dim> fe_values(mapping_in, *fe_grid,
                                    Quadrature<dim>(fe_grid->get_unit_support_points()),
                                    update_quadrature_points);
            std::vector<types::global_dof_index> dof_indices(fe_grid->dofs_per_cell);
            for (const auto & cell : dof_handler_grid.mg_cell_iterators_on_level(level))
              if (cell->level_subdomain_id() != numbers::artificial_subdomain_id)
                {
                  fe_values.reinit(cell);
                  cell->get_active_or_mg_dof_indices(dof_indices);
                  for (unsigned int i=0; i<fe_grid->dofs_per_cell; ++i)
                    {
                      const unsigned int coordinate_direction =
                          fe_grid->system_to_component_index(i).first;
                      const Point<dim> point = fe_values.quadrature_point(i);
                        double displacement=0.0;
                      for (unsigned int d=0; d<dim; ++d)
                          displacement = field_functions->analytical_solution_grid_velocity->displacement(point, coordinate_direction);

                      position_grid_init(dof_indices[i]) = point[coordinate_direction];
                      displacement_grid(dof_indices[i]) = displacement;
                    }
                }

          position_grid_new  = position_grid_init;
          position_grid_new += displacement_grid;
          position_grid_new_multigrid[level] = position_grid_new;
          position_grid_new_multigrid[level].update_ghost_values();
          }
  }

  void
  compute_grid_velocity()
  {
    push_back(d_grid);
    fill_d_grid();
    //time_operator_base->compute_BDF_time_derivative(u_grid_np, d_grid);
  }
  void
  initialize_d_grid_and_u_grid_np()
  {

    for(unsigned int i = 0; i < d_grid.size(); ++i)
      {
      matrix_free.initialize_dof_vector(d_grid[i], moving_mesh_data.dof_index_u);
      d_grid[i].update_ghost_values();
      }
    matrix_free.initialize_dof_vector(u_grid_np, moving_mesh_data.dof_index_u);

    fill_d_grid();

  }





  void
  fill_d_grid()
  {


    IndexSet relevant_dofs_grid;
    DoFTools::extract_locally_relevant_dofs(dof_handler_d_grid,
        relevant_dofs_grid);

    d_grid[0].reinit(dof_handler_d_grid.locally_owned_dofs(), relevant_dofs_grid, MPI_COMM_WORLD);

    FEValues<dim> fe_values(get_mapping(), *fe_u_grid,
                            Quadrature<dim>((*fe_u_grid).get_unit_support_points()),
                            update_quadrature_points);
    std::vector<types::global_dof_index> dof_indices((*fe_u_grid).dofs_per_cell);
    for (const auto & cell : dof_handler_d_grid.active_cell_iterators())
    {

      if (!cell->is_artificial())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(dof_indices);
          for (unsigned int i=0; i<(*fe_u_grid).dofs_per_cell; ++i)
            {
              const unsigned int coordinate_direction =
                  (*fe_u_grid).system_to_component_index(i).first;
              const Point<dim> point = fe_values.quadrature_point(i);
              d_grid[0](dof_indices[i]) = point[coordinate_direction];
            }
        }
    }
    d_grid[0].update_ghost_values();
  }


  MovingMeshData moving_mesh_data;
  std::shared_ptr<Operators::ConvectiveKernel<dim, Number>> convective_kernel;

  std::shared_ptr<FESystem<dim>> fe_grid;
  std::shared_ptr<FESystem<dim>> fe_u_grid;
  DoFHandler<dim> dof_handler_grid;
  DoFHandler<dim> dof_handler_u_grid;

  std::shared_ptr<MappingField> mapping;
  std::vector<VectorType> position_grid_new_multigrid;
  std::shared_ptr<MappingQ> mapping_init;

  std::vector<const DoFHandler<dim> *> dof_handler_vec;
  std::vector<const AffineConstraints<double> *> constraint_matrix_vec;
  UpdateFlags ale_update_flags= (update_gradients |
                                 update_JxW_values |
                                 update_quadrature_points |
                                 update_normal_vectors |
                                 update_values |
                                 update_inverse_jacobians /*CFL*/);

  std::vector<Quadrature<1>> quadratures;
  typename MatrixFree<dim, Number>::AdditionalData additional_data;



  MatrixFree<dim, Number> matrix_free;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  LinearAlgebra::distributed::Vector<Number> u_grid_np;
  std::vector<LinearAlgebra::distributed::Vector<Number>> d_grid;
//
//  AffineConstraints<double> constraint_u;
//        AffineConstraints<double> constraint_p;
//        AffineConstraints<double> constraint_u_scalar;
//        unsigned int dof_index_u;
//        unsigned int dof_index_p;
//        unsigned int dof_index_u_scalar;
  DoFHandler<dim> dof_handler_d_grid;

  //std::shared_ptr<Interface::TimeOperatorBase<Number>> time_operator_base;

};

}/*IncNS*/
#endif /*INCLUDE_MOVING_MESH_H_*/
