#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

//#include <deal.II/fe/mapping_fe_field.h>
#include "../include/time_integration/push_back_vectors.h"
#include "../../../applications/grid_tools/mesh_movement_functions.h"
#include <deal.II/lac/la_parallel_block_vector.h>

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
  typedef LinearAlgebra::distributed::BlockVector<Number> BlockVectorType;

  MovingMesh(InputParameters const & param_in,
              std::shared_ptr<parallel::Triangulation<dim>> triangulation_in,
              std::shared_ptr<FieldFunctions<dim>> const    field_functions_in,
              std::shared_ptr<DGNavierStokesBase<dim, Number>> navier_stokes_operation_in)
  :param(param_in),
   field_functions(field_functions_in),
   navier_stokes_operation(navier_stokes_operation_in),
   fe_grid(new FESystem<dim>(FE_Q<dim>(param_in.degree_u), dim)),
   fe_u_grid(new FESystem<dim>(FE_DGQ<dim>(param_in.degree_u), dim)),
   dof_handler_grid(*triangulation_in),
   dof_handler_u_grid(*triangulation_in),
   dof_handler_d_grid(*triangulation_in),
   position_grid_new_multigrid((*triangulation_in).n_global_levels()),
   d_grid(param_in.order_time_integrator + 1)
  {
    mapping_init = navier_stokes_operation->get_mapping_init();
    initialize_dof_handler();
    initialize_mapping_field();
  }

  void setup()
  {
  initialize_vectors();
  initialize_ALE_update_data();
  }

  void
  move_mesh(const double time_in,
            const double time_step_size, //= 0.0,
            const std::vector<Number> time_integrator_constants)// = std::vector<Number>(0))
  {
    advance_mesh(time_in);

    navier_stokes_operation->ALE_update(dof_handler_vec_ALE, constraint_matrix_vec_ALE, quadratures_ALE, additional_data_ALE);


    if(param.grid_velocity_analytical==true)
      get_analytical_grid_velocity(time_in);
    else
     compute_grid_velocity(time_integrator_constants, time_step_size);

      navier_stokes_operation->set_grid_velocity(u_grid_np);
  }

  void init_d_grid_on_former_mesh(std::vector<double> eval_times)
  {
    //Iterating backwards leaves us with the mesh at start time automatically
    for (unsigned int i=param.order_time_integrator-1;i <param.order_time_integrator;--i)
    {
      advance_mesh(eval_times[i]);
      navier_stokes_operation->ALE_update(dof_handler_vec_ALE, constraint_matrix_vec_ALE, quadratures_ALE, additional_data_ALE);

      get_analytical_grid_velocity(eval_times[i]);

      fill_d_grid(i);
    }

  }

  std::vector<BlockVectorType>
  init_former_solution_on_former_mesh(std::vector<double> eval_times)
  {
  std::vector<BlockVectorType> solution(param.order_time_integrator);

  //Iterating backwards leaves us with the mesh at start time automatically
  for(unsigned int i = 0; i < solution.size(); ++i)
  {
    solution[i].reinit(2);
    navier_stokes_operation->initialize_vector_velocity(solution[i].block(0));
    navier_stokes_operation->initialize_vector_pressure(solution[i].block(1));
  }

  for (unsigned int i=param.order_time_integrator-1;i <param.order_time_integrator;--i)
  {
    advance_mesh(eval_times[i]);
    navier_stokes_operation->ALE_update(dof_handler_vec_ALE, constraint_matrix_vec_ALE, quadratures_ALE, additional_data_ALE);

    navier_stokes_operation->prescribe_initial_conditions(solution[i].block(0),
                                                          solution[i].block(1),
                                                          eval_times[i]);
  }

  return solution;

  }

  std::vector<VectorType>
  init_convective_term_on_former_mesh(std::vector<double> eval_times)
  {
    std::vector<BlockVectorType> solution = init_former_solution_on_former_mesh(eval_times);
    std::vector<VectorType> vec_convective_term(param.order_time_integrator);

    for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
      navier_stokes_operation->initialize_vector_velocity(vec_convective_term[i]);

    //Iterating backwards leaves us with the mesh at start time automatically
    for (unsigned int i=param.order_time_integrator-1;i <param.order_time_integrator;--i)
    {
      advance_mesh(eval_times[i]);
      navier_stokes_operation->ALE_update(dof_handler_vec_ALE, constraint_matrix_vec_ALE, quadratures_ALE, additional_data_ALE);
      get_analytical_grid_velocity(eval_times[i]);
      navier_stokes_operation->set_grid_velocity(u_grid_np);

      navier_stokes_operation->evaluate_convective_term(vec_convective_term[i],
                                                        solution[i].block(0),
                                                        eval_times[i]);

      //IT IS IMPORTANT TO OVERWRITE CONVECTIVE TERM [0] since now we have velocity at this time
    }

    return vec_convective_term;

  }

private:
  void initialize_dof_handler()
  {
    dof_handler_grid.distribute_dofs(*fe_grid);
    dof_handler_grid.distribute_mg_dofs();
    dof_handler_u_grid.distribute_dofs(*fe_u_grid);
    dof_handler_u_grid.distribute_mg_dofs();
    dof_handler_d_grid.distribute_dofs(*fe_u_grid);
    dof_handler_d_grid.distribute_mg_dofs();
  }

    void
    initialize_vectors()
    {
      navier_stokes_operation->initialize_vector_velocity(u_grid_np);

      if(param.grid_velocity_analytical==false)
      {
        for(unsigned int i = 0; i < d_grid.size(); ++i)
          {
          navier_stokes_operation->initialize_vector_velocity(d_grid[i]);
          d_grid[i].update_ghost_values();
          }

        fill_d_grid(0);
      }
    }






  Mapping<dim> const &
  get_mapping() const
  {
      return *mapping;
  }

  void
  get_analytical_grid_velocity(double const evaluation_time)
  {
    field_functions->analytical_solution_grid_velocity->set_time_velocity(evaluation_time);

    // This is necessary if Number == float
    typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

    VectorTypeDouble grid_velocity_double;
    grid_velocity_double = u_grid_np;

    VectorTools::interpolate(*mapping_init,//-->coordinates on which analytical_solution_grid_velocity() relis
                             dof_handler_u_grid,
                             *(field_functions->analytical_solution_grid_velocity),
                             grid_velocity_double);

    u_grid_np = grid_velocity_double;

   // std::cout<<"Set u_grid at time t=" <<evaluation_time<<" leats to L2 norm"<<u_grid_np.l2_norm()<<std::endl;//TEST
  }

  void
  advance_mesh(double time_in)
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
    navier_stokes_operation->set_mapping_ALE(mapping);
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
  compute_grid_velocity(std::vector<double> time_integrator_constants, double time_step_size)
  {
    push_back(d_grid);
    fill_d_grid(0);
    compute_BDF_time_derivative(u_grid_np, d_grid, time_integrator_constants, time_step_size);
  }


  void
  compute_BDF_time_derivative(VectorType & dst,
                              std::vector<VectorType> src,
                              std::vector<double> time_integrator_constants,
                              double time_step_size)
  {

      dst.equ(time_integrator_constants[0]/time_step_size,src[0]);

      for(unsigned int i = 1; i < src.size(); ++i)
        dst.add(-1*time_integrator_constants[i]/time_step_size,src[i]);

  }




  void
  fill_d_grid(int component)
  {
    IndexSet relevant_dofs_grid;
    DoFTools::extract_locally_relevant_dofs(dof_handler_d_grid,
        relevant_dofs_grid);

    d_grid[component].reinit(dof_handler_d_grid.locally_owned_dofs(), relevant_dofs_grid, MPI_COMM_WORLD);

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
              d_grid[component](dof_indices[i]) = point[coordinate_direction];
            }
        }
    }
    d_grid[component].update_ghost_values();
  }


  void
  initialize_ALE_update_data()
  {

    additional_data_ALE.tasks_parallel_scheme = MatrixFree<dim, Number>::AdditionalData::partition_partition;
    additional_data_ALE.initialize_indices = false; //connectivity of elements stays the same
    additional_data_ALE.initialize_mapping = true;
    additional_data_ALE.mapping_update_flags =ale_update_flags;
    additional_data_ALE.mapping_update_flags_inner_faces = ale_update_flags;
    additional_data_ALE.mapping_update_flags_boundary_faces =ale_update_flags;

    auto& dof_handler_u = navier_stokes_operation->get_dof_handler_u();
    auto& dof_handler_u_scalar = navier_stokes_operation->get_dof_handler_u_scalar();
    auto& dof_handler_p = navier_stokes_operation->get_dof_handler_p();

    if(param.use_cell_based_face_loops)
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &dof_handler_u.get_triangulation());
      Categorization::do_cell_based_loops(*tria, additional_data_ALE);
    }

    dof_handler_vec_ALE.resize(3);
    dof_handler_vec_ALE[navier_stokes_operation->get_dof_index_velocity()]        = &dof_handler_u;
    dof_handler_vec_ALE[navier_stokes_operation->get_dof_index_pressure()]        = &dof_handler_p;
    dof_handler_vec_ALE[navier_stokes_operation->get_dof_index_velocity_scalar()] = &dof_handler_u_scalar;

    constraint_matrix_vec_ALE.resize(3);

    constraint_u_ALE.close();
    constraint_p_ALE.close();
    constraint_u_scalar_ALE.close();
    constraint_matrix_vec_ALE[navier_stokes_operation->get_dof_index_velocity()]        = &constraint_u_ALE;
    constraint_matrix_vec_ALE[navier_stokes_operation->get_dof_index_pressure()]        = &constraint_p_ALE;
    constraint_matrix_vec_ALE[navier_stokes_operation->get_dof_index_velocity_scalar()] = &constraint_u_scalar_ALE;

    quadratures_ALE.resize(3);
    quadratures_ALE[navier_stokes_operation->get_quad_index_velocity_linear()] = QGauss<1>(param.degree_u + 1);
    quadratures_ALE[navier_stokes_operation->get_quad_index_pressure()] = QGauss<1>(param.get_degree_p() + 1);
    quadratures_ALE[navier_stokes_operation->get_quad_index_velocity_nonlinear()] = QGauss<1>(param.degree_u + (param.degree_u + 2) / 2);
  }


  InputParameters param;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;
  std::shared_ptr<DGNavierStokesBase<dim, Number>> navier_stokes_operation;


  std::shared_ptr<FESystem<dim>> fe_grid;
  std::shared_ptr<FESystem<dim>> fe_u_grid;
  DoFHandler<dim> dof_handler_grid;
  DoFHandler<dim> dof_handler_u_grid;
  DoFHandler<dim> dof_handler_d_grid;

  std::vector<VectorType> position_grid_new_multigrid;
  LinearAlgebra::distributed::Vector<Number> u_grid_np;
  std::vector<LinearAlgebra::distributed::Vector<Number>> d_grid;

  std::shared_ptr<MappingField> mapping;
  std::shared_ptr<MappingQ> mapping_init;


  std::vector<Quadrature<1>> quadratures_ALE;
  AffineConstraints<double> constraint_u_ALE, constraint_p_ALE, constraint_u_scalar_ALE;
  std::vector<const AffineConstraints<double> *> constraint_matrix_vec_ALE;
  std::vector<const DoFHandler<dim> *> dof_handler_vec_ALE;

  typename MatrixFree<dim, Number>::AdditionalData additional_data_ALE;
  UpdateFlags ale_update_flags= (update_gradients |
                                 update_JxW_values |
                                 update_quadrature_points |
                                 update_normal_vectors |
                                 update_values |
                                 update_inverse_jacobians /*CFL*/);


};

}/*IncNS*/
#endif /*INCLUDE_MOVING_MESH_H_*/
