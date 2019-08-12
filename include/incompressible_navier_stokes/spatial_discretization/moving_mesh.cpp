#include "moving_mesh.h"
#include "../include/time_integration/push_back_vectors.h"

namespace IncNS
{
template<int dim, typename Number>
MovingMesh<dim, Number>::MovingMesh(
  InputParameters const &                          param_in,
  std::shared_ptr<parallel::Triangulation<dim>>    triangulation_in,
  std::shared_ptr<FieldFunctions<dim>> const       field_functions_in,
  std::shared_ptr<DGNavierStokesBase<dim, Number>> navier_stokes_operation_in)
  : param(param_in),
    field_functions(field_functions_in),
    navier_stokes_operation(navier_stokes_operation_in),
    fe_grid(new FESystem<dim>(FE_Q<dim>(param_in.degree_u), dim)),
    fe_u_grid(new FESystem<dim>(FE_DGQ<dim>(param_in.degree_u), dim)),
    dof_handler_grid(*triangulation_in),
    dof_handler_u_grid(*triangulation_in),
    dof_handler_x_grid(*triangulation_in),
    position_grid_new_multigrid((*triangulation_in).n_global_levels()),
    x_grid(param_in.order_time_integrator + 1),
    ale_update_timer(0.0),
    advance_mesh_timer(0.0),
    compute_and_set_mesh_velocity_timer(0.0),
    help_timer(0.0)
{
  mapping = std::make_shared<MappingQGeneric<dim>>(
    dynamic_cast<const MappingQGeneric<dim> &>((navier_stokes_operation_in->get_mapping())));
  initialize_dof_handler();
  initialize_mapping_ale();
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::setup()
{
  initialize_vectors();
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::move_mesh(const double time_in)
{
  help_timer = timer_ale.wall_time();
  advance_mesh(time_in);
  advance_mesh_timer += timer_ale.wall_time() - help_timer;

  help_timer = timer_ale.wall_time();
  navier_stokes_operation->ale_update();
  ale_update_timer += timer_ale.wall_time() - help_timer;
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::update_grid_velocities(const double              time_in,
                                                const double              time_step_size,
                                                const std::vector<Number> time_integrator_constants)
{
  help_timer = timer_ale.wall_time();
  if(param.grid_velocity_analytical == true)
    get_analytical_grid_velocity(time_in);
  else
    compute_grid_velocity(time_integrator_constants, time_step_size);

  navier_stokes_operation->set_grid_velocity(u_grid_np);
  compute_and_set_mesh_velocity_timer += timer_ale.wall_time() - help_timer;
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::initialize_grid_coordinates_on_former_mesh_instances(
  std::vector<double> eval_times)
{
  // Iterating backwards leaves us with the mesh at start time automatically
  for(unsigned int i = param.order_time_integrator - 1; i < param.order_time_integrator; --i)
  {
    advance_mesh(eval_times[i]);
    navier_stokes_operation->ale_update();

    get_analytical_grid_velocity(eval_times[i]);

    fill_grid_coordinates_vector(i);
  }
}

template<int dim, typename Number>
std::vector<LinearAlgebra::distributed::BlockVector<Number>>
MovingMesh<dim, Number>::get_former_solution_on_former_mesh_instances(
  std::vector<double> eval_times)
{
  std::vector<BlockVectorType> solution(param.order_time_integrator);

  // Iterating backwards leaves us with the mesh at start time automatically
  for(unsigned int i = 0; i < solution.size(); ++i)
  {
    solution[i].reinit(2);
    navier_stokes_operation->initialize_vector_velocity(solution[i].block(0));
    navier_stokes_operation->initialize_vector_pressure(solution[i].block(1));
  }

  for(unsigned int i = param.order_time_integrator - 1; i < param.order_time_integrator; --i)
  {
    advance_mesh(eval_times[i]);
    navier_stokes_operation->ale_update();

    navier_stokes_operation->prescribe_initial_conditions(solution[i].block(0),
                                                          solution[i].block(1),
                                                          eval_times[i]);
  }
  return solution;
}

template<int dim, typename Number>
std::vector<LinearAlgebra::distributed::Vector<Number>>
MovingMesh<dim, Number>::get_convective_term_on_former_mesh_instances(
  std::vector<double> eval_times)
{
  std::vector<BlockVectorType> solution = get_former_solution_on_former_mesh_instances(eval_times);
  std::vector<VectorType>      vec_convective_term(param.order_time_integrator);

  for(unsigned int i = 0; i < vec_convective_term.size(); ++i)
    navier_stokes_operation->initialize_vector_velocity(vec_convective_term[i]);

  // Iterating backwards leaves us with the mesh at start time automatically
  for(unsigned int i = param.order_time_integrator - 1; i < param.order_time_integrator; --i)
  {
    advance_mesh(eval_times[i]);
    navier_stokes_operation->ale_update();

    get_analytical_grid_velocity(eval_times[i]);
    navier_stokes_operation->set_grid_velocity(u_grid_np);

    navier_stokes_operation->evaluate_convective_term(vec_convective_term[i],
                                                      solution[i].block(0),
                                                      eval_times[i]);
  }
  return vec_convective_term;
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number>
MovingMesh<dim, Number>::get_grid_velocity() const
{
  return u_grid_np;
}

template<int dim, typename Number>
double
MovingMesh<dim, Number>::get_wall_time_ale_update() const
{
  return ale_update_timer;
}

template<int dim, typename Number>
double
MovingMesh<dim, Number>::get_wall_time_advance_mesh() const
{
  return advance_mesh_timer;
}

template<int dim, typename Number>
double
MovingMesh<dim, Number>::get_wall_time_compute_and_set_mesh_velocity() const
{
  return compute_and_set_mesh_velocity_timer;
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::initialize_dof_handler()
{
  dof_handler_grid.distribute_dofs(*fe_grid);
  dof_handler_grid.distribute_mg_dofs();
  dof_handler_u_grid.distribute_dofs(*fe_u_grid);
  dof_handler_u_grid.distribute_mg_dofs();
  dof_handler_x_grid.distribute_dofs(*fe_u_grid);
  dof_handler_x_grid.distribute_mg_dofs();
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::initialize_vectors()
{
  navier_stokes_operation->initialize_vector_velocity(u_grid_np);

  if(param.grid_velocity_analytical == false)
  {
    for(unsigned int i = 0; i < x_grid.size(); ++i)
    {
      navier_stokes_operation->initialize_vector_velocity(x_grid[i]);
      x_grid[i].update_ghost_values();
    }

    fill_grid_coordinates_vector();
    // if gird velocity is not prescribed analytically it is assumed to be 0 at timesteps
    // t_0 and former timesteps
    navier_stokes_operation->set_grid_velocity(u_grid_np);
  }
  else if(param.grid_velocity_analytical == true)
  {
    // otherwise:
    get_analytical_grid_velocity(param.start_time);
    navier_stokes_operation->set_grid_velocity(u_grid_np);
  }
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::initialize_mapping_ale()
{
  field_functions->analytical_solution_grid_velocity->set_time_displacement(0.0);
  advance_position_grid_new_multigrid(*mapping);
  mapping_ale.reset(new MappingFEField<dim, dim, LinearAlgebra::distributed::Vector<Number>>(
    dof_handler_grid, position_grid_new_multigrid));
  navier_stokes_operation->set_mapping_ale(mapping_ale);
}

template<int dim, typename Number>
Mapping<dim> &
MovingMesh<dim, Number>::get_mapping() const
{
  return *mapping_ale;
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::get_analytical_grid_velocity(double const evaluation_time)
{
  field_functions->analytical_solution_grid_velocity->set_time_velocity(evaluation_time);

  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble grid_velocity_double;
  grid_velocity_double = u_grid_np;

  VectorTools::interpolate(
    *mapping, // coordinates on which analytical_solution_grid_velocity() relies
    dof_handler_u_grid,
    *(field_functions->analytical_solution_grid_velocity),
    grid_velocity_double);

  u_grid_np = grid_velocity_double;
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::advance_mesh(double time_in)
{
  field_functions->analytical_solution_grid_velocity->set_time_displacement(time_in);
  advance_position_grid_new_multigrid(*mapping);
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::advance_position_grid_new_multigrid(Mapping<dim> & mapping_in)
{
  VectorType position_grid_init;
  VectorType displacement_grid;
  VectorType position_grid_new;
  IndexSet   relevant_dofs_grid;


  unsigned int nlevel = dof_handler_grid.get_triangulation().n_global_levels();
  for(unsigned int level = 0; level < nlevel; ++level)
  {
    DoFTools::extract_locally_relevant_level_dofs(dof_handler_grid, level, relevant_dofs_grid);

    position_grid_init.reinit(dof_handler_grid.locally_owned_mg_dofs(level),
                              relevant_dofs_grid,
                              MPI_COMM_WORLD);
    displacement_grid.reinit(dof_handler_grid.locally_owned_mg_dofs(level),
                             relevant_dofs_grid,
                             MPI_COMM_WORLD);
    position_grid_new.reinit(dof_handler_grid.locally_owned_mg_dofs(level),
                             relevant_dofs_grid,
                             MPI_COMM_WORLD);

    // clang-format off
    FEValues<dim>  fe_values(mapping_in,
                             *fe_grid,
                             Quadrature<dim>(fe_grid->get_unit_support_points()),
                             update_quadrature_points);
    // clang-format on

    std::vector<types::global_dof_index> dof_indices(fe_grid->dofs_per_cell);
    for(const auto & cell : dof_handler_grid.mg_cell_iterators_on_level(level))
      if(cell->level_subdomain_id() != numbers::artificial_subdomain_id)
      {
        fe_values.reinit(cell);
        cell->get_active_or_mg_dof_indices(dof_indices);
        for(unsigned int i = 0; i < fe_grid->dofs_per_cell; ++i)
        {
          const unsigned int coordinate_direction = fe_grid->system_to_component_index(i).first;
          const Point<dim>   point                = fe_values.quadrature_point(i);
          double             displacement         = 0.0;
          for(unsigned int d = 0; d < dim; ++d)
            displacement = field_functions->analytical_solution_grid_velocity->displacement(
              point, coordinate_direction);

          position_grid_init(dof_indices[i]) = point[coordinate_direction];
          displacement_grid(dof_indices[i])  = displacement;
        }
      }

    position_grid_new = position_grid_init;
    position_grid_new += displacement_grid;
    position_grid_new_multigrid[level] = position_grid_new;
    position_grid_new_multigrid[level].update_ghost_values();
  }
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::compute_grid_velocity(std::vector<Number> time_integrator_constants,
                                               double              time_step_size)
{
  push_back(x_grid);
  fill_grid_coordinates_vector();
  compute_bdf_time_derivative(u_grid_np, x_grid, time_integrator_constants, time_step_size);
}


template<int dim, typename Number>
void
MovingMesh<dim, Number>::compute_bdf_time_derivative(VectorType &            dst,
                                                     std::vector<VectorType> src,
                                                     std::vector<Number> time_integrator_constants,
                                                     double              time_step_size)
{
  dst.equ(time_integrator_constants[0] / time_step_size, src[0]);

  for(unsigned int i = 1; i < src.size(); ++i)
    dst.add(-1 * time_integrator_constants[i] / time_step_size, src[i]);
}



template<int dim, typename Number>
void
MovingMesh<dim, Number>::fill_grid_coordinates_vector(int component)
{
  IndexSet relevant_dofs_grid;
  DoFTools::extract_locally_relevant_dofs(dof_handler_x_grid, relevant_dofs_grid);

  x_grid[component].reinit(dof_handler_x_grid.locally_owned_dofs(),
                           relevant_dofs_grid,
                           MPI_COMM_WORLD);
  // clang-format off
  FEValues<dim>  fe_values(get_mapping(),
                           *fe_u_grid,
                           Quadrature<dim>((*fe_u_grid).get_unit_support_points()),
                           update_quadrature_points);
  // clang-format on

  std::vector<types::global_dof_index> dof_indices((*fe_u_grid).dofs_per_cell);
  for(const auto & cell : dof_handler_x_grid.active_cell_iterators())
  {
    if(!cell->is_artificial())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(dof_indices);
      for(unsigned int i = 0; i < (*fe_u_grid).dofs_per_cell; ++i)
      {
        const unsigned int coordinate_direction = (*fe_u_grid).system_to_component_index(i).first;
        const Point<dim>   point                = fe_values.quadrature_point(i);
        x_grid[component](dof_indices[i])       = point[coordinate_direction];
      }
    }
  }
  x_grid[component].update_ghost_values();
}


template class MovingMesh<2, float>;
template class MovingMesh<2, double>;

template class MovingMesh<3, float>;
template class MovingMesh<3, double>;


} // namespace IncNS
