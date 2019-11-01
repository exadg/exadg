#include "moving_mesh.h"
#include "../include/time_integration/push_back_vectors.h"

namespace IncNS
{
template<int dim, typename Number>
MovingMesh<dim, Number>::MovingMesh(
  InputParameters const &                           param_in,
  std::shared_ptr<parallel::TriangulationBase<dim>> triangulation_in,
  std::shared_ptr<MeshMovementFunctions<dim>> const mesh_movement_function_in,
  std::shared_ptr<DGNavierStokesBase<dim, Number>>  navier_stokes_operation_in)
  : param(param_in),
    mesh_movement_function(mesh_movement_function_in),
    navier_stokes_operation(navier_stokes_operation_in),
    fe_x_grid_continuous(new FESystem<dim>(FE_Q<dim>(param_in.degree_u), dim)),
    fe_u_grid(new FESystem<dim>(FE_DGQ<dim>(param_in.degree_u), dim)),
    dof_handler_x_grid_continuous(*triangulation_in),
    dof_handler_u_grid(*triangulation_in),
    dof_handler_x_grid_discontinuous(*triangulation_in),
    vec_position_grid_new((*triangulation_in).n_global_levels()),
    vec_x_grid_discontinuous(param_in.order_time_integrator + 1)
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
  navier_stokes_operation->initialize_vector_velocity(grid_velocity);

  // If param.start_with_low_order == true, we only know the grid coordinates
  // at start time t_0 but not at previous times. Hence, it is not possible to
  // calculate the grid velocity from the grid coordinates but it has to be calculated
  // using the analytical function.
  if(param.grid_velocity_analytical == true || param.start_with_low_order == true)
  {
    compute_grid_velocity_analytical(param.start_time);
  }

  // compute grid velocity from grid positions
  if(param.grid_velocity_analytical == false)
  {
    for(unsigned int i = 0; i < vec_x_grid_discontinuous.size(); ++i)
    {
      navier_stokes_operation->initialize_vector_velocity(vec_x_grid_discontinuous[i]);
      vec_x_grid_discontinuous[i].update_ghost_values();
    }
  }

  navier_stokes_operation->set_grid_velocity(grid_velocity);
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::update_grid_velocities(const double              time,
                                                const double              time_step_size,
                                                const std::vector<Number> time_integrator_constants)
{
  if(param.grid_velocity_analytical == true)
    compute_grid_velocity_analytical(time);
  else
    compute_grid_velocity_from_grid_coordinates(time_integrator_constants, time_step_size);

  navier_stokes_operation->set_grid_velocity(grid_velocity);
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number>
MovingMesh<dim, Number>::get_grid_velocity() const
{
  return grid_velocity;
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::initialize_dof_handler()
{
  dof_handler_x_grid_continuous.distribute_dofs(*fe_x_grid_continuous);
  dof_handler_x_grid_continuous.distribute_mg_dofs();
  dof_handler_u_grid.distribute_dofs(*fe_u_grid);
  dof_handler_x_grid_discontinuous.distribute_dofs(*fe_u_grid);
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::initialize_mapping_ale()
{
  // For initialization of mapping_ale a vector is used that describes the mesh movement.
  // If at start time t_0 the displacement of the mesh is 0 this could also be done by
  // VectorTools::get_position_vector. Here the more generic case is covered, namely
  // advance_grid_coordinates() is used, which determines the positions at start time t_0
  // that are prescribed on each multigrid level.
  advance_grid_coordinates(param.start_time);

  mapping_ale.reset(new MappingFEField<dim, dim, LinearAlgebra::distributed::Vector<Number>>(
    dof_handler_x_grid_continuous, vec_position_grid_new));

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
MovingMesh<dim, Number>::compute_grid_velocity_analytical(double const time)
{
  mesh_movement_function->set_time(time);

  // This is necessary if Number == float
  typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

  VectorTypeDouble grid_velocity_double;
  grid_velocity_double = grid_velocity;

  VectorTools::interpolate(*mapping,
                           dof_handler_u_grid,
                           *mesh_movement_function,
                           grid_velocity_double);

  grid_velocity = grid_velocity_double;
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::advance_grid_coordinates(double const time)
{
  mesh_movement_function->set_time(time);

  /*
   * Two cases can occur:
   * Case 1: The user wants to advance the mesh from a displacement vector:
   *         This is the case if a mesh moving algorithm is applied -> MappingField
   * Case 2: The user wants to advance the mesh with an analytical mesh movement function
   *         and wants to use the initial mesh as a reference -> MappingQGeneric
   */
  Mapping<dim> & mapping_in = *mapping;

  VectorType position_grid_init;
  VectorType displacement_grid;
  VectorType position_grid_new;
  IndexSet   relevant_dofs_grid;

  unsigned int nlevel = dof_handler_x_grid_continuous.get_triangulation().n_global_levels();
  for(unsigned int level = 0; level < nlevel; ++level)
  {
    DoFTools::extract_locally_relevant_level_dofs(dof_handler_x_grid_continuous,
                                                  level,
                                                  relevant_dofs_grid);

    position_grid_init.reinit(dof_handler_x_grid_continuous.locally_owned_mg_dofs(level),
                              relevant_dofs_grid,
                              MPI_COMM_WORLD);
    displacement_grid.reinit(dof_handler_x_grid_continuous.locally_owned_mg_dofs(level),
                             relevant_dofs_grid,
                             MPI_COMM_WORLD);
    position_grid_new.reinit(dof_handler_x_grid_continuous.locally_owned_mg_dofs(level),
                             relevant_dofs_grid,
                             MPI_COMM_WORLD);
    // clang-format off
    FEValues<dim>  fe_values(mapping_in,
                             *fe_x_grid_continuous,
                             Quadrature<dim>(fe_x_grid_continuous->get_unit_support_points()),
                             update_quadrature_points);
    // clang-format on

    std::vector<types::global_dof_index> dof_indices(fe_x_grid_continuous->dofs_per_cell);
    for(const auto & cell : dof_handler_x_grid_continuous.mg_cell_iterators_on_level(level))
      if(cell->level_subdomain_id() != numbers::artificial_subdomain_id)
      {
        fe_values.reinit(cell);
        cell->get_active_or_mg_dof_indices(dof_indices);
        for(unsigned int i = 0; i < fe_x_grid_continuous->dofs_per_cell; ++i)
        {
          const unsigned int coordinate_direction =
            fe_x_grid_continuous->system_to_component_index(i).first;
          const Point<dim> point        = fe_values.quadrature_point(i);
          double           displacement = 0.0;
          for(unsigned int d = 0; d < dim; ++d)
            displacement = mesh_movement_function->displacement(point, coordinate_direction);

          position_grid_init(dof_indices[i]) = point[coordinate_direction];
          displacement_grid(dof_indices[i])  = displacement;
        }
      }

    position_grid_new = position_grid_init;
    position_grid_new += displacement_grid;
    vec_position_grid_new[level] = position_grid_new;
    vec_position_grid_new[level].update_ghost_values();
  }
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::compute_grid_velocity_from_grid_coordinates(
  std::vector<Number> time_integrator_constants,
  double              time_step_size)
{
  push_back(vec_x_grid_discontinuous);
  fill_grid_coordinates_vector(0);
  compute_bdf_time_derivative(grid_velocity,
                              vec_x_grid_discontinuous,
                              time_integrator_constants,
                              time_step_size);
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
    dst.add(-time_integrator_constants[i] / time_step_size, src[i]);
}



template<int dim, typename Number>
void
MovingMesh<dim, Number>::fill_grid_coordinates_vector(unsigned int const time_index)
{
  IndexSet relevant_dofs_grid;
  DoFTools::extract_locally_relevant_dofs(dof_handler_x_grid_discontinuous, relevant_dofs_grid);

  vec_x_grid_discontinuous[time_index].reinit(dof_handler_x_grid_discontinuous.locally_owned_dofs(),
                                              relevant_dofs_grid,
                                              MPI_COMM_WORLD);
  // clang-format off
  FEValues<dim>  fe_values(get_mapping(),
                           *fe_u_grid,
                           Quadrature<dim>((*fe_u_grid).get_unit_support_points()),
                           update_quadrature_points);
  // clang-format on

  std::vector<types::global_dof_index> dof_indices((*fe_u_grid).dofs_per_cell);
  for(const auto & cell : dof_handler_x_grid_discontinuous.active_cell_iterators())
  {
    if(!cell->is_artificial())
    {
      fe_values.reinit(cell);
      cell->get_dof_indices(dof_indices);
      for(unsigned int i = 0; i < (*fe_u_grid).dofs_per_cell; ++i)
      {
        const unsigned int coordinate_direction = (*fe_u_grid).system_to_component_index(i).first;
        const Point<dim>   point                = fe_values.quadrature_point(i);
        vec_x_grid_discontinuous[time_index](dof_indices[i]) = point[coordinate_direction];
      }
    }
  }

  vec_x_grid_discontinuous[time_index].update_ghost_values();
}


template class MovingMesh<2, float>;
template class MovingMesh<2, double>;

template class MovingMesh<3, float>;
template class MovingMesh<3, double>;


} // namespace IncNS
