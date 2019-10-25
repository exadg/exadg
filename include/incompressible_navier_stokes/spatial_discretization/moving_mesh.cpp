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
    vec_x_grid_discontinuous(param_in.order_time_integrator + 1),
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

  navier_stokes_operation->set_grid_velocity(grid_velocity);
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
    navier_stokes_operation->set_grid_velocity(grid_velocity);

    navier_stokes_operation->evaluate_convective_term(vec_convective_term[i],
                                                      solution[i].block(0),
                                                      eval_times[i]);
  }
  return vec_convective_term;
}

template<int dim, typename Number>
std::vector<LinearAlgebra::distributed::Vector<Number>>
MovingMesh<dim, Number>::get_vec_rhs_ppe_div_term_convective_term_on_former_mesh_instances(
  std::vector<double> eval_times)
{
  std::vector<BlockVectorType> solution = get_former_solution_on_former_mesh_instances(eval_times);
  std::vector<VectorType>      vec_rhs_ppe_div_term_convective_term(param.order_time_integrator);

  for(unsigned int i = 0; i < vec_rhs_ppe_div_term_convective_term.size(); ++i)
    navier_stokes_operation->initialize_vector_pressure(vec_rhs_ppe_div_term_convective_term[i]);

  auto navier_stokes_operation_ds =
    std::dynamic_pointer_cast<DGNavierStokesDualSplitting<dim, Number>>(navier_stokes_operation);

  // Iterating backwards leaves us with the mesh at start time automatically
  for(unsigned int i = param.order_time_integrator - 1; i < param.order_time_integrator; --i)
  {
    advance_mesh(eval_times[i]);
    navier_stokes_operation->ale_update();
    get_analytical_grid_velocity(eval_times[i]);
    navier_stokes_operation->set_grid_velocity(grid_velocity);

    vec_rhs_ppe_div_term_convective_term[i] = 0.0;
    navier_stokes_operation_ds->rhs_ppe_div_term_convective_term_add(
      vec_rhs_ppe_div_term_convective_term[i], solution[i].block(0));
  }

  return vec_rhs_ppe_div_term_convective_term;
}

template<int dim, typename Number>
std::vector<LinearAlgebra::distributed::Vector<Number>>
MovingMesh<dim, Number>::get_vec_rhs_ppe_convective_on_former_mesh_instances(
  std::vector<double> eval_times)
{
  std::vector<BlockVectorType> solution = get_former_solution_on_former_mesh_instances(eval_times);
  std::vector<VectorType>      vec_rhs_ppe_convective(param.order_extrapolation_pressure_nbc);

  for(unsigned int i = 0; i < vec_rhs_ppe_convective.size(); ++i)
    navier_stokes_operation->initialize_vector_pressure(vec_rhs_ppe_convective[i]);

  auto navier_stokes_operation_ds =
    std::dynamic_pointer_cast<DGNavierStokesDualSplitting<dim, Number>>(navier_stokes_operation);

  // Iterating backwards leaves us with the mesh at start time automatically
  for(unsigned int i = param.order_extrapolation_pressure_nbc - 1;
      i < param.order_extrapolation_pressure_nbc;
      --i)
  {
    advance_mesh(eval_times[i]);
    navier_stokes_operation->ale_update();
    get_analytical_grid_velocity(eval_times[i]);
    navier_stokes_operation->set_grid_velocity(grid_velocity);

    vec_rhs_ppe_convective[i] = 0.0;
    navier_stokes_operation_ds->rhs_ppe_convective_add(vec_rhs_ppe_convective[i],
                                                       solution[i].block(0));
  }

  return vec_rhs_ppe_convective;
}

template<int dim, typename Number>
std::vector<LinearAlgebra::distributed::Vector<Number>>
MovingMesh<dim, Number>::get_vec_rhs_ppe_viscous_on_former_mesh_instances(
  std::vector<double> eval_times)
{
  std::vector<VectorType>      vec_rhs_ppe_viscous(param.order_extrapolation_pressure_nbc);
  VectorType                   vorticity;
  std::vector<BlockVectorType> solution = get_former_solution_on_former_mesh_instances(eval_times);

  for(unsigned int i = 0; i < vec_rhs_ppe_viscous.size(); ++i)
    navier_stokes_operation->initialize_vector_pressure(vec_rhs_ppe_viscous[i]);
  navier_stokes_operation->initialize_vector_velocity(vorticity);

  auto navier_stokes_operation_ds =
    std::dynamic_pointer_cast<DGNavierStokesDualSplitting<dim, Number>>(navier_stokes_operation);

  // Iterating backwards leaves us with the mesh at start time automatically
  for(unsigned int i = param.order_extrapolation_pressure_nbc - 1;
      i < param.order_extrapolation_pressure_nbc;
      --i)
  {
    advance_mesh(eval_times[i]);
    navier_stokes_operation->ale_update();

    vec_rhs_ppe_viscous[i] = 0.0;

    navier_stokes_operation->compute_vorticity(vorticity, solution[i].block(0));
    navier_stokes_operation_ds->rhs_ppe_viscous_add(vec_rhs_ppe_viscous[i], vorticity);
  }

  return vec_rhs_ppe_viscous;
}

template<int dim, typename Number>
LinearAlgebra::distributed::Vector<Number>
MovingMesh<dim, Number>::get_grid_velocity() const
{
  return grid_velocity;
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
  dof_handler_x_grid_continuous.distribute_dofs(*fe_x_grid_continuous);
  dof_handler_x_grid_continuous.distribute_mg_dofs();
  dof_handler_u_grid.distribute_dofs(*fe_u_grid);
  dof_handler_x_grid_discontinuous.distribute_dofs(*fe_u_grid);
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::initialize_vectors()
{
  navier_stokes_operation->initialize_vector_velocity(grid_velocity);

  if(param.start_with_low_order == true)
  {
    get_analytical_grid_velocity(param.start_time);
    AssertThrow(
      grid_velocity.l2_norm() <= std::numeric_limits<Number>::min(),
      ExcMessage(
        "Consider an other mesh moving function (e.g. use MeshMovementAdvanceInTime::SinSquared). For low oder start, the grid velocity has to be 0 at start time to ensure a continuisly differentiable (in time) mesh motion."));
  }

  if(param.grid_velocity_analytical == true)
  {
    get_analytical_grid_velocity(param.start_time);
  }
  else // compute grid velocity from grid positions
  {
    for(unsigned int i = 0; i < vec_x_grid_discontinuous.size(); ++i)
    {
      navier_stokes_operation->initialize_vector_velocity(vec_x_grid_discontinuous[i]);
      vec_x_grid_discontinuous[i].update_ghost_values();
    }

    // fill grid coordinates vector at start time t_0, grid coordinates
    // at previous times have to be computed by a separate function call
    // if the time integrator is started with high order.
    fill_grid_coordinates_vector();
  }

  navier_stokes_operation->set_grid_velocity(grid_velocity);
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::initialize_mapping_ale()
{
  mesh_movement_function->set_time(param.start_time);
  advance_grid_position(*mapping);

  // For initialization a vector is used that is determined by the function that describes the mesh
  // movement. If at start time t_0 the displacement of the mesh is 0 this could also be done by
  // VectorTools::get_position_vector. Here the more generic case is coverd:
  // advance_grid_position(*mapping) is used, which determines the positions at t_0 that is
  // prescribed on each multigrid level.
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
MovingMesh<dim, Number>::get_analytical_grid_velocity(double const evaluation_time)
{
  mesh_movement_function->set_time(evaluation_time);

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
MovingMesh<dim, Number>::advance_mesh(double time_in)
{
  mesh_movement_function->set_time(time_in);
  advance_grid_position(*mapping);
}

template<int dim, typename Number>
void
MovingMesh<dim, Number>::advance_grid_position(Mapping<dim> & mapping_in)
{
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
MovingMesh<dim, Number>::compute_grid_velocity(std::vector<Number> time_integrator_constants,
                                               double              time_step_size)
{
  push_back(vec_x_grid_discontinuous);
  fill_grid_coordinates_vector();
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
MovingMesh<dim, Number>::fill_grid_coordinates_vector(int component)
{
  IndexSet relevant_dofs_grid;
  DoFTools::extract_locally_relevant_dofs(dof_handler_x_grid_discontinuous, relevant_dofs_grid);

  vec_x_grid_discontinuous[component].reinit(dof_handler_x_grid_discontinuous.locally_owned_dofs(),
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
        vec_x_grid_discontinuous[component](dof_indices[i]) = point[coordinate_direction];
      }
    }
  }
  vec_x_grid_discontinuous[component].update_ghost_values();
}


template class MovingMesh<2, float>;
template class MovingMesh<2, double>;

template class MovingMesh<3, float>;
template class MovingMesh<3, double>;


} // namespace IncNS
