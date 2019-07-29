#ifndef INCLUDE_INCLUDE_MESH_MOVEMENT_H_
#define INCLUDE_INCLUDE_MESH_MOVEMENT_H_

#include "../include/time_integration/push_back_vectors.h"
#include "../../include/incompressible_navier_stokes/spatial_discretization/dg_navier_stokes_base.h"
#include <deal.II/fe/mapping_fe_field.h>
#include "function_mesh_movement.h"

namespace IncNS
{
template<int dim, typename Number>
class MeshMovement
{

public:

  MeshMovement(InputParameters & param_in,
               std::shared_ptr<DGNavierStokesBase<dim, Number>> & navier_stokes_operation_in,
               std::shared_ptr<TimeIntBDF<Number>> & time_integrator_in,
               std::shared_ptr<parallel::Triangulation<dim>> triangulation_in,
               std::shared_ptr<FieldFunctions<dim>>      field_functions_in)
  :
  navier_stokes_operation(navier_stokes_operation_in),
  time_integrator(time_integrator_in),
  d_grid(param_in.order_time_integrator + 1),
  param(param_in),
  triangulation(triangulation_in),
  update_time(0.0),
  move_mesh_time(0.0),
  compute_and_set_mesh_velocity_time(0.0),
  timer_help(0.0),
  dof_handler_u_grid(*triangulation_in),
  dof_handler_grid(*triangulation_in),
  fe_u_grid(new FESystem<dim>(FE_DGQ<dim>(param.degree_u), dim)),
  fe_grid(new FESystem<dim>(FE_Q<dim>(param.degree_u), dim)),//FE_Q is enough
  field_functions(field_functions_in)
  {
    initialize_d_grid_and_u_grid_np();
    dof_handler_u_grid.distribute_dofs(*fe_u_grid);
    dof_handler_u_grid.distribute_mg_dofs();
    dof_handler_grid.distribute_dofs(*fe_grid);
    dof_handler_grid.distribute_mg_dofs();


    mesh_movement = std::make_shared<FunctionMeshMovement<dim>>(param_in);
  }

  ~MeshMovement()
  {}


  void
  advance_mesh_to_next_timestep_and_set_grid_velocities()
  {
    timer_help = timer_mesh.wall_time();
    move_mesh(time_integrator->get_next_time());
    move_mesh_time += timer_mesh.wall_time() - timer_help;

    timer_help = timer_mesh.wall_time();
    navier_stokes_operation->update();
    update_time += timer_mesh.wall_time() - timer_help;

    timer_help = timer_mesh.wall_time();

    if(param.grid_velocity_analytical==true)
      get_analytical_grid_velocity(time_integrator->get_next_time());
    else
     compute_grid_velocity();

    navier_stokes_operation->set_grid_velocity_in_convective_operator_kernel(u_grid_np);
    compute_and_set_mesh_velocity_time += timer_mesh.wall_time() - timer_help;

  }

  double
  get_update_time()
  {
    return update_time;
  }

  double
  get_move_mesh_time()
  {
    return move_mesh_time;
  }

  double
  get_compute_and_set_mesh_velocity_time()
  {
    return compute_and_set_mesh_velocity_time;
  }

protected:

  void
  get_analytical_grid_velocity(double const evaluation_time)
  {
    field_functions->analytical_solution_grid_velocity->set_time(evaluation_time);

    // This is necessary if Number == float
    typedef LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

    VectorTypeDouble grid_velocity_double;
    grid_velocity_double = u_grid_np;

    VectorTools::interpolate(navier_stokes_operation->get_mapping_init(),
                             dof_handler_u_grid,
                             *(field_functions->analytical_solution_grid_velocity),
                             grid_velocity_double);

    u_grid_np = grid_velocity_double;
  }



  void
  compute_grid_velocity()
  {
    push_back(d_grid);
    fill_d_grid();
    time_integrator->compute_BDF_time_derivative(u_grid_np, d_grid);
  }


  void
  initialize_d_grid_and_u_grid_np()
  {

    for(unsigned int i = 0; i < d_grid.size(); ++i)
      {
      navier_stokes_operation->initialize_vector_velocity(d_grid[i]);
      d_grid[i].update_ghost_values();
      }
    navier_stokes_operation->initialize_vector_velocity(u_grid_np);

    fill_d_grid();

    if (param.start_with_low_order==false)
    {//only possible when analytical. otherwise lower_order has to be true
      for(unsigned int i = 1; i < d_grid.size(); ++i)
      {
        //TODO: only possible if analytical solution of grid displacement can be provided
       /* move_mesh(time_integrator->get_time());
        update();*/
        push_back(d_grid);
        fill_d_grid();
      }
    }
  }

  void
  fill_d_grid()
  {
    auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation);
    DoFHandler<dim> dof_handler_grid(*tria);

    dof_handler_grid.distribute_dofs(*fe_u_grid);
    dof_handler_grid.distribute_mg_dofs();

    IndexSet relevant_dofs_grid;
    DoFTools::extract_locally_relevant_dofs(dof_handler_grid,
        relevant_dofs_grid);

    d_grid[0].reinit(dof_handler_grid.locally_owned_dofs(), relevant_dofs_grid, MPI_COMM_WORLD);
    d_grid[0].update_ghost_values();

    FEValues<dim> fe_values(navier_stokes_operation->get_mapping(), *fe_u_grid,
                            Quadrature<dim>((*fe_u_grid).get_unit_support_points()),
                            update_quadrature_points);
    std::vector<types::global_dof_index> dof_indices((*fe_u_grid).dofs_per_cell);
    for (const auto & cell : dof_handler_grid.active_cell_iterators())
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
  }


  void
  move_mesh(double t){

      //const double sin_t=std::pow(std::sin(2*numbers::PI*t/T),2);
      mesh_movement->set_time_displacement(t);

        unsigned int nlevel = dof_handler_grid.get_triangulation().n_global_levels();
        for (unsigned int level=0; level<nlevel; ++level)
        {
        DoFTools::extract_locally_relevant_level_dofs(dof_handler_grid, level,
            relevant_dofs_grid);

        position_grid_init.reinit(dof_handler_grid.locally_owned_mg_dofs(level), relevant_dofs_grid, MPI_COMM_WORLD);
        displacement_grid.reinit(dof_handler_grid.locally_owned_mg_dofs(level), relevant_dofs_grid, MPI_COMM_WORLD);
        position_grid_new.reinit(dof_handler_grid.locally_owned_mg_dofs(level), relevant_dofs_grid, MPI_COMM_WORLD);

        FEValues<dim> fe_values(navier_stokes_operation->get_mapping_init(), *fe_grid,
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
                    double displacement=0;
                    for (unsigned int d=0; d<dim; ++d)
                      displacement = mesh_movement->displacement(point, coordinate_direction);

                    position_grid_init(dof_indices[i]) = point[coordinate_direction];
                    displacement_grid(dof_indices[i]) = displacement;
                  }
              }

        position_grid_new  = position_grid_init;
        position_grid_new += displacement_grid;
        navier_stokes_operation->set_position_grid_new_multigrid(level,position_grid_new);

        }
  }


private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef MappingFEField<dim,dim,LinearAlgebra::distributed::Vector<Number>> MappingField;

  std::shared_ptr<DGNavierStokesBase<dim, Number>> navier_stokes_operation;
  std::shared_ptr<TimeIntBDF<Number>> time_integrator;

  LinearAlgebra::distributed::Vector<Number> u_grid_np;
  std::vector<LinearAlgebra::distributed::Vector<Number>> d_grid;
  InputParameters param;
  std::shared_ptr<parallel::Triangulation<dim>> triangulation;

  Timer timer_mesh;
  double update_time;
  double move_mesh_time;
  double compute_and_set_mesh_velocity_time;
  double timer_help;

  DoFHandler<dim> dof_handler_u_grid;
  DoFHandler<dim> dof_handler_grid;
  std::shared_ptr<FESystem<dim>> fe_u_grid;
  std::shared_ptr<FESystem<dim>> fe_grid;
  std::shared_ptr<FieldFunctions<dim>>      field_functions;

  IndexSet relevant_dofs_grid;

  VectorType position_grid_init;
  VectorType displacement_grid;
  VectorType position_grid_new;

  std::shared_ptr<FunctionMeshMovement<dim>> mesh_movement;


};

}

#endif /*INCLUDE_MESH_MOVEMENT_H_*/
