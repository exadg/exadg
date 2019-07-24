#include "../include/time_integration/push_back_vectors.h"

namespace IncNS
{
template<int dim, typename Number>
class MeshMovement
{

public:

  MeshMovement(InputParameters & param_in,
               std::shared_ptr<DGNavierStokesBase<dim, Number>> & navier_stokes_operation_in,
               std::shared_ptr<TimeIntBDF<Number>> & time_integrator_in,
               std::shared_ptr<parallel::Triangulation<dim>> triangulation_in)
  :
  left(param_in.triangulation_left),
  right(param_in.triangulation_right),
  amplitude(param_in.grid_movement_amplitude),
  delta_t(param_in.end_time - param_in.start_time),
  frequency(param_in.grid_movement_frequency),
  navier_stokes_operation(navier_stokes_operation_in),
  time_integrator(time_integrator_in),
  d_grid(param_in.order_time_integrator + 1),
  param(param_in),
  triangulation(triangulation_in)
  {
    initialize_d_grid_and_u_grid_np();
  }

  void
  compute_grid_velocity()
  {
    push_back(d_grid);
    fill_d_grid();
    time_integrator->compute_BDF_time_derivative(u_grid_np, d_grid);
    navier_stokes_operation->set_grid_velocity_in_convective_operator_kernel(u_grid_np);
  }




protected:

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

    FESystem<dim> fe_grid = navier_stokes_operation->get_fe_u_grid(); //DGQ is needed to determine the right velocitys in any point
    auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation);
    DoFHandler<dim> dof_handler_grid(*tria);

    dof_handler_grid.distribute_dofs(fe_grid);
    dof_handler_grid.distribute_mg_dofs();

    IndexSet relevant_dofs_grid;
    DoFTools::extract_locally_relevant_dofs(dof_handler_grid,
        relevant_dofs_grid);

    d_grid[0].reinit(dof_handler_grid.locally_owned_dofs(), relevant_dofs_grid, MPI_COMM_WORLD);
    d_grid[0].update_ghost_values();

    FEValues<dim> fe_values(navier_stokes_operation->get_mapping(), fe_grid,
                            Quadrature<dim>(fe_grid.get_unit_support_points()),
                            update_quadrature_points);
    std::vector<types::global_dof_index> dof_indices(fe_grid.dofs_per_cell);
    for (const auto & cell : dof_handler_grid.active_cell_iterators())
    {

      if (!cell->is_artificial())
        {
          fe_values.reinit(cell);
          cell->get_dof_indices(dof_indices);
          for (unsigned int i=0; i<fe_grid.dofs_per_cell; ++i)
            {
              const unsigned int coordinate_direction =
                  fe_grid.system_to_component_index(i).first;
              const Point<dim> point = fe_values.quadrature_point(i);
              d_grid[0](dof_indices[i]) = point[coordinate_direction];
            }
        }
    }
  }




private:
  const double left;
  const double right;
  const double amplitude;
  const double delta_t;
  const double frequency;
  const double width = right-left;
  const double T = delta_t/frequency; //duration of a period
  //const double sin_t=std::pow(std::sin(2*numbers::PI*t/T),2);
  std::shared_ptr<DGNavierStokesBase<dim, Number>> navier_stokes_operation;
  std::shared_ptr<TimeIntBDF<Number>> time_integrator;

  LinearAlgebra::distributed::Vector<Number> u_grid_np;
  std::vector<LinearAlgebra::distributed::Vector<Number>> d_grid;
  InputParameters param;
  std::shared_ptr<parallel::Triangulation<dim>> triangulation;

};

}
