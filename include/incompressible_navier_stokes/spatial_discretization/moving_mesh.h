#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include "moving_mesh_dat.h"
#include "../../../applications/grid_tools/mesh_movement_functions.h"


using namespace dealii;

namespace IncNS
{

template<int dim, typename Number>
class MovingMesh
{
public:

  void
  setup(MovingMeshData const & data_in)
  {
    moving_mesh_data = data_in;

    if (data_in.type == AnalyicMeshMovement::InteriorSinCos)
      mesh_movement_function.reset(new InteriorSinCos<dim>(moving_mesh_data));
    else if (data_in.type == AnalyicMeshMovement::InteriorSinCosOnlyX)
      mesh_movement_function.reset(new InteriorSinCosOnlyX<dim>(moving_mesh_data));
    else if (data_in.type == AnalyicMeshMovement::InteriorSinCosOnlyY)
      mesh_movement_function.reset(new InteriorSinCosOnlyY<dim>(moving_mesh_data));
    else if (data_in.type == AnalyicMeshMovement::SinCosWithBoundaries)
      mesh_movement_function.reset(new SinCosWithBoundaries<dim>(moving_mesh_data));
    else if (data_in.type == AnalyicMeshMovement::SinCosWithBoundariesOnlyX)
      mesh_movement_function.reset(new SinCosWithBoundariesOnlyX<dim>(moving_mesh_data));
    else if (data_in.type == AnalyicMeshMovement::SinCosWithBoundariesOnlyY)
      mesh_movement_function.reset(new SinCosWithBoundariesOnlyY<dim>(moving_mesh_data));
    else if (data_in.type == AnalyicMeshMovement::InteriorSinCosWithSinInTime)
      mesh_movement_function.reset(new InteriorSinCosWithSinInTime<dim>(moving_mesh_data));
    else if (data_in.type == AnalyicMeshMovement::XSquaredWithBoundaries)
      mesh_movement_function.reset(new XSquaredWithBoundaries<dim>(moving_mesh_data));
    else if (data_in.type == AnalyicMeshMovement::DoubleInteriorSinCos)
      mesh_movement_function.reset(new DoubleInteriorSinCos<dim>(moving_mesh_data));
    else if (data_in.type == AnalyicMeshMovement::DoubleSinCosWithBoundaries)
      mesh_movement_function.reset(new DoubleSinCosWithBoundaries<dim>(moving_mesh_data));
    else if (data_in.type == AnalyicMeshMovement::None)
      mesh_movement_function.reset(new None<dim>(moving_mesh_data));





  }

  void
  advance_mesh_and_set_grid_velocities(double time_in)
  {
    move_mesh(time_in);

    //navier_stokes_operation->update();


//    if(param.grid_velocity_analytical==true)
//      get_analytical_grid_velocity(time_integrator->get_next_time());
//    else
//     compute_grid_velocity();
//
//    navier_stokes_operation->set_grid_velocity_in_convective_operator_kernel(u_grid_np);

  }




private:
  void
  move_mesh(double time_in){

    mesh_movement_function->set_time_displacement(time_in);

//      unsigned int nlevel = dof_handler_grid.get_triangulation().n_global_levels();
//      for (unsigned int level=0; level<nlevel; ++level)
//      {
//      DoFTools::extract_locally_relevant_level_dofs(dof_handler_grid, level,
//          relevant_dofs_grid);
//
//      position_grid_init.reinit(dof_handler_grid.locally_owned_mg_dofs(level), relevant_dofs_grid, MPI_COMM_WORLD);
//      displacement_grid.reinit(dof_handler_grid.locally_owned_mg_dofs(level), relevant_dofs_grid, MPI_COMM_WORLD);
//      position_grid_new.reinit(dof_handler_grid.locally_owned_mg_dofs(level), relevant_dofs_grid, MPI_COMM_WORLD);
//
//      FEValues<dim> fe_values(navier_stokes_operation->get_mapping_init(), *fe_grid,
//                                Quadrature<dim>(fe_grid->get_unit_support_points()),
//                                update_quadrature_points);
//        std::vector<types::global_dof_index> dof_indices(fe_grid->dofs_per_cell);
//        for (const auto & cell : dof_handler_grid.mg_cell_iterators_on_level(level))
//          if (cell->level_subdomain_id() != numbers::artificial_subdomain_id)
//            {
//              fe_values.reinit(cell);
//              cell->get_active_or_mg_dof_indices(dof_indices);
//              for (unsigned int i=0; i<fe_grid->dofs_per_cell; ++i)
//                {
//                  const unsigned int coordinate_direction =
//                      fe_grid->system_to_component_index(i).first;
//                  const Point<dim> point = fe_values.quadrature_point(i);
//                    double displacement=0;
//                  for (unsigned int d=0; d<dim; ++d)
//                      displacement = mesh_movement_function->displacement(point, coordinate_direction);
//
//                  position_grid_init(dof_indices[i]) = point[coordinate_direction];
//                  displacement_grid(dof_indices[i]) = displacement;
//                }
//            }
//
//      position_grid_new  = position_grid_init;
//      position_grid_new += displacement_grid;
//      navier_stokes_operation->set_position_grid_new_multigrid(level,position_grid_new);

  }

private:

  std::shared_ptr<MeshMovementFunctions<dim>> mesh_movement_function;
  MovingMeshData moving_mesh_data;


};

}/*IncNS*/
#endif /*INCLUDE_MOVING_MESH_H_*/
