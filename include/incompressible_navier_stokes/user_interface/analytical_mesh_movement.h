#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ANALYTICAL_MESH_MOVEMENT_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ANALYTICAL_MESH_MOVEMENT_H_

#include "../../../applications/grid_tools/mesh_movement_functions.h"

using namespace dealii;

namespace IncNS
{
template<int dim>
struct AnalyticalMeshMovement
{
  std::shared_ptr<MeshMovementFunctions<dim>> analytical_mesh_movement;
};

} // namespace IncNS

#endif /*INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_USER_INTERFACE_ANALYTICAL_MESH_MOVEMENT_H_*/
