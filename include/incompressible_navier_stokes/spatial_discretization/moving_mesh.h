#ifndef INCLUDE_MOVING_MESH_H_
#define INCLUDE_MOVING_MESH_H_

#include <deal.II/fe/mapping_fe_field.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_block_vector.h>

#include "dg_navier_stokes_base.h"

using namespace dealii;

namespace IncNS
{
template<int dim, typename Number>
class MovingMesh
{
public:
  typedef MappingFEField<dim, dim, LinearAlgebra::distributed::Vector<Number>> MappingField;
  typedef LinearAlgebra::distributed::Vector<Number>                           VectorType;
  typedef LinearAlgebra::distributed::BlockVector<Number>                      BlockVectorType;

  MovingMesh(InputParameters const &                           param_in,
             parallel::TriangulationBase<dim> const &          triangulation_in,
             std::shared_ptr<MeshMovementFunctions<dim>> const mesh_movement_function_in,
             DGNavierStokesBase<dim, Number> &                 navier_stokes_operation_in);

  void
  advance_grid_coordinates(double const time);

  void
  compute_grid_velocity_analytical(VectorType & velocity, double const time);

  void
  fill_grid_coordinates_vector(VectorType & vector);

private:
  void
  initialize_dof_handler();

  void
  initialize_mapping_ale();

  Mapping<dim> &
  get_mapping() const;

  InputParameters                             param;
  std::shared_ptr<MeshMovementFunctions<dim>> mesh_movement_function;
  DGNavierStokesBase<dim, Number> &           navier_stokes_operation;

  // fe systems
  std::shared_ptr<FESystem<dim>> fe_vectorial_continuous;
  std::shared_ptr<FESystem<dim>> fe_vectorial_discontinuous;

  // dof handlers
  DoFHandler<dim> dof_handler_vectorial_continuous;
  DoFHandler<dim> dof_handler_vectorial_discontinuous;

  // vectors with grid coordinates for all multigrid levels
  std::vector<VectorType> grid_coordinates;

  // mappings
  std::shared_ptr<MappingQGeneric<dim>> mapping;
  std::shared_ptr<MappingField>         mapping_ale;
};

} // namespace IncNS
#endif /*INCLUDE_MOVING_MESH_H_*/
