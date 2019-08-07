#ifndef INCLUDE_MOVING_MESH_DAT_H_
#define INCLUDE_MOVING_MESH_DAT_H_

namespace IncNS
{
struct MovingMeshData
{
  MovingMeshData()
    : type(AnalyicMeshMovement::Undefined),
      left(0.0),
      right(0.0),
      f(0.0),
      A(0.0),
      Dt(0.0),
      width(0.0),
      T(0.0),
      u_ana(false),
      degree_u(0.0),
      degree_p(0.0),
      dof_index_u(0.0),
      dof_index_p(0.0),
      dof_index_u_scalar(0.0),
      quad_index_u(0.0),
      quad_index_p(0.0),
      quad_index_u_nonlinear(0.0),
      use_cell_based_face_loops(false),
      order_time_integrator(2),
      start_low_order(true),
      initialize_with_former_mesh_instances(false)
  {
  }

  AnalyicMeshMovement type;
  double              left;
  double              right;
  double              f;
  double              A;
  double              Dt;
  double              width;
  double              T;
  bool                u_ana;
  unsigned int        degree_u;

  unsigned int degree_p;
  unsigned int dof_index_u;
  unsigned int dof_index_p;
  unsigned int dof_index_u_scalar;
  unsigned int quad_index_u;
  unsigned int quad_index_p;
  unsigned int quad_index_u_nonlinear;
  bool         use_cell_based_face_loops;
  unsigned int order_time_integrator;
  bool         start_low_order;
  bool         initialize_with_former_mesh_instances;
};

} // namespace IncNS
#endif /*INCLUDE_MOVING_MESH_DAT_H_*/
