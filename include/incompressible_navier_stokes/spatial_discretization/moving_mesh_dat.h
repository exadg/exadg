#ifndef INCLUDE_MOVING_MESH_DAT_H_
#define INCLUDE_MOVING_MESH_DAT_H_

namespace IncNS
{

struct MovingMeshData
{
  MovingMeshData()
  :type(AnalyicMeshMovement::Undefined),
   left(0.0),
   right(0.0),
   f(0.0),
   A(0.0),
   Dt(0.0),
   width(0.0),
   T(0.0)
  {
  }

  AnalyicMeshMovement type;
  double left;
  double right;
  double f;
  double A;
  double Dt;
  double width;
  double T;

};

}/*IncNS*/
#endif /*INCLUDE_MOVING_MESH_DAT_H_*/
