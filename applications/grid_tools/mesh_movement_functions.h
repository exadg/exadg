#ifndef INCLUDE_FUNCTION_MESH_MOVEMENT_H_
#define INCLUDE_FUNCTION_MESH_MOVEMENT_H_

#include "../../include/incompressible_navier_stokes/spatial_discretization/moving_mesh.h"

namespace IncNS
{

template<int dim>
class MeshMovementFunctions //: public Function<dim>
{

public:
  MeshMovementFunctions(MovingMeshData data_in)
  :sin_t(0.0),
   dat(data_in)
  {
  }

  virtual ~MeshMovementFunctions(){}

  virtual double
  displacement(const Point<dim>    &x,
               const unsigned int  coordinate_direction = 0) const = 0;

  void
  set_time_displacement(double const t)
  {
    if(dat.type == AnalyicMeshMovement::InteriorSinCosWithSinInTime)
      sin_t = std::sin(2*pi*t/dat.T);
    else
      sin_t = std::pow(std::sin(2*pi*t/dat.T),2);


  }

  virtual void
  set_time_velocity(double t) const
  {   //mostly Sin² is used to advance mesh movement in time
      sin_t = (4*pi*std::sin(2*pi*t/dat.T)*std::cos(2*pi*t/dat.T)/dat.T);
  }

  //Velocity doesnt require multigrid support, hence to be able to use interpolation by deal.II this function is called value()
  double
  value(const Point<dim>    &p,
        const unsigned int  component = 0) const{
    //Since displacements are of shape const*f(t), code duplication can be avoided using f(t)=\partial_t f(t)
    set_time_velocity(this->get_time());
    return displacement(p, component);

  }

protected:
  mutable double sin_t;
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class SinCosWithBoundaries : public Function<dim>,
                             public MeshMovementFunctions<dim>
{
  public:
    SinCosWithBoundaries(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in),
     dat(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if (coordinate_direction == 0)
      solution = std::sin(2* pi*(x(1)-dat.left)/dat.width)*this->sin_t*dat.A;
    else if (coordinate_direction == 1)
      solution = std::sin(2* pi*(x(0)-dat.left)/dat.width)*this->sin_t*dat.A;

    return solution;
  }

  private:
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class SinCosWithBoundariesOnlyX : public Function<dim>,
                                  public MeshMovementFunctions<dim>
{
  public:
    SinCosWithBoundariesOnlyX(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in),
     dat(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if (coordinate_direction == 0)
      solution = std::sin(2* pi*(x(1)-dat.left)/dat.width)*this->sin_t*dat.A;
    else if (coordinate_direction == 1)
      solution = 0.0;

    return solution;
  }

  private:
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class SinCosWithBoundariesOnlyY : public Function<dim>,
                                  public MeshMovementFunctions<dim>
{
  public:
    SinCosWithBoundariesOnlyY(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in),
     dat(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if (coordinate_direction == 0)
      solution = 0.0;
    else if (coordinate_direction == 1)
      solution = std::sin(2* pi*(x(0)-dat.left)/dat.width)*this->sin_t*dat.A;

    return solution;
  }

  private:
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class InteriorSinCosOnlyX : public Function<dim>,
                            public MeshMovementFunctions<dim>
{
  public:
  InteriorSinCosOnlyX(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in),
     dat(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if (coordinate_direction == 0)
      solution = std::sin(2* pi*(x(1)-dat.left)/dat.width)*this->sin_t*dat.A*(1- std::pow(x(0)/dat.right,2));
    else if (coordinate_direction == 1)
      solution = 0.0;

    return solution;
  }

  private:
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class InteriorSinCosOnlyY : public Function<dim>,
                            public MeshMovementFunctions<dim>
{
  public:
  InteriorSinCosOnlyY(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in),
     dat(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if (coordinate_direction == 0)
      solution = 0.0;
    else if (coordinate_direction == 1)
      solution = solution = std::sin(2* pi*(x(0)-dat.left)/dat.width)*this->sin_t*dat.A*(1- std::pow(x(1)/dat.right,2));

    return solution;
  }

  private:
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class XSquaredWithBoundaries : public Function<dim>,
                            public MeshMovementFunctions<dim>
{
  public:
  XSquaredWithBoundaries(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in),
     dat(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if (coordinate_direction == 0)
      solution = std::pow(x(1),2)* std::pow((dat.right-std::abs(x(1))),2)*this->sin_t*dat.A;
    else if (coordinate_direction == 1)
      solution = std::pow(x(0),2)* std::pow((dat.right-std::abs(x(0))),2)*this->sin_t*dat.A;

    return solution;
  }

  private:
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class DoubleInteriorSinCos : public Function<dim>,
                            public MeshMovementFunctions<dim>
{
  public:
  DoubleInteriorSinCos(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in),
     dat(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
    double solution = 0.0;

      if (coordinate_direction == 0)
        solution = std::sin(2* pi*(x(1)-dat.left)*2/dat.width)*this->sin_t*dat.A*(1- std::pow(x(0)/dat.right,2));
      else if (coordinate_direction == 1)
        solution = std::sin(2* pi*(x(0)-dat.left)*2/dat.width)*this->sin_t*dat.A*(1- std::pow(x(1)/dat.right,2));

    return solution;
  }

  private:
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class DoubleSinCosWithBoundaries : public Function<dim>,
                            public MeshMovementFunctions<dim>
{
  public:
  DoubleSinCosWithBoundaries(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in),
     dat(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
    double solution = 0.0;

      if (coordinate_direction == 0)
        solution = std::sin(2* pi*(x(1)-dat.left)*2/dat.width)*this->sin_t*dat.A;
      else if (coordinate_direction == 1)
        solution = std::sin(2* pi*(x(0)-dat.left)*2/dat.width)*this->sin_t*dat.A;

    return solution;
  }

  private:
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class InteriorSinCos : public Function<dim>,
                            public MeshMovementFunctions<dim>
{
  public:
  InteriorSinCos(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in),
     dat(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
    double solution = 0.0;

      if (coordinate_direction == 0)
        solution = std::sin(2* pi*(x(1)-dat.left)/dat.width)*this->sin_t*dat.A*(1- std::pow(x(0)/dat.right,2));
      else if (coordinate_direction == 1)
        solution = std::sin(2* pi*(x(0)-dat.left)/dat.width)*this->sin_t*dat.A*(1- std::pow(x(1)/dat.right,2));

    return solution;
  }

  private:
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class InteriorSinCosWithSinInTime : public Function<dim>,
                            public MeshMovementFunctions<dim>
{
  public:
  InteriorSinCosWithSinInTime(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in),
     dat(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
    double solution = 0.0;

      if (coordinate_direction == 0)
        solution = std::sin(2* pi*(x(1)-dat.left)/dat.width)*this->sin_t*dat.A*(1- std::pow(x(0)/dat.right,2));
      else if (coordinate_direction == 1)
        solution = std::sin(2* pi*(x(0)-dat.left)/dat.width)*this->sin_t*dat.A*(1- std::pow(x(1)/dat.right,2));

    return solution;
  }

  void
  set_time_velocity(double t) const override
  {
      this->sin_t = std::cos(2*pi*t/dat.T)*2*pi/dat.T;
  }

  private:
  MovingMeshData dat;
  double pi = numbers::PI;
};

template<int dim>
class None : public Function<dim>,
                            public MeshMovementFunctions<dim>
{
  public:
  None(MovingMeshData data_in)
    :Function<dim>(dim, 0.0),
     MeshMovementFunctions<dim>(data_in)
     {}

  double
  displacement(const Point<dim>    &x,
              const unsigned int  coordinate_direction = 0) const override
  {
     return 0.0;
  }

};

}

#endif /*INCLUDE_FUNCTION_MESH_MOVEMENT_H_*/
