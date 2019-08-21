#ifndef INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_
#define INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_

#include "../../include/incompressible_navier_stokes/user_interface/input_parameters.h"

namespace IncNS
{
template<int dim>
struct MeshMovementData
{
  MeshMovementData()
    : type(AnalyicMeshMovement::Undefined),
      A(0.0),
      f(0.0),
      t_0(0.0),
      t_end(0.0),
      initialize_with_former_mesh_instances(false)
  {
  }

  AnalyicMeshMovement type;
  Tensor<1, dim>      dimensions;
  double              A;
  double              f;
  double              t_0;
  double              t_end;
  bool                initialize_with_former_mesh_instances;
};

template<int dim>
class MeshMovementFunctions : public Function<dim>
{
public:
  MeshMovementFunctions(MeshMovementData<dim> const & data_in)
    : Function<dim>(dim, 0.0),
      dat(data_in),
      f(data_in.f),
      A(data_in.A),
      Dt(data_in.t_end - data_in.t_0),
      dimensions(data_in.dimensions)
  {
  }

  virtual ~MeshMovementFunctions()
  {
  }

  double
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const
  {
    return compute_displacement_share(x, coordinate_direction) * compute_time_share();
  }

  // velocity is called vale since VectorTools::Interpolate can be used to evaluate the velocity.
  // the displacement requires multigrid support and hence, can not be acessed with
  // VectorTools::Interpolate
  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    return compute_velocity(p, component);
  }

  double
  compute_velocity(const Point<dim> & x, const unsigned int coordinate_direction = 0) const
  {
    double velocity = 0.0;

    if(this->get_time() >= dat.t_0 || dat.initialize_with_former_mesh_instances == true)
      velocity = compute_displacement_share(x, coordinate_direction) * compute_time_deriv_share();
    else if(this->get_time() < dat.t_0 && dat.initialize_with_former_mesh_instances == false)
      velocity = 0.0;

    return velocity;
  }

  virtual double
  compute_time_share() const
  {
    // By default the time share equals the sin²
    return std::pow(std::sin(2 * pi * this->get_time() / T), 2);
  }

  virtual double
  compute_time_deriv_share() const
  {
    // By default the time derivative share equals the dsin²/dt
    return (4 * pi * std::sin(2 * pi * this->get_time() / T) *
            std::cos(2 * pi * this->get_time() / T) / T);
  }

  virtual double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const = 0;

protected:
  MeshMovementData<dim> dat;
  double                pi = numbers::PI;
  const double          f;
  const double          A;
  const double          Dt;
  const double          T = Dt / f;
  Tensor<1, dim>        dimensions;
};

template<int dim>
class CubeSinCosWithBoundaries : public MeshMovementFunctions<dim>
{
public:
  CubeSinCosWithBoundaries(MeshMovementData<dim> const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - left) / width) * this->A *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) / width) : 1.0);
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - left) / width) * this->A *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) / width) : 1.0);
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(0) - left) / width) * this->A *
                 std::sin(2 * pi * (x(1) - left) / width);

    return solution;
  }

private:
  double pi    = numbers::PI;
  double left  = -1 / 2 * this->dimensions[0];
  double width = this->dimensions[0];
};

template<int dim>
class CubeInteriorSinCosOnlyX : public MeshMovementFunctions<dim>
{
public:
  CubeInteriorSinCosOnlyX(MeshMovementData<dim> const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution =
        std::sin(2 * pi * (x(1) - left) / width) * this->A * (1 - std::pow(x(0) / right, 2));
    else if(coordinate_direction == 1)
      solution = 0.0;
    else if(coordinate_direction == 2)
      solution = 0.0;

    return solution;
  }

private:
  double pi    = numbers::PI;
  double left  = -1 / 2 * this->dimensions[0];
  double right = 1 / 2 * this->dimensions[0];
  double width = this->dimensions[0];
};

template<int dim>
class CubeInteriorSinCosOnlyY : public MeshMovementFunctions<dim>
{
public:
  CubeInteriorSinCosOnlyY(MeshMovementData<dim> const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = 0.0;
    else if(coordinate_direction == 1)
      solution =
        std::sin(2 * pi * (x(0) - left) / width) * this->A * (1 - std::pow(x(1) / right, 2));
    else if(coordinate_direction == 2)
      solution = 0.0;

    return solution;
  }

private:
  double pi    = numbers::PI;
  double left  = -1 / 2 * this->dimensions[0];
  double right = 1 / 2 * this->dimensions[0];
  double width = this->dimensions[0];
};

template<int dim>
class CubeDoubleInteriorSinCos : public MeshMovementFunctions<dim>
{
public:
  CubeDoubleInteriorSinCos(MeshMovementData<dim> const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    double damp0 = (1 - std::pow(x(0) / right, 2));
    double damp1 = (1 - std::pow(x(1) / right, 2));
    double damp2 = (1 - std::pow(x(2) / right, 2));

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - left) * 2 / width) * this->A * damp0 *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) * 2 / width) : 1.0);
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - left) * 2 / width) * this->A * damp1 *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) * 2 / width) : 1.0);
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(0) - left) * 2 / width) * this->A * damp2 *
                 std::sin(2 * pi * (x(1) - left) * 2 / width);

    return solution;
  }

private:
  double pi    = numbers::PI;
  double left  = -1 / 2 * this->dimensions[0];
  double right = 1 / 2 * this->dimensions[0];
  double width = this->dimensions[0];
};

template<int dim>
class CubeDoubleSinCosWithBoundaries : public MeshMovementFunctions<dim>
{
public:
  CubeDoubleSinCosWithBoundaries(MeshMovementData<dim> const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - left) * 2 / width) * this->A *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) * 2 / width) : 1.0);
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - left) * 2 / width) * this->A *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) * 2 / width) : 1.0);
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(0) - left) * 2 / width) * this->A *
                 std::sin(2 * pi * (x(1) - left) * 2 / width);


    return solution;
  }

private:
  double pi    = numbers::PI;
  double left  = -1 / 2 * this->dimensions[0];
  double width = this->dimensions[0];
};

template<int dim>
class CubeInteriorSinCos : public MeshMovementFunctions<dim>
{
public:
  CubeInteriorSinCos(MeshMovementData<dim> const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    double damp0 = (1 - std::pow(x(0) / right, 2));
    double damp1 = (1 - std::pow(x(1) / right, 2));
    double damp2 = (1 - std::pow(x(2) / right, 2));

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - left) / width) * this->A * damp0 *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) / width) : 1.0);
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - left) / width) * this->A * damp1 *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) / width) : 1.0);
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(0) - left) / width) * this->A * damp2 *
                 std::sin(2 * pi * (x(1) - left) / width);

    return solution;
  }

private:
  double pi    = numbers::PI;
  double left  = -1 / 2 * this->dimensions[0];
  double right = 1 / 2 * this->dimensions[0];
  double width = this->dimensions[0];
};

template<int dim>
class CubeInteriorSinCosWithSinInTime : public MeshMovementFunctions<dim>
{
public:
  CubeInteriorSinCosWithSinInTime(MeshMovementData<dim> const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    double damp0 = (1 - std::pow(x(0) / right, 2));
    double damp1 = (1 - std::pow(x(1) / right, 2));
    double damp2 = (1 - std::pow(x(2) / right, 2));

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - left) / width) * this->A * damp0 *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) / width) : 1.0);
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - left) / width) * this->A * damp1 *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) / width) : 1.0);
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(0) - left) / width) * this->A * damp2 *
                 std::sin(2 * pi * (x(1) - left) / width);

    return solution;
  }

  double
  compute_time_share() const override
  {
    return std::sin(2 * pi * this->get_time() / this->T);
  }

  double
  compute_time_deriv_share() const override
  {
    return std::cos(2 * pi * this->get_time() / this->T) * 2 * pi / this->T;
  }

private:
  double pi    = numbers::PI;
  double left  = -1 / 2 * this->dimensions[0];
  double right = 1 / 2 * this->dimensions[0];
  double width = this->dimensions[0];
};

template<int dim>
class RectangleSinCos : public MeshMovementFunctions<dim>
{
public:
  RectangleSinCos(MeshMovementData<dim> const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x_in,
                             const unsigned int coordinate_direction = 0) const override
  {
    // For 2D and 3D the coordinate system is set differently
    Point<dim> x = x_in;
    if(dim == 2)
      x[0] -= length / 2;

    double solution = 0.0;
    double damp0    = (1 - std::pow((x(0)) / (length / 2), 2));
    double damp1    = (1 - std::pow((x(1)) / (height / 2), 2));
    double damp2    = (1 - std::pow((x(2)) / (depth / 2), 2));

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - (height / 2)) / height) * this->A * damp0 *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - (depth / 2)) / depth) : 1.0);
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - (length / 2)) / length) * this->A * damp1 *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - (depth / 2)) / depth) : 1.0);
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(1) - (height / 2)) / height) * this->A * damp2 *
                 std::sin(2 * pi * (x(0) - (length / 2)) / length);

    return solution;
  }

private:
  double pi     = numbers::PI;
  double length = this->dimensions[0];
  double height = this->dimensions[1];
  double depth  = this->dimensions[2];
};

template<int dim>
class RectangleSinCosWithSinInTime : public MeshMovementFunctions<dim>
{
public:
  RectangleSinCosWithSinInTime(MeshMovementData<dim> const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x_in,
                             const unsigned int coordinate_direction = 0) const override
  {
    // For 2D and 3D the coordinate system is set differently
    Point<dim> x = x_in;
    if(dim == 2)
      x[0] -= length / 2;

    double solution = 0.0;
    double damp0    = (1 - std::pow((x(0)) / (length / 2), 2));
    double damp1    = (1 - std::pow((x(1)) / (height / 2), 2));
    double damp2    = (1 - std::pow((x(2)) / (depth / 2), 2));

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - (height / 2)) / height) * this->A * damp0 *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - (depth / 2)) / depth) : 1.0);
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - (length / 2)) / length) * this->A * damp1 *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - (depth / 2)) / depth) : 1.0);
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(1) - (height / 2)) / height) * this->A * damp2 *
                 std::sin(2 * pi * (x(0) - (length / 2)) / length);

    return solution;
  }

  double
  compute_time_share() const override
  {
    return std::sin(2 * pi * this->get_time() / this->T);
  }

  double
  compute_time_deriv_share() const override
  {
    return std::cos(2 * pi * this->get_time() / this->T) * 2 * pi / this->T;
  }

private:
  double pi     = numbers::PI;
  double length = this->dimensions[0];
  double height = this->dimensions[1];
  double depth  = this->dimensions[2];
};

template<int dim>
class CubeSinCosWithBoundariesWithSinInTime : public MeshMovementFunctions<dim>
{
public:
  CubeSinCosWithBoundariesWithSinInTime(MeshMovementData<dim> const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - left) / width) * this->A *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) / width) : 1.0);
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - left) / width) * this->A *
                 (dim == 3 ? std::sin(2 * pi * (x(2) - left) / width) : 1.0);
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(0) - left) / width) * this->A *
                 std::sin(2 * pi * (x(1) - left) / width);

    return solution;
  }

  double
  compute_time_share() const override
  {
    return std::sin(2 * pi * this->get_time() / this->T);
  }

  double
  compute_time_deriv_share() const override
  {
    return std::cos(2 * pi * this->get_time() / this->T) * 2 * pi / this->T;
  }

private:
  double pi    = numbers::PI;
  double left  = -1 / 2 * this->dimensions[0];
  double right = 1 / 2 * this->dimensions[0];
  double width = this->dimensions[0];
};

} // namespace IncNS

#endif /*INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_*/
