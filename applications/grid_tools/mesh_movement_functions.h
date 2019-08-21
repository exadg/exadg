#ifndef INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_
#define INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_

#include "../../include/incompressible_navier_stokes/user_interface/input_parameters.h"

namespace IncNS
{
struct MeshMovementData
{
  MeshMovementData()
    : type(AnalyicMeshMovement::Undefined),
      left(0.0),
      right(0.0),
      height(0.0),
      length(0.0),
      depth(0.0),
      A(0.0),
      f(0.0),
      t_0(0.0),
      t_end(0.0),
      initialize_with_former_mesh_instances(false)
  {
  }

  AnalyicMeshMovement type;
  double              left;
  double              right;
  double              height;
  double              length;
  double              depth;
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
  MeshMovementFunctions(MeshMovementData const & data_in)
    : Function<dim>(dim, 0.0),
      dat(data_in),
      left(data_in.left),
      right(data_in.right),
      f(data_in.f),
      A(data_in.A),
      Dt(data_in.t_end - data_in.t_0),
      height(data_in.height),
      length(data_in.length),
      depth(data_in.depth)
  {
  }

  virtual ~MeshMovementFunctions()
  {
  }

  double
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const
  {
    return compute_displacement_share(x,coordinate_direction) * compute_time_share();
  }

  // velocity is called vale since VectorTools::Interpolate can be used to evaluate the velocity.
  // the displacement requires multigrid support and hence, can not be acessed with VectorTools::Interpolate
  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    return compute_velocity(p,component);
  }

  double
  compute_velocity(const Point<dim> & x, const unsigned int coordinate_direction = 0) const
  {
    double velocity = 0.0;

    if(this->get_time() >= dat.t_0 || dat.initialize_with_former_mesh_instances == true)
      velocity = compute_displacement_share(x,coordinate_direction) * compute_time_deriv_share();
    else if(this->get_time() < dat.t_0 && dat.initialize_with_former_mesh_instances == false)
      velocity =  0.0;

    return velocity;
  }

  virtual double compute_time_share() const
  {
    //By default the time share equals the sin²
    return std::pow(std::sin(2 * pi * this->get_time() / T), 2);
  }

  virtual double compute_time_deriv_share() const
  {
    //By default the time derivative share equals the dsin²/dt
    return (4 * pi * std::sin(2 * pi * this->get_time() / T) * std::cos(2 * pi * this->get_time() / T) / T);
  }

  virtual double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const = 0;

protected:
  MeshMovementData dat;
  double           pi = numbers::PI;
  const double     left;
  const double     right;
  const double     width = right - left;
  const double     f;
  const double     A;
  const double     Dt;
  const double     T = Dt / f;
  const double     height;
  const double     length;
  const double     depth;
};

template<int dim>
class CubeSinCosWithBoundaries : public MeshMovementFunctions<dim>
{
public:
  CubeSinCosWithBoundaries(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) / this->width) * this->A;
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width) * this->A;

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class CubeSinCosWithBoundaries3D : public MeshMovementFunctions<dim>
{
public:
  CubeSinCosWithBoundaries3D(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) / this->width)
                *std::sin(2 * pi * (x(2) - this->left) / this->width) * this->A;
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width)
                *std::sin(2 * pi * (x(2) - this->left) / this->width) * this->A;
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width)
                *std::sin(2 * pi * (x(1) - this->left) / this->width) * this->A;

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class CubeInteriorSinCosOnlyX : public MeshMovementFunctions<dim>
{
public:
  CubeInteriorSinCosOnlyX(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) / this->width) * this->A *
                 (1 - std::pow(x(0) / this->right, 2));
    else if(coordinate_direction == 1)
      solution = 0.0;

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class CubeInteriorSinCosOnlyY : public MeshMovementFunctions<dim>
{
public:
  CubeInteriorSinCosOnlyY(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = 0.0;
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width) * this->A *
                 (1 - std::pow(x(1) / this->right, 2));

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class CubeXSquaredWithBoundaries : public MeshMovementFunctions<dim>
{
public:
  CubeXSquaredWithBoundaries(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution =
        std::pow(x(1), 2) * std::pow((this->right - std::abs(x(1))), 2) * this->A;
    else if(coordinate_direction == 1)
      solution =
        std::pow(x(0), 2) * std::pow((this->right - std::abs(x(0))), 2) * this->A;

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class CubeDoubleInteriorSinCos : public MeshMovementFunctions<dim>
{
public:
  CubeDoubleInteriorSinCos(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) * 2 / this->width) * this->A *
                 (1 - std::pow(x(0) / this->right, 2));
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) * 2 / this->width) * this->A *
                 (1 - std::pow(x(1) / this->right, 2));

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class CubeDoubleSinCosWithBoundaries : public MeshMovementFunctions<dim>
{
public:
  CubeDoubleSinCosWithBoundaries(MeshMovementData const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) * 2 / this->width) * this->A;
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) * 2 / this->width) * this->A;

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class CubeInteriorSinCos : public MeshMovementFunctions<dim>
{
public:
  CubeInteriorSinCos(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) / this->width) * this->A *
                 (1 - std::pow(x(0) / this->right, 2));
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width) * this->A *
                 (1 - std::pow(x(1) / this->right, 2));

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class CubeInteriorSinCos3D : public MeshMovementFunctions<dim>
{
public:
  CubeInteriorSinCos3D(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    double damp0 = (1 - std::pow(x(0) / this->right, 2));
    double damp1 = (1 - std::pow(x(1) / this->right, 2));
    double damp2 = (1 - std::pow(x(2) / this->right, 2));

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) / this->width) * this->A
                *std::sin(2 * pi * (x(2) - this->left) / this->width) * damp0;
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width) * this->A
                *std::sin(2 * pi * (x(2) - this->left) / this->width) * damp1;
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width) * this->A
                *std::sin(2 * pi * (x(1) - this->left) / this->width) * damp2;

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class CubeInteriorSinCosWithSinInTime : public MeshMovementFunctions<dim>
{
public:
  CubeInteriorSinCosWithSinInTime(MeshMovementData const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) / this->width) * this->A *
                 (1 - std::pow(x(0) / this->right, 2));
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width) * this->A *
                 (1 - std::pow(x(1) / this->right, 2));

    return solution;
  }

  double compute_time_share() const override
  {
    return std::sin(2 * pi * this->get_time() / this->T);
  }

  double compute_time_deriv_share() const override
  {
    return std::cos(2 * pi * this->get_time() / this->T) * 2 * pi / this->T;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class RectangleSinCos : public MeshMovementFunctions<dim>
{
public:
  RectangleSinCos(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - (this->height / 2)) / this->height) *
                 this->A * (1 - std::pow((x(0) - (this->length / 2)) / (this->length / 2), 2));
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0)) / this->length) * this->A *
                 (1 - std::pow(x(1) / (this->height / 2), 2));

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class RectangleSinCos3D : public MeshMovementFunctions<dim>
{
public:
  RectangleSinCos3D(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;
    double damp0 =(1 - std::pow((x(0)) / (this->length / 2), 2));
    double damp1 =(1 - std::pow((x(1)) / (this->height / 2), 2));
    double damp2 =(1 - std::pow((x(2)) / (this->depth / 2), 2));

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - (this->height / 2)) / this->height) * this->A
                *std::sin(2 * pi * (x(2) - (this->depth / 2)) / this->depth) *damp0;
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0)-(this->length/2)) / this->length) * this->A
                *std::sin(2 * pi * (x(2)-(this->depth/2)) / this->depth) * damp1;
    else if(coordinate_direction == 2)
      solution = std::sin(2 * pi * (x(1)-(this->height/2)) / this->height) * this->A
                *std::sin(2 * pi * (x(0)-(this->length/2)) / this->length) * damp2;

    return solution;
  }

private:
  double pi = numbers::PI;
};

template<int dim>
class RectangleSinCosWithSinInTime : public MeshMovementFunctions<dim>
{
public:
  RectangleSinCosWithSinInTime(MeshMovementData const & data_in)
    : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  compute_displacement_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - (this->height / 2)) / this->height) *
                 this->A * (1 - std::pow((x(0) - (this->length / 2)) / (this->length / 2), 2));
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0)) / this->length) * this->A *
                 (1 - std::pow(x(1) / (this->height / 2), 2));

    return solution;
  }

  double compute_time_share() const override
  {
    return std::sin(2 * pi * this->get_time() / this->T);
  }

  double compute_time_deriv_share() const override
  {
    return std::cos(2 * pi * this->get_time() / this->T) * 2 * pi / this->T;
  }

private:
  double pi = numbers::PI;
};

} // namespace IncNS

#endif /*INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_*/
