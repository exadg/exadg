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
      sin_t(0.0),
      dat(data_in),
      t_current(0.0),
      left(data_in.left),
      right(data_in.right),
      f(data_in.f),
      A(data_in.A),
      Dt(data_in.t_end - data_in.t_0),
      height(data_in.height),
      length(data_in.length)
  {
  }

  virtual ~MeshMovementFunctions()
  {
  }

  virtual double
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const = 0;

  virtual void
  set_time_displacement(double const t)
  {
    t_current = t;
    sin_t     = std::pow(std::sin(2 * pi * t / T), 2);
  }

  virtual void
  set_time_velocity(double const t)
  { // mostly Sin² is used to advance mesh movement in time
    t_current = t;
    sin_t     = (4 * pi * std::sin(2 * pi * t / T) * std::cos(2 * pi * t / T) / T);
  }

  // Velocity doesnt require multigrid support, hence to be able to use interpolation by deal.II
  // this function is called value()
  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    // Since displacements are of shape const*f(t), code duplication can be avoided using
    // f(t)=\partial_t f(t)
    double value = 0.0;
    if(t_current >= dat.t_0 || dat.initialize_with_former_mesh_instances == true)
      value = displacement(p, component);
    else if(t_current < dat.t_0 && dat.initialize_with_former_mesh_instances == false)
      value = 0.0;

    return value;
  }

protected:
  double           sin_t;
  MeshMovementData dat;
  double           t_current;
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
};

template<int dim>
class CubeSinCosWithBoundaries : public MeshMovementFunctions<dim>
{
public:
  CubeSinCosWithBoundaries(MeshMovementData const & data_in) : MeshMovementFunctions<dim>(data_in)
  {
  }

  double
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) / this->width) * this->sin_t * this->A;
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width) * this->sin_t * this->A;

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
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) / this->width) * this->sin_t * this->A *
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
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = 0.0;
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width) * this->sin_t * this->A *
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
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution =
        std::pow(x(1), 2) * std::pow((this->right - std::abs(x(1))), 2) * this->sin_t * this->A;
    else if(coordinate_direction == 1)
      solution =
        std::pow(x(0), 2) * std::pow((this->right - std::abs(x(0))), 2) * this->sin_t * this->A;

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
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) * 2 / this->width) * this->sin_t * this->A *
                 (1 - std::pow(x(0) / this->right, 2));
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) * 2 / this->width) * this->sin_t * this->A *
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
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) * 2 / this->width) * this->sin_t * this->A;
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) * 2 / this->width) * this->sin_t * this->A;

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
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) / this->width) * this->sin_t * this->A *
                 (1 - std::pow(x(0) / this->right, 2));
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width) * this->sin_t * this->A *
                 (1 - std::pow(x(1) / this->right, 2));

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
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - this->left) / this->width) * this->sin_t * this->A *
                 (1 - std::pow(x(0) / this->right, 2));
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0) - this->left) / this->width) * this->sin_t * this->A *
                 (1 - std::pow(x(1) / this->right, 2));

    return solution;
  }

  void
  set_time_velocity(double const t) override
  {
    this->t_current = t;
    this->sin_t     = std::cos(2 * pi * t / this->T) * 2 * pi / this->T;
  }

  void
  set_time_displacement(double const t) override
  {
    this->t_current = t;
    this->sin_t     = std::sin(2 * pi * t / this->T);
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
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - (this->height / 2)) / this->height) * this->sin_t *
                 this->A * (1 - std::pow((x(0) - (this->length / 2)) / (this->length / 2), 2));
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0)) / this->length) * this->sin_t * this->A *
                 (1 - std::pow(x(1) / (this->height / 2), 2));

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
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double solution = 0.0;

    if(coordinate_direction == 0)
      solution = std::sin(2 * pi * (x(1) - (this->height / 2)) / this->height) * this->sin_t *
                 this->A * (1 - std::pow((x(0) - (this->length / 2)) / (this->length / 2), 2));
    else if(coordinate_direction == 1)
      solution = std::sin(2 * pi * (x(0)) / this->length) * this->sin_t * this->A *
                 (1 - std::pow(x(1) / (this->height / 2), 2));

    return solution;
  }

  void
  set_time_velocity(double const t) override
  {
    this->t_current = t;
    this->sin_t     = std::cos(2 * pi * t / this->T) * 2 * pi / this->T;
  }

  void
  set_time_displacement(double const t) override
  {
    this->t_current = t;
    this->sin_t     = std::sin(2 * pi * t / this->T);
  }

private:
  double pi = numbers::PI;
};

} // namespace IncNS

#endif /*INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_*/
