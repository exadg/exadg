#ifndef INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_
#define INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_

#include "../../include/incompressible_navier_stokes/user_interface/input_parameters.h"

namespace IncNS
{
template<int dim>
class MeshMovementFunctions : public Function<dim>
{
public:
  MeshMovementFunctions(InputParameters const & data_in)
    : Function<dim>(dim, 0.0),
      sin_t(0.0),
      dat(data_in),
      t_current(0.0),
      left(data_in.triangulation_left),
      right(data_in.triangulation_right),
      f(data_in.grid_movement_frequency),
      A(data_in.grid_movement_amplitude),
      Dt(data_in.end_time - data_in.start_time),
      height(data_in.triangulation_height),
      length(data_in.triangulation_length)
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
    if(t_current >= dat.start_time || dat.initialize_with_former_mesh_instances == true)
      value = displacement(p, component);
    else if(t_current < dat.start_time && dat.initialize_with_former_mesh_instances == false)
      value = 0.0;

    return value;
  }

protected:
  double          sin_t;
  InputParameters dat;
  double          t_current;
  double          pi = numbers::PI;
  const double    left;
  const double    right;
  const double    width = right - left;
  const double    f;
  const double    A;
  const double    Dt;
  const double    T = Dt / f;
  const double    height;
  const double    length;
};

template<int dim>
class CubeSinCosWithBoundaries : public MeshMovementFunctions<dim>
{
public:
  CubeSinCosWithBoundaries(InputParameters const & data_in) : MeshMovementFunctions<dim>(data_in)
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
  CubeInteriorSinCosOnlyX(InputParameters const & data_in) : MeshMovementFunctions<dim>(data_in)
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
  CubeInteriorSinCosOnlyY(InputParameters const & data_in) : MeshMovementFunctions<dim>(data_in)
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
  CubeXSquaredWithBoundaries(InputParameters const & data_in) : MeshMovementFunctions<dim>(data_in)
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
  CubeDoubleInteriorSinCos(InputParameters const & data_in) : MeshMovementFunctions<dim>(data_in)
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
  CubeDoubleSinCosWithBoundaries(InputParameters const & data_in)
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
  CubeInteriorSinCos(InputParameters const & data_in) : MeshMovementFunctions<dim>(data_in)
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
  CubeInteriorSinCosWithSinInTime(InputParameters const & data_in)
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
  RectangleSinCos(InputParameters const & data_in) : MeshMovementFunctions<dim>(data_in)
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
  RectangleSinCosWithSinInTime(InputParameters const & data_in)
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
