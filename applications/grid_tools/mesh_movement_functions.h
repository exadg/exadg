#ifndef INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_
#define INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_

#include "../../include/incompressible_navier_stokes/user_interface/input_parameters.h"

namespace IncNS
{
template<int dim>
struct MeshMovementData
{
  MeshMovementData()
    : temporal(MeshMovementAdvanceInTime::Undefined),
      shape(MeshMovementShape::Undefined),
      amplitude(0.0),
      frequency(0.0),
      t_start(0.0),
      t_end(0.0),
      spatial_number_of_oscillations(1),
      start_with_low_order(true),
      damp_towards_bondaries(true)
  {
  }

  MeshMovementAdvanceInTime temporal;
  MeshMovementShape         shape;
  Tensor<1, dim>            dimensions;
  double                    amplitude;
  double                    frequency;
  double                    t_start;
  double                    t_end;
  double                    spatial_number_of_oscillations;
  bool                      start_with_low_order;
  bool                      damp_towards_bondaries;
};

template<int dim>
class MeshMovementFunctions : public Function<dim>
{
public:
  MeshMovementFunctions() : Function<dim>(dim, 0.0)
  {
  }

  virtual ~MeshMovementFunctions()
  {
  }

  virtual double
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const = 0;

  // velocity is called vale since VectorTools::Interpolate can be used to evaluate the velocity.
  // the displacement requires multigrid support and hence, can not be acessed with
  // VectorTools::Interpolate
  virtual double
  value(const Point<dim> & p, const unsigned int component = 0) const = 0;
};



template<int dim>
class CubeMeshMovementFunctions : public MeshMovementFunctions<dim>
{
public:
  CubeMeshMovementFunctions(MeshMovementData<dim> const & data_in)
    : MeshMovementFunctions<dim>(),
      data(data_in),
      width(data_in.dimensions[0]),
      left(-1.0 / 2.0 * width),
      right(-left),
      runtime(data_in.t_end - data_in.t_start),
      time_period(runtime / data_in.frequency)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const override
  {
    double velocity = 0.0;

    if(this->get_time() >= data.t_start || data.start_with_low_order == false)
      velocity = compute_displacement_share(p, component) * compute_damping_share(p, component) *
                 compute_time_deriv_share();
    else if(this->get_time() < data.t_start && data.start_with_low_order == true)
      velocity = 0.0;

    return velocity;
  }

  double
  displacement(const Point<dim> & x, const unsigned int coordinate_direction = 0) const override
  {
    double displacement = 0.0;

    if(this->get_time() >= data.t_start || data.start_with_low_order == false)
      displacement = compute_displacement_share(x, coordinate_direction) *
                     compute_damping_share(x, coordinate_direction) * compute_time_share();
    else if(this->get_time() < data.t_start && data.start_with_low_order == true)
      displacement = 0.0;

    return displacement;
  }

private:
  double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const
  {
    double solution = 0.0;

    switch(data.shape)
    {
      case MeshMovementShape::Undefined:
        AssertThrow(false,
                    ExcMessage(
                      "You are trying to use a mesh moving function but didn't specify its shape"));
        break;

      case MeshMovementShape::Sin:
        if(coordinate_direction == 0)
          solution =
            std::sin(2.0 * pi * (x(1) - left) * data.spatial_number_of_oscillations / width) *
            data.amplitude *
            (dim == 3 ?
               std::sin(2.0 * pi * (x(2) - left) * data.spatial_number_of_oscillations / width) :
               1.0);
        else if(coordinate_direction == 1)
          solution =
            std::sin(2.0 * pi * (x(0) - left) * data.spatial_number_of_oscillations / width) *
            data.amplitude *
            (dim == 3 ?
               std::sin(2.0 * pi * (x(2) - left) * data.spatial_number_of_oscillations / width) :
               1.0);
        else if(coordinate_direction == 2)
          solution =
            std::sin(2.0 * pi * (x(0) - left) * data.spatial_number_of_oscillations / width) *
            data.amplitude *
            std::sin(2.0 * pi * (x(1) - left) * data.spatial_number_of_oscillations / width);
        break;

      default:
        AssertThrow(false, ExcMessage("Not implemented."));
        break;
    }

    return solution;
  }

  double
  compute_time_deriv_share() const
  {
    double solution = 0.0;

    switch(data.temporal)
    {
      case MeshMovementAdvanceInTime::Undefined:
        AssertThrow(
          false,
          ExcMessage(
            "You are trying to use a mesh moving function but didn't specify how it is advanced in time"));
        break;

      case MeshMovementAdvanceInTime::SinSquared:
        solution = (4.0 * pi * std::sin(2.0 * pi * this->get_time() / time_period) *
                    std::cos(2.0 * pi * this->get_time() / time_period) / time_period);
        break;

      case MeshMovementAdvanceInTime::Sin:
        solution = std::cos(2.0 * pi * this->get_time() / time_period) * 2.0 * pi / time_period;
        break;

      default:
        AssertThrow(false, ExcMessage("Not implemented."));
        break;
    }
    return solution;
  }

  double
  compute_time_share() const
  {
    double solution = 0.0;

    switch(data.temporal)
    {
      case MeshMovementAdvanceInTime::Undefined:
        AssertThrow(
          false,
          ExcMessage(
            "You are trying to use a mesh moving function but didn't specify how it is advanced in time"));
        break;

      case MeshMovementAdvanceInTime::SinSquared:
        solution = std::pow(std::sin(2.0 * pi * this->get_time() / time_period), 2);
        break;

      case MeshMovementAdvanceInTime::Sin:
        solution = std::sin(2.0 * pi * this->get_time() / time_period);
        break;

      default:
        AssertThrow(false, ExcMessage("Not implemented."));
        break;
    }
    return solution;
  }

  double
  compute_damping_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const
  {
    double damp = 0.0;

    if(this->data.damp_towards_bondaries == true)
      damp = (1 - std::pow(x(coordinate_direction) / right, 2));
    else
      damp = 1.0;

    return damp;
  }

protected:
  const double                pi = numbers::PI;
  MeshMovementData<dim> const data;
  double const                width;
  double const                left;
  double const                right;
  double const                runtime;
  double const                time_period;
};

template<int dim>
class RectangleMeshMovementFunctions : public MeshMovementFunctions<dim>
{
public:
  RectangleMeshMovementFunctions(MeshMovementData<dim> const & data_in)
    : MeshMovementFunctions<dim>(),
      data(data_in),
      length(data_in.dimensions[0]),
      height(data_in.dimensions[1]),
      runtime(data_in.t_end - data_in.t_start),
      time_period(runtime / data_in.frequency)
  {
    if(dim == 3)
      depth = data_in.dimensions[2];
  }

  double
  value(const Point<dim> & p_in, const unsigned int component = 0) const override
  {
    // For 2D and 3D the coordinate system is set differently
    Point<dim> p = p_in;
    if(dim == 2)
      p[0] -= length / 2.0;

    double velocity = 0.0;

    if(this->get_time() >= data.t_start || data.start_with_low_order == false)
      velocity = compute_displacement_share(p, component) * compute_damping_share(p, component) *
                 compute_time_deriv_share();
    else if(this->get_time() < data.t_start && data.start_with_low_order == true)
      velocity = 0.0;

    return velocity;
  }

  double
  displacement(const Point<dim> & x_in, const unsigned int coordinate_direction = 0) const override
  {
    // For 2D and 3D the coordinate system is set differently
    Point<dim> x = x_in;
    if(dim == 2)
      x[0] -= length / 2.0;

    double displacement = 0.0;

    if(this->get_time() >= data.t_start || data.start_with_low_order == false)
      displacement = compute_displacement_share(x, coordinate_direction) *
                     compute_damping_share(x, coordinate_direction) * compute_time_share();
    else if(this->get_time() < data.t_start && data.start_with_low_order == true)
      displacement = 0.0;

    return displacement;
  }

private:
  double
  compute_displacement_share(const Point<dim> & x,
                             const unsigned int coordinate_direction = 0) const
  {
    double solution = 0.0;

    switch(data.shape)
    {
      case MeshMovementShape::Undefined:
        AssertThrow(false,
                    ExcMessage(
                      "You are trying to use a mesh moving function but didn't specify its shape"));
        break;

      case MeshMovementShape::Sin:
        if(coordinate_direction == 0)
          solution = std::sin(2.0 * pi * (x(1) - (height / 2.0)) / height) * data.amplitude *
                     (dim == 3 ? std::sin(2 * pi * (x(2.0) - (depth / 2)) / depth) : 1.0);
        else if(coordinate_direction == 1)
          solution = std::sin(2.0 * pi * (x(0) - (length / 2.0)) / length) * data.amplitude *
                     (dim == 3 ? std::sin(2 * pi * (x(2) - (depth / 2.0)) / depth) : 1.0);
        else if(coordinate_direction == 2)
          solution = std::sin(2.0 * pi * (x(1) - (height / 2.0)) / height) * data.amplitude *
                     std::sin(2.0 * pi * (x(0) - (length / 2.0)) / length);
        break;

      default:
        AssertThrow(false, ExcMessage("Not implemented."));
        break;
    }

    return solution;
  }

  double
  compute_time_deriv_share() const
  {
    double solution = 0.0;

    switch(data.temporal)
    {
      case MeshMovementAdvanceInTime::Undefined:
        AssertThrow(
          false,
          ExcMessage(
            "You are trying to use a mesh moving function but didn't specify how it is advanced in time"));
        break;

      case MeshMovementAdvanceInTime::SinSquared:
        solution = (4.0 * pi * std::sin(2.0 * pi * this->get_time() / time_period) *
                    std::cos(2.0 * pi * this->get_time() / time_period) / time_period);
        break;

      case MeshMovementAdvanceInTime::Sin:
        solution = std::cos(2.0 * pi * this->get_time() / time_period) * 2.0 * pi / time_period;
        break;

      default:
        AssertThrow(false, ExcMessage("Not implemented."));
        break;
    }
    return solution;
  }

  double
  compute_time_share() const
  {
    double solution = 0.0;

    switch(data.temporal)
    {
      case MeshMovementAdvanceInTime::Undefined:
        AssertThrow(
          false,
          ExcMessage(
            "You are trying to use a mesh moving function but didn't specify how it is advanced in time"));
        break;

      case MeshMovementAdvanceInTime::SinSquared:
        solution = std::pow(std::sin(2.0 * pi * this->get_time() / time_period), 2);
        break;

      case MeshMovementAdvanceInTime::Sin:
        solution = std::sin(2.0 * pi * this->get_time() / time_period);
        break;

      default:
        AssertThrow(false, ExcMessage("Not implemented."));
        break;
    }
    return solution;
  }

  double
  compute_damping_share(const Point<dim> & x, const unsigned int coordinate_direction = 0) const
  {
    double damp = 0.0;

    if(this->data.damp_towards_bondaries == true)
      damp = (1.0 -
              std::pow(x(coordinate_direction) / (data.dimensions[coordinate_direction] / 2.0), 2));
    else
      damp = 1.0;

    return damp;
  }

protected:
  const double                pi = numbers::PI;
  MeshMovementData<dim> const data;
  double const                length;
  double const                height;
  double const                depth;
  double const                runtime;
  double const                time_period;
};
//
//  template<int dim>
//  class RectangleSinCos : public MeshMovementFunctions<dim>
//  {
//  public:
//    RectangleSinCos(MeshMovementData<dim> const & data_in)
//      : MeshMovementFunctions<dim>(data_in),
//        length(data_in.dimensions[0]),
//        height(data_in.dimensions[1]),
//        dimensions(data_in.dimensions)
//    {
//      if(dim == 3)
//        depth = data_in.dimensions[2];
//    }
//
//    double
//    compute_displacement_share(const Point<dim> & x_in,
//                               const unsigned int coordinate_direction = 0) const override
//    {
//      // For 2D and 3D the coordinate system is set differently
//      Point<dim> x = x_in;
//      if(dim == 2)
//        x[0] -= length / 2;
//
//      double solution = 0.0;
//      double damp =
//        (1 - std::pow(x(coordinate_direction) / (dimensions[coordinate_direction] / 2), 2));
//
//      if(coordinate_direction == 0)
//        solution = std::sin(2 * pi * (x(1) - (height / 2)) / height) * this->A * damp *
//                   (dim == 3 ? std::sin(2 * pi * (x(2) - (depth / 2)) / depth) : 1.0);
//      else if(coordinate_direction == 1)
//        solution = std::sin(2 * pi * (x(0) - (length / 2)) / length) * this->A * damp *
//                   (dim == 3 ? std::sin(2 * pi * (x(2) - (depth / 2)) / depth) : 1.0);
//      else if(coordinate_direction == 2)
//        solution = std::sin(2 * pi * (x(1) - (height / 2)) / height) * this->A * damp *
//                   std::sin(2 * pi * (x(0) - (length / 2)) / length);
//
//      return solution;
//    }
//
//  private:
//    double         pi = numbers::PI;
//    double         length;
//    double         height;
//    double         depth = 0;
//    Tensor<1, dim> dimensions;
//  };
//
//  template<int dim>
//  class RectangleSinCosWithSinInTime : public MeshMovementFunctions<dim>
//  {
//  public:
//    RectangleSinCosWithSinInTime(MeshMovementData<dim> const & data_in)
//      : MeshMovementFunctions<dim>(data_in),
//        length(data_in.dimensions[0]),
//        height(data_in.dimensions[1]),
//        dimensions(data_in.dimensions)
//    {
//      if(dim == 3)
//        depth = data_in.dimensions[2];
//    }
//
//    double
//    compute_displacement_share(const Point<dim> & x_in,
//                               const unsigned int coordinate_direction = 0) const override
//    {
//      // For 2D and 3D the coordinate system is set differently
//      Point<dim> x = x_in;
//      if(dim == 2)
//        x[0] -= length / 2;
//
//      double solution = 0.0;
//
//      double damp =
//        (1 - std::pow(x(coordinate_direction) / (dimensions[coordinate_direction] / 2), 2));
//
//
//      if(coordinate_direction == 0)
//        solution = std::sin(2 * pi * (x(1) - (height / 2)) / height) * this->A * damp *
//                   (dim == 3 ? std::sin(2 * pi * (x(2) - (depth / 2)) / depth) : 1.0);
//      else if(coordinate_direction == 1)
//        solution = std::sin(2 * pi * (x(0) - (length / 2)) / length) * this->A * damp *
//                   (dim == 3 ? std::sin(2 * pi * (x(2) - (depth / 2)) / depth) : 1.0);
//      else if(coordinate_direction == 2)
//        solution = std::sin(2 * pi * (x(1) - (height / 2)) / height) * this->A * damp *
//                   std::sin(2 * pi * (x(0) - (length / 2)) / length);
//
//      return solution;
//    }
//
//    double
//    compute_time_share() const override
//    {
//      return std::sin(2 * pi * this->get_time() / this->T);
//    }
//
//    double
//    compute_time_deriv_share() const override
//    {
//      return std::cos(2 * pi * this->get_time() / this->T) * 2 * pi / this->T;
//    }
//
//  private:
//    double         pi = numbers::PI;
//    double         length;
//    double         height;
//    double         depth = 0;
//    Tensor<1, dim> dimensions;
//  };



} // namespace IncNS

#endif /*INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_*/
