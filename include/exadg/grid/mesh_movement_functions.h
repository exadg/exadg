/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_
#define INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_

#include <deal.II/base/function.h>

namespace ExaDG
{
enum class MeshMovementAdvanceInTime
{
  Undefined,
  SinSquared,
  Sin
};

enum class MeshMovementShape
{
  Undefined,
  Sin,
  SineZeroAtBoundary,
  SineAligned
};

template<int dim>
struct MeshMovementData
{
  MeshMovementData()
    : temporal(MeshMovementAdvanceInTime::Undefined),
      shape(MeshMovementShape::Undefined),
      amplitude(0.0),
      period(1.0),
      t_start(0.0),
      t_end(1.0),
      spatial_number_of_oscillations(1)
  {
  }

  MeshMovementAdvanceInTime temporal;
  MeshMovementShape         shape;
  dealii::Tensor<1, dim>    dimensions;
  double                    amplitude;
  double                    period;
  double                    t_start;
  double                    t_end;
  double                    spatial_number_of_oscillations;
};

template<int dim>
class CubeMeshMovementFunctions : public dealii::Function<dim>
{
public:
  CubeMeshMovementFunctions(MeshMovementData<dim> const & data_in)
    : dealii::Function<dim>(dim),
      data(data_in),
      width(data_in.dimensions[0]),
      left(-1.0 / 2.0 * width),
      right(-left),
      runtime(data_in.t_end - data_in.t_start),
      time_period(data_in.period)
  {
  }

  double
  value(dealii::Point<dim> const & x, unsigned int const coordinate_direction = 0) const override
  {
    double displacement = 0.0;

    displacement = compute_displacement_share(x, coordinate_direction) * compute_time_share();

    return displacement;
  }

private:
  double
  compute_displacement_share(dealii::Point<dim> const & x,
                             unsigned int const         coordinate_direction = 0) const
  {
    double solution = 0.0;

    switch(data.shape)
    {
      case MeshMovementShape::Undefined:
        AssertThrow(false, dealii::ExcMessage("Undefined parameter MeshMovementShape."));
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

      case MeshMovementShape::SineZeroAtBoundary:
        if(coordinate_direction == 0)
          solution =
            std::sin(pi * (x(0) - left) * data.spatial_number_of_oscillations / width) *
            std::sin(2.0 * pi * (x(1) - left) * data.spatial_number_of_oscillations / width) *
            data.amplitude *
            (dim == 3 ?
               std::sin(2.0 * pi * (x(2) - left) * data.spatial_number_of_oscillations / width) :
               1.0);
        else if(coordinate_direction == 1)
          solution =
            std::sin(pi * (x(1) - left) * data.spatial_number_of_oscillations / width) *
            std::sin(2.0 * pi * (x(0) - left) * data.spatial_number_of_oscillations / width) *
            data.amplitude *
            (dim == 3 ?
               std::sin(2.0 * pi * (x(2) - left) * data.spatial_number_of_oscillations / width) :
               1.0);
        else if(coordinate_direction == 2)
          solution =
            std::sin(pi * (x(2) - left) * data.spatial_number_of_oscillations / width) *
            std::sin(2.0 * pi * (x(0) - left) * data.spatial_number_of_oscillations / width) *
            data.amplitude *
            std::sin(2.0 * pi * (x(1) - left) * data.spatial_number_of_oscillations / width);
        break;

      case MeshMovementShape::SineAligned:
        solution = std::sin(2.0 * pi * (x(coordinate_direction) - left) *
                            data.spatial_number_of_oscillations / width) *
                   data.amplitude;
        break;

      default:
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
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
        AssertThrow(false, dealii::ExcMessage("Undefined parameter MeshMovementAdvanceInTime."));
        break;

      case MeshMovementAdvanceInTime::SinSquared:
        solution = std::pow(std::sin(2.0 * pi * this->get_time() / time_period), 2);
        break;

      case MeshMovementAdvanceInTime::Sin:
        solution = std::sin(2.0 * pi * this->get_time() / time_period);
        break;

      default:
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
        break;
    }
    return solution;
  }

protected:
  double const                pi = dealii::numbers::PI;
  MeshMovementData<dim> const data;
  double const                width;
  double const                left;
  double const                right;
  double const                runtime;
  double const                time_period;
};

template<int dim>
class RectangleMeshMovementFunctions : public dealii::Function<dim>
{
public:
  RectangleMeshMovementFunctions(MeshMovementData<dim> const & data_in)
    : dealii::Function<dim>(dim),
      data(data_in),
      length(data_in.dimensions[0]),
      height(data_in.dimensions[1]),
      depth(dim == 3 ? data_in.dimensions[2] : 1.0),
      runtime(data_in.t_end - data_in.t_start),
      time_period(data_in.period)
  {
  }

  double
  value(dealii::Point<dim> const & x_in, unsigned int const coordinate_direction = 0) const override
  {
    // For 2D and 3D the coordinate system is set differently
    dealii::Point<dim> x = x_in;
    if(dim == 2)
      x[0] -= length / 2.0;

    double displacement = 0.0;

    displacement = compute_displacement_share(x, coordinate_direction) * compute_time_share();

    return displacement;
  }

private:
  double
  compute_displacement_share(dealii::Point<dim> const & x,
                             unsigned int const         coordinate_direction = 0) const
  {
    double solution = 0.0;

    switch(data.shape)
    {
      case MeshMovementShape::Undefined:
        AssertThrow(false, dealii::ExcMessage("Undefined parameter MeshMovementShape."));
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
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
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
        AssertThrow(false, dealii::ExcMessage("Undefined parameter MeshMovementAdvanceInTime."));
        break;

      case MeshMovementAdvanceInTime::SinSquared:
        solution = std::pow(std::sin(2.0 * pi * this->get_time() / time_period), 2);
        break;

      case MeshMovementAdvanceInTime::Sin:
        solution = std::sin(2.0 * pi * this->get_time() / time_period);
        break;

      default:
        AssertThrow(false, dealii::ExcMessage("Not implemented."));
        break;
    }
    return solution;
  }

protected:
  double const                pi = dealii::numbers::PI;
  MeshMovementData<dim> const data;
  double const                length;
  double const                height;
  double const                depth;
  double const                runtime;
  double const                time_period;
};


template<int dim>
struct RotationMeshMovementData
{
  RotationMeshMovementData() : counter_clockwise_degrees_per_s(0.0)
  {
  }

  double counter_clockwise_degrees_per_s;
};

template<int dim>
class RotationMeshMovementFunctions : public dealii::Function<dim>
{
public:
  RotationMeshMovementFunctions(RotationMeshMovementData<dim> const & data)
    : dealii::Function<dim>(dim),
      angular_velocity{dealii::numbers::PI / 180. * data.counter_clockwise_degrees_per_s}
  {
  }

  double
  value(dealii::Point<dim> const & x, unsigned int const coordinate_direction = 0) const override
  {
    double displacement = 0.0;

    double const phi = std::atan2(x(1), x(0)) + this->get_time() * angular_velocity;
    double const r   = std::sqrt(x(0) * x(0) + x(1) * x(1));
    if(coordinate_direction == 0)
      displacement = -r * std::cos(phi) - x(0);
    else if(coordinate_direction == 1)
      displacement = -r * std::sin(phi) - x(1);
    else if(coordinate_direction == 2)
      displacement = 0.;

    return displacement;
  }

private:
  double const angular_velocity;
};

} // namespace ExaDG

#endif /*INCLUDE_MESH_MOVEMENT_FUNCTIONS_H_*/
