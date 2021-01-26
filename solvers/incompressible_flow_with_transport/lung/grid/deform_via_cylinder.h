
#ifndef DEFORM_VIA_CYLINDER_H
#define DEFORM_VIA_CYLINDER_H

// C/C++
#include <array>
#include <fstream>
#include <vector>

// deal.II
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>


namespace ExaDG
{
using namespace dealii;

template<int dim>
class DeformTransfinitelyViaCylinder
{
public:
  DeformTransfinitelyViaCylinder() = default;

  DeformTransfinitelyViaCylinder(const DeformTransfinitelyViaCylinder & other) = default;

  DeformTransfinitelyViaCylinder(const std::vector<Point<dim>> &     surrounding_points,
                                 const std::array<unsigned int, 2> & bifurcation_indices_in)
  {
    reinit(surrounding_points, bifurcation_indices_in);
  }

  void
  reinit(const std::vector<Point<dim>> &     surrounding_points,
         const std::array<unsigned int, 2> & bifurcation_indices_in)
  {
    AssertThrow(surrounding_points.size() == GeometryInfo<dim>::vertices_per_cell,
                ExcDimensionMismatch(surrounding_points.size(),
                                     GeometryInfo<dim>::vertices_per_cell));

    bifurcation_indices = bifurcation_indices_in;
    for(unsigned int i = 0; i < bifurcation_indices.size(); ++i)
      AssertThrow(bifurcation_indices[i] < 2,
                  ExcInternalError("Bifurcation index must be either 0 or 1"));

    surrounding_points_parts = surrounding_points;
    // add mid points in front and back of cylinder to the points to determine the
    surrounding_points_parts.push_back(0.5 * (surrounding_points[bifurcation_indices[0]] +
                                              surrounding_points[3 - bifurcation_indices[0]]));
    surrounding_points_parts.push_back(0.5 * (surrounding_points[4 + bifurcation_indices[1]] +
                                              surrounding_points[7 - bifurcation_indices[1]]));

#ifdef DEBUG
    std::cout << "Surrounding points: ";
    for(const auto & p : surrounding_points_parts)
      std::cout << "[" << p << "]   ";
    std::cout << std::endl;
#endif
  }

  virtual Point<dim>
  transform_to_deformed(const Point<dim> & untransformed) const
  {
    unsigned int quadrant      = 0;
    unsigned int best_quadrant = -1;
    Point<dim>   reference(-1, -1, -1);
    double       distance = 3;
    for(; quadrant < 4; ++quadrant)
    {
      Point<dim>   tentative = transform_to_reference_prism(quadrant, untransformed);
      const double my_distance =
        (tentative[0] < -1e-12 ? -tentative[0] :
                                 (tentative[0] > 1 + 1e-12 ? tentative[0] - 1 : 0)) +
        (tentative[1] < -1e-12 ?
           -tentative[1] :
           (tentative[1] > 1 - tentative[0] + 1e-12 ? tentative[1] + tentative[0] - 1 : 0)) +
        (tentative[2] < -1e-12 ? -tentative[2] : (tentative[2] > 1 + 1e-12 ? tentative[2] - 1 : 0));
      if(my_distance < distance)
      {
        reference     = tentative;
        distance      = my_distance;
        best_quadrant = quadrant;
      }
      if(tentative[0] > -1e-12 && tentative[0] < 1 + 1e-12 && tentative[1] > -1e-12 &&
         tentative[1] < 1 + 1e-12 - tentative[0] && tentative[2] > -1e-12 &&
         tentative[2] < 1 + 1e-12)
        break;
    }
    if(quadrant == 4)
    {
      quadrant = best_quadrant;
      std::cout << "Warning: Error in locating point " << untransformed
                << " on reference prism; guesses were ";
      for(unsigned int r = 0; r < 4; ++r)
        std::cout << transform_to_reference_prism(r, untransformed) << "   ";
      std::cout << std::endl;
    }

    if(distance > 0.1)
    {
      std::ostringstream str;
      str << "Error in locating point " << untransformed << " on reference prism; guesses were ";
      for(unsigned int r = 0; r < 4; ++r)
        str << transform_to_reference_prism(r, untransformed) << "   ";
      AssertThrow(false, (typename Mapping<dim, dim>::ExcTransformationFailed(str.str())));
    }

    for(unsigned int d = 0; d < dim; ++d)
    {
      reference[d] = std::max(0., reference[d]);
      reference[d] = std::min(1., reference[d]);
    }
    reference[1] = std::min(1 - reference[0], reference[1]);

    const Point<dim> mid_point = surrounding_points_parts[8] * (1. - reference[2]) +
                                 surrounding_points_parts[9] * reference[2];

    // This follows the transfinite interpolation on the triangle plus the deformation in the z
    // direction. However, there should be a simpler way to achieve this, as attempted by the
    // commented-out code.

    const Point<dim> bounds[4] = {surrounding_points_parts[0] * (1. - reference[2]) +
                                    surrounding_points_parts[4] * reference[2],
                                  surrounding_points_parts[1] * (1. - reference[2]) +
                                    surrounding_points_parts[5] * reference[2],
                                  surrounding_points_parts[3] * (1. - reference[2]) +
                                    surrounding_points_parts[7] * reference[2],
                                  surrounding_points_parts[2] * (1. - reference[2]) +
                                    surrounding_points_parts[6] * reference[2]};

    Point<dim>           lambda(1 - reference[0] - reference[1], reference[0], reference[1]);
    const Tensor<1, dim> vbar12 = mid_point + (bounds[quadrant] - mid_point) * lambda[1];
    const Tensor<1, dim> vbar13 = mid_point + (bounds[(quadrant + 1) % 4] - mid_point) * lambda[2];
    const Tensor<1, dim> vbar23 =
      mid_point + std::cos(0.5 * numbers::PI * lambda[2]) * (bounds[quadrant] - mid_point) +
      std::sin(0.5 * numbers::PI * lambda[2]) * (bounds[(quadrant + 1) % 4] - mid_point);
    const Tensor<1, dim> vbar21 = bounds[quadrant] + (mid_point - bounds[quadrant]) * lambda[0];
    const Tensor<1, dim> vbar31 =
      bounds[(quadrant + 1) % 4] + (mid_point - bounds[(quadrant + 1) % 4]) * lambda[0];
    const Tensor<1, dim> vbar32 =
      mid_point + std::sin(0.5 * numbers::PI * lambda[1]) * (bounds[quadrant] - mid_point) +
      std::cos(0.5 * numbers::PI * lambda[1]) * (bounds[(quadrant + 1) % 4] - mid_point);
    const auto transformed = Point<dim>(lambda[0] * (vbar12 + vbar13 - mid_point) +
                                        lambda[1] * (vbar23 + vbar21 - bounds[quadrant]) +
                                        lambda[2] * (vbar31 + vbar32 - bounds[(quadrant + 1) % 4]));

    if(!this->do_blend)
      return transformed;
    else
      return Point<dim>(untransformed * reference[2] * reference[2] +
                        transformed * (1.0 - reference[2] * reference[2]));
  }

protected:
  std::vector<Point<dim>>     surrounding_points_parts;
  std::array<unsigned int, 2> bifurcation_indices;
  bool                        do_blend = false;

  Point<dim>
  transform_to_reference_prism(const unsigned int section, const Point<dim> & original) const
  {
    Point<dim> result(1. / 3., 1. / 3., 0.5);

    double                                error_norm_sqr = 1;
    std::pair<Point<dim>, Tensor<2, dim>> data           = transform_to_prism(section, result);
    while(error_norm_sqr > 1e-24)
    {
      const Tensor<1, dim> residual = original - data.first;
      error_norm_sqr                = residual.norm_square();
      const double det              = determinant(data.second);
      if(det == 0.)
        return Point<dim>(-1, -1, -1);
      else
      {
        Tensor<1, dim> update = invert(data.second) * residual;
        double         alpha  = 1;
        while(alpha > 1e-7)
        {
          Point<dim> tentative = result + alpha * update;
          data                 = transform_to_prism(section, tentative);
          if((original - data.first).norm_square() <= error_norm_sqr &&
             determinant(data.second) != 0.)
          {
            result = tentative;
            break;
          }
          alpha *= 0.5;
        }
        if(alpha <= 1e-7 && error_norm_sqr > 1e-24)
          return Point<dim>(-1, -1, -1);
      }
    }
    return result;
  }

  std::pair<Point<dim>, Tensor<2, dim>>
  transform_to_prism(const unsigned int section, const Point<dim> & reference) const
  {
    constexpr unsigned int indices_tria[4][2] = {{0, 1}, {1, 3}, {3, 2}, {2, 0}};
    const Point<dim>       point(
      (1 - reference[2]) * ((1 - reference[0] - reference[1]) * surrounding_points_parts[8] +
                            reference[0] * surrounding_points_parts[indices_tria[section][0]] +
                            reference[1] * surrounding_points_parts[indices_tria[section][1]]) +
      reference[2] * ((1 - reference[0] - reference[1]) * surrounding_points_parts[9] +
                      reference[0] * surrounding_points_parts[4 + indices_tria[section][0]] +
                      reference[1] * surrounding_points_parts[4 + indices_tria[section][1]]));
    Tensor<2, dim>       derivative;
    const Tensor<1, dim> der_0 =
      (1 - reference[2]) *
        (surrounding_points_parts[indices_tria[section][0]] - surrounding_points_parts[8]) +
      reference[2] *
        (surrounding_points_parts[4 + indices_tria[section][0]] - surrounding_points_parts[9]);
    const Tensor<1, dim> der_1 =
      (1 - reference[2]) *
        (surrounding_points_parts[indices_tria[section][1]] - surrounding_points_parts[8]) +
      reference[2] *
        (surrounding_points_parts[4 + indices_tria[section][1]] - surrounding_points_parts[9]);
    const Tensor<1, dim> der_2 =
      -((1 - reference[0] - reference[1]) * surrounding_points_parts[8] +
        reference[0] * surrounding_points_parts[indices_tria[section][0]] +
        reference[1] * surrounding_points_parts[indices_tria[section][1]]) +
      ((1 - reference[0] - reference[1]) * surrounding_points_parts[9] +
       reference[0] * surrounding_points_parts[4 + indices_tria[section][0]] +
       reference[1] * surrounding_points_parts[4 + indices_tria[section][1]]);
    for(unsigned int d = 0; d < dim; ++d)
    {
      derivative[d][0] = der_0[d];
      derivative[d][1] = der_1[d];
      derivative[d][2] = der_2[d];
    }
    return std::make_pair(point, derivative);
  }
};

} // namespace ExaDG

#endif
