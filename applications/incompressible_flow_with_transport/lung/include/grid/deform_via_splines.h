
#ifndef DEFORM_VIA_SPLINES_H
#define DEFORM_VIA_SPLINES_H

// C/C++
#include <fstream>

// deal.II
#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/grid_out.h>
#include <deal.II/grid/tria.h>

// ExaDG
#include "deform_via_cylinder.h"
#include "read_bspline.h"

namespace ExaDG
{
using namespace dealii;

template<int dim>
class DeformTransfinitelyViaSplines : public DeformTransfinitelyViaCylinder<dim>
{
public:
  DeformTransfinitelyViaSplines() = default;

  DeformTransfinitelyViaSplines(const DeformTransfinitelyViaSplines & other) = default;

  DeformTransfinitelyViaSplines(const std::string &                 bspline_file,
                                const std::vector<Point<dim>> &     surrounding_points,
                                const std::array<unsigned int, 2> & bifurcation_indices_in)
  {
    reinit(bspline_file, surrounding_points, bifurcation_indices_in);
  }

  void
  reinit(const std::string &                 bspline_file,
         const std::vector<Point<dim>> &     surrounding_points,
         const std::array<unsigned int, 2> & bifurcation_indices_in)
  {
    std::ifstream file(bspline_file.c_str(), std::ios::binary);
    unsigned int  n_splines;
    file.read(reinterpret_cast<char *>(&n_splines), sizeof(unsigned int));
    splines.resize(n_splines);
    for(unsigned int s = 0; s < n_splines; ++s)
      splines[s].read_from_file(file);

    AssertThrow(surrounding_points.size() == GeometryInfo<dim>::vertices_per_cell,
                ExcDimensionMismatch(surrounding_points.size(),
                                     GeometryInfo<dim>::vertices_per_cell));

    this->bifurcation_indices = bifurcation_indices_in;
    for(unsigned int i = 0; i < this->bifurcation_indices.size(); ++i)
      AssertThrow(this->bifurcation_indices[i] < 2,
                  ExcInternalError("Bifurcation index must be either 0 or 1"));

    this->surrounding_points_parts = surrounding_points;
    this->surrounding_points_parts.push_back(
      0.5(surrounding_points[this->bifurcation_indices[0]] +
          surrounding_points[3 - this->bifurcation_indices[0]]));
    this->surrounding_points_parts.push_back(
      0.5(surrounding_points[4 + this->bifurcation_indices[1]] +
          surrounding_points[7 - this->bifurcation_indices[1]]));
  }

  DeformTransfinitelyViaSplines(const std::vector<BSpline2D<dim, 3>> & splines_in,
                                const unsigned int                     first_spline_index,
                                const std::vector<Point<dim>> &        surrounding_points,
                                const std::array<unsigned int, 2> &    bifurcation_indices_in,
                                const bool                             do_blend = false)
  {
    reinit(splines_in, first_spline_index, surrounding_points, bifurcation_indices_in, do_blend);
  }

  void
  reinit(const std::vector<BSpline2D<dim, 3>> & splines_in,
         const unsigned int                     first_spline_index,
         const std::vector<Point<dim>> &        surrounding_points,
         const std::array<unsigned int, 2> &    bifurcation_indices_in,
         const bool                             do_blend = false)
  {
    this->do_blend = do_blend;
    splines.clear();
    splines.insert(splines.end(),
                   splines_in.begin() + first_spline_index,
                   splines_in.begin() + 4 + first_spline_index);
    AssertThrow(surrounding_points.size() == GeometryInfo<dim>::vertices_per_cell,
                ExcDimensionMismatch(surrounding_points.size(),
                                     GeometryInfo<dim>::vertices_per_cell));

    this->bifurcation_indices = bifurcation_indices_in;
    for(unsigned int i = 0; i < this->bifurcation_indices.size(); ++i)
      AssertThrow(this->bifurcation_indices[i] < 2,
                  ExcInternalError("Bifurcation index must be either 0 or 1"));

    this->surrounding_points_parts = surrounding_points;
    this->surrounding_points_parts.push_back(
      0.5 * (surrounding_points[this->bifurcation_indices[0]] +
             surrounding_points[3 - this->bifurcation_indices[0]]));
    this->surrounding_points_parts.push_back(
      0.5 * (surrounding_points[4 + this->bifurcation_indices[1]] +
             surrounding_points[7 - this->bifurcation_indices[1]]));
  }

  Point<dim>
  transform_to_deformed(const Point<dim> & untransformed) const override
  {
    AssertThrow(splines.size() >= 4, ExcNotImplemented("Need at least 4 splines"));

    unsigned int quadrant      = 0;
    unsigned int best_quadrant = -1;
    Point<dim>   reference(-1, -1, -1);
    double       distance = 3;
    for(; quadrant < 4; ++quadrant)
    {
      Point<dim>   tentative = this->transform_to_reference_prism(quadrant, untransformed);
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
        std::cout << this->transform_to_reference_prism(r, untransformed) << "   ";
      std::cout << std::endl;
    }

    if(distance > 0.1)
    {
      std::ostringstream str;
      str << "Error in locating point " << untransformed << " on reference prism; guesses were ";
      for(unsigned int r = 0; r < 4; ++r)
        str << this->transform_to_reference_prism(r, untransformed) << "   ";
      AssertThrow(false, (typename Mapping<dim, dim>::ExcTransformationFailed(str.str())));
    }

    for(unsigned int d = 0; d < dim; ++d)
    {
      reference[d] = std::max(0., reference[d]);
      reference[d] = std::min(1., reference[d]);
    }
    reference[1] = std::min(1 - reference[0], reference[1]);

    const Point<dim> bounds[4] = {splines[0].value(0., reference[2]),
                                  splines[0].value(1., reference[2]),
                                  splines[2].value(0., reference[2]),
                                  splines[2].value(1., reference[2])};
    const Point<dim> mid_point =
      (1. - reference[2]) * 0.5 *
        (bounds[this->bifurcation_indices[0]] + bounds[2 + this->bifurcation_indices[0]]) +
      reference[2] * 0.5 *
        (bounds[this->bifurcation_indices[1]] + bounds[2 + this->bifurcation_indices[1]]);

    Point<dim>           lambda(1 - reference[0] - reference[1], reference[0], reference[1]);
    const Tensor<1, dim> vbar12 = mid_point + (bounds[quadrant] - mid_point) * lambda[1];
    const Tensor<1, dim> vbar13 = mid_point + (bounds[(quadrant + 1) % 4] - mid_point) * lambda[2];
    const Tensor<1, dim> vbar23 = splines[quadrant].value(lambda[2], reference[2]);
    const Tensor<1, dim> vbar21 = bounds[quadrant] + (mid_point - bounds[quadrant]) * lambda[0];
    const Tensor<1, dim> vbar31 =
      bounds[(quadrant + 1) % 4] + (mid_point - bounds[(quadrant + 1) % 4]) * lambda[0];
    const Tensor<1, dim> vbar32 = splines[quadrant].value(1 - lambda[1], reference[2]);
    const Point<dim>     transformed(lambda[0] * (vbar12 + vbar13 - mid_point) +
                                 lambda[1] * (vbar23 + vbar21 - bounds[quadrant]) +
                                 lambda[2] * (vbar31 + vbar32 - bounds[(quadrant + 1) % 4]));

    if(!this->do_blend)
      return transformed;
    else
      return untransformed * reference[2] * reference[2] +
             transformed * (1.0 - reference[2] * reference[2]);

    // compute angle and radius of the new point to correct for the fact that
    // we are going to map back to a circle with a single element
    // const double angle = std::atan2(reference[1]-0.5, reference[0]-0.5);
    // const double radius = std::max(std::abs(reference[1]-0.5),
    //                               std::abs(reference[0]-0.5));
    // const double sin2 = std::sin(2*angle);
    // const double x = std::max(-1.0, std::min(1.0, (2.*reference[0]-1.)));
    // reference[0] = 0.5 + (0.5 * (1. - std::acos(x) * (2. / numbers::PI)) * sin2 * sin2 +
    //  0.5 * x * (1-sin2*sin2));
    // const double y = std::max(-1.0, std::min(1.0, (2.*reference[1]-1.)));
    // reference[1] = 0.5 + (0.5 * (1. - std::acos(y) * (2. / numbers::PI)) * sin2 * sin2 +
    //  0.5 * y * (1-sin2*sin2));

    // reference = GeometryInfo<dim>::project_to_unit_cell(reference);

    // return
    //  Point<dim>((1. - reference[1]) * splines[2].value(1.0-reference[0], reference[2]) +
    //             reference[0] * splines[1].value(1.0-reference[1], reference[2]) +
    //             reference[1] * splines[0].value(reference[0], reference[2]) +
    //             (1. - reference[0]) * splines[3].value(reference[1], reference[2]) -
    //             (1. - reference[0]) * (1.-reference[1]) * splines[2].value(1.0, reference[2]) -
    //             (1. - reference[0]) * reference[1] * splines[0].value(0., reference[2]) -
    //             reference[0] * (1. - reference[1]) * splines[2].value(0., reference[2]) -
    //             reference[0] * reference[1] * splines[0].value(1.0, reference[2]));
  }

private:
  std::vector<BSpline2D<dim, 3>> splines;
};

} // namespace ExaDG

#endif
