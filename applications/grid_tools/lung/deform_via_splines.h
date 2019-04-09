
#ifndef DEFORM_VIA_SPLINES_H
#define DEFORM_VIA_SPLINES_H

#include <deal.II/base/geometry_info.h>
#include <deal.II/base/point.h>
#include <deal.II/grid/tria.h>

#include <fstream>

#include "read_bspline.h"

using namespace dealii;

template <int dim>
class DeformTransfinitelyViaSplines
{
public:
  DeformTransfinitelyViaSplines() = default;

  DeformTransfinitelyViaSplines(const DeformTransfinitelyViaSplines &other)
    :
    splines(other.splines),
    bifurcation_indices(other.bifurcation_indices)
  {
    triangulation.copy_triangulation(other.triangulation);
  }

  DeformTransfinitelyViaSplines(const std::string &bspline_file,
                                const std::vector<Point<dim>> &surrounding_points,
                                const std::array<unsigned int,4> &bifurcation_indices_in)
  {
    reinit(bspline_file, surrounding_points, bifurcation_indices_in);
  }

  void reinit(const std::string &bspline_file,
              const std::vector<Point<dim>> &surrounding_points,
              const std::array<unsigned int,4> &bifurcation_indices_in)
  {
    std::ifstream file(bspline_file.c_str(), std::ios::binary);
    unsigned int n_splines;
    file.read(reinterpret_cast<char*>(&n_splines), sizeof(unsigned int));
    splines.resize(n_splines);
    for (unsigned int s=0; s<n_splines; ++s)
      splines[s].read_from_file(file);

    AssertThrow(surrounding_points.size() == GeometryInfo<dim>::vertices_per_cell,
                ExcDimensionMismatch(surrounding_points.size(),
                                     GeometryInfo<dim>::vertices_per_cell));

    std::vector<CellData<dim>> cell_data(1);
    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
      cell_data[0].vertices[v] = v;

    SubCellData subcell_data;
    triangulation.create_triangulation(surrounding_points, cell_data, subcell_data);

    bifurcation_indices = bifurcation_indices_in;
    for (unsigned int i=0; i<bifurcation_indices.size(); ++i)
      AssertThrow(bifurcation_indices[i] < 4,
                  ExcInternalError("Bifurcation index must be between 0 and 3"));
    AssertThrow(bifurcation_indices[0] != bifurcation_indices[1] &&
                bifurcation_indices[2] != bifurcation_indices[3],
                ExcMessage("Must not use the same index on each side of bifurcation"));
  }

  DeformTransfinitelyViaSplines(const std::vector<BSpline2D<dim,3>> &splines_in,
                                const unsigned int first_spline_index,
                                const std::vector<Point<dim>> &surrounding_points,
                                const std::array<unsigned int,4> &bifurcation_indices_in)
  {
    reinit(splines_in, first_spline_index, surrounding_points, bifurcation_indices_in);
  }

  void reinit(const std::vector<BSpline2D<dim,3>> &splines_in,
              const unsigned int first_spline_index,
              const std::vector<Point<dim>> &surrounding_points,
              const std::array<unsigned int,4> &bifurcation_indices_in)
  {
    splines.clear();
    splines.insert(splines.end(),
                   splines_in.begin()+first_spline_index,
                   splines_in.begin()+4+first_spline_index);
    AssertThrow(surrounding_points.size() == GeometryInfo<dim>::vertices_per_cell,
                ExcDimensionMismatch(surrounding_points.size(),
                                     GeometryInfo<dim>::vertices_per_cell));

    std::vector<CellData<dim>> cell_data(1);
    for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
      cell_data[0].vertices[v] = v;

    SubCellData subcell_data;
    triangulation.create_triangulation(surrounding_points, cell_data, subcell_data);

    bifurcation_indices = bifurcation_indices_in;
    for (unsigned int i=0; i<bifurcation_indices.size(); ++i)
      AssertThrow(bifurcation_indices[i] < 4,
                  ExcInternalError("Bifurcation index must be between 0 and 3"));
    AssertThrow(bifurcation_indices[0] != bifurcation_indices[1] &&
                bifurcation_indices[2] != bifurcation_indices[3],
                ExcMessage("Must not use the same index on each side of bifurcation"));
  }

  Point<dim> transform_to_deformed(const Point<dim> &untransformed) const
  {
    AssertThrow(splines.size() >= 4,
                ExcNotImplemented("Need at least 4 splines"));

    Point<dim> reference = GeometryInfo<dim>::project_to_unit_cell
      (auxiliary_mapping.transform_real_to_unit_cell(triangulation.begin(),
                                                     untransformed));
    const Point<dim> bounds[4] = { splines[0].value(0., reference[2]),
                                   splines[0].value(1., reference[2]),
                                   splines[2].value(1., reference[2]),
                                   splines[2].value(0., reference[2]) };
    const Point<dim> mid_point =
      (1.-reference[2]) * 0.5 * (bounds[bifurcation_indices[0]] + bounds[bifurcation_indices[1]])
      +
      reference[2] * 0.5 * (bounds[bifurcation_indices[2]] + bounds[bifurcation_indices[3]]);

    if (std::abs(reference[0]-0.5) < 1e-12 && std::abs(reference[1]-0.5) < 1e-12)
      return mid_point;

    // rotate by 3 pi / 4 (135 degrees) in counterclockwise direction
    const double x = -std::sqrt(0.5) * (reference[0]-0.5) - std::sqrt(0.5) * (reference[1]-0.5);
    const double y = std::sqrt(0.5) * (reference[0]-0.5) - std::sqrt(0.5) * (reference[1]-0.5);

    // transform point into barycentric coordinates of the triangle
    const unsigned int quadrant = y >= 0. ? (x >= 0. ? 0 : 1 ) : (x < 0. ? 2 : 3);
    Tensor<2,3> mat_lambda;
    for (unsigned d=0; d<3; ++d)
      mat_lambda[2][d] = 1;
    if (quadrant == 0)
      {
        mat_lambda[0][1] = std::sqrt(0.5);
        mat_lambda[1][2] = std::sqrt(0.5);
      }
    else if (quadrant == 1)
      {
        mat_lambda[0][2] = -std::sqrt(0.5);
        mat_lambda[1][1] = std::sqrt(0.5);
      }
    else if (quadrant == 2)
      {
        mat_lambda[0][1] = -std::sqrt(0.5);
        mat_lambda[1][2] = -std::sqrt(0.5);
      }
    else if (quadrant == 3)
      {
        mat_lambda[0][2] = std::sqrt(0.5);
        mat_lambda[1][1] = -std::sqrt(0.5);
      }

    Tensor<1,dim> lambda = (invert(mat_lambda)) * Point<dim>({x, y, 1});
    const Tensor<1,dim> vbar12 = mid_point + (bounds[quadrant] - mid_point) * lambda[1];
    const Tensor<1,dim> vbar13 = mid_point + (bounds[(quadrant+1)%4] - mid_point) * lambda[2];
    const Tensor<1,dim> vbar23 = splines[quadrant].value(lambda[2], reference[2]);
    const Tensor<1,dim> vbar21 = bounds[quadrant] + (mid_point - bounds[quadrant]) * lambda[0];
    const Tensor<1,dim> vbar31 = bounds[(quadrant+1)%4] + (mid_point - bounds[(quadrant+1)%4]) * lambda[0];
    const Tensor<1,dim> vbar32 = splines[quadrant].value(1-lambda[1], reference[2]);
    return Point<dim>(lambda[0] * (vbar12 + vbar13 - mid_point) +
                      lambda[1] * (vbar23 + vbar21 - bounds[quadrant]) +
                      lambda[2] * (vbar31 + vbar32 - bounds[(quadrant+1)%4]));

    // compute angle and radius of the new point to correct for the fact that
    // we are going to map back to a circle with a single element
    //const double angle = std::atan2(reference[1]-0.5, reference[0]-0.5);
    //const double radius = std::max(std::abs(reference[1]-0.5),
    //                               std::abs(reference[0]-0.5));
    //const double sin2 = std::sin(2*angle);
    //const double x = std::max(-1.0, std::min(1.0, (2.*reference[0]-1.)));
    //reference[0] = 0.5 + (0.5 * (1. - std::acos(x) * (2. / numbers::PI)) * sin2 * sin2 +
    //  0.5 * x * (1-sin2*sin2));
    //const double y = std::max(-1.0, std::min(1.0, (2.*reference[1]-1.)));
    //reference[1] = 0.5 + (0.5 * (1. - std::acos(y) * (2. / numbers::PI)) * sin2 * sin2 +
    //  0.5 * y * (1-sin2*sin2));

    //reference = GeometryInfo<dim>::project_to_unit_cell(reference);

    //return
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
  std::vector<BSpline2D<dim,3>> splines;
  Triangulation<dim> triangulation;
  MappingQ1<dim> auxiliary_mapping;
  std::array<unsigned int,4> bifurcation_indices;
};

#endif
