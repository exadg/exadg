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

#ifndef INCLUDE_FUNCTIONALITIES_ONESIDEDSPHERICALMANIFOLD_H_
#define INCLUDE_FUNCTIONALITIES_ONESIDEDSPHERICALMANIFOLD_H_

#include <deal.II/grid/manifold_lib.h>

namespace ExaDG
{
/**
 * Class that provides a spherical manifold applied to one of the faces of a quadrilateral element.
 * On the face subject to the spherical manifold intermediate points are inserted so that an
 * equidistant distribution of points in terms of arclength is obtained. When refining the mesh, all
 * child cells are subject to this "one-sided" spherical volume manifold. This manifold description
 * is available for the two-dimensional case, and for the three-dimensional case with the
 * restriction that the geometry has to be extruded in x3/z-direction.
 */
template<int dim>
class OneSidedCylindricalManifold : public dealii::ChartManifold<dim, dim, dim>
{
public:
  OneSidedCylindricalManifold(dealii::Triangulation<dim> const &                         tria_in,
                              typename dealii::Triangulation<dim>::cell_iterator const & cell_in,
                              unsigned int const                                         face_in,
                              dealii::Point<dim> const &                                 center_in)
    : alpha(1.0), radius(1.0), tria(tria_in), cell(cell_in), face(face_in), center(center_in)
  {
    AssertThrow(tria.all_reference_cells_are_hyper_cube(),
                dealii::ExcMessage("This class is only implemented for hypercube elements."));

    AssertThrow(face <= 3,
                dealii::ExcMessage(
                  "One sided spherical manifold can only be applied to face f=0,1,2,3."));

    // get center coordinates in x1-x2 plane
    x_C[0] = center[0];
    x_C[1] = center[1];

    // determine x_1 and x_2 which denote the end points of the face that is
    // subject to the spherical manifold.
    dealii::Point<dim> x_1, x_2;
    x_1 = cell->vertex(get_vertex_id(0));
    x_2 = cell->vertex(get_vertex_id(1));

    dealii::Point<2> x_1_2d = dealii::Point<2>(x_1[0], x_1[1]);
    dealii::Point<2> x_2_2d = dealii::Point<2>(x_2[0], x_2[1]);

    initialize(x_1_2d, x_2_2d);
  }

  void
  initialize(dealii::Point<2> const & x_1, dealii::Point<2> const & x_2)
  {
    double const tol = 1.e-12;

    v_1 = x_1 - x_C;
    v_2 = x_2 - x_C,

    // calculate radius of spherical manifold
      radius = v_1.norm();

    // check correctness of geometry and parameters
    double radius_check = v_2.norm();
    AssertThrow(std::abs(radius - radius_check) < tol * radius,
                dealii::ExcMessage(
                  "Invalid geometry parameters. To apply a spherical manifold both "
                  "end points of the face must have the same distance from the center."));

    // normalize v_1 and v_2
    v_1 /= v_1.norm();
    v_2 /= v_2.norm();

    // calculate angle between v_1 and v_2
    alpha = std::acos(v_1 * v_2);

    // calculate vector that is perpendicular to v_1 in plane that is spanned by v_1 and v_2
    normal = v_2 - (v_2 * v_1) * v_1;

    AssertThrow(normal.norm() > tol, dealii::ExcMessage("Vector must not have length 0."));

    normal /= normal.norm();
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  dealii::Point<dim>
  push_forward(dealii::Point<dim> const & xi) const override
  {
    dealii::Point<dim> x;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int const v : cell->vertex_indices())
    {
      double shape_function_value = cell->reference_cell().d_linear_shape_function(xi, v);
      x += shape_function_value * cell->vertex(v);
    }

    // Add contribution of spherical manifold.
    // Here, we only operate in the xi1-xi2 plane.

    // set xi_face, xi_other to xi[0],xi[1] depending on the face that is subject to the manifold
    unsigned int index_face  = get_index_face();
    unsigned int index_other = get_index_other();
    double const xi_face     = xi[index_face];
    double const xi_other    = xi[index_other];

    // calculate deformation related to the spherical manifold
    double beta = xi_face * alpha;

    dealii::Tensor<1, 2> direction;
    direction = std::cos(beta) * v_1 + std::sin(beta) * normal;

    Assert(std::abs(direction.norm() - 1.0) < 1.e-12,
           dealii::ExcMessage("Vector must have length 1."));

    // calculate point x_S on spherical manifold
    dealii::Tensor<1, 2> x_S;
    x_S = x_C + radius * direction;

    // calculate displacement as compared to straight sided quadrilateral element
    // on the face that is subject to the manifold
    dealii::Tensor<1, 2> displ, x_lin;
    for(unsigned int v : dealii::ReferenceCells::template get_hypercube<1>().vertex_indices())
    {
      double shape_function_value =
        dealii::ReferenceCells::template get_hypercube<1>().d_linear_shape_function(
          dealii::Point<1>(xi_face), v);

      unsigned int       vertex_id = get_vertex_id(v);
      dealii::Point<dim> vertex    = cell->vertex(vertex_id);

      x_lin[0] += shape_function_value * vertex[0];
      x_lin[1] += shape_function_value * vertex[1];
    }

    displ = x_S - x_lin;

    // deformation decreases linearly in the second (other) direction
    dealii::Point<1> xi_other_1d = dealii::Point<1>(xi_other);
    unsigned int     index_1d    = get_index_1d();
    double           fading_value =
      dealii::ReferenceCells::template get_hypercube<1>().d_linear_shape_function(xi_other_1d,
                                                                                  index_1d);
    x[0] += fading_value * displ[0];
    x[1] += fading_value * displ[1];

    Assert(dealii::numbers::is_finite(x.norm_square()), dealii::ExcMessage("Invalid point found"));

    return x;
  }

  /*
   *  Calculate vertex_id of 2d object (cell in 2d, face4 in 3d)
   *  given the vertex_id of the 1d object (vertex_id_1d = 0,1).
   */
  unsigned int
  get_vertex_id(unsigned int vertex_id_1d) const
  {
    unsigned int vertex_id = 0;

    if(face == 0)
      vertex_id = 2 * vertex_id_1d;
    else if(face == 1)
      vertex_id = 1 + 2 * vertex_id_1d;
    else if(face == 2)
      vertex_id = vertex_id_1d;
    else if(face == 3)
      vertex_id = 2 + vertex_id_1d;

    return vertex_id;
  }

  /*
   *  Calculate index of 1d linear shape function (0 or 1)
   *  that takes a value of 1 on the specified face.
   */
  unsigned int
  get_index_1d() const
  {
    unsigned int index_1d = 0;

    if(face == 0 or face == 2)
      index_1d = 0;
    else if(face == 1 or face == 3)
      index_1d = 1;
    else
      Assert(false, dealii::ExcMessage("Face ID is invalid."));

    return index_1d;
  }

  /*
   *  Calculate which xi-coordinate corresponds to the
   *  tangent direction of the respective face
   */
  unsigned int
  get_index_face() const
  {
    unsigned int index_face = 0;

    if(face == 0 or face == 1)
      index_face = 1;
    else if(face == 2 or face == 3)
      index_face = 0;
    else
      Assert(false, dealii::ExcMessage("Face ID is invalid."));

    return index_face;
  }

  /*
   *  Calculate which xi-coordinate corresponds to
   *  the normal direction of the respective face
   *  in xi1-xi2-plane.
   */
  unsigned int
  get_index_other() const
  {
    return 1 - get_index_face();
  }

  /*
   *  This function calculates the inverse Jacobi matrix dx/d(xi) = d(phi(xi))/d(xi) of
   *  the push-forward operation phi: [0,1]^d -> R^d: xi -> x = phi(xi)
   *  We assume that the gradient of the standard bilinear shape functions is sufficient
   *  to find the solution.
   */
  dealii::Tensor<2, dim>
  get_inverse_jacobian(dealii::Point<dim> const & xi) const
  {
    dealii::Tensor<2, dim> jacobian;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int const v : cell->vertex_indices())
    {
      dealii::Tensor<1, dim> shape_function_gradient =
        cell->reference_cell().d_linear_shape_function_gradient(xi, v);
      jacobian += outer_product(cell->vertex(v), shape_function_gradient);
    }

    return invert(jacobian);
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates [0,1]^d using the
   *  push_forward operation and Newton's method
   */
  dealii::Point<dim>
  pull_back(dealii::Point<dim> const & x) const override
  {
    dealii::Point<dim>     xi;
    dealii::Tensor<1, dim> residual = push_forward(xi) - x;
    dealii::Tensor<1, dim> delta_xi;

    // Newton method to solve nonlinear pull_back operation
    unsigned int n_iter = 0, MAX_ITER = 100;
    double const TOL = 1.e-12;
    while(residual.norm() > TOL and n_iter < MAX_ITER)
    {
      // multiply by -1.0, i.e., shift residual to the rhs
      residual *= -1.0;

      // solve linear problem
      delta_xi = get_inverse_jacobian(xi) * residual;

      // add increment
      xi += delta_xi;

      // make sure that xi is in the valid range [0,1]^d
      if(xi[0] < 0.0)
        xi[0] = 0.0;
      else if(xi[0] > 1.0)
        xi[0] = 1.0;

      if(xi[1] < 0.0)
        xi[1] = 0.0;
      else if(xi[1] > 1.0)
        xi[1] = 1.0;

      // evaluate residual
      residual = push_forward(xi) - x;

      // increment counter
      ++n_iter;
    }

    Assert(n_iter < MAX_ITER,
           dealii::ExcMessage("Newton solver did not converge to given tolerance. "
                              "Maximum number of iterations exceeded."));

    Assert(xi[0] >= 0.0 and xi[0] <= 1.0,
           dealii::ExcMessage("Pull back operation generated invalid xi[0] values."));

    Assert(xi[1] >= 0.0 and xi[1] <= 1.0,
           dealii::ExcMessage("Pull back operation generated invalid xi[1] values."));

    return xi;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const override
  {
    return std::make_unique<OneSidedCylindricalManifold<dim>>(tria, cell, face, center);
  }

private:
  dealii::Point<2>     x_C;
  dealii::Tensor<1, 2> v_1;
  dealii::Tensor<1, 2> v_2;
  dealii::Tensor<1, 2> normal;
  double               alpha;
  double               radius;

  dealii::Triangulation<dim> const &                 tria;
  typename dealii::Triangulation<dim>::cell_iterator cell;
  unsigned int                                       face;
  dealii::Point<dim>                                 center;
};

/**
 * Class that provides a conical manifold applied to one of the faces of a hexahedral element. On
 * the face subject to the conical manifold intermediate points are inserted so that an equidistant
 * distribution of points in terms of arclength is obtained. When refining the mesh, all child cells
 * are subject to this "one-sided" conical volume manifold. This manifold description is only
 * available for the three-dimensional case where the axis of the cone has to be along the
 * x3/z-direction.
 */
template<int dim>
class OneSidedConicalManifold : public dealii::ChartManifold<dim, dim, dim>
{
public:
  OneSidedConicalManifold(dealii::Triangulation<dim> const &                         tria_in,
                          typename dealii::Triangulation<dim>::cell_iterator const & cell_in,
                          unsigned int const                                         face_in,
                          dealii::Point<dim> const &                                 center_in,
                          double const                                               r_0_in,
                          double const                                               r_1_in)
    : alpha(1.0),
      tria(tria_in),
      cell(cell_in),
      face(face_in),
      center(center_in),
      r_0(r_0_in),
      r_1(r_1_in)
  {
    AssertThrow(tria.all_reference_cells_are_hyper_cube(),
                dealii::ExcMessage("This class is only implemented for hypercube elements."));

    AssertThrow(dim == 3,
                dealii::ExcMessage("OneSidedConicalManifold can only be used for 3D problems."));

    AssertThrow(face <= 3,
                dealii::ExcMessage(
                  "One sided spherical manifold can only be applied to face f=0,1,2,3."));

    // get center coordinates in x1-x2 plane
    x_C[0] = center[0];
    x_C[1] = center[1];

    // determine x_1 and x_2 which denote the end points of the face that is
    // subject to the spherical manifold.
    dealii::Point<dim> x_1, x_2;
    x_1 = cell->vertex(get_vertex_id(0));
    x_2 = cell->vertex(get_vertex_id(1));

    dealii::Point<2> x_1_2d = dealii::Point<2>(x_1[0], x_1[1]);
    dealii::Point<2> x_2_2d = dealii::Point<2>(x_2[0], x_2[1]);

    initialize(x_1_2d, x_2_2d);
  }

  void
  initialize(dealii::Point<2> const & x_1, dealii::Point<2> const & x_2)
  {
    double const tol = 1.e-12;

    v_1 = x_1 - x_C;
    v_2 = x_2 - x_C,

    // calculate radius of spherical manifold
      r_0 = v_1.norm();

    // check correctness of geometry and parameters
    double radius_check = v_2.norm();

    AssertThrow(std::abs(r_0 - radius_check) < tol * r_0,
                dealii::ExcMessage(
                  "Invalid geometry parameters. To apply a spherical manifold both "
                  "end points of the face must have the same distance from the center."));

    // normalize v_1 and v_2
    v_1 /= v_1.norm();
    v_2 /= v_2.norm();

    // calculate angle between v_1 and v_2
    alpha = std::acos(v_1 * v_2);

    // calculate vector that is perpendicular to v_1 in plane that is spanned by v_1 and v_2
    normal = v_2 - (v_2 * v_1) * v_1;

    AssertThrow(normal.norm() > tol, dealii::ExcMessage("Vector must not have length 0."));

    normal /= normal.norm();
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  dealii::Point<dim>
  push_forward(dealii::Point<dim> const & xi) const override
  {
    dealii::Point<dim> x;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int const v : cell->vertex_indices())
    {
      double shape_function_value = cell->reference_cell().d_linear_shape_function(xi, v);
      x += shape_function_value * cell->vertex(v);
    }

    // Add contribution of conical manifold.
    // Here, we only operate in the xi1-xi2 plane.

    // set xi_face, xi_other to xi[0],xi[1] depending on the face that is subject to the manifold
    unsigned int index_face  = get_index_face();
    unsigned int index_other = get_index_other();
    double const xi_face     = xi[index_face];
    double const xi_other    = xi[index_other];

    // calculate deformation related to the conical manifold
    double beta = xi_face * alpha;

    dealii::Tensor<1, 2> direction;
    direction = std::cos(beta) * v_1 + std::sin(beta) * normal;

    Assert(std::abs(direction.norm() - 1.0) < 1.e-12,
           dealii::ExcMessage("Vector must have length 1."));

    // calculate point x_S on spherical manifold
    dealii::Tensor<1, 2> x_S;
    x_S = x_C + r_0 * direction;

    // calculate displacement as compared to straight sided quadrilateral element
    // on the face that is subject to the manifold
    dealii::Tensor<1, 2> displ, x_lin;
    for(unsigned int const v : dealii::ReferenceCells::template get_hypercube<1>().vertex_indices())
    {
      double shape_function_value =
        dealii::ReferenceCells::template get_hypercube<1>().d_linear_shape_function(
          dealii::Point<1>(xi_face), v);

      unsigned int       vertex_id = get_vertex_id(v);
      dealii::Point<dim> vertex    = cell->vertex(vertex_id);

      x_lin[0] += shape_function_value * vertex[0];
      x_lin[1] += shape_function_value * vertex[1];
    }

    displ = x_S - x_lin;

    // conical manifold
    displ *= (1 - xi[2] * (r_0 - r_1) / r_0);

    // deformation decreases linearly in the second (other) direction
    dealii::Point<1> xi_other_1d = dealii::Point<1>(xi_other);
    unsigned int     index_1d    = get_index_1d();
    double           fading_value =
      dealii::ReferenceCells::template get_hypercube<1>().d_linear_shape_function(xi_other_1d,
                                                                                  index_1d);
    x[0] += fading_value * displ[0];
    x[1] += fading_value * displ[1];

    Assert(dealii::numbers::is_finite(x.norm_square()), dealii::ExcMessage("Invalid point found"));

    return x;
  }

  /*
   *  Calculate vertex_id of 2d object (cell in 2d, face4 in 3d)
   *  given the vertex_id of the 1d object (vertex_id_1d = 0,1).
   */
  unsigned int
  get_vertex_id(unsigned int vertex_id_1d) const
  {
    unsigned int vertex_id = 0;

    if(face == 0)
      vertex_id = 2 * vertex_id_1d;
    else if(face == 1)
      vertex_id = 1 + 2 * vertex_id_1d;
    else if(face == 2)
      vertex_id = vertex_id_1d;
    else if(face == 3)
      vertex_id = 2 + vertex_id_1d;

    return vertex_id;
  }

  /*
   *  Calculate index of 1d linear shape function (0 or 1)
   *  that takes a value of 1 on the specified face.
   */
  unsigned int
  get_index_1d() const
  {
    unsigned int index_1d = 0;

    if(face == 0 or face == 2)
      index_1d = 0;
    else if(face == 1 or face == 3)
      index_1d = 1;
    else
      Assert(false, dealii::ExcMessage("Face ID is invalid."));

    return index_1d;
  }

  /*
   *  Calculate which xi-coordinate corresponds to the
   *  tangent direction of the respective face
   */
  unsigned int
  get_index_face() const
  {
    unsigned int index_face = 0;

    if(face == 0 or face == 1)
      index_face = 1;
    else if(face == 2 or face == 3)
      index_face = 0;
    else
      Assert(false, dealii::ExcMessage("Face ID is invalid."));

    return index_face;
  }

  /*
   *  Calculate which xi-coordinate corresponds to
   *  the normal direction of the respective face
   *  in xi1-xi2-plane.
   */
  unsigned int
  get_index_other() const
  {
    return 1 - get_index_face();
  }

  /*
   *  This function calculates the inverse Jacobi matrix dx/d(xi) = d(phi(xi))/d(xi) of
   *  the push-forward operation phi: [0,1]^d -> R^d: xi -> x = phi(xi)
   *  We assume that the gradient of the standard bilinear shape functions is sufficient
   *  to find the solution.
   */
  dealii::Tensor<2, dim>
  get_inverse_jacobian(dealii::Point<dim> const & xi) const
  {
    dealii::Tensor<2, dim> jacobian;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int const v : cell->vertex_indices())
    {
      dealii::Tensor<1, dim> shape_function_gradient =
        cell->reference_cell().d_linear_shape_function_gradient(xi, v);
      jacobian += outer_product(cell->vertex(v), shape_function_gradient);
    }

    return invert(jacobian);
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates [0,1]^d using the
   *  push_forward operation and Newton's method
   */
  dealii::Point<dim>
  pull_back(dealii::Point<dim> const & x) const override
  {
    dealii::Point<dim>     xi;
    dealii::Tensor<1, dim> residual = push_forward(xi) - x;
    dealii::Tensor<1, dim> delta_xi;

    // Newton method to solve nonlinear pull_back operation
    unsigned int n_iter = 0, MAX_ITER = 100;
    double const TOL = 1.e-12;
    while(residual.norm() > TOL and n_iter < MAX_ITER)
    {
      // multiply by -1.0, i.e., shift residual to the rhs
      residual *= -1.0;

      // solve linear problem
      delta_xi = get_inverse_jacobian(xi) * residual;

      // add increment
      xi += delta_xi;

      // make sure that xi is in the valid range [0,1]^d
      if(xi[0] < 0.0)
        xi[0] = 0.0;
      else if(xi[0] > 1.0)
        xi[0] = 1.0;

      if(xi[1] < 0.0)
        xi[1] = 0.0;
      else if(xi[1] > 1.0)
        xi[1] = 1.0;

      if(xi[2] < 0.0)
        xi[2] = 0.0;
      else if(xi[2] > 1.0)
        xi[2] = 1.0;

      // evaluate residual
      residual = push_forward(xi) - x;

      // increment counter
      ++n_iter;
    }

    Assert(n_iter < MAX_ITER,
           dealii::ExcMessage("Newton solver did not converge to given tolerance. "
                              "Maximum number of iterations exceeded."));

    Assert(xi[0] >= 0.0 and xi[0] <= 1.0,
           dealii::ExcMessage("Pull back operation generated invalid xi[0] values."));

    Assert(xi[1] >= 0.0 and xi[1] <= 1.0,
           dealii::ExcMessage("Pull back operation generated invalid xi[1] values."));

    Assert(xi[2] >= 0.0 and xi[2] <= 1.0,
           dealii::ExcMessage("Pull back operation generated invalid xi[2] values."));

    return xi;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const override
  {
    return std::make_unique<OneSidedConicalManifold<dim>>(tria, cell, face, center, r_0, r_1);
  }


private:
  dealii::Point<2>     x_C;
  dealii::Tensor<1, 2> v_1;
  dealii::Tensor<1, 2> v_2;
  dealii::Tensor<1, 2> normal;
  double               alpha;

  dealii::Triangulation<dim> const &                 tria;
  typename dealii::Triangulation<dim>::cell_iterator cell;
  unsigned int                                       face;

  dealii::Point<dim> center;

  // radius of cone at xi_3 = 0 (-> r_0) and at xi_3 = 1 (-> r_1)
  double r_0, r_1;
};


/**
 * Own implementation of cylindrical manifold with an equidistant distribution of nodes along the
 * cylinder surface.
 */
template<int dim, int spacedim = dim>
class MyCylindricalManifold : public dealii::ChartManifold<dim, spacedim, spacedim>
{
public:
  MyCylindricalManifold(dealii::Point<spacedim> const center_in)
    : dealii::ChartManifold<dim, spacedim, spacedim>(
        MyCylindricalManifold<dim, spacedim>::get_periodicity()),
      center(center_in)
  {
  }

  dealii::Tensor<1, spacedim>
  get_periodicity()
  {
    dealii::Tensor<1, spacedim> periodicity;

    // angle theta is 2*pi periodic
    periodicity[1] = 2 * dealii::numbers::PI;
    return periodicity;
  }

  dealii::Point<spacedim>
  push_forward(dealii::Point<spacedim> const & ref_point) const override
  {
    double const radius = ref_point[0];
    double const theta  = ref_point[1];

    Assert(ref_point[0] >= 0.0, dealii::ExcMessage("Radius must be positive."));

    dealii::Point<spacedim> space_point;
    if(radius > 1e-10)
    {
      AssertThrow(spacedim == 2 or spacedim == 3,
                  dealii::ExcMessage("Only implemented for 2D and 3D case."));

      space_point[0] = radius * cos(theta);
      space_point[1] = radius * sin(theta);

      if(spacedim == 3)
        space_point[2] = ref_point[2];
    }

    return space_point + center;
  }

  dealii::Point<spacedim>
  pull_back(dealii::Point<spacedim> const & space_point) const override
  {
    dealii::Tensor<1, spacedim> vector;
    vector[0] = space_point[0] - center[0];
    vector[1] = space_point[1] - center[1];
    // for the 3D case: vector[2] will always be 0.

    double const radius = vector.norm();

    dealii::Point<spacedim> ref_point;
    ref_point[0] = radius;
    ref_point[1] = atan2(vector[1], vector[0]);
    if(ref_point[1] < 0)
      ref_point[1] += 2.0 * dealii::numbers::PI;
    if(spacedim == 3)
      ref_point[2] = space_point[2];

    return ref_point;
  }

  std::unique_ptr<dealii::Manifold<dim>>
  clone() const override
  {
    return std::make_unique<MyCylindricalManifold<dim, spacedim>>(center);
  }


private:
  dealii::Point<dim> center;
};
} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_ONESIDEDSPHERICALMANIFOLD_H_ */
