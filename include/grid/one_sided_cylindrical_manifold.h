/*
 * OneSidedSphericalManifold.h
 *
 *  Created on: Feb 13, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_FUNCTIONALITIES_ONESIDEDSPHERICALMANIFOLD_H_
#define INCLUDE_FUNCTIONALITIES_ONESIDEDSPHERICALMANIFOLD_H_

#include <deal.II/grid/manifold_lib.h>

namespace ExaDG
{
using namespace dealii;

/*
 *  Class that provides a spherical manifold applied to one of the faces
 *  of a quadrilateral element.
 *  On the face subject to the spherical manifold intermediate points are
 *  inserted so that an equidistant distribution of points in terms of
 *  arclength is obtained.
 *  When refining the mesh, all child cells are subject to this "one-sided"
 *  spherical volume manifold.
 *  This manifold description is available for the two-dimensional case,
 *  and for the three-dimensional case with the restriction that the geometry
 *  has to be extruded in x3/z-direction.
 */
template<int dim>
class OneSidedCylindricalManifold : public ChartManifold<dim, dim, dim>
{
public:
  OneSidedCylindricalManifold(typename Triangulation<dim>::cell_iterator const & cell_in,
                              unsigned int const                                 face_in,
                              Point<dim> const &                                 center_in)
    : cell(cell_in), face(face_in), center(center_in)
  {
    AssertThrow(face >= 0 && face <= 3,
                ExcMessage("One sided spherical manifold can only be applied to face f=0,1,2,3."));

    // get center coordinates in x1-x2 plane
    x_C[0] = center[0];
    x_C[1] = center[1];

    // determine x_1 and x_2 which denote the end points of the face that is
    // subject to the spherical manifold.
    Point<dim> x_1, x_2;
    x_1 = cell->vertex(get_vertex_id(0));
    x_2 = cell->vertex(get_vertex_id(1));

    Point<2> x_1_2d = Point<2>(x_1[0], x_1[1]);
    Point<2> x_2_2d = Point<2>(x_2[0], x_2[1]);

    initialize(x_1_2d, x_2_2d);
  }

  void initialize(Point<2> const & x_1, Point<2> const & x_2)
  {
    double const tol = 1.e-12;

    v_1 = x_1 - x_C;
    v_2 = x_2 - x_C,

    // calculate radius of spherical manifold
      radius = v_1.norm();

    // check correctness of geometry and parameters
    double radius_check = v_2.norm();
    AssertThrow(std::abs(radius - radius_check) < tol * radius,
                ExcMessage("Invalid geometry parameters. To apply a spherical manifold both "
                           "end points of the face must have the same distance from the center."));

    // normalize v_1 and v_2
    v_1 /= v_1.norm();
    v_2 /= v_2.norm();

    // calculate angle between v_1 and v_2
    alpha = std::acos(v_1 * v_2);

    // calculate vector that is perpendicular to v_1 in plane that is spanned by v_1 and v_2
    normal = v_2 - (v_2 * v_1) * v_1;

    AssertThrow(normal.norm() > tol, ExcMessage("Vector must not have length 0."));

    normal /= normal.norm();
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  Point<dim>
  push_forward(const Point<dim> & xi) const
  {
    Point<dim> x;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      double shape_function_value = GeometryInfo<dim>::d_linear_shape_function(xi, v);
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

    Tensor<1, 2> direction;
    direction = std::cos(beta) * v_1 + std::sin(beta) * normal;

    Assert(std::abs(direction.norm() - 1.0) < 1.e-12, ExcMessage("Vector must have length 1."));

    // calculate point x_S on spherical manifold
    Tensor<1, 2> x_S;
    x_S = x_C + radius * direction;

    // calculate displacement as compared to straight sided quadrilateral element
    // on the face that is subject to the manifold
    Tensor<1, 2> displ, x_lin;
    for(unsigned int v = 0; v < GeometryInfo<1>::vertices_per_cell; ++v)
    {
      double shape_function_value = GeometryInfo<1>::d_linear_shape_function(Point<1>(xi_face), v);

      unsigned int vertex_id = get_vertex_id(v);
      Point<dim>   vertex    = cell->vertex(vertex_id);

      x_lin[0] += shape_function_value * vertex[0];
      x_lin[1] += shape_function_value * vertex[1];
    }

    displ = x_S - x_lin;

    // deformation decreases linearly in the second (other) direction
    Point<1>     xi_other_1d  = Point<1>(xi_other);
    unsigned int index_1d     = get_index_1d();
    double       fading_value = GeometryInfo<1>::d_linear_shape_function(xi_other_1d, index_1d);
    x[0] += fading_value * displ[0];
    x[1] += fading_value * displ[1];

    Assert(numbers::is_finite(x.norm_square()), ExcMessage("Invalid point found"));

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

    if(face == 0 || face == 2)
      index_1d = 0;
    else if(face == 1 || face == 3)
      index_1d = 1;
    else
      Assert(false, ExcMessage("Face ID is invalid."));

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

    if(face == 0 || face == 1)
      index_face = 1;
    else if(face == 2 || face == 3)
      index_face = 0;
    else
      Assert(false, ExcMessage("Face ID is invalid."));

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
  Tensor<2, dim>
  get_inverse_jacobian(Point<dim> const & xi) const
  {
    Tensor<2, dim> jacobian;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      Tensor<1, dim> shape_function_gradient =
        GeometryInfo<dim>::d_linear_shape_function_gradient(xi, v);
      jacobian += outer_product(cell->vertex(v), shape_function_gradient);
    }

    return invert(jacobian);
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates [0,1]^d using the
   *  push_forward operation and Newton's method
   */
  Point<dim>
  pull_back(const Point<dim> & x) const
  {
    Point<dim>     xi;
    Tensor<1, dim> residual = push_forward(xi) - x;
    Tensor<1, dim> delta_xi;

    // Newton method to solve nonlinear pull_back operation
    unsigned int n_iter = 0, MAX_ITER = 100;
    double const TOL = 1.e-12;
    while(residual.norm() > TOL && n_iter < MAX_ITER)
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
           ExcMessage("Newton solver did not converge to given tolerance. "
                      "Maximum number of iterations exceeded."));

    Assert(xi[0] >= 0.0 && xi[0] <= 1.0,
           ExcMessage("Pull back operation generated invalid xi[0] values."));

    Assert(xi[1] >= 0.0 && xi[1] <= 1.0,
           ExcMessage("Pull back operation generated invalid xi[1] values."));

    return xi;
  }

  std::unique_ptr<Manifold<dim>>
  clone() const override
  {
    return std::make_unique<OneSidedCylindricalManifold<dim>>(cell, face, center);
  }

private:
  Point<2>     x_C;
  Tensor<1, 2> v_1;
  Tensor<1, 2> v_2;
  Tensor<1, 2> normal;
  double       alpha;
  double       radius;

  typename Triangulation<dim>::cell_iterator cell;
  unsigned int                               face;
  Point<dim>                                 center;
};

/*
 *  Class that provides a conical manifold applied to one of the faces
 *  of a hexahedral element.
 *  On the face subject to the conical manifold intermediate points are
 *  inserted so that an equidistant distribution of points in terms of
 *  arclength is obtained.
 *  When refining the mesh, all child cells are subject to this "one-sided"
 *  conical volume manifold.
 *  This manifold description is only available for the three-dimensional case
 *  where the axis of the cone has to be along the x3/z-direction.
 */
template<int dim>
class OneSidedConicalManifold : public ChartManifold<dim, dim, dim>
{
public:
  OneSidedConicalManifold(typename Triangulation<dim>::cell_iterator const & cell_in,
                          unsigned int const                                 face_in,
                          Point<dim> const &                                 center_in,
                          double const                                       r_0_in,
                          double const                                       r_1_in)
    : cell(cell_in), face(face_in), center(center_in), r_0(r_0_in), r_1(r_1_in)
  {
    AssertThrow(dim == 3, ExcMessage("OneSidedConicalManifold can only be used for 3D problems."));

    AssertThrow(face >= 0 && face <= 3,
                ExcMessage("One sided spherical manifold can only be applied to face f=0,1,2,3."));

    // get center coordinates in x1-x2 plane
    x_C[0] = center[0];
    x_C[1] = center[1];

    // determine x_1 and x_2 which denote the end points of the face that is
    // subject to the spherical manifold.
    Point<dim> x_1, x_2;
    x_1 = cell->vertex(get_vertex_id(0));
    x_2 = cell->vertex(get_vertex_id(1));

    Point<2> x_1_2d = Point<2>(x_1[0], x_1[1]);
    Point<2> x_2_2d = Point<2>(x_2[0], x_2[1]);

    initialize(x_1_2d, x_2_2d);
  }

  void initialize(Point<2> const & x_1, Point<2> const & x_2)
  {
    double const tol = 1.e-12;

    v_1 = x_1 - x_C;
    v_2 = x_2 - x_C,

    // calculate radius of spherical manifold
      r_0 = v_1.norm();

    // check correctness of geometry and parameters
    double radius_check = v_2.norm();

    AssertThrow(std::abs(r_0 - radius_check) < tol * r_0,
                ExcMessage("Invalid geometry parameters. To apply a spherical manifold both "
                           "end points of the face must have the same distance from the center."));

    // normalize v_1 and v_2
    v_1 /= v_1.norm();
    v_2 /= v_2.norm();

    // calculate angle between v_1 and v_2
    alpha = std::acos(v_1 * v_2);

    // calculate vector that is perpendicular to v_1 in plane that is spanned by v_1 and v_2
    normal = v_2 - (v_2 * v_1) * v_1;

    AssertThrow(normal.norm() > tol, ExcMessage("Vector must not have length 0."));

    normal /= normal.norm();
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  Point<dim>
  push_forward(const Point<dim> & xi) const
  {
    Point<dim> x;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      double shape_function_value = GeometryInfo<dim>::d_linear_shape_function(xi, v);
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

    Tensor<1, 2> direction;
    direction = std::cos(beta) * v_1 + std::sin(beta) * normal;

    Assert(std::abs(direction.norm() - 1.0) < 1.e-12, ExcMessage("Vector must have length 1."));

    // calculate point x_S on spherical manifold
    Tensor<1, 2> x_S;
    x_S = x_C + r_0 * direction;

    // calculate displacement as compared to straight sided quadrilateral element
    // on the face that is subject to the manifold
    Tensor<1, 2> displ, x_lin;
    for(unsigned int v = 0; v < GeometryInfo<1>::vertices_per_cell; ++v)
    {
      double shape_function_value = GeometryInfo<1>::d_linear_shape_function(Point<1>(xi_face), v);

      unsigned int vertex_id = get_vertex_id(v);
      Point<dim>   vertex    = cell->vertex(vertex_id);

      x_lin[0] += shape_function_value * vertex[0];
      x_lin[1] += shape_function_value * vertex[1];
    }

    displ = x_S - x_lin;

    // conical manifold
    displ *= (1 - xi[2] * (r_0 - r_1) / r_0);

    // deformation decreases linearly in the second (other) direction
    Point<1>     xi_other_1d  = Point<1>(xi_other);
    unsigned int index_1d     = get_index_1d();
    double       fading_value = GeometryInfo<1>::d_linear_shape_function(xi_other_1d, index_1d);
    x[0] += fading_value * displ[0];
    x[1] += fading_value * displ[1];

    Assert(numbers::is_finite(x.norm_square()), ExcMessage("Invalid point found"));

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

    if(face == 0 || face == 2)
      index_1d = 0;
    else if(face == 1 || face == 3)
      index_1d = 1;
    else
      Assert(false, ExcMessage("Face ID is invalid."));

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

    if(face == 0 || face == 1)
      index_face = 1;
    else if(face == 2 || face == 3)
      index_face = 0;
    else
      Assert(false, ExcMessage("Face ID is invalid."));

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
  Tensor<2, dim>
  get_inverse_jacobian(Point<dim> const & xi) const
  {
    Tensor<2, dim> jacobian;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
    {
      Tensor<1, dim> shape_function_gradient =
        GeometryInfo<dim>::d_linear_shape_function_gradient(xi, v);
      jacobian += outer_product(cell->vertex(v), shape_function_gradient);
    }

    return invert(jacobian);
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates [0,1]^d using the
   *  push_forward operation and Newton's method
   */
  Point<dim>
  pull_back(const Point<dim> & x) const
  {
    Point<dim>     xi;
    Tensor<1, dim> residual = push_forward(xi) - x;
    Tensor<1, dim> delta_xi;

    // Newton method to solve nonlinear pull_back operation
    unsigned int n_iter = 0, MAX_ITER = 100;
    double const TOL = 1.e-12;
    while(residual.norm() > TOL && n_iter < MAX_ITER)
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
           ExcMessage("Newton solver did not converge to given tolerance. "
                      "Maximum number of iterations exceeded."));

    Assert(xi[0] >= 0.0 && xi[0] <= 1.0,
           ExcMessage("Pull back operation generated invalid xi[0] values."));

    Assert(xi[1] >= 0.0 && xi[1] <= 1.0,
           ExcMessage("Pull back operation generated invalid xi[1] values."));

    Assert(xi[2] >= 0.0 && xi[2] <= 1.0,
           ExcMessage("Pull back operation generated invalid xi[2] values."));

    return xi;
  }

  std::unique_ptr<Manifold<dim>>
  clone() const override
  {
    return std::make_unique<OneSidedConicalManifold<dim>>(cell, face, center, r_0, r_1);
  }


private:
  Point<2>     x_C;
  Tensor<1, 2> v_1;
  Tensor<1, 2> v_2;
  Tensor<1, 2> normal;
  double       alpha;

  typename Triangulation<dim>::cell_iterator cell;
  unsigned int                               face;

  Point<dim> center;

  // radius of cone at xi_3 = 0 (-> r_0) and at xi_3 = 1 (-> r_1)
  double r_0, r_1;
};


/*
 *  Own implementation of cylindrical manifold
 *  with an equidistant distribution of nodes along the
 *  cylinder surface.
 */
template<int dim, int spacedim = dim>
class MyCylindricalManifold : public ChartManifold<dim, spacedim, spacedim>
{
public:
  MyCylindricalManifold(Point<spacedim> const center_in)
    : ChartManifold<dim, spacedim, spacedim>(
        MyCylindricalManifold<dim, spacedim>::get_periodicity()),
      center(center_in)
  {
  }

  Tensor<1, spacedim>
  get_periodicity()
  {
    Tensor<1, spacedim> periodicity;

    // angle theta is 2*pi periodic
    periodicity[1] = 2 * numbers::PI;
    return periodicity;
  }

  Point<spacedim>
  push_forward(Point<spacedim> const & ref_point) const
  {
    double const radius = ref_point[0];
    double const theta  = ref_point[1];

    Assert(ref_point[0] >= 0.0, ExcMessage("Radius must be positive."));

    Point<spacedim> space_point;
    if(radius > 1e-10)
    {
      AssertThrow(spacedim == 2 || spacedim == 3,
                  ExcMessage("Only implemented for 2D and 3D case."));

      space_point[0] = radius * cos(theta);
      space_point[1] = radius * sin(theta);

      if(spacedim == 3)
        space_point[2] = ref_point[2];
    }

    return space_point + center;
  }

  Point<spacedim>
  pull_back(Point<spacedim> const & space_point) const
  {
    Tensor<1, spacedim> vector;
    vector[0] = space_point[0] - center[0];
    vector[1] = space_point[1] - center[1];
    // for the 3D case: vector[2] will always be 0.

    double const radius = vector.norm();

    Point<spacedim> ref_point;
    ref_point[0] = radius;
    ref_point[1] = atan2(vector[1], vector[0]);
    if(ref_point[1] < 0)
      ref_point[1] += 2.0 * numbers::PI;
    if(spacedim == 3)
      ref_point[2] = space_point[2];

    return ref_point;
  }

  std::unique_ptr<Manifold<dim>>
  clone() const override
  {
    return std::make_unique<MyCylindricalManifold<dim, spacedim>>(center);
  }


private:
  Point<dim> center;
};
} // namespace ExaDG

#endif /* INCLUDE_FUNCTIONALITIES_ONESIDEDSPHERICALMANIFOLD_H_ */
