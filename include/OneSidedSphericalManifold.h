/*
 * OneSidedSphericalManifold.h
 *
 *  Created on: Feb 13, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_ONESIDEDSPHERICALMANIFOLD_H_
#define INCLUDE_ONESIDEDSPHERICALMANIFOLD_H_

#include <deal.II/grid/manifold_lib.h>

/*
 *  Implementation of sign function.
 */
template<typename T>
inline T sgn(T const &number)
{
  return (number > T(0)) ? T(1) : T(-1);
}

/*
 *  Class that provides a spherical manifold applied to
 *  one of the faces of a quadrilateral element.
 *  When refining the mesh, all child cells are subject to
 *  this "one-sided" spherical volume manifold.
 *  This manifold description is only available for the
 *  two-dimensional case.
 */
template <int dim, int spacedim = dim, int chartdim=dim>
class OneSidedSphericalManifold : public ChartManifold<dim,spacedim,chartdim>
{
public:
  OneSidedSphericalManifold(typename Triangulation<dim>::cell_iterator &cell_in,
                            unsigned int face_in,
                            Point<dim> const &center)
    : cell(cell_in), face(face_in)
  {
    Assert(dim==2,ExcMessage("OneSidedSphericalManifold only implemented for dim==2"));

    // determine x1 and x2 which denote the end points of the face that is
    // subject to the spherical manifold and calculate the vector pointing from x1 to x2.
    Point<dim> x1, x2;
    x1 = cell->face(face)->vertex(0);
    x2 = cell->face(face)->vertex(1);
    face_vector = x2 - x1;

    // calculate radius of spherical manifold
    radius = x2.distance(center);

    // check correctness of geometry and parameters
    double radius_check = x1.distance(center);
    Assert(std::abs(radius-radius_check) < 1.e-12*radius,
        ExcMessage("Invalid geometry parameters. To apply a spherical manifold both "
            "end points of the face must have the same distance from the center."));

    // calculate normal distance center<->face
    distance_center_face = radius*std::sqrt(1.0-std::pow(face_vector.norm()/(2*radius),2.0));

    // calculate normal vector of the face that is subject to the spherical manifold
    if(dim==2)
    {
      // find a vector that is orthogonal to the face_vector
      normal[0] = face_vector[1];
      normal[1] = -face_vector[0];

      // normalize
      normal /= normal.norm();

      // adjust orientation
      double sign = sgn((x2 - center)*normal);
      normal *= sign;
    }
    else
    {
      Assert(dim==2,ExcMessage("OneSidedSphericalManifold only implemented for dim==2"));
    }
  }


  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  Point<dim> push_forward(const Point<dim> &xi) const
  {
    Point<dim> x;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell;++v)
    {
      double shape_function_value = GeometryInfo<dim>::d_linear_shape_function(xi,v);
      x += shape_function_value*cell->vertex(v);
    }

    // set pointers xi_face, xi_other to xi[0],xi[1] depending on the face that is subject to the manifold
    unsigned int index_face = get_index_face();
    unsigned int index_other = get_index_other();
    double const xi_face = xi[index_face];
    double const xi_other = xi[index_other];

    // add contribution of spherical manifold

    // transform coordinate from reference interval [0,1] to interval [-1,1]: xi_face -> eta
    double eta = 2.0*(xi_face) - 1;

    // calculate deformation related to the spherical manifold
    double h = calculate_h(eta);

    // deformation decreases linearly in the second (other) direction
    Point<1> xi_other_1d = Point<1>(xi_other);
    unsigned int index_1d = get_index_1d();
    double fading_value = GeometryInfo<1>::d_linear_shape_function(xi_other_1d, index_1d);
    x += h*fading_value*normal;

    Assert(numbers::is_finite(x.norm_square()), ExcMessage("Invalid point found"));

    return x;
  }

  /*
   *  Calculate index of 1d linear shape function (0 or 1)
   *  that takes a value of 1 on the specified face.
   */
  unsigned int get_index_1d() const
  {
    unsigned int index_1d = 0;

    if(face==0 || face==2)
      index_1d = 0;
    else if(face==1 || face==3)
      index_1d = 1;
    else
      Assert(false,ExcMessage("Face ID is invalid."));

    return index_1d;
  }

  /*
   *  Calculate which xi-coordinate corresponds to the
   *  tangent direction of the respective face
   */
  unsigned int get_index_face() const
  {
    unsigned int index_face = 0;

    if(face==0 || face==1)
      index_face = 1;
    else if(face==2 || face==3)
      index_face = 0;
    else
      Assert(false,ExcMessage("Face ID is invalid."));

    return index_face;
  }

  /*
   *  Calculate which xi-coordinate corresponds to
   *  the normal direction of the respective face.
   */
  unsigned int get_index_other() const
  {
    return 1-get_index_face();
  }

  /*
   *  Calculate h describing the displacement on the face
   *  that is subject to the spherical manifold.
   */
  double calculate_h(double const eta) const
  {
    double value = 1.0-std::pow(face_vector.norm()*eta/(2*radius),2.0);
    Assert(value > 0.0, ExcMessage("Argument of std::sqrt() must not be negative."));
    double h = radius*std::sqrt(value) - distance_center_face;

    return h;
  }

  /*
   *  Calculate derivative of h with respect to xi_face.
   */
  double calculate_dh_dxi(double const eta) const
  {
    double temp = face_vector.norm()*eta/(2*radius);
    double value = 1.0-std::pow(temp,2.0);
    Assert(value > 0.0, ExcMessage("Argument of std::sqrt() must not be negative."));
    double dh_dxi = -face_vector.norm()*temp/std::sqrt(value);

    return dh_dxi;
  }

  /*
   *  This function calculates the inverse Jacobi matrix dx/d(xi) = d(phi(xi))/d(xi) of
   *  the push-forward operation phi: [0,1]^d -> R^d: xi -> x = phi(xi)
   */
  Tensor<2,dim> get_inverse_jacobian(Point<dim> const &xi) const
  {
    Tensor<2,dim> jacobian;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell;++v)
    {
      Tensor<1,dim> shape_function_gradient = GeometryInfo<dim>::d_linear_shape_function_gradient(xi,v);
      jacobian += outer_product(cell->vertex(v),shape_function_gradient);
    }

    // add contribution of spherical manifold

    // set pointers xi_face, xi_other to xi[0],xi[1] depending on which face is subject to the manifold
    unsigned int index_face = get_index_face();
    unsigned int index_other = get_index_other();
    double const xi_face = xi[index_face];
    double const xi_other = xi[index_other];

    // transform from reference interval [0,1] to interval [-1,1]: xi_face -> eta
    double eta = 2.0*(xi_face) - 1;

    // calculate h and derivative of h with respect to xi_face
    double h =  calculate_h(eta);
    double dh_dxi = calculate_dh_dxi(eta);

    // calculate index specifying which of the two one-dimensional,
    // linear shape functions has to be evaluated.
    unsigned int index_1d = get_index_1d();

    // deformation decreases linearly in the second direction
    Point<1> xi_other_1d = Point<1>(xi_other);
    // 1d shape function with index index_1d evaluated in xi_other
    double fading_value = GeometryInfo<1>::d_linear_shape_function(xi_other_1d, index_1d);
    // gradient of 1d shape function with index index_1d evaluated in xi_other
    Tensor<1,1> fading_gradient = GeometryInfo<1>::d_linear_shape_function_gradient(xi_other_1d, index_1d);

    // derivative of h with respect to xi_face
    for(unsigned int i=0;i<dim;++i)
      jacobian[i][index_face] += dh_dxi*fading_value*normal[i];

    // derivative of 1d shape function with respect to xi_other
    for(unsigned int i=0;i<dim;++i)
      jacobian[i][index_other] += h*fading_gradient[0]*normal[i];

    return invert(jacobian);
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates [0,1]^d using the
   *  push_forward operation and Newton's method
   */
  Point<dim> pull_back(const Point<dim> &x) const
  {
    Point<dim> xi;
    Tensor<1,dim> residual = push_forward(xi) - x;
    Tensor<1,dim> delta_xi;

    // Newton method to solve nonlinear pull_back operation
    unsigned int n_iter = 0, MAX_ITER = 100;
    while(residual.norm() > 1.e-12 && n_iter < MAX_ITER)
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

    Assert(n_iter < MAX_ITER, ExcMessage("Newton solver did not converge to given tolerance. Maximum number of iterations exceeded."));
    Assert(xi[0] >= 0.0 && xi[0] <= 1.0, ExcMessage("Pull back operation generated invalid xi[0] values."));
    Assert(xi[1] >= 0.0 && xi[1] <= 1.0, ExcMessage("Pull back operation generated invalid xi[1] values."));

    return xi;
  }

private:
  double radius;
  double distance_center_face;
  Tensor<1,dim,double> normal;
  Tensor<1,dim,double> face_vector;

  typename Triangulation<dim>::cell_iterator cell;
  unsigned int face;
};


///*
// *  Class that provides a spherical manifold applied to
// *  one of the faces of a quadrilateral element.
// *  When refining the mesh, all child cells are subject to
// *  this "one-sided" spherical volume manifold.
// *  This manifold description is only available for the
// *  two-dimensional case.
// */
//template <int dim, int spacedim = dim, int chartdim=dim>
//class RealOneSidedSphericalManifold : public ChartManifold<dim,spacedim,chartdim>
//{
//public:
//  RealOneSidedSphericalManifold(typename Triangulation<dim>::cell_iterator &cell_in,
//                                unsigned int face_in,
//                                Point<dim> const &center)
//    : x_C(center), cell(cell_in), face(face_in)
//  {
//    Assert(dim==2,ExcMessage("RealOneSidedSphericalManifold only implemented for dim==2"));
//
//    // determine x1 and x2 which denote the end points of the face that is
//    // subject to the spherical manifold and calculate the vector pointing from x1 to x2.
//    Point<dim> x_1, x_2;
//    x_1 = cell->face(face)->vertex(0);
//    x_2 = cell->face(face)->vertex(1);
//    x_1C = x_1 - x_C;
//
//    // calculate radius of spherical manifold and check correctness of geometry and parameters
//    double radius = x_2.distance(center);
//    double radius_check = x_1.distance(center);
//    Assert(std::abs(radius-radius_check) < 1.e-12*radius,
//        ExcMessage("Invalid geometry parameters. To apply a spherical manifold both "
//            "end points of the face must have the same distance from the center."));
//
//    // calculate normal vector of the face that is subject to the spherical manifold
//    Tensor<1,dim> normal;
//    Tensor<1,dim> x_21 = x_2 - x_1;
//    if(dim==2)
//    {
//      // find a vector that is orthogonal to the face_vector
//      normal[0] = x_21[1];
//      normal[1] = -x_21[0];
//
//      // normalize
//      normal /= normal.norm();
//
//      // adjust orientation
//      double sign = sgn((x_2 - x_C)*normal);
//      normal *= sign;
//    }
//
//    // calculate angle
//    alpha = 2*std::acos(x_1C*normal/x_1C.norm());
//
//    AssertThrow(alpha >=0 && alpha <= numbers::PI, ExcMessage("Unexpected result."));
//
//    // calculate the cross product of x_1C and normal to get the correct sign
//    double sign = sgn(x_1C[0]*normal[1]-x_1C[1]*normal[0]);
//    alpha *= sign;
//  }
//
//
//  /*
//   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
//   *  point x in physical coordinates
//   */
//  Point<dim> push_forward(const Point<dim> &xi) const
//  {
//    Point<dim> x;
//
//    // standard mapping from reference space to physical space using d-linear shape functions
//    for(unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell;++v)
//    {
//      double shape_function_value = GeometryInfo<dim>::d_linear_shape_function(xi,v);
//      x += shape_function_value*cell->vertex(v);
//    }
//
//    // set pointers xi_face, xi_other to xi[0],xi[1] depending on the face that is subject to the manifold
//    unsigned int index_face = get_index_face();
//    unsigned int index_other = get_index_other();
//    double const xi_face = xi[index_face];
//    double const xi_other = xi[index_other];
//
//    // add contribution of spherical manifold
//
//    // calculate deformation related to the spherical manifold
//    double beta = xi_face*alpha;
//    Tensor<2,dim> Rot;
//    Rot[0][0] = +std::cos(beta);
//    Rot[0][1] = -std::sin(beta);
//    Rot[1][0] = +std::sin(beta);
//    Rot[1][1] = +std::cos(beta);
//
//    Tensor<1,dim> x_S;
//    x_S = x_C + Rot*x_1C;
//
//    Tensor<1,dim> d, x_lin;
//    for(unsigned int v=0; v<GeometryInfo<1>::vertices_per_cell;++v)
//    {
//      double shape_function_value = GeometryInfo<1>::d_linear_shape_function(Point<1>(xi_face),v);
//      x_lin += shape_function_value*cell->face(face)->vertex(v);
//    }
//
//    d = x_S - x_lin;
//
//    // deformation decreases linearly in the second (other) direction
//    Point<1> xi_other_1d = Point<1>(xi_other);
//    unsigned int index_1d = get_index_1d();
//    double fading_value = GeometryInfo<1>::d_linear_shape_function(xi_other_1d, index_1d);
//    x += fading_value*d;
//
//    Assert(numbers::is_finite(x.norm_square()), ExcMessage("Invalid point found"));
//
//    return x;
//  }
//
//  /*
//   *  Calculate index of 1d linear shape function (0 or 1)
//   *  that takes a value of 1 on the specified face.
//   */
//  unsigned int get_index_1d() const
//  {
//    unsigned int index_1d = 0;
//
//    if(face==0 || face==2)
//      index_1d = 0;
//    else if(face==1 || face==3)
//      index_1d = 1;
//    else
//      Assert(false,ExcMessage("Face ID is invalid."));
//
//    return index_1d;
//  }
//
//  /*
//   *  Calculate which xi-coordinate corresponds to the
//   *  tangent direction of the respective face
//   */
//  unsigned int get_index_face() const
//  {
//    unsigned int index_face = 0;
//
//    if(face==0 || face==1)
//      index_face = 1;
//    else if(face==2 || face==3)
//      index_face = 0;
//    else
//      Assert(false,ExcMessage("Face ID is invalid."));
//
//    return index_face;
//  }
//
//  /*
//   *  Calculate which xi-coordinate corresponds to
//   *  the normal direction of the respective face.
//   */
//  unsigned int get_index_other() const
//  {
//    return 1-get_index_face();
//  }
//
//  /*
//   *  This function calculates the inverse Jacobi matrix dx/d(xi) = d(phi(xi))/d(xi) of
//   *  the push-forward operation phi: [0,1]^d -> R^d: xi -> x = phi(xi)
//   *  We assume that the gradient of the standard bilinear shape functions is sufficient
//   *  to find the solution.
//   */
//  Tensor<2,dim> get_inverse_jacobian(Point<dim> const &xi) const
//  {
//    Tensor<2,dim> jacobian;
//
//    // standard mapping from reference space to physical space using d-linear shape functions
//    for(unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell;++v)
//    {
//      Tensor<1,dim> shape_function_gradient = GeometryInfo<dim>::d_linear_shape_function_gradient(xi,v);
//      jacobian += outer_product(cell->vertex(v),shape_function_gradient);
//    }
//
//    return invert(jacobian);
//  }
//
//  /*
//   *  pull_back operation that maps point x in physical coordinates
//   *  to point xi in reference coordinates [0,1]^d using the
//   *  push_forward operation and Newton's method
//   */
//  Point<dim> pull_back(const Point<dim> &x) const
//  {
//    Point<dim> xi;
//    Tensor<1,dim> residual = push_forward(xi) - x;
//    Tensor<1,dim> delta_xi;
//
//    // Newton method to solve nonlinear pull_back operation
//    unsigned int n_iter = 0, MAX_ITER = 100;
//    while(residual.norm() > 1.e-12 && n_iter < MAX_ITER)
//    {
//      // multiply by -1.0, i.e., shift residual to the rhs
//      residual *= -1.0;
//
//      // solve linear problem
//      delta_xi = get_inverse_jacobian(xi) * residual;
//
//      // add increment
//      xi += delta_xi;
//
//      // make sure that xi is in the valid range [0,1]^d
//      if(xi[0] < 0.0)
//        xi[0] = 0.0;
//      else if(xi[0] > 1.0)
//        xi[0] = 1.0;
//
//      if(xi[1] < 0.0)
//        xi[1] = 0.0;
//      else if(xi[1] > 1.0)
//        xi[1] = 1.0;
//
//      // evaluate residual
//      residual = push_forward(xi) - x;
//
//      // increment counter
//      ++n_iter;
//    }
//
//    Assert(n_iter < MAX_ITER, ExcMessage("Newton solver did not converge to given tolerance. Maximum number of iterations exceeded."));
//    Assert(xi[0] >= 0.0 && xi[0] <= 1.0, ExcMessage("Pull back operation generated invalid xi[0] values."));
//    Assert(xi[1] >= 0.0 && xi[1] <= 1.0, ExcMessage("Pull back operation generated invalid xi[1] values."));
//
//    return xi;
//  }
//
//private:
//  Tensor<1,dim> x_C;
//  Tensor<1,dim> x_1C;
//  double alpha;
//
//  typename Triangulation<dim>::cell_iterator cell;
//  unsigned int face;
//};


///*
// *  Class that provides a spherical manifold applied to
// *  one of the faces of a quadrilateral element.
// *  On the face subject to the spherical manifold intermediate
// *  points are inserted so that an equidistant distribution of points
// *  in terms of arclength is obtained.
// *  When refining the mesh, all child cells are subject to
// *  this "one-sided" spherical volume manifold.
// *  This manifold description is only available for the
// *  two-dimensional case.
// */
//template <int dim>
//class RealOneSidedSphericalManifold : public ChartManifold<dim,dim,dim>
//{
//public:
//  RealOneSidedSphericalManifold(typename Triangulation<dim>::cell_iterator &cell_in,
//                                unsigned int                               face_in,
//                                Point<dim> const                           &center)
//    : x_C(center), cell(cell_in), face(face_in)
//  {
//    double const tol = 1.e-12;
//
////    AssertThrow(dim==2,ExcMessage("RealOneSidedSphericalManifold only implemented for dim==2"));
//
//    // determine x_1 and x_2 which denote the end points of the face that is
//    // subject to the spherical manifold
//    Point<dim> x_1, x_2;
//    x_1 = cell->face(face)->vertex(0);
//    x_2 = cell->face(face)->vertex(1);
//
//    v_1 = x_1 - x_C;
//    v_2 = x_2 - x_C,
//
//    // calculate radius of spherical manifold
//    radius = v_1.norm();
//
//    // check correctness of geometry and parameters
//    double radius_check = v_2.norm();
//    AssertThrow(std::abs(radius-radius_check) < tol*radius,
//        ExcMessage("Invalid geometry parameters. To apply a spherical manifold both "
//            "end points of the face must have the same distance from the center."));
//
//    // normalize v_1 and v_2
//    v_1 /= v_1.norm();
//    v_2 /= v_2.norm();
//
//    // calculate angle between v_1 and v_2
//    alpha = std::acos(v_1*v_2);
//
//    // calculate vector that is perpendicular to v_1 in plane that is spanned by v_1 and v_2
//    normal = v_2 - (v_2*v_1)*v_1;
//
//    AssertThrow(normal.norm()>tol, ExcMessage("Vector must not have length 0."));
//
//    normal /= normal.norm();
//  }
//
//  /*
//   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
//   *  point x in physical coordinates
//   */
//  Point<dim> push_forward(const Point<dim> &xi) const
//  {
//    Point<dim> x;
//
//    // standard mapping from reference space to physical space using d-linear shape functions
//    for(unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell;++v)
//    {
//      double shape_function_value = GeometryInfo<dim>::d_linear_shape_function(xi,v);
//      x += shape_function_value*cell->vertex(v);
//    }
//
//    // set xi_face, xi_other to xi[0],xi[1] depending on the face that is subject to the manifold
//    unsigned int index_face = get_index_face();
//    unsigned int index_other = get_index_other();
//    double const xi_face = xi[index_face];
//    double const xi_other = xi[index_other];
//
//    // add contribution of spherical manifold
//
//    // calculate deformation related to the spherical manifold
//    double beta = xi_face*alpha;
//
//    Tensor<1,dim> direction;
//    direction = std::cos(beta)*v_1 + std::sin(beta)*normal;
//
//    Assert(std::abs(direction.norm()-1.0)<tol, ExcMessage("Vector must have length 1."));
//
//    // calculate point x_S on spherical manifold
//    Tensor<1,dim> x_S;
//    x_S = x_C + radius*direction;
//
//    // calculate displacement as compared to straight sided quadrilateral element
//    // on the face that is subject to the manifold
//    Tensor<1,dim> displ, x_lin;
//    for(unsigned int v=0; v<GeometryInfo<1>::vertices_per_cell;++v)
//    {
//      double shape_function_value = GeometryInfo<1>::d_linear_shape_function(Point<1>(xi_face),v);
//      x_lin += shape_function_value*cell->face(face)->vertex(v);
//    }
//
//    displ = x_S - x_lin;
//
//    // deformation decreases linearly in the second (other) direction
//    Point<1> xi_other_1d = Point<1>(xi_other);
//    unsigned int index_1d = get_index_1d();
//    double fading_value = GeometryInfo<1>::d_linear_shape_function(xi_other_1d, index_1d);
//    x += fading_value*displ;
//
//    Assert(numbers::is_finite(x.norm_square()), ExcMessage("Invalid point found"));
//
//    return x;
//  }
//
//  /*
//   *  Calculate index of 1d linear shape function (0 or 1)
//   *  that takes a value of 1 on the specified face.
//   */
//  unsigned int get_index_1d() const
//  {
//    unsigned int index_1d = 0;
//
//    if(face==0 || face==2)
//      index_1d = 0;
//    else if(face==1 || face==3)
//      index_1d = 1;
//    else
//      Assert(false,ExcMessage("Face ID is invalid."));
//
//    return index_1d;
//  }
//
//  /*
//   *  Calculate which xi-coordinate corresponds to the
//   *  tangent direction of the respective face
//   */
//  unsigned int get_index_face() const
//  {
//    unsigned int index_face = 0;
//
//    if(face==0 || face==1)
//      index_face = 1;
//    else if(face==2 || face==3)
//      index_face = 0;
//    else
//      Assert(false,ExcMessage("Face ID is invalid."));
//
//    return index_face;
//  }
//
//  /*
//   *  Calculate which xi-coordinate corresponds to
//   *  the normal direction of the respective face.
//   */
//  unsigned int get_index_other() const
//  {
//    return 1-get_index_face();
//  }
//
//  /*
//   *  This function calculates the inverse Jacobi matrix dx/d(xi) = d(phi(xi))/d(xi) of
//   *  the push-forward operation phi: [0,1]^d -> R^d: xi -> x = phi(xi)
//   *  We assume that the gradient of the standard bilinear shape functions is sufficient
//   *  to find the solution.
//   */
//  Tensor<2,dim> get_inverse_jacobian(Point<dim> const &xi) const
//  {
//    Tensor<2,dim> jacobian;
//
//    // standard mapping from reference space to physical space using d-linear shape functions
//    for(unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell;++v)
//    {
//      Tensor<1,dim> shape_function_gradient = GeometryInfo<dim>::d_linear_shape_function_gradient(xi,v);
//      jacobian += outer_product(cell->vertex(v),shape_function_gradient);
//    }
//
//    return invert(jacobian);
//  }
//
//  /*
//   *  pull_back operation that maps point x in physical coordinates
//   *  to point xi in reference coordinates [0,1]^d using the
//   *  push_forward operation and Newton's method
//   */
//  Point<dim> pull_back(const Point<dim> &x) const
//  {
//    Point<dim> xi;
//    Tensor<1,dim> residual = push_forward(xi) - x;
//    Tensor<1,dim> delta_xi;
//
//    // Newton method to solve nonlinear pull_back operation
//    unsigned int n_iter = 0, MAX_ITER = 100;
//    double const TOL = 1.e-12;
//    while(residual.norm() > TOL && n_iter < MAX_ITER)
//    {
//      // multiply by -1.0, i.e., shift residual to the rhs
//      residual *= -1.0;
//
//      // solve linear problem
//      delta_xi = get_inverse_jacobian(xi) * residual;
//
//      // add increment
//      xi += delta_xi;
//
//      // make sure that xi is in the valid range [0,1]^d
//      if(xi[0] < 0.0)
//        xi[0] = 0.0;
//      else if(xi[0] > 1.0)
//        xi[0] = 1.0;
//
//      if(xi[1] < 0.0)
//        xi[1] = 0.0;
//      else if(xi[1] > 1.0)
//        xi[1] = 1.0;
//
//      // evaluate residual
//      residual = push_forward(xi) - x;
//
//      // increment counter
//      ++n_iter;
//    }
//
//    Assert(n_iter < MAX_ITER, ExcMessage("Newton solver did not converge to given tolerance. Maximum number of iterations exceeded."));
//    Assert(xi[0] >= 0.0 && xi[0] <= 1.0, ExcMessage("Pull back operation generated invalid xi[0] values."));
//    Assert(xi[1] >= 0.0 && xi[1] <= 1.0, ExcMessage("Pull back operation generated invalid xi[1] values."));
//
//    return xi;
//  }
//
//private:
//  Tensor<1,dim> x_C;
//  Tensor<1,dim> v_1;
//  Tensor<1,dim> v_2;
//  Tensor<1,dim> normal;
//  double alpha;
//  double radius;
//
//  typename Triangulation<dim>::cell_iterator cell;
//  unsigned int face;
//};

/*
 *  Class that provides a spherical manifold applied to one of the faces
 *  of a quadrilateral element.
 *  On the face subject to the spherical manifold intermediate points are
 *  inserted so that an equidistant distribution of points in terms of
 *  arclength is obtained.
 *  When refining the mesh, all child cells are subject to this "one-sided"
 *  spherical volume manifold.
 *  This manifold description is only available for the two-dimensional case.
 */
template <int dim>
class RealOneSidedSphericalManifold : public ChartManifold<dim,dim,dim>
{
public:
  RealOneSidedSphericalManifold(typename Triangulation<dim>::cell_iterator &cell_in,
                                unsigned int                               face_in,
                                Point<dim> const                           &center)
    : cell(cell_in), face(face_in)
  {
    AssertThrow(face >= 0 && face <= 3, ExcMessage("One sided spherical manifold can only be applied to face f=0,1,2,3."));

    // get center coordinates in x1-x2 plane
    x_C[0] = center[0];
    x_C[1] = center[1];

    // determine x_1 and x_2 which denote the end points of the face that is
    // subject to the spherical manifold.
    Point<dim> x_1, x_2;
    x_1 = cell->vertex(get_vertex_id(0));
    x_2 = cell->vertex(get_vertex_id(1));

    Point<2> x_1_2d = Point<2>(x_1[0],x_1[1]);
    Point<2> x_2_2d = Point<2>(x_2[0],x_2[1]);

    initialize(x_1_2d,x_2_2d);
  }

  void initialize(Point<2> const &x_1, Point<2> const &x_2)
  {
    double const tol = 1.e-12;

    v_1 = x_1 - x_C;
    v_2 = x_2 - x_C,

    // calculate radius of spherical manifold
    radius = v_1.norm();

    // check correctness of geometry and parameters
    double radius_check = v_2.norm();
    AssertThrow(std::abs(radius-radius_check) < tol*radius,
        ExcMessage("Invalid geometry parameters. To apply a spherical manifold both "
            "end points of the face must have the same distance from the center."));

    // normalize v_1 and v_2
    v_1 /= v_1.norm();
    v_2 /= v_2.norm();

    // calculate angle between v_1 and v_2
    alpha = std::acos(v_1*v_2);

    // calculate vector that is perpendicular to v_1 in plane that is spanned by v_1 and v_2
    normal = v_2 - (v_2*v_1)*v_1;

    AssertThrow(normal.norm()>tol, ExcMessage("Vector must not have length 0."));

    normal /= normal.norm();
  }

  /*
   *  push_forward operation that maps point xi in reference coordinates [0,1]^d to
   *  point x in physical coordinates
   */
  Point<dim> push_forward(const Point<dim> &xi) const
  {
    Point<dim> x;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell;++v)
    {
      double shape_function_value = GeometryInfo<dim>::d_linear_shape_function(xi,v);
      x += shape_function_value*cell->vertex(v);
    }

    // Add contribution of spherical manifold.
    // Here, we only operate in the xi1-xi2 plane.

    // set xi_face, xi_other to xi[0],xi[1] depending on the face that is subject to the manifold
    unsigned int index_face = get_index_face();
    unsigned int index_other = get_index_other();
    double const xi_face = xi[index_face];
    double const xi_other = xi[index_other];

    // calculate deformation related to the spherical manifold
    double beta = xi_face*alpha;

    Tensor<1,2> direction;
    direction = std::cos(beta)*v_1 + std::sin(beta)*normal;

    Assert(std::abs(direction.norm()-1.0)<tol, ExcMessage("Vector must have length 1."));

    // calculate point x_S on spherical manifold
    Tensor<1,2> x_S;
    x_S = x_C + radius*direction;

    // calculate displacement as compared to straight sided quadrilateral element
    // on the face that is subject to the manifold
    Tensor<1,2> displ, x_lin;
    for(unsigned int v=0; v<GeometryInfo<1>::vertices_per_cell;++v)
    {
      double shape_function_value = GeometryInfo<1>::d_linear_shape_function(Point<1>(xi_face),v);

      unsigned int vertex_id = get_vertex_id(v);
      Point<dim> vertex = cell->vertex(vertex_id);

      x_lin[0] += shape_function_value*vertex[0];
      x_lin[1] += shape_function_value*vertex[1];
    }

    displ = x_S - x_lin;

    // deformation decreases linearly in the second (other) direction
    Point<1> xi_other_1d = Point<1>(xi_other);
    unsigned int index_1d = get_index_1d();
    double fading_value = GeometryInfo<1>::d_linear_shape_function(xi_other_1d, index_1d);
    x[0] += fading_value*displ[0];
    x[1] += fading_value*displ[1];


    Assert(numbers::is_finite(x.norm_square()), ExcMessage("Invalid point found"));

    return x;
  }

  /*
   *  Calculate vertex_id of 2d object (cell in 2d, face4 in 3d)
   *  given the vertex_id of the 1d object (vertex_id_1d = 0,1).
   */
  unsigned int get_vertex_id(unsigned int vertex_id_1d) const
  {
    unsigned int vertex_id = 0;

    if(face == 0)
      vertex_id = 2*vertex_id_1d;
    else if(face == 1)
      vertex_id = 1 + 2*vertex_id_1d;
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
  unsigned int get_index_1d() const
  {
    unsigned int index_1d = 0;

    if(face==0 || face==2)
      index_1d = 0;
    else if(face==1 || face==3)
      index_1d = 1;
    else
      Assert(false,ExcMessage("Face ID is invalid."));

    return index_1d;
  }

  /*
   *  Calculate which xi-coordinate corresponds to the
   *  tangent direction of the respective face
   */
  unsigned int get_index_face() const
  {
    unsigned int index_face = 0;

    if(face==0 || face==1)
      index_face = 1;
    else if(face==2 || face==3)
      index_face = 0;
    else
      Assert(false,ExcMessage("Face ID is invalid."));

    return index_face;
  }

  /*
   *  Calculate which xi-coordinate corresponds to
   *  the normal direction of the respective face
   *  in xi1-xi2-plane.
   */
  unsigned int get_index_other() const
  {
    return 1-get_index_face();
  }

  /*
   *  This function calculates the inverse Jacobi matrix dx/d(xi) = d(phi(xi))/d(xi) of
   *  the push-forward operation phi: [0,1]^d -> R^d: xi -> x = phi(xi)
   *  We assume that the gradient of the standard bilinear shape functions is sufficient
   *  to find the solution.
   */
  Tensor<2,dim> get_inverse_jacobian(Point<dim> const &xi) const
  {
    Tensor<2,dim> jacobian;

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell;++v)
    {
      Tensor<1,dim> shape_function_gradient = GeometryInfo<dim>::d_linear_shape_function_gradient(xi,v);
      jacobian += outer_product(cell->vertex(v),shape_function_gradient);
    }

    return invert(jacobian);
  }

  /*
   *  pull_back operation that maps point x in physical coordinates
   *  to point xi in reference coordinates [0,1]^d using the
   *  push_forward operation and Newton's method
   */
  Point<dim> pull_back(const Point<dim> &x) const
  {
    Point<dim> xi;
    Tensor<1,dim> residual = push_forward(xi) - x;
    Tensor<1,dim> delta_xi;

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

    Assert(n_iter < MAX_ITER, ExcMessage("Newton solver did not converge to given tolerance. Maximum number of iterations exceeded."));
    Assert(xi[0] >= 0.0 && xi[0] <= 1.0, ExcMessage("Pull back operation generated invalid xi[0] values."));
    Assert(xi[1] >= 0.0 && xi[1] <= 1.0, ExcMessage("Pull back operation generated invalid xi[1] values."));

    return xi;
  }

private:
  Point<2> x_C;
  Tensor<1,2> v_1;
  Tensor<1,2> v_2;
  Tensor<1,2> normal;
  double alpha;
  double radius;

  typename Triangulation<dim>::cell_iterator cell;
  unsigned int face;
};

#endif /* INCLUDE_ONESIDEDSPHERICALMANIFOLD_H_ */
