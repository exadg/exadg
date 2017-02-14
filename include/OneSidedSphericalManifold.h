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
    Point<dim> x1 = Point<dim>(), x2 = Point<dim>();
    x1 = cell->face(face)->vertex(0);
    x2 = cell->face(face)->vertex(1);
    face_vector = x2 - x1;

    // calculate radius of spherical manifold
    radius = x2.distance(center);

    // check correctness of geometry and parameters
    double radius_check = x1.distance(center);
    Assert(std::abs(radius-radius_check)<1.e-12,
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
    Point<dim> x = Point<dim>();

    // standard mapping from reference space to physical space using d-linear shape functions
    for(unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell;++v)
    {
      double shape_function_value = GeometryInfo<dim>::d_linear_shape_function(xi,v);
      x += shape_function_value*cell->vertex(v);
    }

    // set pointers xi_face, xi_other to xi[0],xi[1] depending on the face that is subject to the manifold
    unsigned int index_face = get_index_face();
    unsigned int index_other = get_index_other();
    double const *xi_face = &xi[index_face];
    double const *xi_other = &xi[index_other];

    // add contribution of spherical manifold

    // transform coordinate from reference interval [0,1] to interval [-1,1]: xi_face -> eta
    double eta = 2.0*(*xi_face) - 1;

    // calculate deformation related to the spherical manifold
    double h = calculate_h(eta);

    // deformation decreases linearly in the second (other) direction
    Point<1> xi_other_1d = Point<1>(*xi_other);
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
    unsigned int index_other = 0;

    if(face==0 || face==1)
      index_other = 0;
    else if(face==2 || face==3)
      index_other = 1;
    else
      Assert(false,ExcMessage("Face ID is invalid."));

    return index_other;
  }

  /*
   *  Calculate h describing the displacement on the face
   *  that is subject to the spherical manifold.
   */
  double calculate_h(double const &eta) const
  {
    double value = 1.0-std::pow(face_vector.norm()*eta/(2*radius),2.0);
    Assert(value > 0.0, ExcMessage("Argument of std::sqrt() must not be negative."));
    double h = radius*std::sqrt(value) - distance_center_face;

    return h;
  }

  /*
   *  Calculate derivative of h with respect to xi_face.
   */
  double calculate_dh_dxi(double const &eta) const
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
    double const *xi_face = &xi[index_face];
    double const *xi_other = &xi[index_other];

    // transform from reference interval [0,1] to interval [-1,1]: xi_face -> eta
    double eta = 2.0*(*xi_face) - 1;

    // calculate h and derivative of h with respect to xi_face
    double h =  calculate_h(eta);
    double dh_dxi = calculate_dh_dxi(eta);

    // calculate index specifying which of the two one-dimensional,
    // linear shape functions has to be evaluated.
    unsigned int index_1d = get_index_1d();

    // deformation decreases linearly in the second direction
    Point<1> xi_other_1d = Point<1>(*xi_other);
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

#endif /* INCLUDE_ONESIDEDSPHERICALMANIFOLD_H_ */
