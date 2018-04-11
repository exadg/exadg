/*
 * deformed_cube_manifold.h
 *
 *  Created on: Apr 9, 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_DEFORMED_CUBE_MANIFOLD_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_DEFORMED_CUBE_MANIFOLD_H_


template <int dim>
class DeformedCubeManifold : public ChartManifold<dim,dim,dim>
{
public:
  DeformedCubeManifold(const double left,
                       const double right,
                       const double deformation,
                       const unsigned int frequency = 1)
    :
    left(left),
    right(right),
    deformation(deformation),
    frequency(frequency)
  {}

  Point<dim> push_forward(const Point<dim> &chart_point) const
  {
    double sinval = deformation;
    for (unsigned int d=0; d<dim; ++d)
      sinval *= std::sin(frequency*numbers::PI*(chart_point(d)-left)/(right-left));
    Point<dim> space_point;
    for (unsigned int d=0; d<dim; ++d)
      space_point(d) = chart_point(d) + sinval;
    return space_point;
  }

  Point<dim> pull_back(const Point<dim> &space_point) const
  {
    Point<dim> x = space_point;
    Point<dim> one;
    for (unsigned int d=0; d<dim; ++d)
      one(d) = 1.;

    // Newton iteration to solve the nonlinear equation given by the point
    Tensor<1,dim> sinvals;
    for (unsigned int d=0; d<dim; ++d)
      sinvals[d] = std::sin(frequency*numbers::PI*(x(d)-left)/(right-left));

    double sinval = deformation;
    for (unsigned int d=0; d<dim; ++d)
      sinval *= sinvals[d];
    Tensor<1,dim> residual = space_point - x - sinval*one;
    unsigned int its = 0;
    while (residual.norm() > 1e-12 && its < 100)
      {
        Tensor<2,dim> jacobian;
        for (unsigned int d=0; d<dim; ++d)
          jacobian[d][d] = 1.;
        for (unsigned int d=0; d<dim; ++d)
          {
            double sinval_der = deformation * frequency / (right-left) * numbers::PI *
              std::cos(frequency*numbers::PI*(x(d)-left)/(right-left));
            for (unsigned int e=0; e<dim; ++e)
              if (e!=d)
                sinval_der *= sinvals[e];
            for (unsigned int e=0; e<dim; ++e)
              jacobian[e][d] += sinval_der;
          }

        x += invert(jacobian) * residual;

        for (unsigned int d=0; d<dim; ++d)
          sinvals[d] = std::sin(frequency*numbers::PI*(x(d)-left)/(right-left));

        sinval = deformation;
        for (unsigned int d=0; d<dim; ++d)
          sinval *= sinvals[d];
        residual = space_point - x - sinval*one;
        ++its;
      }
    AssertThrow (residual.norm() < 1e-12,
                 ExcMessage("Newton for point did not converge."));
    return x;
  }

private:
  const double left;
  const double right;
  const double deformation;
  const unsigned int frequency;
};


#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_DEFORMED_CUBE_MANIFOLD_H_ */
