// this file is inspired by the file deal.II/source/base/quadrature_lib.cc,
// see www.dealii.org for information about licenses.

#ifndef gauss_formula_h
#define gauss_formula_h

#include <limits>
#include <vector>
#include <cmath>
#include <memory>

namespace dealspectrum{

const double PI = 3.14159265358979323846;

std::pair<double,double>
do_inner_gauss_loop(const unsigned int n_points,
                    const unsigned int i)
{
  const double tolerance = std::numeric_limits<double>::epsilon() * 5.;

  double z = std::cos(PI * (i-.25)/(n_points+.5));

  double pp;
  double p1;

  // Newton iteration
  do
    {
      // compute L_n (z)
      p1 = 1.;
      double p2 = 0.;
      for (unsigned int j=0; j<n_points; ++j)
        {
          const double p3 = p2;
          p2 = p1;
          p1 = ((2.*j+1.)*z*p2-j*p3)/(j+1);
        }
      pp = n_points*(z*p1-p2)/(z*z-1);
      z = z-p1/pp;
    }
  while (std::abs(p1/pp) > tolerance);

  return std::make_pair(0.5*z, 1./((1.-z*z)*pp*pp));
}

std::vector<double>
get_gauss_points(const unsigned int n_points)
{
  std::vector<double> points(n_points);

  //const unsigned int m=(n_points+1)/2;
  for (unsigned int i=1; i<=n_points; ++i)
    {
      const double x = do_inner_gauss_loop(n_points, i).first;
      points[i-1] = .5-x;
      points[n_points-i] = .5+x;
    }
  return points;
}

std::vector<double>
get_gauss_weights(const unsigned int n_points)
{
  std::vector<double> weights(n_points);

  const unsigned int m=(n_points+1)/2;
  for (unsigned int i=1; i<=m; ++i)
    {
      const double w = do_inner_gauss_loop(n_points, i).second;
      weights[i-1] = w;
      weights[n_points-i] = w;
    }
  return weights;
}


double jacobi_polynomial(const double x,
                         const int alpha,
                         const int beta,
                         const unsigned int n)
{
  // the Jacobi polynomial is evaluated
  // using a recursion formula.
  std::vector<double> p(n+1);

  // initial values P_0(x), P_1(x):
  p[0] = 1.0;
  if (n==0) return p[0];
  p[1] = ((alpha+beta+2)*x + (alpha-beta))/2;
  if (n==1) return p[1];

  for (unsigned int i=1; i<=(n-1); ++i)
    {
      const int v  = 2*i + alpha + beta;
      const int a1 = 2*(i+1)*(i + alpha + beta + 1)*v;
      const int a2 = (v + 1)*(alpha*alpha - beta*beta);
      const int a3 = v*(v + 1)*(v + 2);
      const int a4 = 2*(i+alpha)*(i+beta)*(v + 2);

      p[i+1] = static_cast<double>( (a2 + a3*x)*p[i] - a4*p[i-1])/a1;
    } // for
  return p[n];
}

std::vector<double>
get_gauss_lobatto_points(const unsigned int n_points)
{
  std::vector<double> points(n_points);
  const unsigned int m = n_points-2;
  const double tolerance = std::numeric_limits<double>::epsilon() * 5.;

  // initial guess
  std::vector<double> x(m);
  for (unsigned int i=0; i<m; ++i)
    x[i] = - std::cos( (double) (2*i+1)/(2*m) * PI );

  const double alpha = 1;
  const double beta = 1;
  double s, J_x, f, delta;

  for (unsigned int k=0; k<m; ++k)
    {
      long double r = x[k];
      if (k>0)
        r = (r + x[k-1])/2;

      do
        {
          s = 0.;
          for (unsigned int i=0; i<k; ++i)
            s += 1./(r - x[i]);

          J_x   =  0.5*(alpha + beta + m + 1)*jacobi_polynomial(r, alpha+1, beta+1, m-1);
          f     = jacobi_polynomial(r, alpha, beta, m);
          delta = f/(f*s- J_x);
          r += delta;
        }
      while (std::fabs(delta) >= tolerance);

      x[k] = r;
    } // for

  // add boundary points:
  x.insert(x.begin(), -1.);
  x.push_back(+1.);

  for (unsigned int i=0; i<points.size(); ++i)
    points[i] = 0.5 + 0.5*x[i];

  return points;
}

}

#endif
