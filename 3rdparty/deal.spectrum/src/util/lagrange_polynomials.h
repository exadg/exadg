// this file is inspired by the files deal.II/source/base/polynomial.cc,
// see www.dealii.org for information about licenses.

#ifndef lagrange_polynomials_h
#define lagrange_polynomials_h

#include <limits>
#include <vector>

namespace dealspectrum{

class LagrangePolynomialBasis
{
public:
  LagrangePolynomialBasis(const std::vector<double> &points)
    :
    points(points)
  {
    lagrange_denominators.resize(points.size());
    for (unsigned int i=0; i<points.size(); ++i)
      {
        double denominator = 1.;
        for (unsigned int j=0; j<points.size(); ++j)
          if (j!=i)
            denominator *= points[i] - points[j];
        lagrange_denominators[i] = 1./denominator;
      }
  }

  unsigned int degree() const
  {
    return points.size()-1;
  }

  double value(const unsigned int polynomial_index,
               const double x) const
  {
    double value = 1.;
    for (unsigned int i=0; i<points.size(); ++i)
      if (polynomial_index != i)
        value *= x-points[i];
    return value * lagrange_denominators[polynomial_index];
  }

  double derivative(const unsigned int polynomial_index,
                    const double       x) const
  {
    double value = 1;
    double derivative = 0;
    for (unsigned int i=0; i<points.size(); ++i)
      if (i != polynomial_index)
        {
          const double v = x-points[i];
          derivative = derivative * v + value;
          value *= v;
        }
    return derivative * lagrange_denominators[polynomial_index];
  }

private:
  std::vector<double> points;
  std::vector<double> lagrange_denominators;
};

}

#endif
