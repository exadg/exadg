/*
 * energy_spectrum_calculation.h
 *
 *  Created on: Feb 7, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_

#include <limits>
#include <vector>
#include <cmath>
#include <memory>

std::pair<double,double>
do_inner_gauss_loop(const unsigned int n_points,
                    const unsigned int i)
{
  const double tolerance = std::numeric_limits<double>::epsilon() * 5.;

  double z = std::cos(numbers::PI * (i-.25)/(n_points+.5));

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
    x[i] = - std::cos( (double) (2*i+1)/(2*m) * numbers::PI );

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

template<int dim, int fe_degree, typename Number>
class KineticEnergySpectrumCalculator
{
public:
  KineticEnergySpectrumCalculator()
    :
    clear_files(true),
    matrix_free_data(nullptr)
  {}

  void setup(MatrixFree<dim,Number> const    &matrix_free_data_in,
             DofQuadIndexData const          &dof_quad_index_data_in,
             KineticEnergySpectrumData const &data_in)
  {
    matrix_free_data = &matrix_free_data_in;
    dof_quad_index_data = dof_quad_index_data_in;
    data = data_in;

    fill_shape_values(shape_values);
  }

  void evaluate(parallel::distributed::Vector<Number> const &velocity,
                double const                                &time,
                int const                                   &time_step_number)
  {
    if(data.calculate == true)
    {
      if(time_step_number >= 0) // unsteady problem
        do_evaluate(velocity,time,time_step_number);
      else // steady problem (time_step_number = -1)
      {
        AssertThrow(false, ExcMessage("Calculation of kinetic energy spectrum only implemented for unsteady problems."));
      }
    }
  }

private:
  std::vector<double> get_equidistant(const unsigned int n_points)
  {
    std::vector<double> points(n_points);

    for(unsigned int i = 0; i < n_points; i++)
    {
      points[i] = (1.0*i)/(n_points-1);
    }

    return points;
  }

  // equidistant points on range [0,1]: outer points positioned inside of range
  std::vector<double> get_equidistant_inner(const unsigned int n_points)
  {
    std::vector<double> points(n_points);

    for(unsigned int i = 0; i < n_points; i++)
    {
      points[i] = (i+0.5)/n_points;
    }

    return points;
  }

  template <typename Vector>
  void fill_shape_values(Vector& shape_values)
  {
    const unsigned TYPE = 2;
    const unsigned int degree = fe_degree;
    // fill shape values and gradients
    const unsigned int n_q_points_1d = degree + 1;
    const unsigned int stride = (n_q_points_1d + 1) / 2;
    LagrangePolynomialBasis basis_gll(get_gauss_lobatto_points(degree + 1));
    std::vector<double> gauss_points(
        TYPE == 0 ? get_gauss_points(n_q_points_1d) :
        TYPE == 1 ? get_equidistant(n_q_points_1d) :
            get_equidistant_inner(n_q_points_1d));

    AlignedVector<Number> temp;
    temp.resize((degree + 1) * n_q_points_1d);

    for (unsigned int i = 0; i < degree + 1; ++i)
    {
      for (unsigned int q = 0; q < n_q_points_1d; ++q)
        temp[i * n_q_points_1d + q] = basis_gll.value(i, gauss_points[q]);
    }

    // for even odd tensor product kernel
    shape_values.resize((degree + 1) * stride);

    for (unsigned int i = 0; i < (degree + 1) / 2; ++i)
    {
      for (unsigned int q = 0; q < stride; ++q)
      {
        const double p1 = basis_gll.value(i, gauss_points[q]);
        const double p2 = basis_gll.value(i, gauss_points[n_q_points_1d - 1 - q]);
        shape_values[i * stride + q] = 0.5 * (p1 + p2);
        shape_values[(degree - i) * stride + q] = 0.5 * (p1 - p2);
      }
    }
    if (degree % 2 == 0)
    {
      for (unsigned int q = 0; q < stride; ++q)
        shape_values[degree / 2 * stride + q] = basis_gll.value(degree / 2, gauss_points[q]);
    }
  }

  void interpolate(MatrixFree<dim, Number> const                    &data,
                   LinearAlgebra::distributed::Vector<Number>       &dst,
                   LinearAlgebra::distributed::Vector<Number> const &src,
                   std::pair<unsigned int,unsigned int> const       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,Number> phi(data);
    typedef dealii::internal::FEEvaluationImplBasisChange<dim, fe_degree+1, fe_degree+1, 1, VectorizedArray<Number>, VectorizedArray<Number> > eval;

    // get reference to local dof-array
    VectorizedArray<Number> *data_ptr = phi.begin_dof_values();

    // iterate over all macro cells
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      // ... set macro cell
      phi.reinit(cell);

      // ... gather dofs
      phi.read_dof_values(src);

      // ... interpolate
      eval::do_forward(shape_values, data_ptr, data_ptr);

      // ... scatter dofs
      phi.set_dof_values(dst);
    }
  }

  void do_evaluate(parallel::distributed::Vector<Number> const &velocity,
                   double const                                time,
                   unsigned int const                          time_step_number)
  {
    if((time_step_number-1)%data.calculate_every_time_steps == 0)
    {
      // TODO: implement functions that evaluate energy spectrum
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::cout << "Calculate kinetic energy spectrum" << std::endl;
      }

      // interpolate data
      LinearAlgebra::distributed::Vector<Number> dst(velocity);
      matrix_free_data->cell_loop(&KineticEnergySpectrumCalculator<dim, fe_degree, Number>::interpolate, this, dst, velocity);

      // call FFT

      // postprocess results



      // write output file
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::ostringstream filename;
        filename << data.filename_prefix;

        std::ofstream f;
        if(clear_files == true)
        {
          f.open(filename.str().c_str(),std::ios::trunc);

          // TODO: write headlines to file
          /*
          f << "Kinetic energy: E_k = 1/V * 1/2 * (u,u)_Omega, where V=(1,1)_Omega" << std::endl
            << "Dissipation rate: epsilon = nu/V * (grad(u),grad(u))_Omega, where V=(1,1)_Omega" << std::endl
            << "Enstrophy: E = 1/V * 1/2 * (rot(u),rot(u))_Omega, where V=(1,1)_Omega" << std::endl;

          f << std::endl
            << "  Time                Kin. energy         dissipation         enstrophy"<<std::endl;
          */

          clear_files = false;
        }
        else
        {
          f.open(filename.str().c_str(),std::ios::app);
        }

        unsigned int precision = 12;
        // TODO: write output
        f << std::scientific << std::setprecision(precision)
          << std::setw(precision+8) << /*TODO <<*/ std::endl;
      }
    }
  }

  bool clear_files;
  MatrixFree<dim,Number> const * matrix_free_data;
  DofQuadIndexData dof_quad_index_data;
  KineticEnergySpectrumData data;
  AlignedVector<VectorizedArray<Number> > shape_values;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_ */
