
#ifndef READ_BSPLINE_H
#define READ_BSPLINE_H

#include <deal.II/base/point.h>
#include <deal.II/base/tensor.h>

using namespace dealii;

const double convert_mm_to_m = 0.001;

template <int dim, int degree=3>
class BSpline2D
{
public:
  BSpline2D()
  {}

  BSpline2D(const std::vector<Point<dim>> &control_points_in)
  {
    reinit(control_points_in);
  }

  void reinit(const std::vector<Point<dim>> &control_points_in)
  {
    ny = control_points_in.size();
    nx = control_points_in[0].size();
    for (auto i : control_points_in)
      Assert(i.size() == nx,
             ExcInternalError("Expected constant size of control points along x, "
                              "got " + std::to_string(i.size()) + " vs " + std::to_string(nx)));
    knot_vector_x.resize(4+nx);
    knot_vector_y.resize(4+ny);
    control_points.resize(nx * ny);
    for (unsigned int j=0; j<nx; ++j)
      {
        const unsigned int ind_x = j<nx-4 ? j+1 : nx-3;
        knot_vector_x[4+j] = (double)ind_x / (nx-3);
      }
    for (unsigned int i=0; i<ny; ++i)
      {
        const unsigned int ind_y = i<ny-4 ? i+1 : ny-3;
        knot_vector_y[4+i] = (double)ind_y / (ny-3);
        for (unsigned int j=0; j<nx; ++j)
          control_points[i*nx+j] = control_points_in[i][j];
      }
  }

  Point<dim> value (const double x,
                    const double y) const
  {
    constexpr unsigned int p=degree;
    const unsigned int n_intervals_x = knot_vector_x.size()-2*p-1;
    const unsigned int n_intervals_y = knot_vector_y.size()-2*p-1;
    const unsigned int ky = std::min(n_intervals_y-1,
                                     static_cast<unsigned int>(y * n_intervals_y));
    const unsigned int k = std::min(n_intervals_x-1,
                                    static_cast<unsigned int>(x * n_intervals_x));

    if (knot_vector_x[p+k]*(1-1e-13) > x || knot_vector_x[p+1+k]*(1+1e-13) < x)
      std::cout << "Could not identify x knot position "
                << x << " with k = " << k << " " << knot_vector_x[p+k] << " "
                << knot_vector_x[p+1+k] << std::endl;
    if (knot_vector_y[p+ky]*(1-1e-13) > y || knot_vector_y[p+1+ky]*(1+1e-13) < y)
      std::cout << "Could not identify y knot position "
                << y << " with k = " << ky << " " << knot_vector_y[p+ky] << " "
                << knot_vector_y[p+1+ky] << std::endl;

    std::array<std::array<Point<dim>,p+1>,p+1> d;
    for (unsigned int i=0; i<=p; ++i)
      for (unsigned int j=0; j<=p; ++j)
        d[j][i] = control_points[(j+ky)*nx+k+i];
    for (unsigned int r=1; r<=p; ++r)
      {
        for (unsigned int j=p; j>=r; --j)
          {
            const double alpha_j = (y - knot_vector_y[j+ky]) /
              (knot_vector_y[j+1+ky-r+p] - knot_vector_y[j+ky]);
            for (unsigned int i=0; i<=p; ++i)
              d[j][i] = (1-alpha_j)*d[j-1][i] + alpha_j*d[j][i];
          }
      }
    for (unsigned int r=1; r<=p; ++r)
      {
        for (unsigned int i=p; i>=r; --i)
          {
            const double alpha_j = (x - knot_vector_x[i+k]) /
              (knot_vector_x[i+1+k-r+p] - knot_vector_x[i+k]);
            d[p][i] = (1-alpha_j)*d[p][i-1] + alpha_j *d[p][i];
          }
      }
    return d[p][p];
  }

  void write_to_file(std::ostream &file)
  {
    file.write(reinterpret_cast<char*>(&nx), sizeof(unsigned int));
    file.write(reinterpret_cast<char*>(&ny), sizeof(unsigned int));
    file.write(reinterpret_cast<char*>(knot_vector_x.data()),
               sizeof(double)*knot_vector_x.size());
    file.write(reinterpret_cast<char*>(knot_vector_y.data()),
               sizeof(double)*knot_vector_y.size());
    file.write(reinterpret_cast<char*>(control_points.data()),
               sizeof(Point<dim>)*control_points.size());
  }

  void read_from_file(std::istream &file)
  {
    file.read(reinterpret_cast<char*>(&nx), sizeof(unsigned int));
    file.read(reinterpret_cast<char*>(&ny), sizeof(unsigned int));
    knot_vector_x.resize(nx+4);
    file.read(reinterpret_cast<char*>(knot_vector_x.data()),
              sizeof(double)*knot_vector_x.size());
    knot_vector_y.resize(ny+4);
    file.read(reinterpret_cast<char*>(knot_vector_y.data()),
              sizeof(double)*knot_vector_y.size());
    control_points.resize(nx*ny);
    file.read(reinterpret_cast<char*>(control_points.data()),
              sizeof(Point<dim>)*control_points.size());
    for (auto &p : control_points)
      p *= convert_mm_to_m;
  }

private:
  std::vector<Point<dim>> control_points;

  std::vector<double> knot_vector_x;
  std::vector<double> knot_vector_y;

  unsigned int nx;
  unsigned int ny;
};


inline
void create_bspline_from_file(const std::string &filename)
{
  std::ifstream file(filename.c_str());
  unsigned int n_splines;
  file.read(reinterpret_cast<char*>(&n_splines), sizeof(unsigned int));
  std::vector<BSpline2D<3,3>> splines(n_splines);
  for (unsigned int s=0; s<n_splines; ++s)
    {
      splines[s].read_from_file(file);
      std::cout << splines[s].value(0,0) << "   " << splines[s].value(0.1, 0)
                << "    " << splines[s].value(0.2, 0.1) << std::endl;
    }
}

#endif
