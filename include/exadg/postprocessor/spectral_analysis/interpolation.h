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

#ifndef DEAL_SPECTRUM_INTERPOLATION
#define DEAL_SPECTRUM_INTERPOLATION

// std
#include <mpi.h>
#include <stdlib.h>

// deal.II
#include <deal.II/base/aligned_vector.h>
#include <deal.II/base/point.h>
#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/utilities.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/matrix_free/evaluation_kernels.h>

// ExaDG
#include <exadg/postprocessor/spectral_analysis/setup.h>

using namespace dealii;

namespace dealspectrum
{
/**
 * Class which loops over all cells and performs interpolation of every velocity
 * component (without using Matrix-Free).
 */
class Interpolator
{
public:
  MPI_Comm const & comm;
  // reference to DEAL.SPECTRUM setup
  Setup & s;
  // is initialized?
  bool initialized;
  // source vector (values to be interpolated)
  double * src = NULL;
  // destination vector (interpolated values)
  double * dst = NULL;
  // shape function (gauss lobatto to equidistant)
  AlignedVector<double> shape_values;
  // number of cells in each direction
  int cells;
  // dimensions
  int DIM;
  // process local cell range on sfc
  int start;
  int end;
  // points in each direction in cell & points per cell for source...
  unsigned int points_source;
  int          dofs_source;
  // ... and for target
  unsigned int points_target;
  int          dofs_target;

  /**
   * Constructor
   * @param s DEAL.SPECTRUM setup
   */
  Interpolator(MPI_Comm const & comm, Setup & s) : comm(comm), s(s), initialized(false)
  {
  }

  virtual ~Interpolator()
  {
    // not initialized -> nothing to clean up
    if(!initialized)
      return;

    delete[] src;
    delete[] dst;
  }

  template<typename MAPPING>
  void
  init(MAPPING & MAP)
  {
    int start, end;
    MAP.getLocalRange(start, end);

    init(start, end);
  }

  void
  init(int start, int end)
  {
    // check if already initialized
    if(this->initialized)
      return;
    this->initialized = true;

    // get cell range...
    this->start = start;
    this->end   = end;
    // ...number of local cells
    cells = end - start;

    // get dimensions
    DIM = s.dim;

    // number of Gauss-Lobatto points:
    points_source = s.points_src;
    // ... of equidistant points:
    points_target = s.points_dst;

    // number of gauss lobatto points per cell and ...
    dofs_source = dealii::Utilities::pow(points_source, DIM);
    // ...number of equidistant points per cell
    dofs_target = dealii::Utilities::pow(points_target, DIM);

    // allocate memory for source (only needed if values are read by IO)...
    src = new double[cells * dofs_source * DIM];
    // ... and target
    dst = new double[cells * dofs_target * DIM];

    // fill shape values
    fill_shape_values(shape_values);
  }

  /**
   * Create n equidistant points on range [0,1] such that outer points
   * positioned inside of range.
   *
   * @params n_points     nr of points
   */
  std::vector<dealii::Point<1>>
  get_equidistant_inner(unsigned int const n_points)
  {
    std::vector<dealii::Point<1>> points(n_points);

    for(unsigned int i = 0; i < n_points; i++)
      points[i][0] = (i + 0.5) / n_points;

    return points;
  }

  /**
   * Fill shape values
   *
   * @params shape_values     data structure to be filled
   */
  void
  fill_shape_values(AlignedVector<double> & shape_values)
  {
    // determine coefficients with deal.II functions
    FE_DGQ<1>          fe(points_source - 1);
    FullMatrix<double> matrix(points_target, points_source);
    FE_DGQArbitraryNodes<1>(Quadrature<1>(get_equidistant_inner(points_target)))
      .get_interpolation_matrix(fe, matrix);

    // ... and convert to linearized format
    shape_values.resize(points_source * points_target);
    for(unsigned int i = 0; i < points_source; ++i)
      for(unsigned int q = 0; q < points_target; ++q)
        shape_values[i * points_target + q] = matrix(q, i);
  }

  /**
   * Perform interpolation and permute dofs such that u, v, and w
   * for each point are grouped together. Use local source vector.
   */
  void
  interpolate()
  {
    double const * src = this->src;
    interpolate(src);
  }

  /**
   * Perform interpolation and permute dofs such that u, v, and w
   * for each point are grouped together. Source vector is explicitly given
   * by user.
   *
   * @params src      source vector (vector to be interpolated)
   */
  void
  interpolate(double const *& src)
  {
    // allocate dst- and src-vector
    AlignedVector<double> temp1(MAX(dofs_source, dofs_target));
    AlignedVector<double> temp2(MAX(dofs_source, dofs_target));

    // get start point of arrays
    double const * src_ = src;
    double *       dst_ = dst;
    // loop over all cells
    for(int c = 0; c < cells; c++)
    {
      // loop over all velocity directions
      for(int d = 0; d < DIM; d++)
      {
        // perform interpolation
        if(DIM == 2)
        {
          // ... 2D
          dealii::internal::
            EvaluatorTensorProduct<dealii::internal::evaluate_general, 2, 0, 0, double>
              eval_val(shape_values, shape_values, shape_values, points_source, points_target);
          eval_val.template values<0, true, false>(src_, temp2.begin());
          eval_val.template values<1, true, false>(temp2.begin(), temp1.begin());
        }
        else
        {
          // ... 3D
          dealii::internal::
            EvaluatorTensorProduct<dealii::internal::evaluate_general, 3, 0, 0, double>
              eval_val(shape_values, shape_values, shape_values, points_source, points_target);
          eval_val.template values<0, true, false>(src_, temp1.begin());
          eval_val.template values<1, true, false>(temp1.begin(), temp2.begin());
          eval_val.template values<2, true, false>(temp2.begin(), temp1.begin());
        }
        // write interpolated data permuted into destination vector
        for(int i = 0; i < dofs_target; i++)
          dst_[i * DIM + d] = temp1[i];
        // go to next dimension
        src_ += dofs_source;
      }
      // go to next cell
      dst_ += dofs_target * DIM;
    }
  }

  /**
   * Only for testing:
   * Do not perform interpolation and only permute dofs such that u, v, and w
   * for each point are grouped together. Use local source vector.
   */
  void
  interpolate_simple()
  {
    interpolate_simple(src);
  }

  /**
   * Only for testing:
   * Do not perform interpolation and only permute dofs such that u, v, and w
   * for each point are grouped together. Source vector is explicitly given
   * by user.
   *
   * @params src      source vector (vector to be interpolated)
   */
  void
  interpolate_simple(double *& src)
  {
    double * src_ = src;
    double * dst_ = dst;

    // loop over all cells
    for(int c = 0; c < cells; c++)
    {
      // loop over all velocity directions
      for(int d = 0; d < DIM; d++)
      {
        // write data permuted into destination vector
        for(int i = 0; i < dofs_target; i++)
          dst_[i * DIM + d] = src_[i];
        // go to next dimension
        src_ += dofs_source;
      }
      // go to next cell
      dst_ += dofs_target * DIM;
    }
  }

  /**
   * Read local source vector from file.
   *
   * @param filename  filename
   */
  void
  deserialize(char const * filename)
  {
    deserialize(filename, src);
  }

  /**
   * Read explicitly given source vector from file.
   *
   * @param filename  filename
   * @param src       source vector
   */
  void
  deserialize(char const * filename, double *& src)
  {
    io(0, filename, src);
  }

  /**
   * Write local source vector to file.
   *
   * @param filename  filename
   */
  void
  serialize(char const * filename)
  {
    double const * src = this->src;
    serialize(filename, src);
  }

  /**
   * Write explicitly given source vector to file.
   *
   * @param filename  filename
   * @param src       source vector
   */
  void
  serialize(char const * filename, double const *& src)
  {
    io(1, filename, src);
  }

  /**
   * Read/write explicitly given source vector to file.
   *
   * @param type      0: read; 1: write
   * @param filename  filename
   * @param src       source vector
   */
  template<typename VEC>
  void
  io(int type, char const * filename, VEC & src)
  {
    // dofs to read/write per field
    unsigned long int dofs = cells * dofs_source;
    // local displacement in file (in bytes)
    MPI_Offset disp =
      8 * sizeof(int) + static_cast<unsigned long int>(start) * dofs_source * sizeof(double) * DIM;

    // create view
    MPI_Datatype stype;
    MPI_Type_contiguous(dofs, MPI_DOUBLE, &stype);
    MPI_Type_commit(&stype);

    // ooen file ...
    MPI_File fh;
    if(type == 0)
      MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);
    else
      MPI_File_open(comm, filename, MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    // ... set view
    MPI_File_set_view(fh, disp, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);

    if(type == 0)
      // ... read file
      MPI_File_read_all(fh, (void *)src, dofs * DIM, MPI_DOUBLE, MPI_STATUSES_IGNORE);
    else
      // ... write file
      MPI_File_write_all(fh, src, dofs * DIM, MPI_DOUBLE, MPI_STATUSES_IGNORE);

    // ... close file
    MPI_File_close(&fh);

    // clean up
    MPI_Type_free(&stype);
  }
};

} // namespace dealspectrum

#endif
