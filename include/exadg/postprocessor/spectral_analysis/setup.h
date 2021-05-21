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

#ifndef DEAL_SPECTRUM_SETUP
#define DEAL_SPECTRUM_SETUP

// C/C++
#include <mpi.h>

// define helper funtions
#ifndef MIN
#  define MIN(a, b) (((a) < (b)) ? (a) : (b))
#endif

#ifndef MAX
#  define MAX(a, b) (((a) > (b)) ? (a) : (b))
#endif

namespace dealspectrum
{
/**
 * Class containing all settings of DEAL.SPECTRUM
 *
 * Aim of this class:
 *      (1) capsule the configuration of all modules
 *      (2) interface to IO: settings are included in header of files
 *
 * The header of the (binary) file consists of 8 integers:
 *      (0)   type: sfc vs. lexi
 *      (1)   dimension
 *      (2)   cells in each direction
 *      (3)   degree of cells
 *      (4-7) currently not used
 * The header is followed by the payload which is created by the specialized
 * classes.
 */
class Setup
{
public:
  // length of header (8 ints)
  static int const HEADER_LENGTH = 8;

  MPI_Comm const & comm;

  // is initialized?
  bool initialized;
  // file type (0) sfc + cell-wise (1) row-wise
  int type;
  // nr. of dimensions
  int dim;
  // nr. of cells in each direction
  int cells;

private:
  // degree of cell
  int degree;

public:
  // points in each direction in cell (i.e.: degree+1)
  int points_src;
  // points in each direction for evaluation in cell
  int points_dst;
  // mpi-rank
  int rank;
  // mpi-size
  int size;
  // nr. of bins for postprocessing
  int bins;
  // time stemp
  double time = 0.0;

  /**
   * Constructor
   */
  Setup(MPI_Comm const & comm) : comm(comm), initialized(false), points_dst(0)
  {
    // extract mpi-rank and ...
    MPI_Comm_rank(comm, &rank);
    // ... mpi-size
    MPI_Comm_size(comm, &size);
  }

  /**
   * Configure DEAL.SPECTRUM manually
   *
   * @param dim           dimension
   * @param cells         number of cells in each direction
   * @param points_src    points in each direction in cell (i.e.: degree+1)
   * @param points_dst    points in each direction for evaluation in cell
   */
  void
  init(int dim, int cells, int points_src, int points_dst)
  {
    this->initialized = true;

    this->type       = 0;
    this->dim        = dim;
    this->cells      = cells;
    this->degree     = points_src - 1;
    this->points_src = points_src;
    this->points_dst = points_dst;
    this->bins       = 1;
  }

  /**
   * Read header of file
   *
   * @param filename name of file
   */
  int
  readHeader(char *& filename)
  {
    FILE * fp;
    int    crit[1 + 4];
    crit[0] = 0;

    if(this->rank == 0)
    {
      // read header only by rank 0 ...
      if((fp = fopen(filename, "r")) != NULL)
      {
        // ... read header in one go
        fread(crit + 1, sizeof(int), 4, fp);

        // read time
        fread(&this->time, sizeof(double), 1, fp);

        // ... close file
        fclose(fp);
      }
      else
      {
        // ... reading the file failed
        crit[0] = 0;
      }
    }

    // broadcast header to all processes
    MPI_Bcast(&crit, 5, MPI_INT, 0, comm);
    MPI_Bcast(&this->time, 1, MPI_DOUBLE, 0, comm);

    if(this->initialized)
    {
      // ... extract header by each process and set properties
      crit[0] += this->type != crit[1];
      crit[0] += this->dim != crit[2];
      crit[0] += this->cells != crit[3];
      crit[0] += this->degree != crit[4];
    }
    else
    {
      // ... extract header by each process and set properties
      this->type       = crit[1];
      this->dim        = crit[2];
      this->cells      = crit[3];
      this->degree     = crit[4];
      this->points_src = this->degree + 1;
      if(this->points_dst == 0)
        this->points_dst = this->points_src; // equal if nothing specified
    }

    // success or failure?
    return crit[0];
  }

  /**
   * Write header of file
   *
   * @param filename name of file
   */
  void
  writeHeader(char const * filename)
  {
    FILE * fp;

    if(this->rank == 0)
    {
      // write header only by rank 0 ...
      if((fp = fopen(filename, "w")) != NULL)
      {
        // ... write type
        fwrite(&this->type, sizeof(int), 1, fp);
        // ... write dim
        fwrite(&this->dim, sizeof(int), 1, fp);

        if(this->type == 0)
        {
          // ... cell wise:
          fwrite(&this->cells, sizeof(int), 1, fp);
          fwrite(&this->degree, sizeof(int), 1, fp);
        }
        else
        {
          // ... row wise (degree is redundant):
          int temp1 = this->cells * this->points_dst;
          int temp2 = 0;
          fwrite(&temp1, sizeof(int), temp1, fp);
          fwrite(&temp2, sizeof(int), temp2, fp);
        }

        // write time
        fwrite(&this->time, sizeof(double), 1, fp);

        // ... close file
        fclose(fp);
      }
    }

    // synchronize all processes (not necessary)
    MPI_Barrier(comm);
  }
};

} // namespace dealspectrum

#endif
