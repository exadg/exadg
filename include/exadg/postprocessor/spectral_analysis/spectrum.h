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

#ifndef DEAL_SPECTRUM_SPECTRUM
#define DEAL_SPECTRUM_SPECTRUM

#include <fftw3-mpi.h>
#include <mpi.h>
#include <cmath>

// ExaDG
#include <exadg/postprocessor/spectral_analysis/setup.h>

// shortcuts for accessing linearized arrays
#define u_comp2(i, j) u_comp[(j - local_start) * (N / 2 + 1) + i]
#define v_comp2(i, j) v_comp[(j - local_start) * (N / 2 + 1) + i]
#define u_comp3(i, j, k) u_comp[((k - local_start) * N + j) * (N / 2 + 1) + i]
#define v_comp3(i, j, k) v_comp[((k - local_start) * N + j) * (N / 2 + 1) + i]
#define w_comp3(i, j, k) w_comp[((k - local_start) * N + j) * (N / 2 + 1) + i]

namespace dealspectrum
{
/**
 * Class wrapping FFTW and performing energy spectral analysis.
 */
class SpectralAnalysis
{
  MPI_Comm const comm;

  // reference to DEAL.SPECTRUM setup
  Setup & s;
  // is initialized?
  bool initialized;
  // rank of process which owns row
  int * _indices_proc_rows;

public:
  /**
   * Constructor
   * @param s DEAL.SPECTRUM setup
   */
  SpectralAnalysis(MPI_Comm const & comm, Setup & s) : comm(comm), s(s), initialized(false)
  {
  }

  /**
   * Determines rank of process owning specified row (2D) or plane (3D)
   *
   * @param i     position on sfc
   * @return      rank of process owning cell
   */
  inline int
  indices_proc_rows(int i)
  {
    return _indices_proc_rows[i];
  }

  /**
   * Get process local range of rows/plane this process owns
   *
   * @param start     start point
   * @param end       end point
   */
  void
  getLocalRange(int & start, int & end)
  {
    start = local_start;
    end   = local_end;
  }

  /**
   * Initialize data structures
   */
  void
  init()
  {
    // check if already initialized
    if(this->initialized)
      return;
    this->initialized = true;

    // extract settings
    this->N    = s.cells * s.points_dst;
    this->dim  = s.dim;
    this->rank = s.rank;
    this->size = s.size;
    this->bins = s.bins;

    // setup global size of output arrays...
    n = new ptrdiff_t[dim];
    for(int i = 0; i < dim - 1; i++)
      n[i] = N;
    n[dim - 1] = N / 2 + 1;

    // ...get local size of local output arrays
    ptrdiff_t local_elements = 0;
    alloc_local              = fftw_mpi_local_size(dim, n, comm, &local_elements, &local_start);
    local_end                = local_start + local_elements;

    // determine how many rows each process has
    int * global_elements = new int[size];
    MPI_Allgather(&local_elements, 1, MPI_INTEGER, global_elements, 1, MPI_INTEGER, comm);

    // ... save for each row by whom it is owned
    _indices_proc_rows = new int[N];

    for(int i = 0, c = 0; i < size; i++)
      for(int j = 0; j < global_elements[i]; j++, c++)
        _indices_proc_rows[c] = i;

    // ... clean up
    delete[] global_elements;

    // modify global size for input array
    n[dim - 1] = N;

    // allocate memory
    // ... for input array (real) - allocated together for all directions
    u_real = fftw_alloc_real(2 * alloc_local * dim);
    // ... and save required size
    this->bsize = 2 * alloc_local;

    // initialize input array with zero (not needed: only useful for IO -> hard zero)
    for(int i = 0; i < 2 * alloc_local * dim; i++)
      u_real[i] = 0;

    // set pointer for v input field
    v_real = u_real + 2 * alloc_local;

    // allocate memory for output array (complex)
    u_comp = fftw_alloc_complex(alloc_local);
    v_comp = fftw_alloc_complex(alloc_local);

    // do the same for 3D
    if(dim == 3)
    {
      w_real = v_real + 2 * alloc_local;
      w_comp = fftw_alloc_complex(alloc_local);
    }

    // allocate memory and ...
    this->e = new double[N];
    this->E = new double[N];
    this->k = new double[N];
    this->K = new double[N];
    this->c = new double[N];
    this->C = new double[N];
  }

  /**
   * Destructor
   */
  ~SpectralAnalysis()
  {
    // not initialized -> nothing to clean up
    if(not initialized)
      return;

    // free data structures
    delete[] _indices_proc_rows;

    free(n);
    fftw_free(u_comp);
    fftw_free(v_comp);
    free(u_real);

    if(dim == 3)
    {
      fftw_free(w_comp);
    }

    delete e;
    delete E;
    delete k;
    delete K;
    delete c;
    delete C;
  }

  /**
   * Perform FFT with FFTW
   */
  void
  execute()
  {
    // perform FFT for u
    fftw_plan pu = fftw_mpi_plan_dft_r2c(dim, n, u_real, u_comp, comm, FFTW_ESTIMATE);
    fftw_execute(pu);
    fftw_destroy_plan(pu);

    // ... for v
    fftw_plan pv = fftw_mpi_plan_dft_r2c(dim, n, v_real, v_comp, comm, FFTW_ESTIMATE);
    fftw_execute(pv);
    fftw_destroy_plan(pv);

    // ... for w
    if(dim == 3)
    {
      fftw_plan pw = fftw_mpi_plan_dft_r2c(dim, n, w_real, w_comp, comm, FFTW_ESTIMATE);
      fftw_execute(pw);
      fftw_destroy_plan(pw);
    }
  }

  void
  calculate_energy()
  {
    double scaling    = pow(N, dim);
    double e_physical = 0.0, e_spectral = 0.0;

    for(int i = 0; i < 2 * alloc_local * dim; i++)
    {
      e_physical += u_real[i] * u_real[i];
    }

    // scale: integrate cell wise...
    e_physical /= pow(N, dim);
    // ... and make to energy 0.5*u^2
    e_physical *= 0.5;

    if(dim == 2)
    {
      for(int j = local_start; j < local_end; j++)
      {
        for(int i = 0; i < N; i++)
        {
          e_spectral += u_comp2(MIN(i, N - i), j)[0] * u_comp2(MIN(i, N - i), j)[0] +
                        u_comp2(MIN(i, N - i), j)[1] * u_comp2(MIN(i, N - i), j)[1] +
                        v_comp2(MIN(i, N - i), j)[0] * v_comp2(MIN(i, N - i), j)[0] +
                        v_comp2(MIN(i, N - i), j)[1] * v_comp2(MIN(i, N - i), j)[1];
        }
      }
    }
    else if(dim == 3)
    {
      for(int k_ = local_start; k_ < local_end; k_++)
      {
        for(int j = 0; j < N; j++)
        {
          for(int i = 0; i < N; i++)
          {
            e_spectral += u_comp3(MIN(i, N - i), j, k_)[0] * u_comp3(MIN(i, N - i), j, k_)[0] +
                          u_comp3(MIN(i, N - i), j, k_)[1] * u_comp3(MIN(i, N - i), j, k_)[1] +
                          v_comp3(MIN(i, N - i), j, k_)[0] * v_comp3(MIN(i, N - i), j, k_)[0] +
                          v_comp3(MIN(i, N - i), j, k_)[1] * v_comp3(MIN(i, N - i), j, k_)[1] +
                          w_comp3(MIN(i, N - i), j, k_)[0] * w_comp3(MIN(i, N - i), j, k_)[0] +
                          w_comp3(MIN(i, N - i), j, k_)[1] * w_comp3(MIN(i, N - i), j, k_)[1];
          }
        }
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    // scale: due to FFT...
    e_spectral /= scaling * scaling;
    // ... and make to energy 0.5*u^2
    e_spectral *= 0.5;

    MPI_Reduce(&e_physical, &this->e_d, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(&e_spectral, &this->e_s, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
  }

  /**
   * Perform spectral analysis
   */
  void
  calculate_energy_spectrum()
  {
    double scaling = pow(N, dim);

    // ... init with zero
    for(int i = 0; i < N; i++)
    {
      e[i] = 0;
      k[i] = 0;
      c[i] = 0;
    }

    // collect energy for local domain...
    if(dim == 2)
    {
      for(int j = local_start; j < local_end; j++)
      {
        for(int i = 0; i < N; i++)
        {
          // determine wavenumber...
          double r = sqrt(pow(MIN(i, N - i), 2.0) + pow(MIN(j, N - j), 2.0));
          // ... use for binning
          int p = static_cast<int>(std::round(r));
          // ... update energy
          e[p] += u_comp2(MIN(i, N - i), j)[0] * u_comp2(MIN(i, N - i), j)[0] +
                  u_comp2(MIN(i, N - i), j)[1] * u_comp2(MIN(i, N - i), j)[1] +
                  v_comp2(MIN(i, N - i), j)[0] * v_comp2(MIN(i, N - i), j)[0] +
                  v_comp2(MIN(i, N - i), j)[1] * v_comp2(MIN(i, N - i), j)[1];

          // ... update kappa results
          k[p] += r;
          c[p]++;
        }
      }
    }
    else if(dim == 3)
    {
      for(int k_ = local_start; k_ < local_end; k_++)
      {
        for(int j = 0; j < N; j++)
        {
          for(int i = 0; i < N; i++)
          {
            // determine wavenumber...
            double r =
              sqrt(pow(MIN(i, N - i), 2.0) + pow(MIN(j, N - j), 2.0) + pow(MIN(k_, N - k_), 2.0));
            // ... use for binning
            int p = static_cast<int>(std::round(r));
            // ... update energy
            e[p] += u_comp3(MIN(i, N - i), j, k_)[0] * u_comp3(MIN(i, N - i), j, k_)[0] +
                    u_comp3(MIN(i, N - i), j, k_)[1] * u_comp3(MIN(i, N - i), j, k_)[1] +
                    v_comp3(MIN(i, N - i), j, k_)[0] * v_comp3(MIN(i, N - i), j, k_)[0] +
                    v_comp3(MIN(i, N - i), j, k_)[1] * v_comp3(MIN(i, N - i), j, k_)[1] +
                    w_comp3(MIN(i, N - i), j, k_)[0] * w_comp3(MIN(i, N - i), j, k_)[0] +
                    w_comp3(MIN(i, N - i), j, k_)[1] * w_comp3(MIN(i, N - i), j, k_)[1];

            // ... update kappa results
            k[p] += r;
            c[p]++;
          }
        }
      }
    }

    // ... sum up local results to global result
    MPI_Reduce(e, E, N, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(k, K, N, MPI_DOUBLE, MPI_SUM, 0, comm);
    MPI_Reduce(c, C, N, MPI_DOUBLE, MPI_SUM, 0, comm);

    // ... normalize results
    for(int i = 0; i < N; i++)
    {
      // ... average energy and kappa
      K[i] /= C[i];
      // ... factor from fft
      E[i] /= scaling * scaling;
      // ... and make to energy 0.5*u^2
      E[i] *= 0.5;
    }
  }

  /**
   * Access spectral analysis results in table format
   *
   * @param kappa     kappa
   * @param E         energy
   * @return          length of table
   */
  int
  get_results(double *& K, double *& E, double *& C, double & e_d, double & e_s)
  {
    K   = this->K;
    E   = this->E;
    C   = this->C;
    e_d = this->e_d;
    e_s = this->e_s;
    return N / 2 + 1;
  }

  /**
   * Write arrays to file
   *
   * @param filename name of file
   */
  void
  serialize(char const * filename)
  {
    int start     = local_start;
    int end       = local_end;
    int delta     = 2 * (N / 2 + 1) * dealii::Utilities::pow(N, dim - 2);
    int delta_all = 2 * (N / 2 + 1) * dealii::Utilities::pow(N, dim - 1) * sizeof(double);

    //
    int        dofs = (end - start) * delta;
    MPI_Offset disp = 8 * sizeof(int) + start * delta * sizeof(double); // bytes

    // create view
    MPI_Datatype stype;
    MPI_Type_contiguous(dofs, MPI_DOUBLE, &stype);
    MPI_Type_commit(&stype);

    // read file
    MPI_File fh;
    MPI_File_open(comm, filename, MPI_MODE_WRONLY, MPI_INFO_NULL, &fh);

    MPI_File_set_view(fh, disp + delta_all * 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, u_real, dofs, MPI_DOUBLE, MPI_STATUSES_IGNORE);

    MPI_File_set_view(fh, disp + delta_all * 1, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_write_all(fh, v_real, dofs, MPI_DOUBLE, MPI_STATUSES_IGNORE);

    if(dim == 3)
    {
      MPI_File_set_view(fh, disp + delta_all * 2, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
      MPI_File_write_all(fh, w_real, dofs, MPI_DOUBLE, MPI_STATUSES_IGNORE);
    }

    MPI_File_close(&fh);
    MPI_Type_free(&stype);
  }

  /**
   * Read arrays from file
   *
   * @param filename name of file
   */
  void
  deserialize(char *& filename)
  {
    int start     = local_start;
    int end       = local_end;
    int delta     = 2 * (N / 2 + 1) * dealii::Utilities::pow(N, dim - 2);
    int delta_all = 2 * (N / 2 + 1) * dealii::Utilities::pow(N, dim - 1) * sizeof(double);

    //
    int        dofs = (end - start) * delta;
    MPI_Offset disp = 8 * sizeof(int) + start * delta * sizeof(double); // bytes

    // create view
    MPI_Datatype stype;
    MPI_Type_contiguous(dofs, MPI_DOUBLE, &stype);
    MPI_Type_commit(&stype);

    // read file
    MPI_File fh;
    MPI_File_open(comm, filename, MPI_MODE_RDONLY, MPI_INFO_NULL, &fh);

    MPI_File_set_view(fh, disp + delta_all * 0, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, u_real, dofs, MPI_DOUBLE, MPI_STATUSES_IGNORE);

    MPI_File_set_view(fh, disp + delta_all * 1, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
    MPI_File_read_all(fh, v_real, dofs, MPI_DOUBLE, MPI_STATUSES_IGNORE);

    if(dim == 3)
    {
      MPI_File_set_view(fh, disp + delta_all * 2, MPI_DOUBLE, MPI_DOUBLE, "native", MPI_INFO_NULL);
      MPI_File_read_all(fh, w_real, dofs, MPI_DOUBLE, MPI_STATUSES_IGNORE);
    }

    MPI_File_close(&fh);
    MPI_Type_free(&stype);
  }

private:
  // number of dofs in each direction
  int N;
  // dimensions
  int dim;
  // rank of this process
  int rank;
  // number of processes
  int size;
  // bin count
  int bins;
  // number of dofs in each direction (for FFTW)
  ptrdiff_t * n;
  // local row/plane range: start
  ptrdiff_t local_start;
  // ... end
  ptrdiff_t local_end;
  ptrdiff_t alloc_local;

public:
  // size of each real field
  int bsize;
  // real field for u
  double * u_real;

private:
  // ... for v
  double * v_real;
  // ... for w
  double * w_real;
  // complex field for u
  fftw_complex * u_comp;
  // ... for v
  fftw_complex * v_comp;
  // ... for w
  fftw_complex * w_comp;

private:
  // array for locally collecting energy
  double * e;
  // ... kappa
  double * k;
  // ... kappa count
  double * c;
  // array for globally collecting energy (on rank 0)
  double * E;
  // ... kappa
  double * K;
  // ... kappa count
  double * C;
  double   e_d;
  double   e_s;
};

} // namespace dealspectrum

#endif
