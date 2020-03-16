/*
 * <DEAL.SPECTRUM>/deal-spectrum.h
 *
 *  Created on: Mar 02, 2018
 *      Author: muench
 */

#ifndef DEAL_SPECTRUM
#define DEAL_SPECTRUM

// includes: all DEAL.SPECTRUM modules
#include "./includes/bijection.h"
#include "./includes/interpolation.h"
#include "./includes/permutation.h"
#include "./includes/setup.h"
#include "./includes/spectrum.h"
#include "./util/timer.h"

#include <deal.II/base/mpi.templates.h>
#include <deal.II/base/mpi_noncontiguous_partitioner.h>

namespace dealspectrum
{
/**
 * Class ordering the calls of the deal.spectrum-components in a meaningful order
 * such that the user can simply integrate in his/her own application.
 *
 * Purpose is twofold:
 *      (1) create files which can be processed by the DEAL.SPECTRUM standalone
 *          application
 *      (2) post process on the fly
 */
class DealSpectrumWrapper
{
public:
  /**
   * Constructor
   *
   * @param write     flush simulation data to hard drive for later post processing
   * @param inplace   create energy spectrum at run time
   *
   */
  DealSpectrumWrapper(MPI_Comm const & comm, bool write, bool inplace)
    : comm(comm), write(write), inplace(inplace), s(comm), ipol(comm, s), fftw(comm, s)
  {
  }

  virtual ~DealSpectrumWrapper()
  {
  }


  /**
   * Initialize data structures
   *
   * @param dim           number of dimension
   * @param cells         number of cells in each direction
   * @param points_src    number of Gauss-Lobatto points (order + 1)
   * @param points_dst    number of equidisant points (for post processing)
   * @param local_cells   number of cells this process owns
   *
   */
  template<class Tria>
  void
  init(types::global_dof_index dim,
       types::global_dof_index n_cells_1D,
       types::global_dof_index points_src,
       types::global_dof_index points_dst,
       Tria &                  tria)
  {
    // init setup ...
    s.init(dim, n_cells_1D, points_src, points_dst);
    
    std::vector<types::global_dof_index> local_cells;

    auto norm_point_to_lex = [&](const auto c) {
      // convert normalized point [0, 1] to lex
      if(dim == 2)
        return std::floor(c[0]) + n_cells_1D * std::floor(c[1]);
      else
        return std::floor(c[0]) + n_cells_1D * std::floor(c[1]) +
               n_cells_1D * n_cells_1D * std::floor(c[2]);
    };

    for(const auto & cell : tria.active_cell_iterators())
      if(cell->is_active() && cell->is_locally_owned())
      {
        auto c = cell->center();
        for(types::global_dof_index i = 0; i < dim; i++)
          c[i] = (c[i] + dealii::numbers::PI) / (2 * dealii::numbers::PI / n_cells_1D);

        local_cells.push_back(norm_point_to_lex(c));
      }

    types::global_dof_index n_local_cells = local_cells.size();
    types::global_dof_index global_offset = 0;

    MPI_Exscan(&n_local_cells,
               &global_offset,
               1,
               Utilities::MPI::internal::mpi_type_id(&global_offset),
               MPI_SUM,
               comm);

    // ... interpolation
    timer.start("Init-Ipol");
    ipol.init(global_offset, global_offset + n_local_cells);
    timer.stop("Init-Ipol");

    // FFTW will be called externally
    if(!inplace)
      return;

    // ... fftw
    timer.start("Init-FFTW");
    fftw.init();
    timer.stop("Init-FFTW");

    int start_;
    int end_;
    fftw.getLocalRange(start_, end_);
    const types::global_dof_index start = start_;
    const types::global_dof_index end   = end_;

    std::vector<types::global_dof_index> indices_has, indices_want;

    for(const auto & I : local_cells)
      for(types::global_dof_index i = 0; i < Utilities::pow(points_dst, dim); i++)
        for(types::global_dof_index d = 0; d < dim; d++)
        {
          types::global_dof_index index =
            (I / (n_cells_1D * n_cells_1D) * points_dst + i / (points_dst * points_dst)) *
              Utilities::pow(n_cells_1D * points_dst, 2) +
            (((I / n_cells_1D) % n_cells_1D) * points_dst + ((i / points_dst) % points_dst)) *
              Utilities::pow(n_cells_1D * points_dst, 1) +
            (I % (n_cells_1D)*points_dst + i % (points_dst));

          indices_has.push_back(d * Utilities::pow(points_dst * n_cells_1D, dim) + index);
        }

    types::global_dof_index N  = s.cells * s.points_dst;
    types::global_dof_index Nx = (N / 2 + 1) * 2;
    
    for(types::global_dof_index d = 0; d < static_cast<types::global_dof_index>(s.dim); d++)
    {
      types::global_dof_index c = 0;
      for(types::global_dof_index k = 0; k < (end - start); k++)
        for(types::global_dof_index j = 0; j < N; j++)
          for(types::global_dof_index i = 0; i < Nx; i++, c++)
            if(i < N )
              indices_want.push_back(d * Utilities::pow(points_dst * n_cells_1D, dim) + (k+start) * Utilities::pow(points_dst * n_cells_1D, 2) + j * Utilities::pow(points_dst * n_cells_1D, 1) + i);
            else
              indices_want.push_back(numbers::invalid_dof_index); // x-padding
        
      for(; c < static_cast<types::global_dof_index>(fftw.bsize) ; c++)
        indices_want.push_back(numbers::invalid_dof_index); // z-padding
    }

    nonconti = std::make_shared<Utilities::MPI::NoncontiguousPartitioner<double>>(indices_has,
                                                                                  indices_want,
                                                                                  comm);
  }

  /**
   * Process current velocity field:
   *      - flush to hard drive
   *      - compute energy spectrum
   *
   * @param src   current velocity field
   *
   */
  void
  execute(const double * src, const std::string & file_name = "", const double time = 0.0)
  {
    if(write)
    {
      // flush flow field to hard drive
      AssertThrow(file_name != "", ExcMessage("No file name has been provided!"));
      s.time = time;
      s.writeHeader(file_name.c_str());
      ipol.serialize(file_name.c_str(), src);
    }

    if(inplace)
    {
      // compute energy spectrum: interpolate ...
      timer.start("Interpolation");
      ipol.interpolate(src);
      timer.append("Interpolation");

      // ... permute
      timer.start("Permutation");
      ArrayView<double> dst(
        fftw.u_real,
        s.dim *
          Utilities::pow(static_cast<types::global_dof_index>(s.cells * s.points_dst), s.dim) * 2);
      ArrayView<double> src_(ipol.dst,
                             s.dim * Utilities::pow(static_cast<types::global_dof_index>(
                                                      s.cells * s.points_dst),
                                                    s.dim));
      nonconti->update_values(dst, src_);

      timer.append("Permutation");

      // ... fft
      timer.start("FFT");
      fftw.execute();
      timer.append("FFT");

      // ... spectral analysis
      timer.start("Postprocessing");
      fftw.calculate_energy_spectrum();
      fftw.calculate_energy();
      timer.append("Postprocessing");
    }
  }

  /**
   * Return spectral data in a table format
   *
   * @param kappa     wave length
   * @param E         energy
   * @return length of table
   *
   */
  int
  get_results(double *& K, double *& E, double *& C, double & e_d, double & e_s)
  {
    // simply return values from the spectral analysis tool
    return this->fftw.get_results(K, E, C, e_d, e_s);
  }

  void
  print_timings()
  {
    if(s.rank == 0)
      timer.printTimings();
  }

private:
  MPI_Comm const & comm;

  // flush flow field to hard drive?
  const bool write;

  // perform spectral analysis
  const bool inplace;

  // struct containing the setup
  Setup s;
  
  // ... for interpolation
  Interpolator ipol;
  
  // ... for spectral analysis
  SpectralAnalysis fftw;

  // Timer
  DealSpectrumTimer timer;

  std::shared_ptr<Utilities::MPI::NoncontiguousPartitioner<double>> nonconti;
};

} // namespace dealspectrum

#endif
