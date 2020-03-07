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
    : comm(comm), write(write), inplace(inplace), s(comm), h(comm,s), ipol(comm, s), fftc(comm, s), fftw(comm, s)
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
  init(int dim, int n_cells_1D, int points_src, int points_dst, Tria & tria)
  {
    // init setup ...
    s.init(dim, n_cells_1D, points_src, points_dst);

    // ... mapper
    timer.start("Init-Map");
    h.init(tria);
    timer.stop("Init-Map");
    
    std::vector<types::global_dof_index> local_cells;
    
    auto norm_point_to_lex = [&](const auto c) {
      // convert normalized point [0, 1] to lex
      if (dim == 2)
        return std::floor(c[0]) + n_cells_1D * std::floor(c[1]);
      else
        return std::floor(c[0]) + n_cells_1D * std::floor(c[1]) +
               n_cells_1D * n_cells_1D * std::floor(c[2]);
    };
    
    for (const auto &cell : tria.active_cell_iterators())
      if (cell->is_active() && cell->is_locally_owned())
        {
          auto c = cell->center();
          for (int i = 0; i < dim; i++)
            c[i] = (c[i]+dealii::numbers::PI) / (2 * dealii::numbers::PI / n_cells_1D);
    
          local_cells.push_back(norm_point_to_lex(c));
    
        }
    
    //for(auto i : local_cells)
    //    std::cout << i << std::endl;
    //std::cout << std::endl << std::endl;
    
    int n_local_cells = local_cells.size();
    int global_offset = 0;
    
    MPI_Exscan(&n_local_cells, &global_offset, 1, MPI_INT, MPI_SUM, comm);
    
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
    
    int start;
    int end;
    fftw.getLocalRange(start, end);
    
    std::vector<types::global_dof_index> indices_has, indices_want;
    
    for(const auto & I : local_cells)
      for(unsigned int i = 0; i < Utilities::pow(points_dst, dim); i++)
        for(types::global_dof_index d =0; d  < dim ; d++)
        {
            
            types::global_dof_index index = 
                    (I / (n_cells_1D * n_cells_1D) * points_dst + i / (points_dst * points_dst)) * Utilities::pow(n_cells_1D * points_dst, 2) + 
                    (((I / n_cells_1D) % n_cells_1D) * points_dst + ((i / points_dst) % points_dst)) * Utilities::pow(n_cells_1D * points_dst, 1) + 
                    (I % (n_cells_1D             ) * points_dst + i % (points_dst             )); 
                    
            
            //std::cout << index << std::endl;
            indices_has.push_back(d * Utilities::pow(points_dst * n_cells_1D, dim) + index);
        }
    
    for(types::global_dof_index d =0; d  < dim ; d++)
      for(int j = start; j < end; j++)
        for(unsigned int i = 0; i < Utilities::pow(points_dst*n_cells_1D, dim-1); i++)
            indices_want.push_back(d * Utilities::pow(points_dst * n_cells_1D, dim) + j * Utilities::pow(points_dst*n_cells_1D, dim-1) + i);
    
    nonconti = std::make_shared<Utilities::MPI::NoncontiguousPartitioner<double>>(indices_has, indices_want, comm);

    // ... permutation
    timer.start("Init-Perm");
    fftc.init(h, fftw);
    timer.stop("Init-Perm");
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
      ArrayView<double> dst(fftw.u_real, s.dim * Utilities::pow(static_cast<types::global_dof_index>(s.cells * s.points_dst), s.dim) *2);
      ArrayView<double> src_(ipol.dst, s.dim * Utilities::pow(static_cast<types::global_dof_index>(s.cells * s.points_dst), s.dim));
      nonconti->update_values(dst, src_);
      
      types::global_dof_index N  = s.cells * s.points_dst;
      types::global_dof_index Nx = (N / 2 + 1) * 2;
      
      int start;
      int end;
      fftw.getLocalRange(start, end);
      
      //for (int k = 0; k < (end - start)*s.dim ; k++)
      //  for (int j = 0; j < N; j++)
      //    for (int i = 0; i < N; i++)
      //      std::cout << dst[N * N * k + N * j + i] << " ";
      //  
      //std::cout << std::endl << std::endl;
      
      for (int k = (end - start)*s.dim - 1; k >= 0 ; k--)
        for (int j = N - 1; j >= 0; j--)
          for (int i = N-1; i >= 0; i--)
            dst[N * Nx * k + Nx * j + i] = dst[N * N * k + N * j + i];
      
      //for (int k = 0; k < (end - start)*s.dim ; k++)
      //  for (int j = 0; j < N; j++)
      //    for (int i = 0; i < N; i++)
      //      std::cout << dst[N * Nx * k + Nx * j + i] << " ";
      //  
      //std::cout << std::endl << std::endl;
      
      //fftc.ipermute(ipol.dst, fftw.u_real);
      //fftc.iwait();
      
      //for (int k = 0; k < (end - start)*s.dim ; k++)
      //  for (int j = 0; j < N; j++)
      //    for (int i = 0; i < N; i++)
      //      std::cout << dst[N * Nx * k + Nx * j + i] << " ";
      //  
      //std::cout << std::endl << std::endl;
        
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

  // class for translating: lexicographical and morton order cell numbering
  Bijection h;

  // ... for interpolation
  Interpolator ipol;

  // ... for permutation
  Permutator fftc;

  // ... for spectral analysis
  SpectralAnalysis fftw;

  // Timer
  DealSpectrumTimer timer;
  
  std::shared_ptr<Utilities::MPI::NoncontiguousPartitioner<double>> nonconti;
};

} // namespace dealspectrum

#endif
