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
  DealSpectrumWrapper(bool write, bool inplace)
    : write(write), inplace(inplace), s(), h(s), ipol(s), fftc(s), fftw(s)
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
  init(int dim, int cells, int points_src, int points_dst, Tria & tria)
  {
    // init setup ...
    s.init(dim, cells, points_src, points_dst);

    // ... mapper
    timer.start("Init-Map");
    h.init(tria);
    timer.stop("Init-Map");

    // ... fftw
    timer.start("Init-FFTW");
    fftw.init();
    timer.stop("Init-FFTW");

    // ... interpolation
    timer.start("Init-Ipol");
    ipol.init(h);
    timer.stop("Init-Ipol");

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
  execute(const double * src)
  {
    if(write)
    {
      // flush flow field to hard drive
      s.writeHeader("TODO");
      ipol.serialize("TODO", src);
    }

    if(inplace)
    {
      // compute energy spectrum: interpolate ...
      timer.start("Interpolation");
      ipol.interpolate(src);
      timer.append("Interpolation");

      // ... permute
      timer.start("Permutation");
      fftc.ipermute(ipol.dst, fftw.u_real);
      fftc.iwait();
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
};

} // namespace dealspectrum

#endif
