/*
 * energy_spectrum_calculation.h
 *
 *  Created on: Feb 7, 2018
 *      Author: fehn/muench
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_

#include <deal.II/lac/la_parallel_vector.h>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include "incompressible_navier_stokes/postprocessor/kinetic_energy_spectrum_data.h"
#include "incompressible_navier_stokes/postprocessor/postprocessor_base.h"

#ifdef USE_DEAL_SPECTRUM
#  include "../../../3rdparty/deal.spectrum/src/deal-spectrum.h"
#else
namespace dealspectrum
{
class DealSpectrumWrapper
{
public:
  DealSpectrumWrapper(bool, bool)
  {
  }

  virtual ~DealSpectrumWrapper()
  {
  }

  template<typename T>
  void
  init(int, int, int, int, T &)
  {
  }

  void
  execute(const double *)
  {
  }

  int
  get_results(double *&, double *&, double *&, double &, double &)
  {
    return 0;
  }
};

} // namespace dealspectrum
#endif

template<int dim, typename Number>
class KineticEnergySpectrumCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  KineticEnergySpectrumCalculator()
    : clear_files(true), deal_spectrum_wrapper(false, true), counter(0), reset_counter(true)
  {
  }

  void
  setup(MatrixFree<dim, Number> const &   matrix_free_data_in,
        Triangulation<dim, dim> const &   tria,
        KineticEnergySpectrumData const & data_in)
  {
    data = data_in;

    if(data.calculate == true)
    {
      int local_cells = matrix_free_data_in.n_physical_cells();
      int cells       = local_cells;
      MPI_Allreduce(MPI_IN_PLACE, &cells, 1, MPI_INTEGER, MPI_SUM, MPI_COMM_WORLD);
      cells = round(pow(cells, 1.0 / dim));

      unsigned int evaluation_points = std::max(data.degree + 1, data.evaluation_points_per_cell);

      deal_spectrum_wrapper.init(dim, cells, data.degree + 1, evaluation_points, tria);
    }
  }

  void
  evaluate(VectorType const & velocity, double const & time, int const & time_step_number)
  {
    if(data.calculate == true)
    {
      if(time_step_number >= 0) // unsteady problem
      {
        do_evaluate(velocity, time, time_step_number);
      }
      else // steady problem (time_step_number = -1)
      {
        AssertThrow(
          false,
          ExcMessage(
            "Calculation of kinetic energy spectrum only implemented for unsteady problems."));
      }
    }
  }

private:
  void
  do_evaluate(VectorType const & velocity, double const time, unsigned int const time_step_number)
  {
    bool evaluate = false;

    if(data.calculate_every_time_steps > 0)
    {
      if(time > data.start_time && (time_step_number - 1) % data.calculate_every_time_steps == 0)
      {
        evaluate = true;
      }

      AssertThrow(data.calculate_every_time_interval < 0.0,
                  ExcMessage("Input parameters are in conflict."));
    }
    else if(data.calculate_every_time_interval > 0.0)
    {
      // small number which is much smaller than the time step size
      const double EPSILON = 1.0e-10;

      // The current time might be larger than output_start_time. In that case, we first have to
      // reset the counter in order to avoid that output is written every time step.
      if(reset_counter)
      {
        counter += int((time - data.start_time + EPSILON) / data.calculate_every_time_interval);
        reset_counter = false;
      }

      if((time > (data.start_time + counter * data.calculate_every_time_interval - EPSILON)))
      {
        evaluate = true;
        ++counter;
      }

      AssertThrow(data.calculate_every_time_steps < 0,
                  ExcMessage("Input parameters are in conflict."));
    }
    else
    {
      AssertThrow(false,
                  ExcMessage(
                    "Invalid parameters specified. Use either "
                    "calculate_every_time_interval > 0.0 or calculate_every_time_steps > 0."));
    }

    if(evaluate)
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        std::cout << std::endl
                  << "Calculate kinetic energy spectrum at time t = " << time << ":" << std::endl;

      // extract beginning of vector...
      const double * temp = velocity.begin();
      deal_spectrum_wrapper.execute(temp);

      // write output file
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ostringstream filename;
        filename << data.filename_prefix;

        std::ofstream f;
        if(clear_files == true)
        {
          f.open(filename.str().c_str(), std::ios::trunc);
          clear_files = false;
        }
        else
        {
          f.open(filename.str().c_str(), std::ios::app);
        }

        // get tabularized results ...
        double * kappa;
        double * E;
        double * C;
        double   e_physical = 0.0;
        double   e_spectral = 0.0;
        int len = deal_spectrum_wrapper.get_results(kappa, E, C /*unused*/, e_physical, e_spectral);

        f << std::endl
          << "Calculate kinetic energy spectrum at time t = " << time << ":" << std::endl
          << std::scientific << std::setprecision(precision) << std::setw(precision + 8)
          << std::endl
          << "  Energy physical space e_phy = " << e_physical << std::endl
          << "  Energy spectral space e_spe = " << e_spectral << std::endl
          << "  Difference  |e_phy - e_spe| = " << std::abs(e_physical - e_spectral) << std::endl
          << std::endl
          << "    k  k (avg)              E(k)" << std::endl;

        // ... and print it line by line:
        for(int i = 0; i < len; i++)
        {
          if(E[i] > data.output_tolerance)
          {
            f << std::scientific << std::setprecision(0)
              << std::setw(2 + ceil(std::max(3.0, log(len) / log(10)))) << i << std::scientific
              << std::setprecision(precision) << std::setw(precision + 8) << kappa[i] << "   "
              << E[i] << std::endl;
          }
        }
      }
    }
  }

  bool                              clear_files;
  dealspectrum::DealSpectrumWrapper deal_spectrum_wrapper;
  KineticEnergySpectrumData         data;
  unsigned int                      counter;
  bool                              reset_counter;
  const unsigned int                precision = 12;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_ */
