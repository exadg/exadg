/*
 * kinetic_energy_spectrum.cpp
 *
 *  Created on: May 17, 2019
 *      Author: fehn
 */

// C/C++
#include <fstream>

// ExaDG
#include <exadg/postprocessor/kinetic_energy_spectrum.h>
#include <exadg/postprocessor/mirror_dof_vector_taylor_green.h>

#ifdef USE_DEAL_SPECTRUM
#  include "../../../3rdparty/deal.spectrum/src/deal-spectrum.h"
#else

namespace dealspectrum
{
class DealSpectrumWrapper
{
public:
  DealSpectrumWrapper(MPI_Comm const &, bool, bool)
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
  execute(const double *, const std::string & = "", const double = 0.0)
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

static std::shared_ptr<dealspectrum::DealSpectrumWrapper> deal_spectrum_wrapper;

namespace ExaDG
{
template<int dim, typename Number>
KineticEnergySpectrumCalculator<dim, Number>::KineticEnergySpectrumCalculator(MPI_Comm const & comm)
  : mpi_comm(comm), clear_files(true), counter(0), reset_counter(true)
{
}

template<int dim, typename Number>
void
KineticEnergySpectrumCalculator<dim, Number>::setup(
  MatrixFree<dim, Number> const &   matrix_free_data_in,
  DoFHandler<dim> const &           dof_handler_in,
  KineticEnergySpectrumData const & data_in)
{
  data        = data_in;
  dof_handler = &dof_handler_in;

  clear_files = data.clear_file;

  if(data.calculate)
  {
    if(data.write_raw_data_to_files)
    {
      AssertThrow(data.n_cells_1d_coarse_grid == 1,
                  ExcMessage(
                    "Choosing write_raw_data_to_files = true requires n_cells_1d_coarse_grid == 1. "
                    "If you want to use n_cells_1d_coarse_grid != 1 (subdivided hypercube), set "
                    "do_fftw = true and write_raw_data_to_files = false."));
    }

    if(deal_spectrum_wrapper == nullptr)
    {
      deal_spectrum_wrapper.reset(new dealspectrum::DealSpectrumWrapper(
        mpi_comm, data.write_raw_data_to_files, data.do_fftw));
    }

    unsigned int evaluation_points = std::max(data.degree + 1, data.evaluation_points_per_cell);

    // create data structures for full system
    if(data.exploit_symmetry)
    {
      tria_full.reset(new parallel::distributed::Triangulation<dim>(
        mpi_comm,
        Triangulation<dim>::limit_level_difference_at_vertices,
        parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy));
      GridGenerator::subdivided_hyper_cube(*tria_full,
                                           data.n_cells_1d_coarse_grid,
                                           -data.length_symmetric_domain,
                                           +data.length_symmetric_domain);
      tria_full->refine_global(data.refine_level + 1);

      fe_full.reset(new FESystem<dim>(FE_DGQ<dim>(data.degree), dim));
      dof_handler_full.reset(new DoFHandler<dim>(*tria_full));
      dof_handler_full->distribute_dofs(*fe_full);

      int cells = tria_full->n_global_active_cells();
      cells     = round(pow(cells, 1.0 / dim));
      deal_spectrum_wrapper->init(dim, cells, data.degree + 1, evaluation_points, *tria_full);
    }
    else
    {
      int local_cells = matrix_free_data_in.n_physical_cells();
      int cells       = local_cells;
      MPI_Allreduce(MPI_IN_PLACE, &cells, 1, MPI_INTEGER, MPI_SUM, mpi_comm);
      cells = round(pow(cells, 1.0 / dim));
      deal_spectrum_wrapper->init(
        dim, cells, data.degree + 1, evaluation_points, dof_handler->get_triangulation());
    }
  }
}

template<int dim, typename Number>
void
KineticEnergySpectrumCalculator<dim, Number>::evaluate(VectorType const & velocity,
                                                       double const &     time,
                                                       int const &        time_step_number)
{
  if(data.calculate == true)
  {
    if(time_step_number >= 0) // unsteady problem
    {
      if(needs_to_be_evaluated(time, time_step_number))
      {
        if(data.exploit_symmetry)
        {
          unsigned int n_cells_1d = data.n_cells_1d_coarse_grid * std::pow(2, data.refine_level);

          velocity_full.reset(new VectorType());
          initialize_dof_vector(*velocity_full, *dof_handler_full);

          apply_taylor_green_symmetry(*dof_handler,
                                      *dof_handler_full,
                                      n_cells_1d,
                                      data.length_symmetric_domain /
                                        static_cast<double>(n_cells_1d),
                                      velocity,
                                      *velocity_full);

          do_evaluate(*velocity_full, time);
        }
        else
        {
          do_evaluate(velocity, time);
        }
      }
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

template<int dim, typename Number>
bool
KineticEnergySpectrumCalculator<dim, Number>::needs_to_be_evaluated(
  double const       time,
  unsigned int const time_step_number)
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

  return evaluate;
}

template<int dim, typename Number>
void
KineticEnergySpectrumCalculator<dim, Number>::do_evaluate(VectorType const & velocity,
                                                          double const       time)
{
  // extract beginning of vector...
  const Number * temp = velocity.begin();

  const std::string file_name = data.filename + "_" + Utilities::int_to_string(counter, 4);

  deal_spectrum_wrapper->execute((double *)temp, file_name, time);

  if(data.do_fftw)
  {
    // write output file
    if(Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::cout << std::endl
                << "Write kinetic energy spectrum at time t = " << time << ":" << std::endl;

      // get tabularized results ...
      double * kappa;
      double * E;
      double * C;
      double   e_physical = 0.0;
      double   e_spectral = 0.0;
      int len = deal_spectrum_wrapper->get_results(kappa, E, C /*unused*/, e_physical, e_spectral);

      std::ostringstream filename;
      filename << data.filename;

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

      f << std::endl
        << "Calculate kinetic energy spectrum at time t = " << time << ":" << std::endl
        << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << std::endl
        << "  Energy physical space e_phy = " << e_physical << std::endl
        << "  Energy spectral space e_spe = " << e_spectral << std::endl
        << "  Difference  |e_phy - e_spe| = " << std::abs(e_physical - e_spectral) << std::endl
        << std::endl
        << "    k  k (avg)              E(k)" << std::endl;

      // ... and print it line by line:
      for(int i = 0; i < len; i++)
      {
        f << std::scientific << std::setprecision(0)
          << std::setw(2 + ceil(std::max(3.0, log(len) / log(10)))) << i << std::scientific
          << std::setprecision(precision) << std::setw(precision + 8) << kappa[i] << "   " << E[i]
          << std::endl;
      }
    }
  }
}

template class KineticEnergySpectrumCalculator<2, float>;
template class KineticEnergySpectrumCalculator<2, double>;

template class KineticEnergySpectrumCalculator<3, float>;
template class KineticEnergySpectrumCalculator<3, double>;

} // namespace ExaDG
