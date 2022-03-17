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

// C/C++
#include <fstream>

// ExaDG
#include <exadg/postprocessor/kinetic_energy_spectrum.h>
#include <exadg/postprocessor/mirror_dof_vector_taylor_green.h>
#include <exadg/utilities/create_directories.h>

#ifdef USE_FFTW
// deal.II
#  include <deal.II/base/mpi.templates.h>
#  include <deal.II/base/mpi_noncontiguous_partitioner.h>

// ExaDG
#  include <exadg/postprocessor/spectral_analysis/interpolation.h>
#  include <exadg/postprocessor/spectral_analysis/permutation.h>
#  include <exadg/postprocessor/spectral_analysis/setup.h>
#  include <exadg/postprocessor/spectral_analysis/spectrum.h>
#  include <exadg/postprocessor/spectral_analysis/timer.h>

namespace ExaDG
{
using namespace dealspectrum;

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
    print_timings();
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
  init(dealii::types::global_dof_index dim,
       dealii::types::global_dof_index n_cells_1D,
       dealii::types::global_dof_index points_src,
       dealii::types::global_dof_index points_dst,
       Tria &                          tria)
  {
    // init setup ...
    s.init(dim, n_cells_1D, points_src, points_dst);

    std::vector<dealii::types::global_dof_index> local_cells;
    for(auto const & cell : tria.active_cell_iterators())
      if(cell->is_active() && cell->is_locally_owned())
      {
        auto c = cell->center();
        for(dealii::types::global_dof_index i = 0; i < dim; i++)
          c[i] = (c[i] + dealii::numbers::PI) / (2 * dealii::numbers::PI / n_cells_1D);

        local_cells.push_back(norm_point_to_lex(c, n_cells_1D));
      }

    dealii::types::global_dof_index n_local_cells = local_cells.size();
    dealii::types::global_dof_index global_offset = 0;

    MPI_Exscan(&n_local_cells,
               &global_offset,
               1,
#  if DEAL_II_VERSION_GTE(10, 0, 0)
               dealii::Utilities::MPI::mpi_type_id_for_type<decltype(global_offset)>,
#  else
               dealii::Utilities::MPI::mpi_type_id(&global_offset),
#  endif
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
    const dealii::types::global_dof_index start = start_;
    const dealii::types::global_dof_index end   = end_;

    std::vector<dealii::types::global_dof_index> indices_has, indices_want;

    for(auto const & I : local_cells)
      for(dealii::types::global_dof_index i = 0; i < dealii::Utilities::pow(points_dst, dim); i++)
        for(dealii::types::global_dof_index d = 0; d < dim; d++)
        {
          dealii::types::global_dof_index index =
            (I / (n_cells_1D * n_cells_1D) * points_dst + i / (points_dst * points_dst)) *
              dealii::Utilities::pow(n_cells_1D * points_dst, 2) +
            (((I / n_cells_1D) % n_cells_1D) * points_dst + ((i / points_dst) % points_dst)) *
              dealii::Utilities::pow(n_cells_1D * points_dst, 1) +
            (I % (n_cells_1D)*points_dst + i % (points_dst));

          indices_has.push_back(d * dealii::Utilities::pow(points_dst * n_cells_1D, dim) + index);
        }

    dealii::types::global_dof_index N  = s.cells * s.points_dst;
    dealii::types::global_dof_index Nx = (N / 2 + 1) * 2;

    for(dealii::types::global_dof_index d = 0;
        d < static_cast<dealii::types::global_dof_index>(s.dim);
        d++)
    {
      dealii::types::global_dof_index c = 0;
      for(dealii::types::global_dof_index k = 0; k < (end - start); k++)
        for(dealii::types::global_dof_index j = 0; j < N; j++)
          for(dealii::types::global_dof_index i = 0; i < Nx; i++, c++)
            if(i < N)
              indices_want.push_back(d * dealii::Utilities::pow(points_dst * n_cells_1D, dim) +
                                     (k + start) *
                                       dealii::Utilities::pow(points_dst * n_cells_1D, 2) +
                                     j * dealii::Utilities::pow(points_dst * n_cells_1D, 1) + i);
            else
              indices_want.push_back(dealii::numbers::invalid_dof_index); // x-padding

      for(; c < static_cast<dealii::types::global_dof_index>(fftw.bsize); c++)
        indices_want.push_back(dealii::numbers::invalid_dof_index); // z-padding
    }

    nonconti = std::make_shared<dealii::Utilities::MPI::NoncontiguousPartitioner>(indices_has,
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
  execute(double const * src, std::string const & file_name = "", double const time = 0.0)
  {
    if(write)
    {
      // flush flow field to hard drive
      AssertThrow(file_name != "", dealii::ExcMessage("No file name has been provided!"));
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
      dealii::types::global_dof_index const size =
        dealii::Utilities::pow(static_cast<dealii::types::global_dof_index>(s.cells * s.points_dst),
                               s.dim) *
        s.dim;
      dealii::ArrayView<double>       dst(fftw.u_real, size * 2);
      dealii::ArrayView<double const> src_(ipol.dst, size);
      nonconti->export_to_ghosted_array(src_, dst);

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
  template<int dim>
  std::size_t
  norm_point_to_lex(dealii::Point<dim> const & c, unsigned int const & n_cells_1D)
  {
    // convert normalized point [0, 1] to lex
    if(dim == 2)
      return static_cast<std::size_t>(std::floor(c[0]) + n_cells_1D * std::floor(c[1]));
    else if(dim == 3)
      return static_cast<std::size_t>(std::floor(c[0]) + n_cells_1D * std::floor(c[1]) +
                                      n_cells_1D * n_cells_1D * std::floor(c[2]));
    else
      Assert(false, dealii::ExcMessage("not implemented"));

    return 0;
  }

  MPI_Comm const & comm;

  // flush flow field to hard drive?
  bool const write;

  // perform spectral analysis
  bool const inplace;

  // struct containing the setup
  Setup s;

  // ... for interpolation
  Interpolator ipol;

  // ... for spectral analysis
  SpectralAnalysis fftw;

  // Timer
  DealSpectrumTimer timer;

  std::shared_ptr<dealii::Utilities::MPI::NoncontiguousPartitioner> nonconti;
};
} // namespace ExaDG
#else
namespace ExaDG
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
  execute(double const *, std::string const & = "", double const = 0.0)
  {
  }

  int
  get_results(double *&, double *&, double *&, double &, double &)
  {
    return 0;
  }
};
} // namespace ExaDG
#endif

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
  dealii::MatrixFree<dim, Number> const & matrix_free_data_in,
  dealii::DoFHandler<dim> const &         dof_handler_in,
  KineticEnergySpectrumData const &       data_in)
{
  data        = data_in;
  dof_handler = &dof_handler_in;

  clear_files = data.clear_file;

  if(data.calculate)
  {
    if(data.write_raw_data_to_files)
    {
      AssertThrow(data.n_cells_1d_coarse_grid == 1,
                  dealii::ExcMessage(
                    "Choosing write_raw_data_to_files = true requires n_cells_1d_coarse_grid == 1. "
                    "If you want to use n_cells_1d_coarse_grid != 1 (subdivided hypercube), set "
                    "do_fftw = true and write_raw_data_to_files = false."));
    }

    if(deal_spectrum_wrapper == nullptr)
    {
      deal_spectrum_wrapper =
        std::make_shared<DealSpectrumWrapper>(mpi_comm, data.write_raw_data_to_files, data.do_fftw);
    }

    unsigned int evaluation_points = std::max(data.degree + 1, data.evaluation_points_per_cell);

    // create data structures for full system
    if(data.exploit_symmetry)
    {
      tria_full = std::make_shared<dealii::parallel::distributed::Triangulation<dim>>(
        mpi_comm,
        dealii::Triangulation<dim>::limit_level_difference_at_vertices,
        dealii::parallel::distributed::Triangulation<dim>::construct_multigrid_hierarchy);
      dealii::GridGenerator::subdivided_hyper_cube(*tria_full,
                                                   data.n_cells_1d_coarse_grid,
                                                   -data.length_symmetric_domain,
                                                   +data.length_symmetric_domain);
      tria_full->refine_global(data.refine_level + 1);

      fe_full = std::make_shared<dealii::FESystem<dim>>(dealii::FE_DGQ<dim>(data.degree), dim);
      dof_handler_full = std::make_shared<dealii::DoFHandler<dim>>(*tria_full);
      dof_handler_full->distribute_dofs(*fe_full);

      int cells = tria_full->n_global_active_cells();
      cells     = static_cast<int>(std::round(std::pow(cells, 1.0 / dim)));
      deal_spectrum_wrapper->init(dim, cells, data.degree + 1, evaluation_points, *tria_full);
    }
    else
    {
      int local_cells = matrix_free_data_in.n_physical_cells();
      int cells       = local_cells;
      MPI_Allreduce(MPI_IN_PLACE, &cells, 1, MPI_INTEGER, MPI_SUM, mpi_comm);
      cells = static_cast<int>(std::round(std::pow(cells, 1.0 / dim)));
      deal_spectrum_wrapper->init(
        dim, cells, data.degree + 1, evaluation_points, dof_handler->get_triangulation());
    }

    create_directories(data.directory, mpi_comm);
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
          unsigned int n_cells_1d =
            data.n_cells_1d_coarse_grid * dealii::Utilities::pow(2, data.refine_level);

          velocity_full = std::make_shared<VectorType>();
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
        dealii::ExcMessage(
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
                dealii::ExcMessage("Input parameters are in conflict."));
  }
  else if(data.calculate_every_time_interval > 0.0)
  {
    // small number which is much smaller than the time step size
    double const EPSILON = 1.0e-10;

    // The current time might be larger than start_time. In that case, we first have to
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
                dealii::ExcMessage("Input parameters are in conflict."));
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage(
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
  Number const * temp = velocity.begin();

  std::string const file_name = data.filename + "_" + dealii::Utilities::int_to_string(counter, 4);

  deal_spectrum_wrapper->execute((double *)temp, file_name, time);

  if(data.do_fftw)
  {
    // write output file
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
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
      filename << data.directory + data.filename;

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
          << std::setw(2 + static_cast<unsigned int>(std::ceil(std::max(3.0, log(len) / log(10)))))
          << i << std::scientific << std::setprecision(precision) << std::setw(precision + 8)
          << kappa[i] << "   " << E[i] << std::endl;
      }
    }
  }
}

template class KineticEnergySpectrumCalculator<2, float>;
template class KineticEnergySpectrumCalculator<2, double>;

template class KineticEnergySpectrumCalculator<3, float>;
template class KineticEnergySpectrumCalculator<3, double>;

} // namespace ExaDG
