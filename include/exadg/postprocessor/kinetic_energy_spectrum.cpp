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

// deal.II
#include <deal.II/base/mpi.h>
#include <deal.II/base/mpi_compute_index_owner_internal.h>
#include <deal.II/distributed/tria.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/postprocessor/kinetic_energy_spectrum.h>
#include <exadg/utilities/create_directories.h>

#ifdef EXADG_WITH_FFTW
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
    {
      if(cell->is_locally_owned())
      {
        auto c = cell->center();
        for(dealii::types::global_dof_index i = 0; i < dim; i++)
          c[i] = (c[i] + dealii::numbers::PI) / (2 * dealii::numbers::PI / n_cells_1D);

        local_cells.push_back(norm_point_to_lex(c, n_cells_1D));
      }
    }

    dealii::types::global_dof_index n_local_cells = local_cells.size();
    dealii::types::global_dof_index global_offset = 0;

    MPI_Exscan(&n_local_cells,
               &global_offset,
               1,
               dealii::Utilities::MPI::mpi_type_id_for_type<decltype(global_offset)>,
               MPI_SUM,
               comm);

    // ... interpolation
    timer.start("Init-Ipol");
    ipol.init(global_offset, global_offset + n_local_cells);
    timer.stop("Init-Ipol");

    // FFTW will be called externally
    if(not inplace)
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

  MPI_Comm const comm;

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
/*
 * Taylor-Green symmetries:
 *
 *  v_i (..., - x_j,..., t) = - v_i(..., x_j,..., t) if i = j
 *
 * and
 *
 *  v_i (..., - x_j,..., t) = + v_i(..., x_j,..., t) if i != j
 */
template<int dim, typename Number>
void
apply_taylor_green_symmetry(dealii::DoFHandler<dim> const & dof_handler_symm,
                            dealii::DoFHandler<dim> const & dof_handler,
                            double const                    n_cells_1d,
                            double const                    delta,
                            const dealii::LinearAlgebra::distributed::Vector<Number> & vector_symm,
                            dealii::LinearAlgebra::distributed::Vector<Number> &       vector)
{
  AssertThrow(
    dof_handler_symm.get_triangulation().all_reference_cells_are_hyper_cube(),
    dealii::ExcMessage(
      "This functionality is only available for meshes consisting of hypercube elements."));
  AssertThrow(
    dof_handler.get_triangulation().all_reference_cells_are_hyper_cube(),
    dealii::ExcMessage(
      "This functionality is only available for meshes consisting of hypercube elements."));

  // determine some useful constants
  auto const & fe = dof_handler.get_fe();

  MPI_Comm const comm = dof_handler.get_communicator();

  // determine which process has which index (lex numbering) and wants which
  dealii::IndexSet range_has_lex(dof_handler_symm.n_dofs());  // has in symm system
  dealii::IndexSet range_want_lex(dof_handler_symm.n_dofs()); // want in full system

  // ... and create a map: lex to cell iterators
  std::map<unsigned int, typename dealii::DoFHandler<dim>::active_cell_iterator>
    map_lex_to_cell_symm;
  std::map<unsigned int, std::vector<typename dealii::DoFHandler<dim>::active_cell_iterator>>
    map_lex_to_cell_full;

  {
    auto norm_point_to_lex = [&](dealii::Point<dim> const c) {
      // convert normalized point [0, 1] to lex
      if(dim == 2)
        return static_cast<std::size_t>(std::floor(c[0]) + n_cells_1d * std::floor(c[1]));
      else
        return static_cast<std::size_t>(std::floor(c[0]) + n_cells_1d * std::floor(c[1]) +
                                        n_cells_1d * n_cells_1d * std::floor(c[2]));
    };

    // ... has (symm)
    for(auto const & cell : dof_handler_symm.active_cell_iterators())
      if(cell->is_active() and cell->is_locally_owned())
      {
        auto c = cell->center();
        for(unsigned int i = 0; i < dim; i++)
          c[i] = c[i] / delta;

        unsigned int const lid = norm_point_to_lex(c);

        range_has_lex.add_index(lid);
        map_lex_to_cell_symm[lid] = cell;
      }

    // want (full)
    for(auto const & cell : dof_handler.active_cell_iterators())
      if(cell->is_active() and cell->is_locally_owned())
      {
        auto c = cell->center();
        for(unsigned int i = 0; i < dim; i++)
          c[i] = std::abs(c[i]) / delta;

        unsigned int const lex = norm_point_to_lex(c);

        range_want_lex.add_index(lex);
        map_lex_to_cell_full[lex].emplace_back(cell);
      }
  }

  // determine who has and who wants data
  std::map<unsigned int, std::vector<unsigned int>> recv_map_proc_to_lex_offset;
  std::map<unsigned int, dealii::IndexSet>          send_map_proc_to_lex;

  {
    std::vector<unsigned int> owning_ranks_of_ghosts(range_want_lex.n_elements());

    // set up dictionary
    dealii::Utilities::MPI::internal::ComputeIndexOwner::ConsensusAlgorithmsPayload process(
      range_has_lex, range_want_lex, comm, owning_ranks_of_ghosts, true);

    dealii::Utilities::MPI::ConsensusAlgorithms::Selector<
      std::vector<std::pair<dealii::types::global_dof_index, dealii::types::global_dof_index>>,
      std::vector<unsigned int>>
      consensus_algorithm;
    consensus_algorithm.run(process, comm);

    for(auto const & owner : owning_ranks_of_ghosts)
      recv_map_proc_to_lex_offset[owner] = std::vector<unsigned int>();

    for(unsigned int i = 0; i < owning_ranks_of_ghosts.size(); i++)
      recv_map_proc_to_lex_offset[owning_ranks_of_ghosts[i]].push_back(i);

    send_map_proc_to_lex = process.get_requesters();
  }

  // perform data exchange and fill this buffer
  std::vector<double> data_buffer(range_want_lex.n_elements() * fe.n_dofs_per_cell());
  {
    // data structure for MPI exchange
    std::map<unsigned int, std::vector<double>> recv_buffer;
    {
      std::map<unsigned int, std::vector<double>> send_buffers;

      std::vector<MPI_Request> recv_requests(recv_map_proc_to_lex_offset.size());
      std::vector<MPI_Request> send_requests(send_map_proc_to_lex.size());

      unsigned int recv_couter = 0;
      unsigned int send_couter = 0;

      // post recv
      for(auto const & recv_offset : recv_map_proc_to_lex_offset)
      {
        recv_buffer[recv_offset.first].resize(recv_offset.second.size() * fe.n_dofs_per_cell());
        MPI_Irecv(recv_buffer[recv_offset.first].data(),
                  recv_buffer[recv_offset.first].size(),
                  MPI_DOUBLE,
                  recv_offset.first,
                  0,
                  comm,
                  &recv_requests[recv_couter++]);
      }

      // post send
      for(auto const & send_index_set : send_map_proc_to_lex)
      {
        // allocate memory
        auto & send_buffer = send_buffers[send_index_set.first];
        send_buffer.resize(send_index_set.second.n_elements() * fe.n_dofs_per_cell());

        // collect data to be send
        auto                                         send_buffer_ptr = &send_buffer[0];
        std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
        for(auto const cell_index : send_index_set.second)
        {
          auto const & cell_accessor = map_lex_to_cell_symm[cell_index];
          cell_accessor->get_dof_indices(dof_indices);

          for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++)
            send_buffer_ptr[i] = vector_symm[dof_indices[i]];
          send_buffer_ptr += fe.n_dofs_per_cell();
        }

        // send data
        MPI_Isend(send_buffer.data(),
                  send_buffer.size(),
                  MPI_DOUBLE,
                  send_index_set.first,
                  0,
                  comm,
                  &send_requests[send_couter++]);
      }

      // wait that data has been send and received
      MPI_Waitall(recv_couter, recv_requests.data(), MPI_STATUSES_IGNORE);
      MPI_Waitall(send_couter, send_requests.data(), MPI_STATUSES_IGNORE);

      // copy received data into a single buffer
      for(auto const & recv_offset : recv_map_proc_to_lex_offset)
      {
        auto const & buffer = recv_buffer[recv_offset.first];

        unsigned int counter = 0;
        for(auto const & offset : recv_offset.second)
          for(unsigned int i = 0; i < fe.n_dofs_per_cell(); i++)
            data_buffer[offset * fe.n_dofs_per_cell() + i] = buffer[counter++];
      }
    }
  }

  // read buffer and fill full vector
  {
    auto send_buffer_ptr = &data_buffer[0];

    unsigned int const n_dofs_per_component = fe.n_dofs_per_cell() / dim;

    std::vector<dealii::types::global_dof_index> dof_indices(fe.n_dofs_per_cell());
    for(auto const cell_index : range_want_lex)
    {
      auto const & cell_accessors = map_lex_to_cell_full[cell_index];

      for(auto const & cell_accessor : cell_accessors)
      {
        cell_accessor->get_dof_indices(dof_indices);

        for(unsigned int i = 0; i < n_dofs_per_component; i++)
        {
          dealii::Point<dim, unsigned int> p =
            dim == 2 ? dealii::Point<dim, unsigned int>(i % (fe.degree + 1), i / (fe.degree + 1)) :
                       dealii::Point<dim, unsigned int>(i % (fe.degree + 1),
                                                        (i % ((fe.degree + 1) * (fe.degree + 1))) /
                                                          (fe.degree + 1),
                                                        i / (fe.degree + 1) / (fe.degree + 1));

          auto c = cell_accessor->center();

          for(unsigned int v = 0; v < dim; v++)
            if(c[v] < 0)
              p[v] = fe.degree - p[v];

          unsigned int const shift =
            dim == 2 ? (p[0] + p[1] * (1 + fe.degree)) :
                       (p[0] + p[1] * (1 + fe.degree) + p[2] * (1 + fe.degree) * (1 + fe.degree));

          for(unsigned int d = 0; d < dim; d++)
            vector[dof_indices[i + d * n_dofs_per_component]] =
              send_buffer_ptr[shift + d * n_dofs_per_component] * (c[d] < 0.0 ? -1.0 : +1.0);
        }
      }
      send_buffer_ptr += fe.n_dofs_per_cell();
    }
  }
}

template<typename MeshType, typename Number>
void
initialize_dof_vector(dealii::LinearAlgebra::distributed::Vector<Number> & vec,
                      const MeshType &                                     dof_handler)
{
  dealii::IndexSet locally_relevant_dofs;
  dealii::DoFTools::extract_locally_relevant_dofs(dof_handler, locally_relevant_dofs);

  MPI_Comm const comm = dof_handler.get_communicator();

  vec.reinit(dof_handler.locally_owned_dofs(), locally_relevant_dofs, comm);
}

template<int dim, typename Number>
KineticEnergySpectrumCalculator<dim, Number>::KineticEnergySpectrumCalculator(MPI_Comm const & comm)
  : mpi_comm(comm), clear_files(true)
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

  time_control.setup(data_in.time_control_data);

  if(data_in.time_control_data.is_active)
  {
    AssertThrow(
      dof_handler->get_triangulation().all_reference_cells_are_hyper_cube(),
      dealii::ExcMessage(
        "This postprocessing utility is only available for meshes consisting of hypercube elements."));

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
                                                       double const       time,
                                                       bool const         unsteady)
{
  if(unsteady)
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
                                  data.length_symmetric_domain / static_cast<double>(n_cells_1d),
                                  velocity,
                                  *velocity_full);

      do_evaluate(*velocity_full, time);
    }
    else
    {
      do_evaluate(velocity, time);
    }
  }
  else
  {
    AssertThrow(
      false,
      dealii::ExcMessage(
        "Calculation of kinetic energy spectrum only implemented for unsteady problems."));
  }
}

template<int dim, typename Number>
void
KineticEnergySpectrumCalculator<dim, Number>::do_evaluate(VectorType const & velocity,
                                                          double const       time)
{
  // extract beginning of vector...
  Number const * temp = velocity.begin();

  std::string const file_name =
    data.filename + "_" + dealii::Utilities::int_to_string(time_control.get_counter(), 4);

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
