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
#include <exadg/incompressible_navier_stokes/postprocessor/mean_velocity_calculator.h>
#include <exadg/utilities/create_directories.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
MeanVelocityCalculator<dim, Number>::MeanVelocityCalculator(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_in,
  unsigned int const                      quad_index_in,
  MeanVelocityCalculatorData<dim> const & data_in,
  MPI_Comm const &                        comm_in)
  : data(data_in),
    matrix_free(matrix_free_in),
    dof_index(dof_index_in),
    quad_index(quad_index_in),
    area_has_been_initialized(false),
    volume_has_been_initialized(false),
    area(0.0),
    volume(0.0),
    clear_files(true),
    mpi_comm(comm_in)
{
  if(data.calculate and data.write_to_file)
    create_directories(data.directory, mpi_comm);
}

template<int dim, typename Number>
Number
MeanVelocityCalculator<dim, Number>::calculate_mean_velocity_area(VectorType const & velocity,
                                                                  double const &     time)
{
  if(data.calculate == true)
  {
    if(area_has_been_initialized == false)
    {
      this->area = calculate_area();

      area_has_been_initialized = true;
    }

    Number flow_rate = do_calculate_flow_rate_area(velocity);

    AssertThrow(area_has_been_initialized == true,
                dealii::ExcMessage("Area has not been initialized."));
    AssertThrow(this->area != 0.0, dealii::ExcMessage("Area has not been initialized."));
    Number mean_velocity = flow_rate / this->area;

    write_output(mean_velocity, time, "Mean velocity [m/s]");

    return mean_velocity;
  }
  else
  {
    return -1.0;
  }
}

template<int dim, typename Number>
Number
MeanVelocityCalculator<dim, Number>::calculate_mean_velocity_volume(VectorType const & velocity,
                                                                    double const &     time)
{
  if(data.calculate == true)
  {
    if(volume_has_been_initialized == false)
    {
      this->volume = calculate_volume();

      volume_has_been_initialized = true;
    }

    Number mean_velocity = do_calculate_mean_velocity_volume(velocity);

    write_output(mean_velocity, time, "Mean velocity [m/s]");

    return mean_velocity;
  }
  else
  {
    return -1.0;
  }
}

template<int dim, typename Number>
Number
MeanVelocityCalculator<dim, Number>::calculate_flow_rate_volume(VectorType const & velocity,
                                                                double const &     time,
                                                                double const &     length) const
{
  if(data.calculate == true)
  {
    Number flow_rate = 1.0 / length * do_calculate_flow_rate_volume(velocity);

    write_output(flow_rate, time, "Flow rate [m^3/s]");

    return flow_rate;
  }
  else
  {
    return -1.0;
  }
}

template<int dim, typename Number>
Number
MeanVelocityCalculator<dim, Number>::calculate_flow_rate_area(VectorType const & velocity,
                                                              double const &     time) const
{
  if(data.calculate == true)
  {
    Number flow_rate = do_calculate_flow_rate_area(velocity);

    write_output(flow_rate, time, "Flow rate [m^3/s]");

    return flow_rate;
  }
  else
  {
    return -1.0;
  }
}

template<int dim, typename Number>
void
MeanVelocityCalculator<dim, Number>::write_output(Number const &      value,
                                                  double const &      time,
                                                  std::string const & name) const
{
  // write output file
  if(data.write_to_file == true and dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::string filename = data.directory + data.filename;

    std::ofstream f;
    if(clear_files == true)
    {
      f.open(filename.c_str(), std::ios::trunc);
      f << std::endl << "  Time                " + name << std::endl;

      clear_files = false;
    }
    else
    {
      f.open(filename.c_str(), std::ios::app);
    }

    unsigned int precision = 12;
    f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
      << std::setw(precision + 8) << value << std::endl;
  }
}

template<int dim, typename Number>
Number
MeanVelocityCalculator<dim, Number>::calculate_area() const
{
  FaceIntegratorU integrator(matrix_free, true, dof_index, quad_index);

  Number area = 0.0;

  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
      face++)
  {
    typename std::set<dealii::types::boundary_id>::iterator it;
    dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

    it = data.boundary_IDs.find(boundary_id);
    if(it != data.boundary_IDs.end())
    {
      integrator.reinit(face);

      scalar area_local = dealii::make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        area_local += integrator.JxW(q);
      }

      // sum over all entries of dealii::VectorizedArray
      for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
        area += area_local[n];
    }
  }

  area = dealii::Utilities::MPI::sum(area, mpi_comm);

  return area;
}

template<int dim, typename Number>
Number
MeanVelocityCalculator<dim, Number>::calculate_volume() const
{
  std::vector<Number> dst(1, 0.0);

  VectorType src_dummy;
  matrix_free.cell_loop(&This::local_calculate_volume, this, dst, src_dummy);

  // sum over all MPI processes
  Number volume = 1.0;
  volume        = dealii::Utilities::MPI::sum(dst.at(0), mpi_comm);

  return volume;
}

template<int dim, typename Number>
void
MeanVelocityCalculator<dim, Number>::local_calculate_volume(
  dealii::MatrixFree<dim, Number> const & data,
  std::vector<Number> &                   dst,
  VectorType const &,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  CellIntegratorU integrator(data, dof_index, quad_index);

  Number volume = 0.;

  // Loop over all elements
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);

    scalar volume_vec = dealii::make_vectorized_array<Number>(0.);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      volume_vec += integrator.JxW(q);
    }

    // sum over entries of dealii::VectorizedArray, but only over those that are "active"
    for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
    {
      volume += volume_vec[v];
    }
  }

  dst.at(0) += volume;
}

template<int dim, typename Number>
Number
MeanVelocityCalculator<dim, Number>::do_calculate_flow_rate_area(VectorType const & velocity) const
{
  FaceIntegratorU integrator(matrix_free, true, dof_index, quad_index);

  // initialize with zero since we accumulate into this variable
  Number flow_rate = 0.0;

  for(unsigned int face = matrix_free.n_inner_face_batches();
      face < (matrix_free.n_inner_face_batches() + matrix_free.n_boundary_face_batches());
      face++)
  {
    typename std::set<dealii::types::boundary_id>::iterator it;
    dealii::types::boundary_id boundary_id = matrix_free.get_boundary_id(face);

    it = data.boundary_IDs.find(boundary_id);
    if(it != data.boundary_IDs.end())
    {
      integrator.reinit(face);
      integrator.read_dof_values(velocity);
      integrator.evaluate(dealii::EvaluationFlags::values);

      scalar flow_rate_face = dealii::make_vectorized_array<Number>(0.0);

      for(unsigned int q = 0; q < integrator.n_q_points; ++q)
      {
        flow_rate_face +=
          integrator.JxW(q) * integrator.get_value(q) * integrator.get_normal_vector(q);
      }

      // sum over all entries of dealii::VectorizedArray
      for(unsigned int n = 0; n < matrix_free.n_active_entries_per_face_batch(face); ++n)
        flow_rate += flow_rate_face[n];
    }
  }

  flow_rate = dealii::Utilities::MPI::sum(flow_rate, mpi_comm);

  return flow_rate;
}

template<int dim, typename Number>
Number
MeanVelocityCalculator<dim, Number>::do_calculate_mean_velocity_volume(
  VectorType const & velocity) const
{
  std::vector<Number> dst(1, 0.0);
  matrix_free.cell_loop(&This::local_calculate_flow_rate_volume, this, dst, velocity);

  // sum over all MPI processes
  Number mean_velocity = dealii::Utilities::MPI::sum(dst.at(0), mpi_comm);

  AssertThrow(volume_has_been_initialized == true,
              dealii::ExcMessage("Volume has not been initialized."));
  AssertThrow(this->volume != 0.0, dealii::ExcMessage("Volume has not been initialized."));

  mean_velocity /= this->volume;

  return mean_velocity;
}

template<int dim, typename Number>
Number
MeanVelocityCalculator<dim, Number>::do_calculate_flow_rate_volume(
  VectorType const & velocity) const
{
  std::vector<Number> dst(1, 0.0);
  matrix_free.cell_loop(&This::local_calculate_flow_rate_volume, this, dst, velocity);

  // sum over all MPI processes
  Number flow_rate_times_length = dealii::Utilities::MPI::sum(dst.at(0), mpi_comm);

  return flow_rate_times_length;
}

template<int dim, typename Number>
void
MeanVelocityCalculator<dim, Number>::local_calculate_flow_rate_volume(
  dealii::MatrixFree<dim, Number> const &       data,
  std::vector<Number> &                         dst,
  VectorType const &                            src,
  std::pair<unsigned int, unsigned int> const & cell_range) const
{
  CellIntegratorU integrator(data, dof_index, quad_index);

  Number flow_rate = 0.;

  // Loop over all elements
  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(src);
    integrator.evaluate(dealii::EvaluationFlags::values);

    scalar flow_rate_vec = dealii::make_vectorized_array<Number>(0.);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      scalar velocity_direction = this->data.direction * integrator.get_value(q);

      flow_rate_vec += velocity_direction * integrator.JxW(q);
    }

    // sum over entries of dealii::VectorizedArray, but only over those
    // that are "active"
    for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
    {
      flow_rate += flow_rate_vec[v];
    }
  }

  dst.at(0) += flow_rate;
}

template class MeanVelocityCalculator<2, float>;
template class MeanVelocityCalculator<2, double>;

template class MeanVelocityCalculator<3, float>;
template class MeanVelocityCalculator<3, double>;

} // namespace IncNS
} // namespace ExaDG
