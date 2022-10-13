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
#include <exadg/incompressible_navier_stokes/postprocessor/divergence_and_mass_error.h>
#include <exadg/utilities/create_directories.h>
#include <exadg/utilities/numbers.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
DivergenceAndMassErrorCalculator<dim, Number>::DivergenceAndMassErrorCalculator(
  MPI_Comm const & comm)
  : mpi_comm(comm),
    clear_files_mass_error(true),
    number_of_samples(0),
    divergence_sample(0.0),
    mass_sample(0.0),
    matrix_free(nullptr),
    dof_index(0),
    quad_index(0)
{
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::setup(
  dealii::MatrixFree<dim, Number> const & matrix_free_in,
  unsigned int const                      dof_index_in,
  unsigned int const                      quad_index_in,
  MassConservationData const &            data_in)
{
  matrix_free = &matrix_free_in;
  dof_index   = dof_index_in;
  quad_index  = quad_index_in;
  data        = data_in;

  if(data.calculate)
    create_directories(data.directory, mpi_comm);
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::evaluate(VectorType const & velocity,
                                                        double const &     time,
                                                        int const &        time_step_number)
{
  if(data.calculate)
  {
    if(Utilities::is_unsteady_timestep(time_step_number))
      analyze_div_and_mass_error_unsteady(velocity, time, time_step_number);
    else
      analyze_div_and_mass_error_steady(velocity);
  }
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::do_evaluate(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType const &                      velocity,
  Number &                                div_error,
  Number &                                div_error_reference,
  Number &                                mass_error,
  Number &                                mass_error_reference)
{
  std::vector<Number> dst(4, 0.0);
  matrix_free.loop(&This::local_compute_div,
                   &This::local_compute_div_face,
                   &This::local_compute_div_boundary_face,
                   this,
                   dst,
                   velocity);

  div_error            = dealii::Utilities::MPI::sum(dst.at(0), mpi_comm);
  div_error_reference  = dealii::Utilities::MPI::sum(dst.at(1), mpi_comm);
  mass_error           = dealii::Utilities::MPI::sum(dst.at(2), mpi_comm);
  mass_error_reference = dealii::Utilities::MPI::sum(dst.at(3), mpi_comm);
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::local_compute_div(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  std::vector<Number> &                         dst,
  VectorType const &                            source,
  const std::pair<unsigned int, unsigned int> & cell_range)
{
  CellIntegratorU integrator(matrix_free, dof_index, quad_index);

  Number div = 0.;
  Number ref = 0.;

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(source);
    integrator.evaluate(dealii::EvaluationFlags::values | dealii::EvaluationFlags::gradients);

    scalar div_vec = dealii::make_vectorized_array<Number>(0.);
    scalar ref_vec = dealii::make_vectorized_array<Number>(0.);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector velocity = integrator.get_value(q);
      ref_vec += integrator.JxW(q) * velocity.norm();
      div_vec += integrator.JxW(q) * std::abs(integrator.get_divergence(q));
    }

    // sum over entries of dealii::VectorizedArray, but only over those that are "active"
    for(unsigned int v = 0; v < matrix_free.n_active_entries_per_cell_batch(cell); ++v)
    {
      div += div_vec[v];
      ref += ref_vec[v];
    }
  }

  dst.at(0) += div * data.reference_length_scale;
  dst.at(1) += ref;
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::local_compute_div_face(
  dealii::MatrixFree<dim, Number> const &       matrix_free,
  std::vector<Number> &                         dst,
  VectorType const &                            source,
  const std::pair<unsigned int, unsigned int> & face_range)
{
  FaceIntegratorU integrator_m(matrix_free, true, dof_index, quad_index);
  FaceIntegratorU integrator_p(matrix_free, false, dof_index, quad_index);

  Number diff_mass_flux = 0.;
  Number mean_mass_flux = 0.;

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    integrator_m.reinit(face);
    integrator_m.read_dof_values(source);
    integrator_m.evaluate(dealii::EvaluationFlags::values);
    integrator_p.reinit(face);
    integrator_p.read_dof_values(source);
    integrator_p.evaluate(dealii::EvaluationFlags::values);

    scalar diff_mass_flux_vec = dealii::make_vectorized_array<Number>(0.);
    scalar mean_mass_flux_vec = dealii::make_vectorized_array<Number>(0.);

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      diff_mass_flux_vec +=
        integrator_m.JxW(q) * std::abs((integrator_m.get_value(q) - integrator_p.get_value(q)) *
                                       integrator_m.get_normal_vector(q));
      mean_mass_flux_vec += integrator_m.JxW(q) *
                            std::abs(0.5 * (integrator_m.get_value(q) + integrator_p.get_value(q)) *
                                     integrator_m.get_normal_vector(q));
    }

    // sum over entries of dealii::VectorizedArray, but only over those that are "active"
    for(unsigned int v = 0; v < matrix_free.n_active_entries_per_face_batch(face); ++v)
    {
      diff_mass_flux += diff_mass_flux_vec[v];
      mean_mass_flux += mean_mass_flux_vec[v];
    }
  }

  dst.at(2) += diff_mass_flux;
  dst.at(3) += mean_mass_flux;
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::local_compute_div_boundary_face(
  dealii::MatrixFree<dim, Number> const &,
  std::vector<Number> &,
  VectorType const &,
  const std::pair<unsigned int, unsigned int> &)
{
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::analyze_div_and_mass_error_unsteady(
  VectorType const & velocity,
  double const       time,
  unsigned int const time_step_number)
{
  AssertThrow(Utilities::is_unsteady_timestep(time_step_number),
              dealii::ExcMessage("Can not be used in steady problem."));

  if(time > data.start_time - 1.e-10)
  {
    Number div_error = 1.0, div_error_reference = 1.0, mass_error = 1.0, mass_error_reference = 1.0;

    // calculate divergence and mass error
    do_evaluate(
      *matrix_free, velocity, div_error, div_error_reference, mass_error, mass_error_reference);

    Number div_error_normalized  = div_error / div_error_reference;
    Number mass_error_normalized = 1.0;
    if(mass_error_reference > 1.e-12)
      mass_error_normalized = mass_error / mass_error_reference;
    else
      mass_error_normalized = mass_error;

    // write output file
    if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
    {
      std::string filename = data.directory + data.filename + ".div_mass_error_timeseries";

      std::ofstream f;
      if(clear_files_mass_error == true)
      {
        f.open(filename.c_str(), std::ios::trunc);
        f << "Error incompressibility constraint:" << std::endl
          << std::endl
          << "  (1,|divu|)_Omega/(1,1)_Omega" << std::endl
          << std::endl
          << "Error mass flux over interior element faces:" << std::endl
          << std::endl
          << "  (1,|(um - up)*n|)_dOmegaI / (1,|0.5(um + up)*n|)_dOmegaI" << std::endl
          << std::endl
          << "       t        |  divergence  |    mass       " << std::endl;

        clear_files_mass_error = false;
      }
      else
      {
        f.open(filename.c_str(), std::ios::app);
      }

      f << std::scientific << std::setprecision(7) << std::setw(15) << time << std::setw(15)
        << div_error_normalized << std::setw(15) << mass_error_normalized << std::endl;
    }

    if(time_step_number % data.sample_every_time_steps == 0)
    {
      // calculate average error
      ++number_of_samples;
      divergence_sample += div_error_normalized;
      mass_sample += mass_error_normalized;

      // write output file
      if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
      {
        std::string filename = data.directory + data.filename + ".div_mass_error_average";

        std::ofstream f;

        f.open(filename.c_str(), std::ios::trunc);
        f << "Divergence and mass error (averaged over time)" << std::endl;
        f << "Number of samples:   " << number_of_samples << std::endl;
        f << "Mean error incompressibility constraint:   " << divergence_sample / number_of_samples
          << std::endl;
        f << "Mean error mass flux over interior element faces:  "
          << mass_sample / number_of_samples << std::endl;
        f.close();
      }
    }
  }
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::analyze_div_and_mass_error_steady(
  VectorType const & velocity)
{
  Number div_error = 1.0, div_error_reference = 1.0, mass_error = 1.0, mass_error_reference = 1.0;

  // calculate divergence and mass error
  do_evaluate(
    *matrix_free, velocity, div_error, div_error_reference, mass_error, mass_error_reference);

  Number div_error_normalized  = div_error / div_error_reference;
  Number mass_error_normalized = 1.0;
  if(mass_error_reference > 1.e-12)
    mass_error_normalized = mass_error / mass_error_reference;
  else
    mass_error_normalized = mass_error;

  // write output file
  if(dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)
  {
    std::string filename = data.directory + data.filename + ".div_mass_error";

    std::ofstream f;

    f.open(filename.c_str(), std::ios::trunc);
    f << "Divergence and mass error:" << std::endl;
    f << "Error incompressibility constraint:   " << div_error_normalized << std::endl;
    f << "Error mass flux over interior element faces:  " << mass_error_normalized << std::endl;
    f.close();
  }
}

template class DivergenceAndMassErrorCalculator<2, float>;
template class DivergenceAndMassErrorCalculator<2, double>;

template class DivergenceAndMassErrorCalculator<3, float>;
template class DivergenceAndMassErrorCalculator<3, double>;

} // namespace IncNS
} // namespace ExaDG
