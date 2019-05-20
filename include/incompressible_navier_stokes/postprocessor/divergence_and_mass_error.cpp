/*
 * divergence_and_mass_error.cpp
 *
 *  Created on: May 18, 2019
 *      Author: fehn
 */

#include "divergence_and_mass_error.h"

namespace IncNS
{
template<int dim, typename Number>
DivergenceAndMassErrorCalculator<dim, Number>::DivergenceAndMassErrorCalculator()
  : clear_files_mass_error(true),
    number_of_samples(0),
    divergence_sample(0.0),
    mass_sample(0.0),
    matrix_free_data(nullptr),
    dof_index(0),
    quad_index(0)
{
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::setup(
  MatrixFree<dim, Number> const & matrix_free_data_in,
  unsigned int const              dof_index_in,
  unsigned int const              quad_index_in,
  MassConservationData const &    div_and_mass_data_in)
{
  matrix_free_data  = &matrix_free_data_in;
  dof_index         = dof_index_in;
  quad_index        = quad_index_in;
  div_and_mass_data = div_and_mass_data_in;
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::evaluate(VectorType const & velocity,
                                                        double const &     time,
                                                        int const &        time_step_number)
{
  if(div_and_mass_data.calculate_error == true)
  {
    if(time_step_number >= 0) // unsteady problem
      analyze_div_and_mass_error_unsteady(velocity, time, time_step_number);
    else // steady problem (time_step_number = -1)
      analyze_div_and_mass_error_steady(velocity);
  }
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::do_evaluate(
  MatrixFree<dim, Number> const & matrix_free_data,
  VectorType const &              velocity,
  Number &                        div_error,
  Number &                        div_error_reference,
  Number &                        mass_error,
  Number &                        mass_error_reference)
{
  std::vector<Number> dst(4, 0.0);
  matrix_free_data.loop(&This::local_compute_div,
                        &This::local_compute_div_face,
                        &This::local_compute_div_boundary_face,
                        this,
                        dst,
                        velocity);

  div_error            = Utilities::MPI::sum(dst.at(0), MPI_COMM_WORLD);
  div_error_reference  = Utilities::MPI::sum(dst.at(1), MPI_COMM_WORLD);
  mass_error           = Utilities::MPI::sum(dst.at(2), MPI_COMM_WORLD);
  mass_error_reference = Utilities::MPI::sum(dst.at(3), MPI_COMM_WORLD);
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::local_compute_div(
  const MatrixFree<dim, Number> &               data,
  std::vector<Number> &                         dst,
  const VectorType &                            source,
  const std::pair<unsigned int, unsigned int> & cell_range)
{
  CellIntegratorU integrator(data, dof_index, quad_index);

  AlignedVector<scalar> JxW_values(integrator.n_q_points);

  Number div = 0.;
  Number ref = 0.;

  for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
  {
    integrator.reinit(cell);
    integrator.read_dof_values(source);
    integrator.evaluate(true, true);
    integrator.fill_JxW_values(JxW_values);

    scalar div_vec = make_vectorized_array<Number>(0.);
    scalar ref_vec = make_vectorized_array<Number>(0.);

    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      vector velocity = integrator.get_value(q);
      ref_vec += JxW_values[q] * velocity.norm();
      div_vec += JxW_values[q] * std::abs(integrator.get_divergence(q));
    }

    // sum over entries of VectorizedArray, but only over those that are "active"
    for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
    {
      div += div_vec[v];
      ref += ref_vec[v];
    }
  }

  dst.at(0) += div * this->div_and_mass_data.reference_length_scale;
  dst.at(1) += ref;
}

template<int dim, typename Number>
void
DivergenceAndMassErrorCalculator<dim, Number>::local_compute_div_face(
  const MatrixFree<dim, Number> &               data,
  std::vector<Number> &                         dst,
  const VectorType &                            source,
  const std::pair<unsigned int, unsigned int> & face_range)
{
  FaceIntegratorU integrator_m(data, true, dof_index, quad_index);
  FaceIntegratorU integrator_p(data, false, dof_index, quad_index);

  AlignedVector<scalar> JxW_values(integrator_m.n_q_points);

  Number diff_mass_flux = 0.;
  Number mean_mass_flux = 0.;

  for(unsigned int face = face_range.first; face < face_range.second; ++face)
  {
    integrator_m.reinit(face);
    integrator_m.read_dof_values(source);
    integrator_m.evaluate(true, false);
    integrator_p.reinit(face);
    integrator_p.read_dof_values(source);
    integrator_p.evaluate(true, false);
    integrator_m.fill_JxW_values(JxW_values);

    scalar diff_mass_flux_vec = make_vectorized_array<Number>(0.);
    scalar mean_mass_flux_vec = make_vectorized_array<Number>(0.);

    for(unsigned int q = 0; q < integrator_m.n_q_points; ++q)
    {
      diff_mass_flux_vec +=
        JxW_values[q] * std::abs((integrator_m.get_value(q) - integrator_p.get_value(q)) *
                                 integrator_m.get_normal_vector(q));
      mean_mass_flux_vec +=
        JxW_values[q] * std::abs(0.5 * (integrator_m.get_value(q) + integrator_p.get_value(q)) *
                                 integrator_m.get_normal_vector(q));
    }

    // sum over entries of VectorizedArray, but only over those that are "active"
    for(unsigned int v = 0; v < data.n_active_entries_per_face_batch(face); ++v)
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
  const MatrixFree<dim, Number> &,
  std::vector<Number> &,
  const VectorType &,
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
  if(time > div_and_mass_data.start_time - 1.e-10)
  {
    Number div_error = 1.0, div_error_reference = 1.0, mass_error = 1.0, mass_error_reference = 1.0;

    // calculate divergence and mass error
    do_evaluate(*matrix_free_data,
                velocity,
                div_error,
                div_error_reference,
                mass_error,
                mass_error_reference);

    Number div_error_normalized  = div_error / div_error_reference;
    Number mass_error_normalized = 1.0;
    if(mass_error_reference > 1.e-12)
      mass_error_normalized = mass_error / mass_error_reference;
    else
      mass_error_normalized = mass_error;

    // write output file
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::ostringstream filename;
      filename << div_and_mass_data.filename_prefix << ".div_mass_error_timeseries";

      std::ofstream f;
      if(clear_files_mass_error == true)
      {
        f.open(filename.str().c_str(), std::ios::trunc);
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
        f.open(filename.str().c_str(), std::ios::app);
      }

      f << std::scientific << std::setprecision(7) << std::setw(15) << time << std::setw(15)
        << div_error_normalized << std::setw(15) << mass_error_normalized << std::endl;
    }

    if(time_step_number % div_and_mass_data.sample_every_time_steps == 0)
    {
      // calculate average error
      ++number_of_samples;
      divergence_sample += div_error_normalized;
      mass_sample += mass_error_normalized;

      // write output file
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::ostringstream filename;
        filename << div_and_mass_data.filename_prefix << ".div_mass_error_average";

        std::ofstream f;

        f.open(filename.str().c_str(), std::ios::trunc);
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
    *matrix_free_data, velocity, div_error, div_error_reference, mass_error, mass_error_reference);

  Number div_error_normalized  = div_error / div_error_reference;
  Number mass_error_normalized = 1.0;
  if(mass_error_reference > 1.e-12)
    mass_error_normalized = mass_error / mass_error_reference;
  else
    mass_error_normalized = mass_error;

  // write output file
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::ostringstream filename;
    filename << div_and_mass_data.filename_prefix << ".div_mass_error";

    std::ofstream f;

    f.open(filename.str().c_str(), std::ios::trunc);
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
