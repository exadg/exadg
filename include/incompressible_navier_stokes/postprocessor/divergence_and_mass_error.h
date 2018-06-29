/*
 * DivergenceAndMassError.h
 *
 *  Created on: Oct 14, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_DIVERGENCE_AND_MASS_ERROR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_DIVERGENCE_AND_MASS_ERROR_H_

#include <fstream>
#include <sstream>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>
#include <deal.II/lac/parallel_vector.h>

#include "incompressible_navier_stokes/postprocessor/postprocessor_base.h"
#include "incompressible_navier_stokes/user_interface/input_parameters.h"


using namespace dealii;

namespace IncNS
{

template<int dim, int fe_degree, typename Number>
class DivergenceAndMassErrorCalculator
{
public:
  DivergenceAndMassErrorCalculator()
    :
    clear_files_mass_error(true),
    number_of_samples(0),
    divergence_sample(0.0),
    mass_sample(0.0),
    matrix_free_data(nullptr)
    {}

  void setup(MatrixFree<dim,Number> const &matrix_free_data_in,
             DofQuadIndexData const       &dof_quad_index_data_in,
             MassConservationData const   &div_and_mass_data_in)
  {
    matrix_free_data = &matrix_free_data_in;
    dof_quad_index_data = dof_quad_index_data_in;
    div_and_mass_data = div_and_mass_data_in;
  }

  void evaluate(parallel::distributed::Vector<Number> const &velocity,
                double const                                &time,
                int const                                   &time_step_number)
  {
    if(div_and_mass_data.calculate_error == true)
    {
      if(time_step_number >= 0) // unsteady problem
        analyze_div_and_mass_error_unsteady(velocity,time,time_step_number);
      else // steady problem (time_step_number = -1)
        analyze_div_and_mass_error_steady(velocity);
    }
  }

private:
  bool clear_files_mass_error;
  int number_of_samples;
  Number divergence_sample;
  Number mass_sample;

  MatrixFree<dim,Number> const * matrix_free_data;
  DofQuadIndexData dof_quad_index_data;
  MassConservationData div_and_mass_data;

  /*
   *  This function calculates the divergence error and the error of mass flux
   *  over interior element faces.
   *
   *  Divergence error: (1,|divu|)_Omega
   *  Reference value for divergence error: (1,1)_Omega
   *
   *  or
   *
   *  Divergence error: L * (1,|divu|)_Omega, L is a reference length scale
   *  Reference value for divergence error: (1,|| u ||)_Omega
   *
   *  and
   *
   *  Mass error: (1,|(um - up)*n|)_dOmegaI
   *  Reference value for mass error: (1,|0.5(um + up)*n|)_dOmegaI
   */
  void do_evaluate(MatrixFree<dim,Number> const                &matrix_free_data,
                   parallel::distributed::Vector<Number> const &velocity,
                   Number                                      &div_error,
                   Number                                      &div_error_reference,
                   Number                                      &mass_error,
                   Number                                      &mass_error_reference)
  {
    std::vector<Number> dst(4,0.0);
    matrix_free_data.loop (&DivergenceAndMassErrorCalculator<dim,fe_degree,Number>::local_compute_div,
                           &DivergenceAndMassErrorCalculator<dim,fe_degree,Number>::local_compute_div_face,
                           &DivergenceAndMassErrorCalculator<dim,fe_degree,Number>::local_compute_div_boundary_face,
                           this, dst, velocity);

    div_error = Utilities::MPI::sum (dst.at(0), MPI_COMM_WORLD);
    div_error_reference = Utilities::MPI::sum (dst.at(1), MPI_COMM_WORLD);
    mass_error = Utilities::MPI::sum (dst.at(2), MPI_COMM_WORLD);
    mass_error_reference = Utilities::MPI::sum (dst.at(3), MPI_COMM_WORLD);
  }

  void local_compute_div(const MatrixFree<dim,Number>                &data,
                         std::vector<Number>                         &dst,
                         const parallel::distributed::Vector<Number> &source,
                         const std::pair<unsigned int,unsigned int>  &cell_range)
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number> phi(data,dof_quad_index_data.dof_index_velocity,dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number> > JxW_values(phi.n_q_points);

    Number div = 0.;
    Number ref = 0.;

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(source);
//      phi.evaluate(false,true);
      phi.evaluate(true,true);
      phi.fill_JxW_values(JxW_values);

      VectorizedArray<Number> div_vec = make_vectorized_array<Number>(0.);
      VectorizedArray<Number> ref_vec = make_vectorized_array<Number>(0.);

      for (unsigned int q=0; q<phi.n_q_points; ++q)
      {
//        ref_vec += JxW_values[q];
        Tensor<1,dim,VectorizedArray<Number> > velocity = phi.get_value(q);
        ref_vec += JxW_values[q]*velocity.norm();
        div_vec += JxW_values[q]*std::abs(phi.get_divergence(q));
      }

      // sum over entries of VectorizedArray, but only over those
      // that are "active"
      for (unsigned int v=0; v<data.n_active_entries_per_cell_batch(cell); ++v)
      {
        div += div_vec[v];
        ref += ref_vec[v];
      }
    }

//    dst.at(0) += div;
    dst.at(0) += div * this->div_and_mass_data.reference_length_scale;
    dst.at(1) += ref;
  }

  void local_compute_div_face (const MatrixFree<dim,Number>                &data,
                               std::vector<Number>                         &dst,
                               const parallel::distributed::Vector<Number> &source,
                               const std::pair<unsigned int,unsigned int>  &face_range)
  {

    FEFaceEvaluation<dim,fe_degree, fe_degree+1,dim,Number> fe_eval(data,true,dof_quad_index_data.dof_index_velocity,dof_quad_index_data.quad_index_velocity);
    FEFaceEvaluation<dim,fe_degree, fe_degree+1,dim,Number> fe_eval_neighbor(data,false,dof_quad_index_data.dof_index_velocity,dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number> > JxW_values(fe_eval.n_q_points);
    Number diff_mass_flux = 0.;
    Number mean_mass_flux = 0.;

    for (unsigned int face=face_range.first; face<face_range.second; ++face)
    {
      fe_eval.reinit(face);
      fe_eval.read_dof_values(source);
      fe_eval.evaluate(true,false);
      fe_eval_neighbor.reinit(face);
      fe_eval_neighbor.read_dof_values(source);
      fe_eval_neighbor.evaluate(true,false);
      fe_eval.fill_JxW_values(JxW_values);

      VectorizedArray<Number> diff_mass_flux_vec = make_vectorized_array<Number>(0.);
      VectorizedArray<Number> mean_mass_flux_vec = make_vectorized_array<Number>(0.);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        diff_mass_flux_vec += JxW_values[q]*std::abs((fe_eval.get_value(q)-fe_eval_neighbor.get_value(q))*fe_eval.get_normal_vector(q));
        mean_mass_flux_vec += JxW_values[q]*std::abs(0.5*(fe_eval.get_value(q)+fe_eval_neighbor.get_value(q))*fe_eval.get_normal_vector(q));
      }

      // sum over entries of VectorizedArray, but only over those
      // that are "active"
      for (unsigned int v=0; v<data.n_active_entries_per_face_batch(face); ++v)
      {
        diff_mass_flux += diff_mass_flux_vec[v];
        mean_mass_flux += mean_mass_flux_vec[v];
      }
    }

    dst.at(2) += diff_mass_flux;
    dst.at(3) += mean_mass_flux;
  }

  void local_compute_div_boundary_face (const MatrixFree<dim,Number>                &,
                                        std::vector<Number>                         &,
                                        const parallel::distributed::Vector<Number> &,
                                        const std::pair<unsigned int,unsigned int>  &)
  {

  }

  void analyze_div_and_mass_error_unsteady(parallel::distributed::Vector<Number> const &velocity,
                                           double const                                time,
                                           unsigned int const                          time_step_number)
  {
    if(time > div_and_mass_data.start_time - 1.e-10)
    {
      Number div_error = 1.0, div_error_reference = 1.0, mass_error = 1.0, mass_error_reference = 1.0;

      // calculate divergence and mass error
      do_evaluate(*matrix_free_data, velocity, div_error, div_error_reference, mass_error, mass_error_reference);
      Number div_error_normalized = div_error/div_error_reference;
      Number mass_error_normalized = 1.0;
      if(mass_error_reference > 1.e-12)
        mass_error_normalized = mass_error/mass_error_reference;
      else
        mass_error_normalized = mass_error;

      // write output file
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::ostringstream filename;
        filename << div_and_mass_data.filename_prefix << ".div_mass_error_timeseries";

        std::ofstream f;
        if(clear_files_mass_error == true)
        {
          f.open(filename.str().c_str(),std::ios::trunc);
          f << "Error incompressibility constraint:" << std::endl << std::endl
            << "  (1,|divu|)_Omega/(1,1)_Omega" << std::endl << std::endl
            << "Error mass flux over interior element faces:" << std::endl << std::endl
            << "  (1,|(um - up)*n|)_dOmegaI / (1,|0.5(um + up)*n|)_dOmegaI" << std::endl << std::endl
            << "       t        |  divergence  |    mass       " << std::endl;

          clear_files_mass_error = false;
        }
        else
        {
          f.open(filename.str().c_str(),std::ios::app);
        }

        f << std::scientific << std::setprecision(7)
          << std::setw(15) << time
          << std::setw(15) << div_error_normalized
          << std::setw(15) << mass_error_normalized << std::endl;
      }

      if(time_step_number % div_and_mass_data.sample_every_time_steps == 0)
      {
        // calculate average error
        ++number_of_samples;
        divergence_sample += div_error_normalized;
        mass_sample += mass_error_normalized;

        // write output file
        if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
        {
          std::ostringstream filename;
          filename << div_and_mass_data.filename_prefix << ".div_mass_error_average";

          std::ofstream f;

          f.open(filename.str().c_str(),std::ios::trunc);
          f << "Divergence and mass error (averaged over time)" << std::endl;
          f << "Number of samples:   " << number_of_samples << std::endl;
          f << "Mean error incompressibility constraint:   " << divergence_sample/number_of_samples << std::endl;
          f << "Mean error mass flux over interior element faces:  " << mass_sample/number_of_samples << std::endl;
          f.close();
        }
      }
    }
  }

   void analyze_div_and_mass_error_steady(parallel::distributed::Vector<Number> const &velocity)
   {
     Number div_error = 1.0, div_error_reference = 1.0, mass_error = 1.0, mass_error_reference = 1.0;

     // calculate divergence and mass error
     do_evaluate(*matrix_free_data,velocity, div_error, div_error_reference, mass_error, mass_error_reference);
     Number div_error_normalized = div_error/div_error_reference;
     Number mass_error_normalized = 1.0;
     if(mass_error_reference > 1.e-12)
       mass_error_normalized = mass_error/mass_error_reference;
     else
       mass_error_normalized = mass_error;

     // write output file
     if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
     {
       std::ostringstream filename;
       filename << div_and_mass_data.filename_prefix << ".div_mass_error";

       std::ofstream f;

       f.open(filename.str().c_str(),std::ios::trunc);
       f << "Divergence and mass error:" << std::endl;
       f << "Error incompressibility constraint:   " << div_error_normalized << std::endl;
       f << "Error mass flux over interior element faces:  " << mass_error_normalized << std::endl;
       f.close();
     }
  }
};


}

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_DIVERGENCE_AND_MASS_ERROR_H_ */
