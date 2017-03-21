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

template<int dim, int fe_degree>
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

  void setup(MatrixFree<dim> const      &matrix_free_data_in,
             DofQuadIndexData const     &dof_quad_index_data_in,
             MassConservationData const &div_and_mass_data_in)
  {
    matrix_free_data = &matrix_free_data_in;
    dof_quad_index_data = dof_quad_index_data_in;
    div_and_mass_data = div_and_mass_data_in;
  }

  void evaluate(parallel::distributed::Vector<double> const &velocity,
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
  double divergence_sample;
  double mass_sample;

  MatrixFree<dim,double> const * matrix_free_data;
  DofQuadIndexData dof_quad_index_data;
  MassConservationData div_and_mass_data;

  /*
   *  This function calculates the divergence error and the error of mass flux
   *  over interior element faces.
   *
   *  Divergence error: (1,|divu|)_Omega
   *  Reference value for divergence error: (1,1)_Omega
   *
   *  Mass error: (1,|(um - up)*n|)_dOmegaI
   *  Reference value for mass error: (1,|0.5(um + up)*n|)_dOmegaI
   */
  void do_evaluate(MatrixFree<dim,double> const                &matrix_free_data,
                   parallel::distributed::Vector<double> const &velocity,
                   double                                      &div_error,
                   double                                      &div_error_reference,
                   double                                      &mass_error,
                   double                                      &mass_error_reference)
  {
    std::vector<double> dst(4,0.0);
    matrix_free_data.loop (&DivergenceAndMassErrorCalculator<dim,fe_degree>::local_compute_div,
                           &DivergenceAndMassErrorCalculator<dim,fe_degree>::local_compute_div_face,
                           &DivergenceAndMassErrorCalculator<dim,fe_degree>::local_compute_div_boundary_face,
                           this, dst, velocity);

    div_error = Utilities::MPI::sum (dst.at(0), MPI_COMM_WORLD);
    div_error_reference = Utilities::MPI::sum (dst.at(1), MPI_COMM_WORLD);
    mass_error = Utilities::MPI::sum (dst.at(2), MPI_COMM_WORLD);
    mass_error_reference = Utilities::MPI::sum (dst.at(3), MPI_COMM_WORLD);
  }

  void local_compute_div(const MatrixFree<dim,double>                &data,
                         std::vector<double>                         &dst,
                         const parallel::distributed::Vector<double> &source,
                         const std::pair<unsigned int,unsigned int>  &cell_range)
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,double> phi(data,dof_quad_index_data.dof_index_velocity,dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<double> > JxW_values(phi.n_q_points);
    VectorizedArray<double> div_vec = make_vectorized_array(0.);
    VectorizedArray<double> vol_vec = make_vectorized_array(0.);
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit(cell);
      phi.read_dof_values(source);
      phi.evaluate(false,true);
      phi.fill_JxW_values(JxW_values);

      for (unsigned int q=0; q<phi.n_q_points; ++q)
      {
        vol_vec += JxW_values[q];
        div_vec += JxW_values[q]*std::abs(phi.get_divergence(q));
      }
    }
    double div = 0.;
    double vol = 0.;
    for (unsigned int v=0;v<VectorizedArray<double>::n_array_elements;v++)
    {
      div += div_vec[v];
      vol += vol_vec[v];
    }
    dst.at(0) += div;
    dst.at(1) += vol;
  }

  void local_compute_div_face (const MatrixFree<dim,double>                &data,
                               std::vector<double >                        &dst,
                               const parallel::distributed::Vector<double> &source,
                               const std::pair<unsigned int,unsigned int>  &face_range)
  {

    FEFaceEvaluation<dim,fe_degree, fe_degree+1,dim,double> fe_eval(data,true,dof_quad_index_data.dof_index_velocity,dof_quad_index_data.quad_index_velocity);
    FEFaceEvaluation<dim,fe_degree, fe_degree+1,dim,double> fe_eval_neighbor(data,false,dof_quad_index_data.dof_index_velocity,dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<double> > JxW_values(fe_eval.n_q_points);
    VectorizedArray<double> diff_mass_flux_vec = make_vectorized_array(0.);
    VectorizedArray<double> mean_mass_flux_vec = make_vectorized_array(0.);
    for (unsigned int face=face_range.first; face<face_range.second; ++face)
    {
      fe_eval.reinit(face);
      fe_eval.read_dof_values(source);
      fe_eval.evaluate(true,false);
      fe_eval_neighbor.reinit(face);
      fe_eval_neighbor.read_dof_values(source);
      fe_eval_neighbor.evaluate(true,false);
      fe_eval.fill_JxW_values(JxW_values);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        mean_mass_flux_vec += JxW_values[q]*std::abs(0.5*(fe_eval.get_value(q)+fe_eval_neighbor.get_value(q))*fe_eval.get_normal_vector(q));

        diff_mass_flux_vec += JxW_values[q]*std::abs((fe_eval.get_value(q)-fe_eval_neighbor.get_value(q))*fe_eval.get_normal_vector(q));
      }
    }
    double diff_mass_flux = 0.;
    double mean_mass_flux = 0.;
    for (unsigned int v=0;v<VectorizedArray<double>::n_array_elements;v++)
    {
      diff_mass_flux += diff_mass_flux_vec[v];
      mean_mass_flux += mean_mass_flux_vec[v];
    }
    dst.at(2) += diff_mass_flux;
    dst.at(3) += mean_mass_flux;
  }

  void local_compute_div_boundary_face (const MatrixFree<dim,double>                 &,
                                        std::vector<double >                         &,
                                        const parallel::distributed::Vector<double>  &,
                                        const std::pair<unsigned int,unsigned int>   &)
  {

  }

  void analyze_div_and_mass_error_unsteady(parallel::distributed::Vector<double> const &velocity,
                                           double const                                time,
                                           unsigned int const                          time_step_number)
  {
    if(time > div_and_mass_data.start_time - 1.e-10)
    {
      double div_error = 1.0, div_error_reference = 1.0, mass_error = 1.0, mass_error_reference = 1.0;

      // calculate divergence and mass error
      do_evaluate(*matrix_free_data,velocity, div_error, div_error_reference, mass_error, mass_error_reference);
      double div_error_normalized = div_error/div_error_reference;
      double mass_error_normalized = 1.0;
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

   void analyze_div_and_mass_error_steady(parallel::distributed::Vector<double> const &velocity)
   {
     double div_error = 1.0, div_error_reference = 1.0, mass_error = 1.0, mass_error_reference = 1.0;

     // calculate divergence and mass error
     do_evaluate(*matrix_free_data,velocity, div_error, div_error_reference, mass_error, mass_error_reference);
     double div_error_normalized = div_error/div_error_reference;
     double mass_error_normalized = 1.0;
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



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_DIVERGENCE_AND_MASS_ERROR_H_ */
