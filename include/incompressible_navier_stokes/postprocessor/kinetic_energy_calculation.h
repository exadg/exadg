/*
 * kinetic_energy_calculation.h
 *
 *  Created on: Jul 17, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_CALCULATION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_CALCULATION_H_

#include "../../incompressible_navier_stokes/spatial_discretization/curl_compute.h"

template<int dim, int fe_degree, typename Number>
class KineticEnergyCalculator
{
public:
  KineticEnergyCalculator()
    :
    clear_files(true),
    counter(0),
    matrix_free_data(nullptr)
  {}

  void setup(MatrixFree<dim,Number> const &matrix_free_data_in,
             DofQuadIndexData const       &dof_quad_index_data_in,
             KineticEnergyData const      &data_in)
  {
    matrix_free_data = &matrix_free_data_in;
    dof_quad_index_data = dof_quad_index_data_in;
    data = data_in;
  }

  void evaluate(parallel::distributed::Vector<Number> const &velocity,
                double const                                &time,
                int const                                   &time_step_number)
  {
    if(data.calculate == true)
    {
      if(time_step_number >= 0) // unsteady problem
        calculate_unsteady(velocity,time,time_step_number);
      else // steady problem (time_step_number = -1)
        calculate_steady(velocity);
    }
  }

private:
  bool clear_files;
  unsigned int counter;

  MatrixFree<dim,Number> const * matrix_free_data;
  DofQuadIndexData dof_quad_index_data;
  KineticEnergyData data;

  void calculate_unsteady(parallel::distributed::Vector<Number> const &velocity,
                          double const                                time,
                          unsigned int const                          time_step_number)
  {
    if((time_step_number-1)%data.calculate_every_time_steps == 0)
    {
      Number kinetic_energy = 1.0, enstrophy = 1.0, dissipation = 1.0;

      integrate(*matrix_free_data,velocity, kinetic_energy, enstrophy, dissipation);

      // write output file
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::ostringstream filename;
        filename << data.filename_prefix;

        std::ofstream f;
        if(clear_files == true)
        {
          f.open(filename.str().c_str(),std::ios::trunc);
          f << "Kinetic energy: E_k = 1/V * 1/2 * (u,u)_Omega, where V=(1,1)_Omega" << std::endl
            << "Dissipation rate: epsilon = nu/V * (grad(u),grad(u))_Omega, where V=(1,1)_Omega" << std::endl
            << "Enstrophy: E = 1/V * 1/2 * (rot(u),rot(u))_Omega, where V=(1,1)_Omega" << std::endl;

          f << std::endl
            << "  Time           Kin. energy    dissipation    enstrophy"<<std::endl;

          clear_files = false;
        }
        else
        {
          f.open(filename.str().c_str(),std::ios::app);
        }

        f << std::scientific << std::setprecision(7)
          << std::setw(15) << time
          << std::setw(15) << kinetic_energy
          << std::setw(15) << dissipation
          << std::setw(15) << enstrophy << std::endl;
      }
    }
  }

  void calculate_steady(parallel::distributed::Vector<Number> const &velocity)
  {
    Number kinetic_energy = 1.0, enstrophy = 1.0, dissipation = 1.0;

    integrate(*matrix_free_data,velocity, kinetic_energy, enstrophy, dissipation);

    // write output file
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
    {
      std::ostringstream filename;
      filename << data.filename_prefix;

      std::ofstream f;
      if(clear_files == true)
      {
        f.open(filename.str().c_str(),std::ios::trunc);
        f << "Kinetic energy: E_k = 1/V * 1/2 * (u,u)_Omega, where V=(1,1)_Omega" << std::endl
          << "Dissipation rate: epsilon = nu/V * (grad(u),grad(u))_Omega, where V=(1,1)_Omega" << std::endl
          << "Enstrophy: E = 1/V * 1/2 * (rot(u),rot(u))_Omega, where V=(1,1)_Omega" << std::endl;

        f << std::endl
          << "          Kin. energy    dissipation    enstrophy"<<std::endl;

        clear_files = false;
      }
      else
      {
        f.open(filename.str().c_str(),std::ios::app);
      }

      f << (counter++==0 ? "initial " : "solution")
        << std::scientific << std::setprecision(7)
        << std::setw(15) << kinetic_energy
        << std::setw(15) << dissipation
        << std::setw(15) << enstrophy << std::endl;
    }
  }

  /*
   *  This function calculates the kinetic energy
   *
   *  Kinetic energy: E_k = 1/V * 1/2 * (1,u*u)_Omega, V=(1,1)_Omega is the volume
   *
   *  Enstrophy: 1/V * 0.5 (1,rot(u)*rot(u))_Omega, V=(1,1)_Omega is the volume
   *
   *  Dissipation rate: epsilon = nu/V * (1, grad(u):grad(u))_Omega, V=(1,1)_Omega is the volume
   *
   *  Note that
   *
   *    epsilon = 2 * nu * Enstrophy
   *
   *  for incompressible flows (div(u)=0) and periodic boundary conditions.
   */
  void integrate(MatrixFree<dim,Number> const                &matrix_free_data,
                 parallel::distributed::Vector<Number> const &velocity,
                 Number                                      &energy,
                 Number                                      &enstrophy,
                 Number                                      &dissipation)
  {
    std::vector<Number> dst(4,0.0);
    matrix_free_data.cell_loop (&KineticEnergyCalculator<dim,fe_degree,Number>::local_compute,this, dst, velocity);

    // sum over all MPI processes
    Number volume = 1.0;
    volume = Utilities::MPI::sum (dst.at(0), MPI_COMM_WORLD);
    energy = Utilities::MPI::sum (dst.at(1), MPI_COMM_WORLD);
    enstrophy = Utilities::MPI::sum (dst.at(2), MPI_COMM_WORLD);
    dissipation = Utilities::MPI::sum (dst.at(3), MPI_COMM_WORLD);

    energy /= volume;
    enstrophy /= volume;
    dissipation /= volume;
  }

  void local_compute(const MatrixFree<dim,Number>                &data,
                     std::vector<Number>                         &dst,
                     const parallel::distributed::Vector<Number> &src,
                     const std::pair<unsigned int,unsigned int>  &cell_range)
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number> fe_eval(data,
                                                           dof_quad_index_data.dof_index_velocity,
                                                           dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number> > JxW_values(fe_eval.n_q_points);

    Number volume = 0.;
    Number energy = 0.;
    Number enstrophy = 0.;
    Number dissipation = 0.;

    // Loop over all elements
    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);
      fe_eval.fill_JxW_values(JxW_values);

      VectorizedArray<Number> volume_vec = make_vectorized_array<Number>(0.);
      VectorizedArray<Number> energy_vec = make_vectorized_array<Number>(0.);
      VectorizedArray<Number> enstrophy_vec = make_vectorized_array<Number>(0.);
      VectorizedArray<Number> dissipation_vec = make_vectorized_array<Number>(0.);

      for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
      {
        volume_vec += JxW_values[q];

        Tensor<1,dim,VectorizedArray<Number> > velocity = fe_eval.get_value(q);
        energy_vec += JxW_values[q]*make_vectorized_array<Number>(0.5)*velocity*velocity;

        Tensor<2,dim,VectorizedArray<Number> > velocity_gradient = fe_eval.get_gradient(q);
        dissipation_vec += JxW_values[q]*make_vectorized_array<Number>(this->data.viscosity)
                           * scalar_product(velocity_gradient,velocity_gradient);

        Tensor<1,dim,VectorizedArray<Number> > rotation =
            CurlCompute<dim,FEEvaluation<dim,fe_degree,fe_degree+1,dim,Number> >::compute(fe_eval,q);
        enstrophy_vec += JxW_values[q]*make_vectorized_array<Number>(0.5)*rotation*rotation;
      }

      // sum over entries of VectorizedArray, but only over those
      // that are "active"
      for (unsigned int v=0; v<data.n_active_entries_per_cell_batch(cell); ++v)
      {
        volume += volume_vec[v];
        energy += energy_vec[v];
        enstrophy += enstrophy_vec[v];
        dissipation += dissipation_vec[v];
      }
    }

    dst.at(0) += volume;
    dst.at(1) += energy;
    dst.at(2) += enstrophy;
    dst.at(3) += dissipation;
  }
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_CALCULATION_H_ */
