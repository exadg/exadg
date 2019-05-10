/*
 * perturbation_energy_orr_sommerfeld.h
 *
 *  Created on: Sep 1, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PERTURBATION_ENERGY_ORR_SOMMERFELD_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PERTURBATION_ENERGY_ORR_SOMMERFELD_H_

#include "deal.II/matrix_free/fe_evaluation_notemplate.h"

template<int dim, typename Number>
class PerturbationEnergyCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef PerturbationEnergyCalculator<dim, Number> This;

  PerturbationEnergyCalculator()
    : clear_files(true),
      initial_perturbation_energy_has_been_calculated(false),
      initial_perturbation_energy(1.0),
      matrix_free_data(nullptr)
  {
  }

  void
  setup(MatrixFree<dim, Number> const & matrix_free_data_in,
        DofQuadIndexData const &        dof_quad_index_data_in,
        PerturbationEnergyData const &  data_in)
  {
    matrix_free_data    = &matrix_free_data_in;
    dof_quad_index_data = dof_quad_index_data_in;
    energy_data         = data_in;
  }

  void
  evaluate(VectorType const & velocity, double const & time, int const & time_step_number)
  {
    if(energy_data.calculate == true)
    {
      if(time_step_number >= 0) // unsteady problem
        calculate_unsteady(velocity, time, time_step_number);
      else // steady problem (time_step_number = -1)
        calculate_steady(velocity);
    }
  }

private:
  bool   clear_files;
  bool   initial_perturbation_energy_has_been_calculated;
  Number initial_perturbation_energy;

  MatrixFree<dim, Number> const * matrix_free_data;
  DofQuadIndexData                dof_quad_index_data;
  PerturbationEnergyData          energy_data;

  void
  calculate_unsteady(VectorType const & velocity,
                     double const       time,
                     unsigned int const time_step_number)
  {
    if((time_step_number - 1) % energy_data.calculate_every_time_steps == 0)
    {
      Number perturbation_energy = 0.0;

      integrate(*matrix_free_data, velocity, perturbation_energy);

      if(!initial_perturbation_energy_has_been_calculated)
      {
        initial_perturbation_energy = perturbation_energy;

        initial_perturbation_energy_has_been_calculated = true;
      }

      // write output file
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        // clang-format off
        unsigned int l = matrix_free_data->get_dof_handler(dof_quad_index_data.dof_index_velocity)
                           .get_triangulation().n_global_levels() - 1;
        // clang-format on

        std::ostringstream filename;
        filename << energy_data.filename_prefix + "_l" + Utilities::int_to_string(l);

        std::ofstream f;
        if(clear_files == true)
        {
          f.open(filename.str().c_str(), std::ios::trunc);
          f << "Perturbation energy: E = (1,(u-u_base)^2)_Omega" << std::endl
            << "Error:               e = |exp(2*omega_i*t) - E(t)/E(0)|" << std::endl;

          f << std::endl << "  Time                energy              error" << std::endl;

          clear_files = false;
        }
        else
        {
          f.open(filename.str().c_str(), std::ios::app);
        }

        Number const rel   = perturbation_energy / initial_perturbation_energy;
        Number const error = std::abs(std::exp(2 * energy_data.omega_i * time) - rel);

        unsigned int const precision = 12;
        f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
          << std::setw(precision + 8) << perturbation_energy << std::setw(precision + 8) << error
          << std::endl;
      }
    }
  }

  void
  calculate_steady(VectorType const & /*velocity*/)
  {
    AssertThrow(false,
                ExcMessage("Calculation of perturbation energy for "
                           "Orr-Sommerfeld problem only makes sense for unsteady problems."));
  }

  /*
   *  This function calculates the perturbation energy
   *
   *  Perturbation energy: E = (1,u*u)_Omega
   */
  void
  integrate(MatrixFree<dim, Number> const & matrix_free_data,
            VectorType const &              velocity,
            Number &                        energy)
  {
    std::vector<Number> dst(1, 0.0);
    matrix_free_data.cell_loop(&This::local_compute, this, dst, velocity);

    // sum over all MPI processes
    energy = Utilities::MPI::sum(dst.at(0), MPI_COMM_WORLD);
  }

  void
  local_compute(MatrixFree<dim, Number> const &               data,
                std::vector<Number> &                         dst,
                VectorType const &                            src,
                std::pair<unsigned int, unsigned int> const & cell_range)
  {
    CellIntegrator<dim, dim, Number> fe_eval(data,
                                             dof_quad_index_data.dof_index_velocity,
                                             dof_quad_index_data.quad_index_velocity);

    AlignedVector<VectorizedArray<Number>> JxW_values(fe_eval.n_q_points);

    // Loop over all elements
    for(unsigned int cell = cell_range.first; cell < cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true, false);
      fe_eval.fill_JxW_values(JxW_values);

      VectorizedArray<Number> energy_vec = make_vectorized_array<Number>(0.);
      for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
      {
        Tensor<1, dim, VectorizedArray<Number>> velocity = fe_eval.get_value(q);

        Point<dim, VectorizedArray<Number>> q_points = fe_eval.quadrature_point(q);

        VectorizedArray<Number> y = q_points[1] / energy_data.h;

        Tensor<1, dim, VectorizedArray<Number>> velocity_base;
        velocity_base[0] = energy_data.U_max * (1.0 - y * y);
        energy_vec += JxW_values[q] * (velocity - velocity_base) * (velocity - velocity_base);
      }

      // sum over entries of VectorizedArray, but only over those
      // that are "active"
      for(unsigned int v = 0; v < data.n_active_entries_per_cell_batch(cell); ++v)
      {
        dst.at(0) += energy_vec[v];
      }
    }
  }
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PERTURBATION_ENERGY_ORR_SOMMERFELD_H_ \
        */
