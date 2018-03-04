/*
 * energy_spectrum_calculation.h
 *
 *  Created on: Feb 7, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_

#include <limits>
#include <vector>
#include <cmath>
#include <memory>

template<int dim, int fe_degree, typename Number>
class KineticEnergySpectrumCalculator
{
public:
  KineticEnergySpectrumCalculator()
    :
    clear_files(true),
    matrix_free_data(nullptr)
  {}

  void setup(MatrixFree<dim,Number> const    &matrix_free_data_in,
             DofQuadIndexData const          &dof_quad_index_data_in,
             KineticEnergySpectrumData const &data_in)
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
        do_evaluate(velocity,time,time_step_number);
      else // steady problem (time_step_number = -1)
      {
        AssertThrow(false, ExcMessage("Calculation of kinetic energy spectrum only implemented for unsteady problems."));
      }
    }
  }

private:
  void do_evaluate(parallel::distributed::Vector<Number> const &velocity,
                   double const                                time,
                   unsigned int const                          time_step_number)
  {
    if((time_step_number-1)%data.calculate_every_time_steps == 0)
    {
      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      {
        std::cout << "Calculate kinetic energy spectrum" << std::endl;
      }
    }
  }

  bool clear_files;
  MatrixFree<dim,Number> const * matrix_free_data;
  DofQuadIndexData dof_quad_index_data;
  KineticEnergySpectrumData data;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_ENERGY_SPECTRUM_CALCULATION_H_ */
