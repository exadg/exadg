/*
 * perturbation_energy_orr_sommerfeld.h
 *
 *  Created on: Sep 1, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PERTURBATION_ENERGY_ORR_SOMMERFELD_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PERTURBATION_ENERGY_ORR_SOMMERFELD_H_

// deal.II
#include "deal.II/matrix_free/fe_evaluation_notemplate.h"

#include "../../utilities/print_functions.h"

struct PerturbationEnergyData
{
  PerturbationEnergyData()
    : calculate(false),
      calculate_every_time_steps(std::numeric_limits<unsigned int>::max()),
      filename_prefix("orr_sommerfeld"),
      omega_i(0.0),
      h(1.0),
      U_max(1.0)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << "  Calculate perturbation energy:" << std::endl;
      print_parameter(pcout, "Calculate perturbation energy", calculate);
      print_parameter(pcout, "Calculate every time steps", calculate_every_time_steps);
      print_parameter(pcout, "Filename output", filename_prefix);
      print_parameter(pcout, "Amplification omega_i", omega_i);
      print_parameter(pcout, "Channel height h", h);
      print_parameter(pcout, "Maximum velocity U_max", U_max);
    }
  }

  bool         calculate;
  unsigned int calculate_every_time_steps;
  std::string  filename_prefix;
  double       omega_i;
  double       h;
  double       U_max;
};

/*
 * Calculation of perturbation energy for Orr-Sommerfeld problem
 */
template<int dim, typename Number>
class PerturbationEnergyCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef PerturbationEnergyCalculator<dim, Number> This;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;

  PerturbationEnergyCalculator(MPI_Comm const & comm);

  void
  setup(MatrixFree<dim, Number> const & matrix_free_data_in,
        unsigned int const              dof_index_in,
        unsigned int const              quad_index_in,
        PerturbationEnergyData const &  data_in);

  void
  evaluate(VectorType const & velocity, double const & time, int const & time_step_number);

private:
  void
  do_evaluate(VectorType const & velocity, double const time, unsigned int const time_step_number);

  /*
   *  This function calculates the perturbation energy
   *
   *  Perturbation energy: E = (1,u*u)_Omega
   */
  void
  integrate(MatrixFree<dim, Number> const & matrix_free_data,
            VectorType const &              velocity,
            Number &                        energy);

  void
  local_compute(MatrixFree<dim, Number> const &               data,
                std::vector<Number> &                         dst,
                VectorType const &                            src,
                std::pair<unsigned int, unsigned int> const & cell_range);

  MPI_Comm const & mpi_comm;

  bool   clear_files;
  bool   initial_perturbation_energy_has_been_calculated;
  Number initial_perturbation_energy;

  MatrixFree<dim, Number> const * matrix_free_data;
  unsigned int                    dof_index, quad_index;
  PerturbationEnergyData          energy_data;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PERTURBATION_ENERGY_ORR_SOMMERFELD_H_ \
        */
