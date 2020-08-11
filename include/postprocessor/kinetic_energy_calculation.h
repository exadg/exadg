/*
 * kinetic_energy_calculation.h
 *
 *  Created on: Jul 17, 2017
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_KINETIC_ENERGY_CALCULATION_H_
#define INCLUDE_POSTPROCESSOR_KINETIC_ENERGY_CALCULATION_H_

#include <deal.II/matrix_free/matrix_free.h>
#include "../matrix_free/integrators.h"

#include "../incompressible_navier_stokes/spatial_discretization/curl_compute.h"

#include "../utilities/print_functions.h"

struct KineticEnergyData
{
  KineticEnergyData()
    : calculate(false),
      evaluate_individual_terms(false),
      calculate_every_time_steps(std::numeric_limits<unsigned int>::max()),
      viscosity(1.0),
      filename("kinetic_energy"),
      clear_file(true)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << std::endl << "  Calculate kinetic energy:" << std::endl;

      print_parameter(pcout, "Calculate energy", calculate);
      print_parameter(pcout, "Evaluate individual terms", evaluate_individual_terms);
      print_parameter(pcout, "Calculate every timesteps", calculate_every_time_steps);
      print_parameter(pcout, "Filename", filename);
      print_parameter(pcout, "Clear file", clear_file);
    }
  }

  // calculate kinetic energy (dissipation)?
  bool calculate;

  // perform detailed analysis and evaluate contribution of individual terms (e.g., convective term,
  // viscous term) to overall kinetic energy dissipation?
  bool evaluate_individual_terms;

  // calculate every ... time steps
  unsigned int calculate_every_time_steps;

  // kinematic viscosity
  double viscosity;

  // filename
  std::string filename;
  bool        clear_file;
};

template<int dim, typename Number>
class KineticEnergyCalculator
{
public:
  static const unsigned int number_vorticity_components = (dim == 2) ? 1 : dim;

  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  KineticEnergyCalculator(MPI_Comm const & comm);

  void
  setup(MatrixFree<dim, Number> const & matrix_free_in,
        unsigned int const              dof_index_in,
        unsigned int const              quad_index_in,
        KineticEnergyData const &       kinetic_energy_data_in);

  void
  evaluate(VectorType const & velocity, double const & time, int const & time_step_number);

protected:
  void
  calculate_basic(VectorType const & velocity,
                  double const       time,
                  unsigned int const time_step_number);

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
  Number
  integrate(MatrixFree<dim, Number> const & matrix_free_data,
            VectorType const &              velocity,
            Number &                        energy,
            Number &                        enstrophy,
            Number &                        dissipation,
            Number &                        max_vorticity);

  void
  cell_loop(MatrixFree<dim, Number> const &               data,
            std::vector<Number> &                         dst,
            VectorType const &                            src,
            std::pair<unsigned int, unsigned int> const & cell_range);

  MPI_Comm const & mpi_comm;

  bool clear_files;

  MatrixFree<dim, Number> const * matrix_free;
  unsigned int                    dof_index, quad_index;
  KineticEnergyData               data;
};

#endif /* INCLUDE_POSTPROCESSOR_KINETIC_ENERGY_CALCULATION_H_ */
