/*
 * kinetic_energy_data.h
 *
 *  Created on: Mar 15, 2018
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_DATA_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_DATA_H_

#include <deal.II/base/conditional_ostream.h>

#include "functionalities/print_functions.h"

struct KineticEnergyData
{
  KineticEnergyData()
    : calculate(false),
      evaluate_individual_terms(false),
      calculate_every_time_steps(std::numeric_limits<unsigned int>::max()),
      viscosity(1.0),
      filename_prefix("kinetic_energy")
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
      print_parameter(pcout, "Filename", filename_prefix);
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
  std::string filename_prefix;
};


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_KINETIC_ENERGY_DATA_H_ */
