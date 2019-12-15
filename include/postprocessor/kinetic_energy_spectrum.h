/*
 * energy_spectrum_calculation.h
 *
 *  Created on: Feb 7, 2018
 *      Author: fehn/muench
 */

#ifndef INCLUDE_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_H_
#define INCLUDE_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_H_

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include "deal.II/matrix_free/matrix_free.h"

#include "../functionalities/print_functions.h"

// kinetic energy spectrum data

struct KineticEnergySpectrumData
{
  KineticEnergySpectrumData()
    : calculate(false),
      start_time(0.0),
      calculate_every_time_steps(-1),
      calculate_every_time_interval(-1.0),
      filename_prefix("energy_spectrum"),
      output_tolerance(std::numeric_limits<double>::min()),
      degree(0),
      evaluation_points_per_cell(0)
  {
  }

  void
  print(ConditionalOStream & pcout)
  {
    if(calculate == true)
    {
      pcout << std::endl << "  Calculate kinetic energy spectrum:" << std::endl;
      print_parameter(pcout, "Start time", start_time);
      if(calculate_every_time_steps >= 0)
        print_parameter(pcout, "Calculate every timesteps", calculate_every_time_steps);
      if(calculate_every_time_interval >= 0.0)
        print_parameter(pcout, "Calculate every time interval", calculate_every_time_interval);
      print_parameter(pcout, "Output precision", output_tolerance);
      print_parameter(pcout, "Evaluation points per cell", evaluation_points_per_cell);
    }
  }

  bool         calculate;
  double       start_time;
  int          calculate_every_time_steps;
  double       calculate_every_time_interval;
  std::string  filename_prefix;
  double       output_tolerance;
  unsigned int degree;
  unsigned int evaluation_points_per_cell;
};

template<int dim, typename Number>
class KineticEnergySpectrumCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  KineticEnergySpectrumCalculator();

  void
  setup(MatrixFree<dim, Number> const &   matrix_free_data_in,
        Triangulation<dim, dim> const &   tria,
        KineticEnergySpectrumData const & data_in);

  void
  evaluate(VectorType const & velocity, double const & time, int const & time_step_number);

private:
  void
  do_evaluate(VectorType const & velocity, double const time, unsigned int const time_step_number);

  bool                      clear_files;
  KineticEnergySpectrumData data;
  unsigned int              counter;
  bool                      reset_counter;
  const unsigned int        precision = 12;
};


#endif /* INCLUDE_POSTPROCESSOR_KINETIC_ENERGY_SPECTRUM_H_ */
