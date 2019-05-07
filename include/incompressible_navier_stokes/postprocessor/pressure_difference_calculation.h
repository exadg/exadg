/*
 * pressure_difference_calculation.h
 *
 *  Created on: Oct 14, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PRESSURE_DIFFERENCE_CALCULATION_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PRESSURE_DIFFERENCE_CALCULATION_H_

#include <fstream>
#include <sstream>

#include "postprocessor/evaluate_solution_in_given_point.h"

#include "../postprocessor/pressure_difference_data.h"

template<int dim, typename Number>
class PressureDifferenceCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  PressureDifferenceCalculator() : clear_files_pressure_difference(true)
  {
  }

  void
  setup(DoFHandler<dim> const &             dof_handler_pressure_in,
        Mapping<dim> const &                mapping_in,
        PressureDifferenceData<dim> const & pressure_difference_data_in)
  {
    dof_handler_pressure     = &dof_handler_pressure_in;
    mapping                  = &mapping_in;
    pressure_difference_data = pressure_difference_data_in;
  }

  void
  evaluate(VectorType const & pressure, double const & time) const
  {
    if(pressure_difference_data.calculate_pressure_difference == true)
    {
      Number pressure_1 = 0.0, pressure_2 = 0.0;

      Point<dim> point_1, point_2;
      point_1 = pressure_difference_data.point_1;
      point_2 = pressure_difference_data.point_2;

      evaluate_scalar_quantity_in_point(
        *dof_handler_pressure, *mapping, pressure, point_1, pressure_1);
      evaluate_scalar_quantity_in_point(
        *dof_handler_pressure, *mapping, pressure, point_2, pressure_2);

      Number const pressure_difference = pressure_1 - pressure_2;

      if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      {
        std::string filename = pressure_difference_data.filename;

        std::ofstream f;
        if(clear_files_pressure_difference)
        {
          f.open(filename.c_str(), std::ios::trunc);
          clear_files_pressure_difference = false;
        }
        else
        {
          f.open(filename.c_str(), std::ios::app);
        }

        unsigned int precision = 12;

        f << std::scientific << std::setprecision(precision) << time << "\t" << pressure_difference
          << std::endl;
        f.close();
      }
    }
  }

private:
  mutable bool clear_files_pressure_difference;

  SmartPointer<DoFHandler<dim> const> dof_handler_pressure;
  SmartPointer<Mapping<dim> const>    mapping;

  PressureDifferenceData<dim> pressure_difference_data;
};



#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_PRESSURE_DIFFERENCE_CALCULATION_H_ */
