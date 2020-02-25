/*
 * pressure_difference_calculation.h
 *
 *  Created on: Oct 14, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POSTPROCESSOR_PRESSURE_DIFFERENCE_CALCULATION_H_
#define INCLUDE_POSTPROCESSOR_PRESSURE_DIFFERENCE_CALCULATION_H_

// deal.II
#include <deal.II/base/point.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>

using namespace dealii;

template<int dim>
struct PressureDifferenceData
{
  PressureDifferenceData() : calculate_pressure_difference(false), filename("pressure_difference")
  {
  }

  /*
   *  active or not
   */
  bool calculate_pressure_difference;

  /*
   *  Points:
   *  calculation of pressure difference: p(point_1) - p(point_2)
   */
  Point<dim> point_1;
  Point<dim> point_2;

  /*
   *  filenames
   */
  std::string filename;
};

template<int dim, typename Number>
class PressureDifferenceCalculator
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  PressureDifferenceCalculator(MPI_Comm const & comm);

  void
  setup(DoFHandler<dim> const &             dof_handler_pressure_in,
        Mapping<dim> const &                mapping_in,
        PressureDifferenceData<dim> const & pressure_difference_data_in);

  void
  evaluate(VectorType const & pressure, double const & time) const;

private:
  MPI_Comm const & mpi_comm;

  mutable bool clear_files_pressure_difference;

  SmartPointer<DoFHandler<dim> const> dof_handler_pressure;
  SmartPointer<Mapping<dim> const>    mapping;

  PressureDifferenceData<dim> pressure_difference_data;
};



#endif /* INCLUDE_POSTPROCESSOR_PRESSURE_DIFFERENCE_CALCULATION_H_ */
