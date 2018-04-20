/*
 * PostProcessorCompNavierStokes.h
 *
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_

// C++
#include <stdio.h>
#include <fstream>
#include <sstream>

// deal.II
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "../user_interface/analytical_solution.h"
#include "incompressible_navier_stokes/postprocessor/postprocessor_base.h"
#include "incompressible_navier_stokes/postprocessor/lift_and_drag_calculation.h"
#include "incompressible_navier_stokes/postprocessor/lift_and_drag_data.h"
#include "incompressible_navier_stokes/postprocessor/pressure_difference_calculation.h"
#include "incompressible_navier_stokes/postprocessor/pressure_difference_data.h"
#include "incompressible_navier_stokes/postprocessor/kinetic_energy_calculation.h"
#include "incompressible_navier_stokes/postprocessor/energy_spectrum_calculation.h"
#include "write_output.h"
#include "postprocessor/error_calculation.h"
#include "../user_interface/input_parameters.h"

namespace CompNS
{

template<int dim>
struct PostProcessorData
{
  PostProcessorData(){}

  OutputDataCompNavierStokes output_data;
  ErrorCalculationData error_data;
  LiftAndDragData lift_and_drag_data;
  PressureDifferenceData<dim> pressure_difference_data;
  KineticEnergyData kinetic_energy_data;
  KineticEnergySpectrumData kinetic_energy_spectrum_data;
};

template<int dim, int fe_degree>
class PostProcessor
{
public:
  PostProcessor(PostProcessorData<dim> const &postprocessor_data)
    :
    pp_data(postprocessor_data)
  {}

  virtual ~PostProcessor(){}

  virtual void setup(DoFHandler<dim> const                             &dof_handler_in,
                     DoFHandler<dim> const                             &dof_handler_vector_in,
                     DoFHandler<dim> const                             &dof_handler_scalar_in,
                     Mapping<dim> const                                &mapping_in,
                     MatrixFree<dim,double> const                      &matrix_free_data_in,
                     DofQuadIndexData const                            &dof_quad_index_data_in,
                     std::shared_ptr<CompNS::AnalyticalSolution<dim> > analytical_solution_in)
  {
    error_calculator.setup(dof_handler_in,
                           mapping_in,
                           analytical_solution_in->solution,
                           pp_data.error_data);

    output_generator.setup(dof_handler_in,
                           mapping_in,
                           pp_data.output_data);

    lift_and_drag_calculator.setup(dof_handler_in,
                                   matrix_free_data_in,
                                   dof_quad_index_data_in,
                                   pp_data.lift_and_drag_data);

    pressure_difference_calculator.setup(dof_handler_scalar_in,
                                         mapping_in,
                                         pp_data.pressure_difference_data);

    kinetic_energy_calculator.setup(matrix_free_data_in,
                                    dof_quad_index_data_in,
                                    pp_data.kinetic_energy_data);

    kinetic_energy_spectrum_calculator.setup(matrix_free_data_in,
                                             dof_quad_index_data_in,
                                             pp_data.kinetic_energy_spectrum_data);
  }

  virtual void do_postprocessing(parallel::distributed::Vector<double> const   &solution,
                                 parallel::distributed::Vector<double> const   &velocity,
                                 parallel::distributed::Vector<double> const   &pressure,
                                 std::vector<SolutionField<dim,double> > const &additional_fields,
                                 double const                                  time,
								                 int const                                     time_step_number);

private:
  PostProcessorData<dim> pp_data;

  OutputGenerator<dim> output_generator;
  ErrorCalculator<dim, double> error_calculator;
  LiftAndDragCalculator<dim, fe_degree, fe_degree, double> lift_and_drag_calculator;
  PressureDifferenceCalculator<dim, fe_degree, fe_degree, double> pressure_difference_calculator;
  KineticEnergyCalculator<dim, fe_degree, double> kinetic_energy_calculator;
  KineticEnergySpectrumCalculator<dim, fe_degree, double> kinetic_energy_spectrum_calculator;
};

template<int dim, int fe_degree>
void PostProcessor<dim, fe_degree>::
do_postprocessing(parallel::distributed::Vector<double> const   &solution,
                  parallel::distributed::Vector<double> const   &velocity,
                  parallel::distributed::Vector<double> const   &pressure,
                  std::vector<SolutionField<dim,double> > const &additional_fields,
                  double const                                  time,
                  int const							                        time_step_number)
{
  /*
   *  write output
   */
  output_generator.evaluate(solution,additional_fields,time,time_step_number);

  /*
   *  calculate error
   */
  error_calculator.evaluate(solution,time,time_step_number);

  /*
   *  calculation of lift and drag coefficients
   */
  lift_and_drag_calculator.evaluate(velocity, pressure, time);

  /*
   *  calculation of pressure difference
   */
  pressure_difference_calculator.evaluate(pressure,time);

  /*
   *  calculation of kinetic energy
   */
  kinetic_energy_calculator.evaluate(velocity,time,time_step_number);

  /*
   *  calculation of kinetic energy spectrum
   */
  kinetic_energy_spectrum_calculator.evaluate(velocity,time,time_step_number);
}

}

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_ */
