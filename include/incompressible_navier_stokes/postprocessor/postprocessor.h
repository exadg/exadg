/*
 * PostProcessor.h
 *
 *  Created on: Aug 8, 2016
 *      Author: krank
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_

#include "../../incompressible_navier_stokes/postprocessor/divergence_and_mass_error.h"
#include "../../incompressible_navier_stokes/postprocessor/error_calculation_navier_stokes.h"
#include "../../incompressible_navier_stokes/postprocessor/lift_and_drag_calculation.h"
#include "../../incompressible_navier_stokes/postprocessor/lift_and_drag_data.h"
#include "../../incompressible_navier_stokes/postprocessor/output_data_navier_stokes.h"
#include "../../incompressible_navier_stokes/postprocessor/postprocessor_base.h"
#include "../../incompressible_navier_stokes/postprocessor/pressure_difference_calculation.h"
#include "../../incompressible_navier_stokes/postprocessor/pressure_difference_data.h"
#include "../../incompressible_navier_stokes/postprocessor/turbulence_statistics_data.h"
#include "../../incompressible_navier_stokes/postprocessor/write_output_navier_stokes.h"
#include "../../incompressible_navier_stokes/user_interface/input_parameters.h"

template<int dim>
struct PostProcessorData
{
  PostProcessorData(){}

  OutputDataNavierStokes output_data;
  ErrorCalculationData error_data;
  LiftAndDragData lift_and_drag_data;
  PressureDifferenceData<dim> pressure_difference_data;
  MassConservationData mass_data;
  TurbulenceStatisticsData turb_stat_data;
};

template<int dim, int fe_degree_u, int fe_degree_p, typename Number>
class PostProcessor : public PostProcessorBase<dim, Number>
{
public:
  PostProcessor(PostProcessorData<dim> const &postprocessor_data)
    :
    pp_data(postprocessor_data)
  {}

  virtual ~PostProcessor(){}

  virtual void setup(DoFHandler<dim> const                                 &dof_handler_velocity_in,
                     DoFHandler<dim> const                                 &dof_handler_pressure_in,
                     Mapping<dim> const                                    &mapping_in,
                     MatrixFree<dim,Number> const                          &matrix_free_data_in,
                     DofQuadIndexData const                                &dof_quad_index_data_in,
                     std::shared_ptr<AnalyticalSolutionNavierStokes<dim> > analytical_solution_in)
  {
    output_generator.setup(dof_handler_velocity_in,
                           dof_handler_pressure_in,
                           mapping_in,
                           pp_data.output_data);

    error_calculator.setup(dof_handler_velocity_in,
                           dof_handler_pressure_in,
                           mapping_in,
                           analytical_solution_in,
                           pp_data.error_data);

    lift_and_drag_calculator.setup(dof_handler_velocity_in,
                                   matrix_free_data_in,
                                   dof_quad_index_data_in,
                                   pp_data.lift_and_drag_data);

    pressure_difference_calculator.setup(dof_handler_pressure_in,
                                         mapping_in,
                                         pp_data.pressure_difference_data);

    div_and_mass_error_calculator.setup(matrix_free_data_in,
                                        dof_quad_index_data_in,
                                        pp_data.mass_data);
  }

  virtual void do_postprocessing(parallel::distributed::Vector<Number> const &velocity,
                                 parallel::distributed::Vector<Number> const &intermediate_velocity,
                                 parallel::distributed::Vector<Number> const &pressure,
                                 parallel::distributed::Vector<Number> const &vorticity,
                                 parallel::distributed::Vector<Number> const &divergence,
                                 double const                                time,
                                 int const                                   time_step_number)
  {
    /*
     *  write output
     */
    output_generator.write_output(velocity,pressure,vorticity,divergence,time,time_step_number);

    /*
     *  calculate error
     */
    error_calculator.evaluate(velocity,pressure,time,time_step_number);

    /*
     *  calculation of lift and drag coefficients
     */
    lift_and_drag_calculator.evaluate(velocity,pressure,time);

    /*
     *  calculation of pressure difference
     */
    pressure_difference_calculator.evaluate(pressure,time);

    /*
     *  Analysis of divergence and mass error
     */
    div_and_mass_error_calculator.evaluate(intermediate_velocity,time,time_step_number);
  };


private:
  PostProcessorData<dim> pp_data;

  OutputGenerator<dim,Number> output_generator;
  ErrorCalculator<dim,Number> error_calculator;
  LiftAndDragCalculator<dim,fe_degree_u,fe_degree_p,Number> lift_and_drag_calculator;
  PressureDifferenceCalculator<dim,fe_degree_u,fe_degree_p,Number> pressure_difference_calculator;
  DivergenceAndMassErrorCalculator<dim,fe_degree_u,Number> div_and_mass_error_calculator;
};




#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_ */
