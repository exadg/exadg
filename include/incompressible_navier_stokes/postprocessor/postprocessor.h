/*
 * PostProcessor.h
 *
 *  Created on: Aug 8, 2016
 *      Author: krank
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_

#include "../../incompressible_navier_stokes/postprocessor/divergence_and_mass_error.h"
#include "../../incompressible_navier_stokes/postprocessor/energy_spectrum_calculation.h"
#include "../../incompressible_navier_stokes/postprocessor/error_calculation_navier_stokes.h"
#include "../../incompressible_navier_stokes/postprocessor/kinetic_energy_calculation.h"
#include "../../incompressible_navier_stokes/postprocessor/lift_and_drag_calculation.h"
#include "../../incompressible_navier_stokes/postprocessor/lift_and_drag_data.h"
#include "../../incompressible_navier_stokes/postprocessor/line_plot_data.h"
#include "../../incompressible_navier_stokes/postprocessor/output_data_navier_stokes.h"
#include "../../incompressible_navier_stokes/postprocessor/postprocessor_base.h"
#include "../../incompressible_navier_stokes/postprocessor/pressure_difference_calculation.h"
#include "../../incompressible_navier_stokes/postprocessor/pressure_difference_data.h"
#include "../../incompressible_navier_stokes/postprocessor/turbulence_statistics_data.h"
#include "../../incompressible_navier_stokes/postprocessor/write_output_navier_stokes.h"

#include "../../incompressible_navier_stokes/user_interface/input_parameters.h"
#include "line_plot_calculation.h"

namespace IncNS
{
template<int dim>
struct PostProcessorData
{
  PostProcessorData()
  {
  }

  OutputDataNavierStokes      output_data;
  ErrorCalculationData        error_data;
  LiftAndDragData             lift_and_drag_data;
  PressureDifferenceData<dim> pressure_difference_data;
  MassConservationData        mass_data;
  TurbulenceStatisticsData    turb_stat_data;
  KineticEnergyData           kinetic_energy_data;
  KineticEnergySpectrumData   kinetic_energy_spectrum_data;
  LinePlotData<dim>           line_plot_data;
};

template<int dim, int fe_degree_u, int fe_degree_p, typename Number>
class PostProcessor : public PostProcessorBase<dim, fe_degree_u, fe_degree_p, Number>
{
public:
  typedef PostProcessorBase<dim, fe_degree_u, fe_degree_p, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef typename Base::NavierStokesOperator NavierStokesOperator;

  PostProcessor(PostProcessorData<dim> const & postprocessor_data) : pp_data(postprocessor_data)
  {
  }

  virtual ~PostProcessor()
  {
  }

  virtual void
  setup(NavierStokesOperator const &             navier_stokes_operator_in,
        DoFHandler<dim> const &                  dof_handler_velocity_in,
        DoFHandler<dim> const &                  dof_handler_pressure_in,
        Mapping<dim> const &                     mapping_in,
        MatrixFree<dim, Number> const &          matrix_free_data_in,
        DofQuadIndexData const &                 dof_quad_index_data_in,
        std::shared_ptr<AnalyticalSolution<dim>> analytical_solution_in)
  {
    output_generator.setup(navier_stokes_operator_in,
                           dof_handler_velocity_in,
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

    kinetic_energy_calculator.setup(navier_stokes_operator_in,
                                    matrix_free_data_in,
                                    dof_quad_index_data_in,
                                    pp_data.kinetic_energy_data);

    kinetic_energy_spectrum_calculator.setup(matrix_free_data_in,
                                             dof_handler_velocity_in.get_triangulation(),
                                             dof_quad_index_data_in,
                                             pp_data.kinetic_energy_spectrum_data);

    line_plot_calculator.setup(dof_handler_velocity_in,
                               dof_handler_pressure_in,
                               mapping_in,
                               pp_data.line_plot_data);
  }

  virtual void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & intermediate_velocity,
                    VectorType const & pressure,
                    double const       time             = 0.0,
                    int const          time_step_number = -1)
  {
    /*
     *  write output
     */
    output_generator.evaluate(velocity, intermediate_velocity, pressure, time, time_step_number);

    /*
     *  calculate error
     */
    error_calculator.evaluate(velocity, pressure, time, time_step_number);

    /*
     *  calculation of lift and drag coefficients
     */
    lift_and_drag_calculator.evaluate(velocity, pressure, time);

    /*
     *  calculation of pressure difference
     */
    pressure_difference_calculator.evaluate(pressure, time);

    /*
     *  Analysis of divergence and mass error
     */
    div_and_mass_error_calculator.evaluate(intermediate_velocity, time, time_step_number);

    /*
     *  calculation of kinetic energy
     */
    kinetic_energy_calculator.evaluate(velocity, time, time_step_number);

    /*
     *  calculation of kinetic energy spectrum
     */
    kinetic_energy_spectrum_calculator.evaluate(velocity, time, time_step_number);

    /*
     *  Evaluate fields along lines
     */
    line_plot_calculator.evaluate(velocity, pressure);
  };


private:
  PostProcessorData<dim> pp_data;

  // write output for visualization of results (e.g., using paraview)
  OutputGenerator<dim, fe_degree_u, fe_degree_p, Number> output_generator;

  // calculate errors for verification purposes for problems with known analytical solution
  ErrorCalculator<dim, Number> error_calculator;

  // calculate lift and drag forces for flow around bodies
  LiftAndDragCalculator<dim, fe_degree_u, fe_degree_p, Number> lift_and_drag_calculator;

  // calculate pressure difference between two points, e.g., the leading and trailing edge of a body
  PressureDifferenceCalculator<dim, fe_degree_u, fe_degree_p, Number>
    pressure_difference_calculator;

  // calculate divergence and continuity errors as a measure of mass conservation (particularly
  // relevant for turbulent flows)
  DivergenceAndMassErrorCalculator<dim, fe_degree_u, Number> div_and_mass_error_calculator;

  // calculate kinetic energy as well as dissipation rates (particularly relevant for turbulent
  // flows)
  KineticEnergyCalculatorDetailed<dim, fe_degree_u, fe_degree_p, Number> kinetic_energy_calculator;

  // evaluate kinetic energy in spectral space (i.e., as a function of the wavenumber)
  KineticEnergySpectrumCalculator<dim, fe_degree_u, Number> kinetic_energy_spectrum_calculator;

  // evaluate quantities along lines through the domain
  LinePlotCalculator<dim, fe_degree_u, fe_degree_p, Number> line_plot_calculator;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_ */
