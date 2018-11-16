/*
 * postprocessor.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_
#define INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_

// C++
#include <stdio.h>
#include <fstream>
#include <sstream>

// deal.II
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include "../user_interface/analytical_solution.h"
#include "../user_interface/input_parameters.h"
#include "incompressible_navier_stokes/postprocessor/energy_spectrum_calculation.h"
#include "incompressible_navier_stokes/postprocessor/kinetic_energy_calculation.h"
#include "incompressible_navier_stokes/postprocessor/lift_and_drag_calculation.h"
#include "incompressible_navier_stokes/postprocessor/lift_and_drag_data.h"
#include "incompressible_navier_stokes/postprocessor/postprocessor_base.h"
#include "incompressible_navier_stokes/postprocessor/pressure_difference_calculation.h"
#include "incompressible_navier_stokes/postprocessor/pressure_difference_data.h"
#include "postprocessor/error_calculation.h"
#include "write_output.h"

namespace CompNS
{
// forward declarations
template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
class DGOperator;

template<int dim>
struct PostProcessorData
{
  PostProcessorData() : calculate_velocity(false), calculate_pressure(false)
  {
  }

  bool calculate_velocity;
  bool calculate_pressure;

  OutputDataCompNavierStokes  output_data;
  ErrorCalculationData        error_data;
  LiftAndDragData             lift_and_drag_data;
  PressureDifferenceData<dim> pressure_difference_data;
  KineticEnergyData           kinetic_energy_data;
  KineticEnergySpectrumData   kinetic_energy_spectrum_data;
};

template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
class PostProcessor
{
public:
  typedef LinearAlgebra::distributed::Vector<double> VectorType;

  typedef DGOperator<dim, degree, n_q_points_conv, n_q_points_vis, Number> NavierStokesOperator;

  PostProcessor(PostProcessorData<dim> const & postprocessor_data) : pp_data(postprocessor_data)
  {
  }

  virtual ~PostProcessor()
  {
  }

  // TODO check if we need dof_handler_vector_in
  virtual void
  setup(NavierStokesOperator const & navier_stokes_operator_in,
        DoFHandler<dim> const &      dof_handler_in,
        DoFHandler<dim> const & /*dof_handler_vector_in*/,
        DoFHandler<dim> const &                          dof_handler_scalar_in,
        Mapping<dim> const &                             mapping_in,
        MatrixFree<dim, double> const &                  matrix_free_data_in,
        DofQuadIndexData const &                         dof_quad_index_data_in,
        std::shared_ptr<CompNS::AnalyticalSolution<dim>> analytical_solution_in)
  {
    navier_stokes_operator = &navier_stokes_operator_in;

    initialize_additional_vectors();

    output_generator.setup(dof_handler_in, mapping_in, pp_data.output_data);

    error_calculator.setup(dof_handler_in,
                           mapping_in,
                           analytical_solution_in->solution,
                           pp_data.error_data);

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

  virtual void
  do_postprocessing(VectorType const & solution, double const time, int const time_step_number)
  {
    /*
     * calculate derived quantities such as velocity, pressure, etc.
     */
    calculate_additional_vectors(solution);

    /*
     *  write output
     */
    output_generator.evaluate(solution, additional_fields, time, time_step_number);

    /*
     *  calculate error
     */
    error_calculator.evaluate(solution, time, time_step_number);

    /*
     *  calculation of lift and drag coefficients
     */
    lift_and_drag_calculator.evaluate(velocity, pressure, time);

    /*
     *  calculation of pressure difference
     */
    pressure_difference_calculator.evaluate(pressure, time);

    /*
     *  calculation of kinetic energy
     */
    kinetic_energy_calculator.evaluate(velocity, time, time_step_number);

    /*
     *  calculation of kinetic energy spectrum
     */
    kinetic_energy_spectrum_calculator.evaluate(velocity, time, time_step_number);
  }

protected:
  // DoF vectors for derived quantities: (p, u, T)
  VectorType pressure;
  VectorType velocity;
  VectorType temperature;
  VectorType vorticity;
  VectorType divergence;

  std::vector<SolutionField<dim, Number>> additional_fields;

private:
  void
  initialize_additional_vectors()
  {
    if(pp_data.output_data.write_pressure == true)
    {
      navier_stokes_operator->initialize_dof_vector_scalar(pressure);

      SolutionField<dim, Number> field;
      field.type        = SolutionFieldType::scalar;
      field.name        = "pressure";
      field.dof_handler = &navier_stokes_operator->get_dof_handler_scalar();
      field.vector      = &pressure;
      additional_fields.push_back(field);
    }

    // velocity
    if(pp_data.output_data.write_velocity == true)
    {
      navier_stokes_operator->initialize_dof_vector_dim_components(velocity);

      SolutionField<dim, Number> field;
      field.type        = SolutionFieldType::vector;
      field.name        = "velocity";
      field.dof_handler = &navier_stokes_operator->get_dof_handler_vector();
      field.vector      = &velocity;
      additional_fields.push_back(field);
    }

    // temperature
    if(pp_data.output_data.write_temperature == true)
    {
      navier_stokes_operator->initialize_dof_vector_scalar(temperature);

      SolutionField<dim, Number> field;
      field.type        = SolutionFieldType::scalar;
      field.name        = "temperature";
      field.dof_handler = &navier_stokes_operator->get_dof_handler_scalar();
      field.vector      = &temperature;
      additional_fields.push_back(field);
    }

    // vorticity
    if(pp_data.output_data.write_vorticity == true)
    {
      navier_stokes_operator->initialize_dof_vector_dim_components(vorticity);

      SolutionField<dim, Number> field;
      field.type        = SolutionFieldType::vector;
      field.name        = "vorticity";
      field.dof_handler = &navier_stokes_operator->get_dof_handler_vector();
      field.vector      = &vorticity;
      additional_fields.push_back(field);
    }

    // divergence
    if(pp_data.output_data.write_divergence == true)
    {
      navier_stokes_operator->initialize_dof_vector_scalar(divergence);

      SolutionField<dim, Number> field;
      field.type        = SolutionFieldType::scalar;
      field.name        = "velocity_divergence";
      field.dof_handler = &navier_stokes_operator->get_dof_handler_scalar();
      field.vector      = &divergence;
      additional_fields.push_back(field);
    }
  }

  void
  calculate_additional_vectors(VectorType const & solution)
  {
    if((pp_data.output_data.write_output == true && pp_data.output_data.write_pressure == true) ||
       pp_data.calculate_pressure == true)
    {
      navier_stokes_operator->compute_pressure(pressure, solution);
    }

    if((pp_data.output_data.write_output == true && pp_data.output_data.write_velocity == true) ||
       pp_data.calculate_velocity == true)
    {
      navier_stokes_operator->compute_velocity(velocity, solution);
    }

    if(pp_data.output_data.write_output == true && pp_data.output_data.write_temperature == true)
    {
      navier_stokes_operator->compute_temperature(temperature, solution);
    }

    if(pp_data.output_data.write_output == true && pp_data.output_data.write_vorticity == true)
    {
      AssertThrow(pp_data.calculate_velocity == true,
                  ExcMessage(
                    "The velocity field has to be computed in order to calculate the vorticity."));

      navier_stokes_operator->compute_vorticity(vorticity, velocity);
    }

    if(pp_data.output_data.write_output == true && pp_data.output_data.write_divergence == true)
    {
      AssertThrow(
        pp_data.calculate_velocity == true,
        ExcMessage(
          "The velocity field has to be computed in order to calculate the divergence of the velocity."));

      navier_stokes_operator->compute_divergence(divergence, velocity);
    }
  }

  PostProcessorData<dim> pp_data;

  SmartPointer<NavierStokesOperator const> navier_stokes_operator;

  OutputGenerator<dim>                                      output_generator;
  ErrorCalculator<dim, double>                              error_calculator;
  LiftAndDragCalculator<dim, degree, degree, double>        lift_and_drag_calculator;
  PressureDifferenceCalculator<dim, degree, degree, double> pressure_difference_calculator;
  KineticEnergyCalculator<dim, degree, double>              kinetic_energy_calculator;
  KineticEnergySpectrumCalculator<dim, degree, double>      kinetic_energy_spectrum_calculator;
};

} // namespace CompNS

#endif /* INCLUDE_COMPRESSIBLE_NAVIER_STOKES_POSTPROCESSOR_POSTPROCESSOR_H_ */
