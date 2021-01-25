/*
 * lung.h
 *
 *  Created on: March 18, 2019
 *      Author: fehn
 */

#ifndef APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_LUNG_H_
#define APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_LUNG_H_

// ExaDG
#include <exadg/convection_diffusion/postprocessor/postprocessor.h>
#include <exadg/convection_diffusion/spatial_discretization/dg_operator.h>
#include <exadg/incompressible_flow_with_transport/user_interface/application_base.h>
#include <exadg/incompressible_navier_stokes/postprocessor/flow_rate_calculator.h>
#include <exadg/incompressible_navier_stokes/postprocessor/mean_velocity_calculator.h>
#include <exadg/postprocessor/mean_scalar_calculation.h>

// lung application
#include "include/grid/lung_environment.h"
#include "include/grid/lung_grid.h"

namespace ExaDG
{
namespace FTI
{
using namespace dealii;

// problem specific parameters

// which lung
//#define BABY
#define ADULT

// clang-format off

// triangulation type
TriangulationType const TRIANGULATION_TYPE = TriangulationType::Distributed;

// set problem specific parameters
double const VISCOSITY = 1.7e-5;  // m^2/s
double const D_OXYGEN = 0.219e-4; // 0.219 cm^2/s = 0.219e-4 m^2/s
double const DENSITY = 1.2;       // kg/m^3 (@ 20°C)

#ifdef BABY // preterm infant
double const PERIOD = 0.1; // 100 ms
unsigned int const N_PERIODS = 10;
double const START_TIME = 0.0;
double const END_TIME = PERIOD*N_PERIODS;
double const PEEP_KINEMATIC = 8.0 * 98.0665 / DENSITY;      // 8 cmH20, 1 cmH20 = 98.0665 Pa, transform to kinematic pressure
double const TIDAL_VOLUME = 6.6e-6;                         // 6.6 ml = 6.6 * 10^{-6} m^3
double const C_RS_KINEMATIC = DENSITY * 20.93e-9;           // total respiratory compliance C_rs = 20.93 ml/kPa (see Roth et al. (2018))
double const DELTA_P_INITIAL = TIDAL_VOLUME/C_RS_KINEMATIC; // initialize pressure difference in order to obtain desired tidal volume

// Menache et al. (2008): Extract diameter and length of airways from Table A1 (0.25-year-old female)
// and compute resistance of airways assuming laminar flow
unsigned int const MAX_GENERATION = 24;
double const RESISTANCE_VECTOR_DYNAMIC[MAX_GENERATION+1] = // resistance [Pa/(m^3/s)]
{
    9.59E+03, // GENERATION 0
    1.44E+04,
    3.66E+04,
    1.37E+05,
    5.36E+05,
    1.78E+06,
    4.36E+06,
    1.13E+07,
    2.60E+07,
    4.30E+07,
    8.46E+07,
    1.38E+08,
    2.29E+08,
    3.06E+08,
    3.64E+08,
    6.24E+08,
    9.02E+08,
    1.08E+09,
    1.36E+09,
    1.75E+09,
    2.41E+09,
    3.65E+09,
    3.45E+09,
    5.54E+09,
    1.62E+09 // MAX_GENERATION
};
#endif
#ifdef ADULT // adult lung
double const PERIOD = 3; // one period lasts 3 s
unsigned int const N_PERIODS = 10;
double const START_TIME = 0.0;
double const END_TIME = PERIOD*N_PERIODS;
double const PEEP_KINEMATIC = 8.0 * 98.0665 / DENSITY;      // 8 cmH20, 1 cmH20 = 98.0665 Pa, transform to kinematic pressure
double const TIDAL_VOLUME = 500.0e-6;                       // 500 ml = 500 * 10^{-6} m^3
double const C_RS_KINEMATIC = DENSITY * 100.0e-6/98.0665;   // total respiratory compliance C_rs = 100 ml/cm H20
double const DELTA_P_INITIAL = TIDAL_VOLUME/C_RS_KINEMATIC; // initialize pressure difference in order to obtain desired tidal volume


double const TOTAL_RESISTANCE = 0.15 * 1e6; // 0.15 kPa * l^{-1} * s, total lung resistance
double const TISSUE_RESISTANCE = 0.2 * TOTAL_RESISTANCE; // around 20% of total lung resistance (West2015)

// Menache et al. (2008): Extract diameter and length of airways from Table A11 (21-year-old male)
// and compute resistance of airways assuming laminar flow

// default is 1.0, but it was found experimentally that a larger resistance gives way better results
// for flow rate and volume profiles
double const SCALING_FACTOR_RESISTANCE = 1.0;
unsigned int const MAX_GENERATION = 25;
double const RESISTANCE_VECTOR_DYNAMIC[MAX_GENERATION+1] = // resistance [Pa/(m^3/s)]
{
    5.96E+02, // GENERATION 0
    3.87E+02,
    1.06E+03,
    2.57E+03,
    7.93E+03,
    3.04E+04,
    7.82E+04,
    2.35E+05,
    5.60E+05,
    1.50E+06,
    2.06E+06,
    3.24E+06,
    4.57E+06,
    6.38E+06,
    8.53E+06,
    1.11E+07,
    1.58E+07,
    2.08E+07,
    2.62E+07,
    3.39E+07,
    4.11E+07,
    5.04E+07,
    5.61E+07,
    6.34E+07,
    7.11E+07,
    4.73E+07 // MAX_GENERATION
};
#endif

// clang-format on

// time stepping
bool const   ADAPTIVE_TIME_STEPPING = true;
double const CFL                    = 0.4;
double const MAX_VELOCITY           = 1.0;
double const TIME_STEP_SIZE_MAX     = 1.e-4;

// solver tolerances
double const ABS_TOL = 1.e-12;
double const REL_TOL = 1.e-3;

// boundary ID trachea
types::boundary_id const TRACHEA_ID = 1;

// outlet boundary IDs
types::boundary_id const OUTLET_ID_FIRST = TRACHEA_ID + 1;
// initialize with OUTLET_ID_FIRST, changed later
types::boundary_id OUTLET_ID_LAST = OUTLET_ID_FIRST;

// restart
bool const   WRITE_RESTART         = false;
double const RESTART_INTERVAL_TIME = PERIOD;

// boundary conditions prescribed at the outlets require an effective resistance for each outlet
double
get_equivalent_resistance(unsigned int const max_resolved_generation,
                          unsigned int const max_generation)
{
  double resistance = 0.0;

  unsigned int const min_unresolved_generation = max_resolved_generation + 1;

  // calculate effective resistance for all higher generations not being resolved
  // assuming that all airways of a specific generation have the same resistance and that the flow
  // is laminar!
  for(unsigned int i = 0; i <= (max_generation - min_unresolved_generation); ++i)
  {
    resistance +=
      RESISTANCE_VECTOR_DYNAMIC[min_unresolved_generation + i] / std::pow(2.0, (double)i);
  }

  // beyond the current outflow boundary, we have two branches from generation
  // max_resolved_generation to generation max_generation, but the resistance computed above
  // corresponds to only one of the two branches
  resistance /= 2.0;


  // TODO: scale airway resistance by constant factor
  resistance *= SCALING_FACTOR_RESISTANCE;

#ifdef DEBUG
  std::cout << std::endl;
  std::cout << "airway outflow resistance: " << resistance << std::endl;
  // percentage of airway resistance at boundaries to total resistance
  double percent_of_total =
    resistance / std::pow(2.0, (double)max_resolved_generation) / TOTAL_RESISTANCE;
  std::cout << "percent of total: " << percent_of_total << std::endl << std::endl;
#endif

  // add tissue resistance
  resistance += TISSUE_RESISTANCE * std::pow(2.0, (double)max_resolved_generation);

#ifdef DEBUG
  std::cout << "total outflow resistance: " << resistance << std::endl;
  // percentage of modeled resistance to total resistance
  double total_percent_of_total =
    resistance / std::pow(2.0, (double)max_resolved_generation) / TOTAL_RESISTANCE;
  std::cout << "percent of total: " << total_percent_of_total << std::endl << std::endl;
#endif

  // the solver uses the kinematic pressure and therefore we have to transform the resistance
  resistance /= DENSITY;

  return resistance;
}

/*
 * This class controls the pressure at the inlet to obtain a desired tidal volume
 */
class Ventilator
{
public:
  Ventilator()
    : pressure_difference(DELTA_P_INITIAL),
      pressure_difference_last_period(DELTA_P_INITIAL),
      pressure_difference_damping(0.0),
      volume_max(std::numeric_limits<double>::min()),
      volume_min(std::numeric_limits<double>::max()),
      tidal_volume_last(TIDAL_VOLUME),
      C_I(0.4), // choose C_I = 0.1-1.0 (larger value might improve speed of convergence to desired
                // value; instabilities detected for C_I = 1 and larger)
      C_D(C_I * 0.2),
      counter(0),
      counter_last(0)
  {
  }

  double
  get_pressure(double const & time) const
  {
    const int    n_period = int(time / PERIOD);
    const double t_period = time - n_period * PERIOD;
    const double t_ramp   = PERIOD / 100.0;

    // TODO: generalize breathing periods
    // 0 <= (t-t_period_start) <= PERIOD/3
    if((int(time / (PERIOD / 3))) % 3 == 0) // inhaling
    {
      if(t_period <= t_ramp)
        return PEEP_KINEMATIC +
               t_period / t_ramp * (pressure_difference + pressure_difference_damping);
      else
        return PEEP_KINEMATIC + pressure_difference + pressure_difference_damping;
    }
    else // rest of the period (exhaling)
    {
      const double t_exhale = t_period - PERIOD / 3.0;

      if(t_exhale <= t_ramp)
        return PEEP_KINEMATIC +
               (1 - t_exhale / t_ramp) * (pressure_difference + pressure_difference_damping);
      else
        return PEEP_KINEMATIC;
    }
  }

  void
  update_pressure_difference(double const time, double const volume)
  {
    // always update volumes
    volume_max = std::max(volume, volume_max);
    volume_min = std::min(volume, volume_min);

    // recalculate pressure difference only once every period
    if(new_period(time))
    {
      // we first have to measure the tidal volume obtained in the first period before the
      // controller can be applied
      if(counter >= 1)
      {
        recalculate_pressure_difference();
      }

      // reset volumes
      volume_max = std::numeric_limits<double>::min();
      volume_min = std::numeric_limits<double>::max();
    }
  }

private:
  bool
  new_period(double const time)
  {
    counter = int(time / PERIOD);
    if(counter > counter_last)
    {
      counter_last = counter;
      return true;
    }
    else
    {
      return false;
    }
  }

  void
  recalculate_pressure_difference()
  {
    pressure_difference =
      pressure_difference_last_period + C_I * (TIDAL_VOLUME - (volume_max - volume_min)) /
                                          TIDAL_VOLUME * PEEP_KINEMATIC; // I-controller


    double const factor = 2.0;
    // limit pressure difference to a change of factor (assuming positive pressure_difference)
    pressure_difference = std::min(pressure_difference, pressure_difference_last_period * factor);
    pressure_difference = std::max(pressure_difference, pressure_difference_last_period / factor);

    // the damping part first be applied once we can compute a discrete derivative, i.e., after two
    // full periods
    if(counter >= 2)
      pressure_difference_damping = -C_D * ((volume_max - volume_min) - tidal_volume_last) /
                                    TIDAL_VOLUME * PEEP_KINEMATIC; // D-controller
    else
      pressure_difference_damping = 0.0;

    pressure_difference_last_period = pressure_difference;
    tidal_volume_last               = volume_max - volume_min;
  }

  double       pressure_difference;
  double       pressure_difference_last_period;
  double       pressure_difference_damping;
  double       volume_max;
  double       volume_min;
  double       tidal_volume_last;
  double const C_I, C_D;
  unsigned int counter;
  unsigned int counter_last;
};

std::shared_ptr<Ventilator> VENTILATOR;

/*
 * This class computes the pressure difference resulting from the resistance of the tubus
 */
class Tubus
{
public:
  Tubus()
    : K1I(8.41 * 1e3 * 98.0665 / DENSITY), // cmH2O*l^{-1}*s
      K2I(1.96),
      K1E(9.28 * 1e3 * 98.0665 / DENSITY), // cmH2O*l^{-1}*s
      K2E(1.81),
      pressure_difference_tubus(0.0)
  {
  }

  double
  get_pressure_difference() const
  {
    return pressure_difference_tubus;
  }

  void
  set_pressure_difference(double const flow_rate_trachea)
  {
    const double sign_flow = std::copysign(1.0, flow_rate_trachea);

    // sign trachea flow rate < 0 -> inhale
    if(sign_flow < 0)
      pressure_difference_tubus =
        -sign_flow * K1I * std::pow(std::abs(flow_rate_trachea) * 1e3, K2I) * 1e-3;
    else // -> exhale
      pressure_difference_tubus =
        -sign_flow * K1E * std::pow(std::abs(flow_rate_trachea) * 1e3, K2E) * 1e-3;
  }

private:
  const double K1I, K2I;
  const double K1E, K2E;

  double pressure_difference_tubus;
};

std::shared_ptr<Tubus> TUBUS;

template<int dim>
class PressureInlet : public Function<dim>
{
public:
  PressureInlet(std::shared_ptr<Ventilator> ventilator_,
                std::shared_ptr<Tubus>      tubus_,
                const double                time = 0.)
    : Function<dim>(1 /*n_components*/, time),
      ventilator(std::move(ventilator_)),
      tubus(std::move(tubus_))
  {
  }

  double
  value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
  {
    double t        = this->get_time();
    double pressure = ventilator->get_pressure(t) - tubus->get_pressure_difference();

    return pressure;
  }

private:
  std::shared_ptr<Ventilator> ventilator;
  std::shared_ptr<Tubus>      tubus;
};

class OutflowBoundary
{
public:
  OutflowBoundary(types::boundary_id const id,
                  unsigned int const       max_resolved_generation,
                  unsigned int const       max_generation)
    : boundary_id(id),
      resistance(get_equivalent_resistance(max_resolved_generation, max_generation)),
      // note that one could use a statistical distribution as in Roth et al. (2018)
      compliance(C_RS_KINEMATIC / std::pow(2.0, max_resolved_generation)),
      // p = 1/C * V -> V = C * p (initialize volume so that p(t=0) = PEEP_KINEMATIC)
      volume(compliance * PEEP_KINEMATIC),
      flow_rate(0.0),
      time_old(START_TIME)
  {
  }

  void
  set_flow_rate(double const flow_rate_)
  {
    flow_rate = flow_rate_;
  }

  void
  integrate_volume(double const time)
  {
    // currently use BDF1 time integration // TODO one could use a higher order time integrator
    volume += flow_rate * (time - time_old);
    time_old = time;
  }

  double
  get_pressure() const
  {
    return resistance * flow_rate + volume / compliance;
  }

  double
  get_volume() const
  {
    return volume;
  }

  types::boundary_id
  get_boundary_id() const
  {
    return boundary_id;
  }

private:
  types::boundary_id const boundary_id;
  double                   resistance;
  double                   compliance;
  double                   volume;
  double                   flow_rate;
  double                   time_old;
};

// we need individual outflow boundary conditions for each outlet
std::vector<std::shared_ptr<OutflowBoundary>> OUTFLOW_BOUNDARIES;

template<int dim>
class PressureOutlet : public Function<dim>
{
public:
  explicit PressureOutlet(std::shared_ptr<OutflowBoundary> outflow_boundary_,
                          double const                     time = 0.)
    : Function<dim>(1 /*n_components*/, time), outflow_boundary(outflow_boundary_)
  {
  }

  double
  value(const Point<dim> & /*p*/, const unsigned int /*component*/) const
  {
    return outflow_boundary->get_pressure();
  }

private:
  std::shared_ptr<OutflowBoundary> outflow_boundary;
};

template<int dim>
class InhaleDirichletBCScalar : public Function<dim>
{
public:
  explicit InhaleDirichletBCScalar(const unsigned int n_components = 1, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & /*p*/, const unsigned int /*component = 0*/) const
  {
    return 1.0;
  }
};

template<int dim>
class ExhaleDirichletBCScalar : public Function<dim>
{
public:
  explicit ExhaleDirichletBCScalar(const double       mean_scalar,
                                   const unsigned int n_components = 1,
                                   const double       time         = 0.)
    : Function<dim>(n_components, time), mean_scalar(mean_scalar)
  {
  }

  double
  value(const Point<dim> & /*p*/, const unsigned int /*component = 0*/) const
  {
    return mean_scalar;
  }

private:
  double mean_scalar;
};

template<int dim>
void
set_inhale_bc_scalar(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor)
{
  // 1 = inlet
  boundary_descriptor->dirichlet_bc.insert({1, std::make_shared<InhaleDirichletBCScalar<dim>>()});

  // outlets
  for(types::boundary_id id = OUTLET_ID_FIRST; id < OUTLET_ID_LAST; ++id)
  {
    boundary_descriptor->neumann_bc.insert({id, std::make_shared<Functions::ZeroFunction<dim>>(1)});
  }
}

template<int dim>
struct PostProcessorDataLung
{
  IncNS::PostProcessorData<dim>      pp_data;
  IncNS::FlowRateCalculatorData<dim> flow_rate_data;
  IncNS::FlowRateCalculatorData<dim> flow_rate_data_trachea;
};

template<int dim, typename Number>
class PostProcessorLung : public IncNS::PostProcessor<dim, Number>
{
public:
  typedef IncNS::PostProcessor<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef typename Base::Operator Operator;

  PostProcessorLung(PostProcessorDataLung<dim> const & pp_data_in,
                    MPI_Comm const &                   comm,
                    std::string const &                output_directory,
                    std::string const &                output_name)
    : Base(pp_data_in.pp_data, comm),
      pp_data_lung(pp_data_in),
      time_last(START_TIME),
      output_directory(output_directory),
      output_name(output_name)
  {
  }

  void
  setup(Operator const & pde_operator)
  {
    // call setup function of base class
    Base::setup(pde_operator);

    // fill flow_rates map
    for(auto & iterator : OUTFLOW_BOUNDARIES)
    {
      flow_rates.insert({iterator->get_boundary_id(), 0.0});
    }

    flow_rate_calculator.reset(
      new IncNS::FlowRateCalculator<dim, Number>(pde_operator.get_matrix_free(),
                                                 pde_operator.get_dof_index_velocity(),
                                                 pde_operator.get_quad_index_velocity_linear(),
                                                 pp_data_lung.flow_rate_data,
                                                 this->mpi_comm));

    flow_rate_calculator_trachea.reset(
      new IncNS::FlowRateCalculator<dim, Number>(pde_operator.get_matrix_free(),
                                                 pde_operator.get_dof_index_velocity(),
                                                 pde_operator.get_quad_index_velocity_linear(),
                                                 pp_data_lung.flow_rate_data_trachea,
                                                 this->mpi_comm));
  }

  void
  do_postprocessing(VectorType const & velocity,
                    VectorType const & pressure,
                    double const       time,
                    int const          time_step_number)
  {
    Base::do_postprocessing(velocity, pressure, time, time_step_number);

    // calculate flow rates for all outflow boundaries
    AssertThrow(pp_data_lung.flow_rate_data.calculate == true,
                ExcMessage("Activate flow rate computation."));

    flow_rate_calculator->calculate_flow_rates(velocity, time, flow_rates);

    // set flow rate for all outflow boundaries and update volume (i.e., integrate flow rate over
    // time)
    Number volume = 0.0;
    for(auto & iterator : OUTFLOW_BOUNDARIES)
    {
      iterator->set_flow_rate(flow_rates.at(iterator->get_boundary_id()));
      iterator->integrate_volume(time);
      volume += iterator->get_volume();
    }

    // write volume to file
    if(pp_data_lung.flow_rate_data.write_to_file)
    {
      std::ostringstream filename;
      filename << output_directory + output_name + "_volume";
      write_output(volume, time, "Volume in [m^3]", time_step_number, filename);

      // write time step size
      std::ostringstream filename_dt;
      filename_dt << output_directory + output_name + "_time_step_size";
      write_output(time - time_last, time, "Time step size in [s]", time_step_number, filename_dt);
      time_last = time;
    }

    // calculate the flow rate of the trachea inlet
    std::map<types::boundary_id, Number> map_trachea = {{TRACHEA_ID, 0.0}};

    flow_rate_calculator_trachea->calculate_flow_rates(velocity, time, map_trachea);

    // extract flow rate from map
    Number flow_rate_trachea = map_trachea.at(TRACHEA_ID);

    TUBUS->set_pressure_difference(flow_rate_trachea);

    // update the ventilator using the new volume
    VENTILATOR->update_pressure_difference(time, volume);

    // write pressure to file
    if(pp_data_lung.flow_rate_data.write_to_file)
    {
      double const pressure_trachea =
        VENTILATOR->get_pressure(time) - TUBUS->get_pressure_difference();
      std::ostringstream filename;
      filename << output_directory + output_name + "_pressure";
      write_output(pressure_trachea, time, "Pressure in [m^2/s^2]", time_step_number, filename);
    }
  }

private:
  void
  write_output(double const &             value,
               double const &             time,
               std::string const &        name,
               unsigned int const         time_step_number,
               std::ostringstream const & filename)
  {
    // write output file
    if(Utilities::MPI::this_mpi_process(this->mpi_comm) == 0)
    {
      std::ofstream f;
      if(time_step_number == 1)
      {
        f.open(filename.str().c_str(), std::ios::trunc);
        f << std::endl << "  Time                " + name << std::endl;
      }
      else
      {
        f.open(filename.str().c_str(), std::ios::app);
      }

      int precision = 12;
      f << std::scientific << std::setprecision(precision) << std::setw(precision + 8) << time
        << std::setw(precision + 8) << value << std::endl;
    }
  }

  // postprocessor data supplemented with data required for lung
  PostProcessorDataLung<dim> pp_data_lung;

  // we need to compute the flow rate for each outlet
  std::map<types::boundary_id, Number> flow_rates;

  // calculate flow rates for all outflow boundaries
  std::shared_ptr<IncNS::FlowRateCalculator<dim, Number>> flow_rate_calculator;

  // calculate flow rate for trachea
  std::shared_ptr<IncNS::FlowRateCalculator<dim, Number>> flow_rate_calculator_trachea;

  double time_last;

  std::string const output_directory, output_name;
};

template<int dim, typename Number>
class PostProcessorLungScalar : public ConvDiff::PostProcessor<dim, Number>
{
public:
  typedef ConvDiff::PostProcessor<dim, Number> Base;

  typedef typename Base::VectorType VectorType;

  typedef typename Base::Operator Operator;

  PostProcessorLungScalar(
    ConvDiff::PostProcessorData<dim> const &                        pp_data_in,
    MPI_Comm const &                                                comm,
    std::vector<std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>>> boundary_descriptor_in)
    : Base(pp_data_in, comm),
      boundary_descriptor_scalar(boundary_descriptor_in),
      is_inhale_old(true)
  {
  }

  void
  setup(Operator const & pde_operator, const Mapping<dim> & mapping) override
  {
    Base::setup(pde_operator, mapping);

    // fill mean_scalar map
    for(auto & iterator : OUTFLOW_BOUNDARIES)
    {
      mean_scalar.insert({iterator->get_boundary_id(), 0.0});
    }

    mean_scalar_calculator.reset(
      new MeanScalarCalculator<dim, Number>(pde_operator.get_matrix_free(),
                                            pde_operator.get_dof_index(),
                                            pde_operator.get_quad_index(),
                                            this->mpi_comm));
  }

  void
  do_postprocessing(const VectorType & solution,
                    const double       time,
                    const int          time_step_number) override
  {
    Base::do_postprocessing(solution, time, time_step_number);

    bool is_inhale_now = is_inhale(time);

    if(is_inhale_old != is_inhale_now)
    {
      if(is_inhale_now)
        for(auto & bc_des : boundary_descriptor_scalar)
        {
          erase_exhale_bc_scalar(bc_des);
          set_inhale_bc_scalar(bc_des);
        }
      else // exhale
        for(auto & bc_des : boundary_descriptor_scalar)
        {
          mean_scalar_calculator->calculate_mean_scalar(solution, mean_scalar);
          erase_inhale_bc_scalar(bc_des);
          set_exhale_bc_scalar(bc_des);
        }
    }
  }

private:
  bool
  is_inhale(const double time)
  {
    // TODO: generalize breathing periods
    return (int(time / (PERIOD / 3))) % 3 == 0;
  }

  void
  erase_inhale_bc_scalar(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor)
  {
    // 1 = inlet
    boundary_descriptor->dirichlet_bc.erase(TRACHEA_ID);

    // outlets
    for(types::boundary_id id = OUTLET_ID_FIRST; id < OUTLET_ID_LAST; ++id)
    {
      boundary_descriptor->neumann_bc.erase(id);
    }
  }

  void
  erase_exhale_bc_scalar(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor)
  {
    // 1 = inlet
    boundary_descriptor->neumann_bc.erase(TRACHEA_ID);

    // outlets
    for(types::boundary_id id = OUTLET_ID_FIRST; id < OUTLET_ID_LAST; ++id)
    {
      boundary_descriptor->dirichlet_bc.erase(id);
    }
  }

  void
  set_exhale_bc_scalar(std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor)
  {
    // 1 = inlet
    boundary_descriptor->neumann_bc.insert_or_assign(
      1, std::make_shared<Functions::ZeroFunction<dim>>(TRACHEA_ID));

    // outlets
    for(types::boundary_id id = OUTLET_ID_FIRST; id < OUTLET_ID_LAST; ++id)
    {
      boundary_descriptor->dirichlet_bc.insert_or_assign(
        id, std::make_shared<ExhaleDirichletBCScalar<dim>>(mean_scalar.at(id)));
    }
  }

  // we need to compute the mean scalar for each outlet
  std::map<types::boundary_id, Number> mean_scalar;

  std::shared_ptr<MeanScalarCalculator<dim, Number>> mean_scalar_calculator;

  std::vector<std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>>> boundary_descriptor_scalar;

  bool is_inhale_old;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  explicit Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
    prm.add_parameter("DirectoryLungFiles", directory_lung_files, "Directory where to find files for lung geometry.");
    prm.add_parameter("MaxGeneration", max_resolved_generation, "Highest resolved generation, starting with 0 for the trachea.");
    prm.leave_subsection();
    // clang-format on
  }

  std::string  directory_lung_files;
  unsigned int max_resolved_generation = 5;

  // output
  bool const   high_order_output    = true;
  double const output_start_time    = START_TIME;
  double const output_interval_time = PERIOD / 30;

  std::vector<std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>>> boundary_descriptor_scalar;

  void
  set_input_parameters(IncNS::InputParameters & param)
  {
    using namespace IncNS;

    // MATHEMATICAL MODEL
    param.problem_type                   = ProblemType::Unsteady;
    param.equation_type                  = EquationType::NavierStokes;
    param.formulation_viscous_term       = FormulationViscousTerm::LaplaceFormulation;
    param.formulation_convective_term    = FormulationConvectiveTerm::DivergenceFormulation;
    param.use_outflow_bc_convective_term = true;
    param.right_hand_side                = false;

    // PHYSICAL QUANTITIES
    param.start_time = START_TIME;
    param.end_time   = END_TIME;
    param.viscosity  = VISCOSITY;

    // TEMPORAL DISCRETIZATION
    param.solver_type                     = SolverType::Unsteady;
    param.temporal_discretization         = TemporalDiscretization::BDFDualSplittingScheme;
    param.treatment_of_convective_term    = TreatmentOfConvectiveTerm::Explicit;
    param.time_integrator_oif             = TimeIntegratorOIF::ExplRK2Stage2;
    param.calculation_of_time_step_size   = TimeStepCalculation::CFL;
    param.adaptive_time_stepping          = ADAPTIVE_TIME_STEPPING;
    param.time_step_size_max              = TIME_STEP_SIZE_MAX;
    param.max_velocity                    = MAX_VELOCITY;
    param.cfl                             = CFL;
    param.cfl_oif                         = CFL;
    param.cfl_exponent_fe_degree_velocity = 1.5;
    param.order_time_integrator           = 2;
    param.start_with_low_order            = true;

    // output of solver information
    param.solver_info_data.interval_time = PERIOD / 30;

    // NUMERICAL PARAMETERS
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;

    // SPATIAL DISCRETIZATION
    param.triangulation_type = TRIANGULATION_TYPE;
    param.degree_p           = DegreePressure::MixedOrder;
    param.mapping            = MappingType::Affine;

    // convective term
    if(param.formulation_convective_term == FormulationConvectiveTerm::DivergenceFormulation)
      param.upwind_factor = 0.5; // allows using larger CFL values for explicit formulations

    // viscous term
    param.IP_formulation_viscous = InteriorPenaltyFormulation::SIPG;

    // div-div and continuity penalty
    param.use_divergence_penalty                     = true;
    param.divergence_penalty_factor                  = 1.0e0;
    param.use_continuity_penalty                     = true;
    param.continuity_penalty_factor                  = param.divergence_penalty_factor;
    param.apply_penalty_terms_in_postprocessing_step = true;

    // PROJECTION METHODS

    // pressure Poisson equation
    param.solver_data_pressure_poisson               = SolverData(1000, ABS_TOL, REL_TOL, 100);
    param.preconditioner_pressure_poisson            = PreconditionerPressurePoisson::Multigrid;
    param.multigrid_data_pressure_poisson.type       = MultigridType::cphMG;
    param.multigrid_data_pressure_poisson.p_sequence = PSequenceType::Bisect;
    param.multigrid_data_pressure_poisson.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    param.multigrid_data_pressure_poisson.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;

    // projection step
    param.solver_projection                = SolverProjection::CG;
    param.solver_data_projection           = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_projection        = PreconditionerProjection::InverseMassMatrix;
    param.update_preconditioner_projection = false;

    // HIGH-ORDER DUAL SPLITTING SCHEME

    // formulations
    param.order_extrapolation_pressure_nbc =
      param.order_time_integrator <= 2 ? param.order_time_integrator : 2;

    // viscous step
    param.solver_viscous         = SolverViscous::CG;
    param.solver_data_viscous    = SolverData(1000, ABS_TOL, REL_TOL);
    param.preconditioner_viscous = PreconditionerViscous::InverseMassMatrix;

    // PRESSURE-CORRECTION SCHEME

    // momentum step

    // Newton solver
    param.newton_solver_data_momentum = Newton::SolverData(100, 1.e-20, 1.e-6);

    // linear solver
    param.solver_momentum = SolverMomentum::GMRES;
    if(param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
      param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-2, 100);
    else
      param.solver_data_momentum = SolverData(1e4, 1.e-12, 1.e-6, 100);

    param.preconditioner_momentum        = MomentumPreconditioner::InverseMassMatrix;
    param.update_preconditioner_momentum = true;

    // formulation
    param.order_pressure_extrapolation = param.order_time_integrator - 1;
    param.rotational_formulation       = true;


    // COUPLED NAVIER-STOKES SOLVER
    param.use_scaling_continuity = false;

    // nonlinear solver (Newton solver)
    param.newton_solver_data_coupled = Newton::SolverData(100, 1.e-12, 1.e-6);

    // linear solver
    param.solver_coupled      = SolverCoupled::GMRES;
    param.solver_data_coupled = SolverData(1e3, 1.e-12, 1.e-6, 100);

    // preconditioning linear solver
    param.preconditioner_coupled = PreconditionerCoupled::BlockTriangular;

    // preconditioner velocity/momentum block
    param.preconditioner_velocity_block = MomentumPreconditioner::InverseMassMatrix;

    // preconditioner Schur-complement block
    param.preconditioner_pressure_block =
      SchurComplementPreconditioner::CahouetChabard; // PressureConvectionDiffusion;
  }

  void
  set_input_parameters_scalar(ConvDiff::InputParameters & param, unsigned int const scalar_index)
  {
    using namespace ConvDiff;

    // MATHEMATICAL MODEL
    param.problem_type                = ProblemType::Unsteady;
    param.equation_type               = EquationType::ConvectionDiffusion;
    param.analytical_velocity_field   = false;
    param.right_hand_side             = false;
    param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;

    // PHYSICAL QUANTITIES
    param.start_time = START_TIME;
    param.end_time   = END_TIME;
    if(scalar_index == 0)
    {
      param.diffusivity = D_OXYGEN;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization       = TemporalDiscretization::BDF;
    param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Explicit;
    param.adaptive_time_stepping        = ADAPTIVE_TIME_STEPPING;
    param.order_time_integrator         = 2;
    param.time_integrator_oif           = TimeIntegratorRK::ExplRK3Stage7Reg2;
    param.start_with_low_order          = true;
    param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    param.cfl                           = CFL;
    param.cfl_oif                       = CFL;
    param.max_velocity                  = MAX_VELOCITY;
    param.time_step_size_max            = TIME_STEP_SIZE_MAX;
    param.exponent_fe_degree_convection = 1.5;
    param.diffusion_number              = 0.01;

    // restart
    param.restart_data.write_restart = WRITE_RESTART;
    param.restart_data.interval_time = RESTART_INTERVAL_TIME;
    param.restart_data.filename =
      this->output_directory + this->output_name + "_scalar_" + std::to_string(scalar_index);

    // SPATIAL DISCRETIZATION

    // triangulation
    param.triangulation_type = TRIANGULATION_TYPE;

    // mapping
    param.mapping = MappingType::Affine;

    // convective term
    param.numerical_flux_convective_operator = NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    param.IP_factor = 1.0;

    // SOLVER
    param.solver         = ConvDiff::Solver::GMRES;
    param.solver_data    = SolverData(1e4, 1.e-12, 1.e-6, 100);
    param.preconditioner = Preconditioner::InverseMassMatrix; // BlockJacobi; //Multigrid;
    param.implement_block_diagonal_preconditioner_matrix_free = false;
    param.use_cell_based_face_loops                           = false;
    param.update_preconditioner                               = false;

    param.multigrid_data.type = MultigridType::hMG;
    param.mg_operator_type    = MultigridOperatorType::ReactionConvectionDiffusion;
    // MG smoother
    param.multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi;
    // MG smoother data
    param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
    param.multigrid_data.smoother_data.iterations     = 5;

    // MG coarse grid solver
    param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::GMRES;

    // output of solver information
    param.solver_info_data.interval_time = PERIOD / 30;

    // NUMERICAL PARAMETERS
    param.use_overintegration = false;
  }

  void
  create_grid(
    std::shared_ptr<parallel::TriangulationBase<dim>>,
    unsigned int const,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &) override
  {
#ifdef DEBUG
    std::cout << "create_grid() is empty for lung application" << std::endl;
#endif
  }

  void
  create_grid_and_mesh(
    std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
    unsigned int const                                n_refine_space,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                                 periodic_faces,
    std::shared_ptr<Mesh<dim>> & mesh) override
  {
    (void)periodic_faces;

    AssertThrow(dim == 3, ExcMessage("This test case can only be used for dim==3!"));

    std::vector<std::string> files;
    files.push_back(directory_lung_files + "airways");

    // call to setup root
    auto tree_factory = ExaDG::GridGen::lung_files_to_node(files);

    // TODO: automate deform via spline
    std::string spline_file = directory_lung_files + "splines_raw6.dat";

    std::map<std::string, double> timings;

    // create triangulation
    if(auto tria =
         dynamic_cast<parallel::fullydistributed::Triangulation<dim> *>(triangulation.get()))
    {
      ExaDG::GridGen::lung(*tria,
                           n_refine_space,
                           n_refine_space,
                           tree_factory,
                           mesh,
                           timings,
                           OUTLET_ID_FIRST,
                           OUTLET_ID_LAST,
                           spline_file,
                           max_resolved_generation);
    }
    else if(auto tria =
              dynamic_cast<parallel::distributed::Triangulation<dim> *>(triangulation.get()))
    {
      ExaDG::GridGen::lung(*tria,
                           n_refine_space,
                           tree_factory,
                           mesh,
                           timings,
                           OUTLET_ID_FIRST,
                           OUTLET_ID_LAST,
                           spline_file,
                           max_resolved_generation);
    }
    else
    {
      AssertThrow(false, ExcMessage("Unknown triangulation!"));
    }
  }

  void
  set_boundary_conditions(
    std::shared_ptr<IncNS::BoundaryDescriptorU<dim>> boundary_descriptor_velocity,
    std::shared_ptr<IncNS::BoundaryDescriptorP<dim>> boundary_descriptor_pressure)
  {
    // set boundary conditions
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // 0 = walls
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(dim)));

    // 1 = inlet
    VENTILATOR.reset(new Ventilator());
    TUBUS.reset(new Tubus());
    boundary_descriptor_velocity->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_pressure->dirichlet_bc.insert(
      pair(1, new PressureInlet<dim>(VENTILATOR, TUBUS)));

    // outlets
    for(types::boundary_id id = OUTLET_ID_FIRST; id < OUTLET_ID_LAST; ++id)
    {
      std::shared_ptr<OutflowBoundary> outflow_boundary;
      outflow_boundary.reset(new OutflowBoundary(id, max_resolved_generation, MAX_GENERATION));
      OUTFLOW_BOUNDARIES.push_back(outflow_boundary);

      boundary_descriptor_velocity->neumann_bc.insert(
        pair(id, new Functions::ZeroFunction<dim>(dim)));
      boundary_descriptor_pressure->dirichlet_bc.insert(
        pair(id, new PressureOutlet<dim>(outflow_boundary)));
    }
  }

  void
  set_field_functions(std::shared_ptr<IncNS::FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->initial_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->analytical_solution_pressure.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<IncNS::PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    IncNS::PostProcessorData<dim> pp_data;

    // write output for visualization of results
    pp_data.output_data.write_output              = this->write_output;
    pp_data.output_data.output_folder             = this->output_directory + "vtu/";
    pp_data.output_data.output_name               = this->output_name + "_fluid";
    pp_data.output_data.output_start_time         = output_start_time;
    pp_data.output_data.output_interval_time      = output_interval_time;
    pp_data.output_data.write_vorticity           = true;
    pp_data.output_data.write_divergence          = true;
    pp_data.output_data.write_velocity_magnitude  = true;
    pp_data.output_data.write_vorticity_magnitude = true;
    pp_data.output_data.write_q_criterion         = true;
    pp_data.output_data.write_cfl                 = true;
    pp_data.output_data.write_aspect_ratio        = true;
    pp_data.output_data.write_processor_id        = true;
    pp_data.output_data.write_boundary_IDs        = true;
    pp_data.output_data.degree                    = degree;
    pp_data.output_data.write_higher_order        = high_order_output;

    // Lung specific modules
    PostProcessorDataLung<dim> pp_data_lung;
    pp_data_lung.pp_data = pp_data;

    // calculation of flow rate at outlets
    pp_data_lung.flow_rate_data.calculate     = true;
    pp_data_lung.flow_rate_data.write_to_file = true;
    pp_data_lung.flow_rate_data.filename_prefix =
      this->output_directory + this->output_name + "_flow_rate";

    // calculation of flow rate of the trachea
    pp_data_lung.flow_rate_data_trachea.calculate     = true;
    pp_data_lung.flow_rate_data_trachea.write_to_file = true;
    pp_data_lung.flow_rate_data_trachea.filename_prefix =
      this->output_directory + this->output_name + "_flow_rate_trachea";

    std::shared_ptr<IncNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessorLung<dim, Number>(
      pp_data_lung, mpi_comm, this->output_directory, this->output_name));

    return pp;
  }

  void
  set_boundary_conditions_scalar(
    std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor,
    unsigned int                                       scalar_index = 0)
  {
    (void)scalar_index;

    boundary_descriptor_scalar.push_back(boundary_descriptor);

    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    // 0 = walls
    boundary_descriptor->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));

    set_inhale_bc_scalar(boundary_descriptor);
  }

  void
  set_field_functions_scalar(std::shared_ptr<ConvDiff::FieldFunctions<dim>> field_functions,
                             unsigned int                                   scalar_index = 0)
  {
    (void)scalar_index; // only one scalar quantity considered

    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>>
  construct_postprocessor_scalar(unsigned int const degree,
                                 MPI_Comm const &   mpi_comm,
                                 unsigned int const scalar_index)
  {
    ConvDiff::PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output  = this->write_output;
    pp_data.output_data.output_folder = this->output_directory + "vtu/";
    pp_data.output_data.output_name = this->output_name + "_scalar_" + std::to_string(scalar_index);
    pp_data.output_data.output_start_time    = output_start_time;
    pp_data.output_data.output_interval_time = output_interval_time;
    pp_data.output_data.degree               = degree;
    pp_data.output_data.write_higher_order   = high_order_output;

    std::shared_ptr<ConvDiff::PostProcessorBase<dim, Number>> pp;
    pp.reset(
      new PostProcessorLungScalar<dim, Number>(pp_data, mpi_comm, boundary_descriptor_scalar));

    return pp;
  }
};

} // namespace FTI
} // namespace ExaDG

#endif /* APPLICATIONS_INCOMPRESSIBLE_NAVIER_STOKES_TEST_CASES_LUNG_H_ */
