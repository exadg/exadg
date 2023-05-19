/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2021 by the ExaDG authors
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_EULER_VORTEX_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_EULER_VORTEX_H_

namespace ExaDG
{
namespace CompNS
{
// problem specific parameters
double const DYN_VISCOSITY  = 0.0;
double const GAMMA          = 1.4;
double const LAMBDA         = 0.0;
double const R              = 1.0;
double const U_0            = 1.0;
double const MACH           = 0.5;
double const SPEED_OF_SOUND = U_0 / MACH;
double const T_0            = SPEED_OF_SOUND * SPEED_OF_SOUND / GAMMA / R;

double const X_0  = 0.0;
double const Y_0  = 0.0;
double const H    = 10.0;
double const L    = 10.0;
double const BETA = 5.0;

double
get_r_square(double const x, double const y, double const t)
{
  return (x - t - X_0) * (x - t - X_0) + (y - Y_0) * (y - Y_0);
}

double
get_rho(double const r_sq)
{
  double const pi = dealii::numbers::PI;
  return std::pow(1.0 - ((GAMMA - 1.0) / (16.0 * GAMMA * pi * pi) * BETA * BETA *
                         std::exp(2.0 * (1.0 - r_sq))),
                  1 / (GAMMA - 1.0));
}

double
get_u(double const y, double const r_sq)
{
  double const pi = dealii::numbers::PI;
  return 1.0 - BETA * std::exp(1.0 - r_sq) * (y - Y_0) / (2.0 * pi);
}

double
get_v(double const x, double const t, double const r_sq)
{
  double const pi = dealii::numbers::PI;
  return BETA * std::exp(1.0 - r_sq) * (x - t - X_0) / (2.0 * pi);
}

double
get_energy(double const rho, double const u, double const v)
{
  double const pressure = std::pow(rho, GAMMA);

  return pressure / (rho * (GAMMA - 1.0)) + 0.5 * (u * u + v * v);
}

template<int dim>
class Solution : public dealii::Function<dim>
{
public:
  Solution(unsigned int const n_components = dim + 2, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double const t    = this->get_time();
    double const r_sq = get_r_square(p[0], p[1], t);
    double const rho  = get_rho(r_sq);
    double const u    = get_u(p[1], r_sq);
    double const v    = get_v(p[0], t, r_sq);

    double result = 0.0;
    if(component == 0)
    {
      result = rho;
    }
    else if(component == 1)
    {
      result = rho * u;
    }
    else if(component == 2)
    {
      result = rho * v;
    }
    else if(component == 1 + dim)
    {
      result = rho * get_energy(rho, u, v);
    }

    return result;
  }
};


/*
 *  prescribe a parabolic velocity profile at the inflow and
 *  zero velocity at the wall boundaries
 */
template<int dim>
class VelocityBC : public dealii::Function<dim>
{
public:
  VelocityBC(unsigned int const n_components = dim, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double const t    = this->get_time();
    double const r_sq = get_r_square(p[0], p[1], t);

    double result = 0.0;
    if(component == 0)
      result = get_u(p[1], r_sq);
    else if(component == 1)
      result = get_v(p[0], t, r_sq);

    return result;
  }
};

/*
 *  prescribe a constant temperature at the channel walls
 */
template<int dim>
class EnergyBC : public dealii::Function<dim>
{
public:
  EnergyBC(double const time = 0.) : dealii::Function<dim>(1, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    (void)component;

    double const t      = this->get_time();
    double const r_sq   = get_r_square(p[0], p[1], t);
    double const rho    = get_rho(r_sq);
    double const u      = get_u(p[1], r_sq);
    double const v      = get_v(p[0], t, r_sq);
    double       energy = get_energy(rho, u, v);

    return energy;
  }
};

template<int dim>
class DensityBC : public dealii::Function<dim>
{
public:
  DensityBC(double const time = 0.) : dealii::Function<dim>(1, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    (void)component;

    double const t    = this->get_time();
    double const r_sq = get_r_square(p[0], p[1], t);
    double const rho  = get_rho(r_sq);

    return rho;
  }
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.equation_type   = EquationType::Euler;
    this->param.right_hand_side = true;

    // PHYSICAL QUANTITIES
    this->param.start_time            = start_time;
    this->param.end_time              = end_time;
    this->param.dynamic_viscosity     = DYN_VISCOSITY;
    this->param.reference_density     = 1.0;
    this->param.heat_capacity_ratio   = GAMMA;
    this->param.thermal_conductivity  = LAMBDA;
    this->param.specific_gas_constant = R;
    this->param.max_temperature       = T_0;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization       = TemporalDiscretization::ExplRK3Stage7Reg2;
    this->param.order_time_integrator         = 2;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.time_step_size                = 1.0e-3;
    this->param.max_velocity                  = U_0;
    this->param.cfl_number                    = 1.0;
    this->param.diffusion_number              = 0.01;
    this->param.exponent_fe_degree_cfl        = 2.0;
    this->param.exponent_fe_degree_viscous    = 4.0;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 10;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = this->param.degree;
    this->param.n_q_points_convective   = QuadratureRule::Standard;
    this->param.n_q_points_viscous      = QuadratureRule::Standard;

    // viscous term
    this->param.IP_factor = 1.0e0;

    // NUMERICAL PARAMETERS
    this->param.use_combined_operator = false;
  }

  void
  create_grid() final
  {
    std::vector<unsigned int> repetitions({1, 1});
    dealii::Point<dim> point1(X_0 - L / 2.0, Y_0 - H / 2.0), point2(X_0 + L / 2.0, Y_0 + H / 2.0);
    dealii::GridGenerator::subdivided_hyper_rectangle(*this->grid->triangulation,
                                                      repetitions,
                                                      point1,
                                                      point2);

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                   pair;
    typedef typename std::pair<dealii::types::boundary_id, EnergyBoundaryVariable> pair_variable;

    this->boundary_descriptor->density.dirichlet_bc.insert(pair(0, new DensityBC<dim>()));
    this->boundary_descriptor->velocity.dirichlet_bc.insert(pair(0, new VelocityBC<dim>()));
    this->boundary_descriptor->pressure.neumann_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(1)));
    // energy: prescribe energy
    this->boundary_descriptor->energy.boundary_variable.insert(
      pair_variable(0, EnergyBoundaryVariable::Energy));
    this->boundary_descriptor->energy.dirichlet_bc.insert(pair(0, new EnergyBC<dim>()));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new Solution<dim>());
    this->field_functions->right_hand_side_density.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side_velocity.reset(
      new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->right_hand_side_energy.reset(
      new dealii::Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active        = this->output_parameters.write;
    pp_data.output_data.time_control_data.start_time       = start_time;
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 10.0;
    pp_data.output_data.directory         = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename          = this->output_parameters.filename;
    pp_data.output_data.write_pressure    = true;
    pp_data.output_data.write_velocity    = true;
    pp_data.output_data.write_temperature = true;
    pp_data.output_data.write_vorticity   = true;
    pp_data.output_data.write_divergence  = true;
    pp_data.output_data.degree            = this->param.degree;

    pp_data.error_data.time_control_data.is_active        = true;
    pp_data.error_data.time_control_data.start_time       = start_time;
    pp_data.error_data.time_control_data.trigger_interval = (end_time - start_time) / 10.0;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>());

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  double const start_time = 0.0;
  double const end_time   = 1.0;
};

} // namespace CompNS

} // namespace ExaDG

#include <exadg/compressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_EULER_VORTEX_H_ */
