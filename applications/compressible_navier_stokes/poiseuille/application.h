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

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_POISEUILLE_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_POISEUILLE_H_

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

// problem specific parameters
double const DYN_VISCOSITY  = 0.1;
double const GAMMA          = 1.4;
double const LAMBDA         = 0.0;
double const R              = 1.0;
double const U_0            = 1.0;
double const MACH           = 0.2;
double const SPEED_OF_SOUND = U_0 / MACH;
double const RHO_0          = 1.0;
double const T_0            = SPEED_OF_SOUND * SPEED_OF_SOUND / GAMMA / R;
double const E_0            = R / (GAMMA - 1.0) * T_0;

double const H = 2.0;
double const L = 4.0;

double
parabolic_velocity_profile(double const y, double const t)
{
  double const pi = numbers::PI;
  double const T  = 10.0;

  double result = U_0 * (1.0 - pow(y / (H / 2.0), 2.0)) * (t < T ? std::sin(pi / 2. * t / T) : 1.0);

  return result;
}

template<int dim>
class InitialSolution : public Function<dim>
{
public:
  InitialSolution(unsigned int const n_components = dim + 2, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    double const t = this->get_time();

    double result = 0.0;

    if(component == 0)
      result = RHO_0;
    else if(component == 1)
      result = RHO_0 * parabolic_velocity_profile(p[1], t);
    else if(component == 2)
      result = 0.0;
    else if(component == 1 + dim)
      result = RHO_0 * E_0;

    return result;
  }
};


/*
 *  prescribe a parabolic velocity profile at the inflow and
 *  zero velocity at the wall boundaries
 */
template<int dim>
class VelocityBC : public Function<dim>
{
public:
  VelocityBC(unsigned int const n_components = dim, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    double const t = this->get_time();

    double result = 0.0;

    // copied from analytical solution
    if(component == 0)
      result = parabolic_velocity_profile(p[1], t);
    else if(component == 1)
      result = 0.0;

    return result;
  }
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  double const start_time = 0.0;
  double const end_time   = 25.0;

  void
  set_input_parameters(unsigned int const degree) final
  {
    // MATHEMATICAL MODEL
    this->param.equation_type   = EquationType::NavierStokes;
    this->param.right_hand_side = true;

    // PHYSICAL QUANTITIES
    this->param.start_time            = start_time;
    this->param.end_time              = end_time;
    this->param.dynamic_viscosity     = DYN_VISCOSITY;
    this->param.reference_density     = RHO_0;
    this->param.heat_capacity_ratio   = GAMMA;
    this->param.thermal_conductivity  = LAMBDA;
    this->param.specific_gas_constant = R;
    this->param.max_temperature       = T_0;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization       = TemporalDiscretization::ExplRK;
    this->param.order_time_integrator         = 2;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
    this->param.time_step_size                = 1.0e-3;
    this->param.max_velocity                  = U_0;
    this->param.cfl_number                    = 0.1;
    this->param.diffusion_number              = 0.01;
    this->param.exponent_fe_degree_cfl        = 2.0;
    this->param.exponent_fe_degree_viscous    = 4.0;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 10;

    // SPATIAL DISCRETIZATION
    this->param.triangulation_type    = TriangulationType::Distributed;
    this->param.mapping               = MappingType::Isoparametric;
    this->param.degree                = degree;
    this->param.n_q_points_convective = QuadratureRule::Standard;
    this->param.n_q_points_viscous    = QuadratureRule::Standard;

    // viscous term
    this->param.IP_factor = 1.0e0;

    // NUMERICAL PARAMETERS
    this->param.use_combined_operator = false;
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid(GridData const & grid_data) final
  {
    std::shared_ptr<Grid<dim, Number>> grid =
      std::make_shared<Grid<dim, Number>>(grid_data, this->mpi_comm);

    std::vector<unsigned int> repetitions({2, 1});
    Point<dim>                point1(0.0, -H / 2.), point2(L, H / 2.);
    GridGenerator::subdivided_hyper_rectangle(*grid->triangulation, repetitions, point1, point2);

    // set boundary indicator
    for(auto cell : *grid->triangulation)
    {
      for(unsigned int face = 0; face < GeometryInfo<dim>::faces_per_cell; ++face)
      {
        if((std::fabs(cell.face(face)->center()(0) - L) < 1e-12))
          cell.face(face)->set_boundary_id(1);
      }
    }

    grid->triangulation->refine_global(grid_data.n_refine_global);

    return grid;
  }

  void
  set_boundary_conditions() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
    typedef typename std::pair<types::boundary_id, EnergyBoundaryVariable>         pair_variable;

    // zero function vectorial
    std::shared_ptr<Function<dim>> zero_function_vectorial;
    zero_function_vectorial.reset(new Functions::ZeroFunction<dim>(dim));

    // density
    // For Neumann boundaries, no value is prescribed (only first derivative of density occurs in
    // equations). Hence the specified function is irrelevant (i.e., it is not used).
    this->boundary_descriptor->density.neumann_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(1)));
    this->boundary_descriptor->density.neumann_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(1)));

    // velocity
    this->boundary_descriptor->velocity.dirichlet_bc.insert(pair(0, new VelocityBC<dim>()));
    this->boundary_descriptor->velocity.neumann_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(dim)));

    // pressure
    this->boundary_descriptor->pressure.dirichlet_bc.insert(
      pair(1, new Functions::ConstantFunction<dim>(RHO_0 * R * T_0, 1)));
    this->boundary_descriptor->pressure.neumann_bc.insert(
      pair(0, new Functions::ZeroFunction<dim>(1)));

    // energy: prescribe temperature
    this->boundary_descriptor->energy.boundary_variable.insert(
      pair_variable(0, EnergyBoundaryVariable::Temperature));
    this->boundary_descriptor->energy.boundary_variable.insert(
      pair_variable(1, EnergyBoundaryVariable::Temperature));

    this->boundary_descriptor->energy.dirichlet_bc.insert(
      pair(0, new Functions::ConstantFunction<dim>(T_0, 1)));
    this->boundary_descriptor->energy.neumann_bc.insert(
      pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new InitialSolution<dim>());
    this->field_functions->right_hand_side_density.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    this->field_functions->right_hand_side_energy.reset(new Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = this->output_name;
    pp_data.output_data.write_pressure     = true;
    pp_data.output_data.write_velocity     = true;
    pp_data.output_data.write_temperature  = true;
    pp_data.output_data.write_vorticity    = true;
    pp_data.output_data.write_divergence   = true;
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = (end_time - start_time) / 20;
    pp_data.output_data.degree             = this->param.degree;
    pp_data.output_data.write_higher_order = false;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace CompNS

} // namespace ExaDG

#include <exadg/compressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_POISEUILLE_H_ */
