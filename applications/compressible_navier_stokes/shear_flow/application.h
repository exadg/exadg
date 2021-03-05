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

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_SHEAR_FLOW_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_SHEAR_FLOW_H_

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

// problem specific parameters
double const DYN_VISCOSITY  = 0.01;
double const GAMMA          = 1.5;
double const LAMBDA         = 0.0;
double const R              = 1.0;
double const U_0            = 1.0;
double const MACH           = 0.2;
double const SPEED_OF_SOUND = U_0 / MACH;
double const RHO_0          = 1.0;
double const T_0            = SPEED_OF_SOUND * SPEED_OF_SOUND / GAMMA / R;

double const H = 1.0;
double const L = 1.0;


/*
 *  Analytical solutions
 */
template<int dim>
class Solution : public Function<dim>
{
public:
  Solution(unsigned int const n_components = dim + 2, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    double result = 0.0;

    if(component == 0)
      result = RHO_0;
    else if(component == 1)
      result = RHO_0 * p[1] * p[1];
    else if(component == 2)
      result = 0.0;
    else if(component == 1 + dim)
      result =
        RHO_0 * (2.0 * DYN_VISCOSITY * p[0] + 10.0) / (GAMMA - 1.0) + std::pow(p[1], 4.0) / 2.0;

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
    double result = 0.0;

    // copied from analytical solution
    if(component == 0)
      result = RHO_0 * p[1] * p[1];
    else if(component == 1)
      result = 0.0;

    return result;
  }
};

/*
 *  prescribe a constant temperature at the channel walls
 */
template<int dim>
class EnergyBC : public Function<dim>
{
public:
  EnergyBC(double const time = 0.) : Function<dim>(1, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const component = 0) const
  {
    (void)component;

    double result = (2.0 * DYN_VISCOSITY * p[0] + 10.0) / (GAMMA - 1.0) + std::pow(p[1], 4.0) / 2.0;

    return result;
  }
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  typedef typename ApplicationBase<dim, Number>::PeriodicFaces PeriodicFaces;

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  double const start_time = 0.0;
  double const end_time   = 10.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.equation_type   = EquationType::NavierStokes;
    param.right_hand_side = true;

    // PHYSICAL QUANTITIES
    param.start_time            = start_time;
    param.end_time              = end_time;
    param.dynamic_viscosity     = DYN_VISCOSITY;
    param.reference_density     = RHO_0;
    param.heat_capacity_ratio   = GAMMA;
    param.thermal_conductivity  = LAMBDA;
    param.specific_gas_constant = R;
    param.max_temperature       = T_0;

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization       = TemporalDiscretization::ExplRK;
    param.order_time_integrator         = 2;
    param.calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
    param.time_step_size                = 1.0e-3;
    param.max_velocity                  = U_0;
    param.cfl_number                    = 0.1;
    param.diffusion_number              = 0.01;
    param.exponent_fe_degree_cfl        = 2.0;
    param.exponent_fe_degree_viscous    = 4.0;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 10;

    // SPATIAL DISCRETIZATION
    param.triangulation_type    = TriangulationType::Distributed;
    param.mapping               = MappingType::Isoparametric;
    param.n_q_points_convective = QuadratureRule::Standard;
    param.n_q_points_viscous    = QuadratureRule::Standard;

    // viscous term
    param.IP_factor = 1.0e0;

    // NUMERICAL PARAMETERS
    param.use_combined_operator = false;
  }

  void
  create_grid(std::shared_ptr<Triangulation<dim>> triangulation,
              PeriodicFaces &                     periodic_faces,
              unsigned int const                  n_refine_space,
              std::shared_ptr<Mapping<dim>> &     mapping,
              unsigned int const                  mapping_degree)
  {
    (void)periodic_faces;

    std::vector<unsigned int> repetitions({1, 1});
    Point<dim>                point1(0.0, 0.0), point2(L, H);
    GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);

    triangulation->refine_global(n_refine_space);

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
  }

  void
  set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor_density,
                          std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor_velocity,
                          std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor_pressure,
                          std::shared_ptr<BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
    typedef typename std::pair<types::boundary_id, EnergyBoundaryVariable>         pair_variable;

    boundary_descriptor_density->dirichlet_bc.insert(
      pair(0, new Functions::ConstantFunction<dim>(RHO_0, 1)));
    boundary_descriptor_velocity->dirichlet_bc.insert(pair(0, new VelocityBC<dim>()));
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    // energy: prescribe energy
    boundary_descriptor_energy->boundary_variable.insert(
      pair_variable(0, EnergyBoundaryVariable::Energy));
    boundary_descriptor_energy->dirichlet_bc.insert(pair(0, new EnergyBC<dim>()));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new Solution<dim>());
    field_functions->right_hand_side_density.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->right_hand_side_energy.reset(new Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.output_folder        = this->output_directory + "vtu/";
    pp_data.output_data.output_name          = this->output_name;
    pp_data.output_data.write_pressure       = true;
    pp_data.output_data.write_velocity       = true;
    pp_data.output_data.write_temperature    = true;
    pp_data.output_data.write_vorticity      = true;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 10;
    pp_data.output_data.degree               = degree;
    pp_data.output_data.write_higher_order   = false;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>());
    pp_data.error_data.error_calc_start_time    = start_time;
    pp_data.error_data.error_calc_interval_time = (end_time - start_time) / 10;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace CompNS

template<int dim, typename Number>
std::shared_ptr<CompNS::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<CompNS::Application<dim, Number>>(input_file);
}

} // namespace ExaDG


#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_SHEAR_FLOW_H_ */
