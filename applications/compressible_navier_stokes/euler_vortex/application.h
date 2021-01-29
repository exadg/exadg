/*
 * euler_vortex.h
 *
 *  Created on: 22.03.2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_EULER_VORTEX_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_EULER_VORTEX_H_

namespace ExaDG
{
namespace CompNS
{
using namespace dealii;

// problem specific parameters
const double DYN_VISCOSITY  = 0.0;
const double GAMMA          = 1.4;
const double LAMBDA         = 0.0;
const double R              = 1.0;
const double U_0            = 1.0;
const double MACH           = 0.5;
const double SPEED_OF_SOUND = U_0 / MACH;
const double T_0            = SPEED_OF_SOUND * SPEED_OF_SOUND / GAMMA / R;

const double X_0  = 0.0;
const double Y_0  = 0.0;
const double H    = 10.0;
const double L    = 10.0;
const double BETA = 5.0;

double
get_r_square(double const x, double const y, double const t)
{
  return (x - t - X_0) * (x - t - X_0) + (y - Y_0) * (y - Y_0);
}

double
get_rho(double const r_sq)
{
  const double pi = numbers::PI;
  return std::pow(1.0 - ((GAMMA - 1.0) / (16.0 * GAMMA * pi * pi) * BETA * BETA *
                         std::exp(2.0 * (1.0 - r_sq))),
                  1 / (GAMMA - 1.0));
}

double
get_u(double const y, double const r_sq)
{
  const double pi = numbers::PI;
  return 1.0 - BETA * std::exp(1.0 - r_sq) * (y - Y_0) / (2.0 * pi);
}

double
get_v(double const x, double const t, double const r_sq)
{
  const double pi = numbers::PI;
  return BETA * std::exp(1.0 - r_sq) * (x - t - X_0) / (2.0 * pi);
}

double
get_energy(double const rho, double const u, double const v)
{
  const double pressure = std::pow(rho, GAMMA);

  return pressure / (rho * (GAMMA - 1.0)) + 0.5 * (u * u + v * v);
}

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution(const unsigned int n_components = dim + 2, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    const double t    = this->get_time();
    const double r_sq = get_r_square(p[0], p[1], t);
    const double rho  = get_rho(r_sq);
    const double u    = get_u(p[1], r_sq);
    const double v    = get_v(p[0], t, r_sq);

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
class VelocityBC : public Function<dim>
{
public:
  VelocityBC(const unsigned int n_components = dim, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    const double t    = this->get_time();
    const double r_sq = get_r_square(p[0], p[1], t);

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
class EnergyBC : public Function<dim>
{
public:
  EnergyBC(const double time = 0.) : Function<dim>(1, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    (void)component;

    const double t      = this->get_time();
    const double r_sq   = get_r_square(p[0], p[1], t);
    const double rho    = get_rho(r_sq);
    const double u      = get_u(p[1], r_sq);
    const double v      = get_v(p[0], t, r_sq);
    double       energy = get_energy(rho, u, v);

    return energy;
  }
};

template<int dim>
class DensityBC : public Function<dim>
{
public:
  DensityBC(const double time = 0.) : Function<dim>(1, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    (void)component;

    const double t    = this->get_time();
    const double r_sq = get_r_square(p[0], p[1], t);
    const double rho  = get_rho(r_sq);

    return rho;
  }
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  double const start_time = 0.0;
  double const end_time   = 1.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.equation_type   = EquationType::Euler;
    param.right_hand_side = true;

    // PHYSICAL QUANTITIES
    param.start_time            = start_time;
    param.end_time              = end_time;
    param.dynamic_viscosity     = DYN_VISCOSITY;
    param.reference_density     = 1.0;
    param.heat_capacity_ratio   = GAMMA;
    param.thermal_conductivity  = LAMBDA;
    param.specific_gas_constant = R;
    param.max_temperature       = T_0;

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization       = TemporalDiscretization::ExplRK3Stage7Reg2;
    param.order_time_integrator         = 2;
    param.calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
    param.time_step_size                = 1.0e-3;
    param.max_velocity                  = U_0;
    param.cfl_number                    = 1.0;
    param.diffusion_number              = 0.01;
    param.exponent_fe_degree_cfl        = 2.0;
    param.exponent_fe_degree_viscous    = 4.0;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 20;

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
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    std::vector<unsigned int> repetitions({1, 1});
    Point<dim> point1(X_0 - L / 2.0, Y_0 - H / 2.0), point2(X_0 + L / 2.0, Y_0 + H / 2.0);
    GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);

    triangulation->refine_global(n_refine_space);
  }

  void
  set_boundary_conditions(
    std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       boundary_descriptor_density,
    std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       boundary_descriptor_velocity,
    std::shared_ptr<CompNS::BoundaryDescriptor<dim>>       boundary_descriptor_pressure,
    std::shared_ptr<CompNS::BoundaryDescriptorEnergy<dim>> boundary_descriptor_energy)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;
    typedef typename std::pair<types::boundary_id, EnergyBoundaryVariable>         pair_variable;

    boundary_descriptor_density->dirichlet_bc.insert(pair(0, new DensityBC<dim>()));
    boundary_descriptor_velocity->dirichlet_bc.insert(pair(0, new VelocityBC<dim>()));
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    // energy: prescribe energy
    boundary_descriptor_energy->boundary_variable.insert(
      pair_variable(0, CompNS::EnergyBoundaryVariable::Energy));
    boundary_descriptor_energy->dirichlet_bc.insert(pair(0, new EnergyBC<dim>()));
  }

  void
  set_field_functions(std::shared_ptr<CompNS::FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new Solution<dim>());
    field_functions->right_hand_side_density.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side_velocity.reset(new Functions::ZeroFunction<dim>(dim));
    field_functions->right_hand_side_energy.reset(new Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<CompNS::PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    CompNS::PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output         = this->write_output;
    pp_data.output_data.output_folder        = this->output_directory + "vtu/";
    pp_data.output_data.output_name          = this->output_name;
    pp_data.output_data.write_pressure       = true;
    pp_data.output_data.write_velocity       = true;
    pp_data.output_data.write_temperature    = true;
    pp_data.output_data.write_vorticity      = true;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 20;
    pp_data.output_data.degree               = degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>());
    pp_data.error_data.error_calc_start_time    = start_time;
    pp_data.error_data.error_calc_interval_time = (end_time - start_time) / 20;

    std::shared_ptr<CompNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new CompNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

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


#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_EULER_VORTEX_H_ */
