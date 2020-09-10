/*
 * poiseuille.h
 *
 *  Created on: 22.03.2020
 *      Author: fehn
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_POISEUILLE_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_POISEUILLE_H_

namespace ExaDG
{
namespace CompNS
{
namespace Poiseuille
{
using namespace dealii;

// problem specific parameters
const double DYN_VISCOSITY  = 0.1;
const double GAMMA          = 1.4;
const double LAMBDA         = 0.0;
const double R              = 1.0;
const double U_0            = 1.0;
const double MACH           = 0.2;
const double SPEED_OF_SOUND = U_0 / MACH;
const double RHO_0          = 1.0;
const double T_0            = SPEED_OF_SOUND * SPEED_OF_SOUND / GAMMA / R;
const double E_0            = R / (GAMMA - 1.0) * T_0;

const double H = 2.0;
const double L = 4.0;

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
  InitialSolution(const unsigned int n_components = dim + 2, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
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
  VelocityBC(const unsigned int n_components = dim, const double time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
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
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    this->add_parameters(prm);
    prm.parse_input(input_file, "", true, true);
  }

  double const start_time = 0.0;
  double const end_time   = 25.0;

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
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;

    std::vector<unsigned int> repetitions({2, 1});
    Point<dim>                point1(0.0, -H / 2.), point2(L, H / 2.);
    GridGenerator::subdivided_hyper_rectangle(*triangulation, repetitions, point1, point2);

    // set boundary indicator
    typename Triangulation<dim>::cell_iterator cell = triangulation->begin(),
                                               endc = triangulation->end();
    for(; cell != endc; ++cell)
    {
      for(unsigned int face_number = 0; face_number < GeometryInfo<dim>::faces_per_cell;
          ++face_number)
      {
        if((std::fabs(cell->face(face_number)->center()(0) - L) < 1e-12))
          cell->face(face_number)->set_boundary_id(1);
      }
    }

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

    // zero function vectorial
    std::shared_ptr<Function<dim>> zero_function_vectorial;
    zero_function_vectorial.reset(new Functions::ZeroFunction<dim>(dim));

    // density
    // For Neumann boundaries, no value is prescribed (only first derivative of density occurs in
    // equations). Hence the specified function is irrelevant (i.e., it is not used).
    boundary_descriptor_density->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor_density->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));

    // velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(pair(0, new VelocityBC<dim>()));
    boundary_descriptor_velocity->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));

    // pressure
    boundary_descriptor_pressure->dirichlet_bc.insert(
      pair(1, new Functions::ConstantFunction<dim>(RHO_0 * R * T_0, 1)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));

    // energy: prescribe temperature
    boundary_descriptor_energy->boundary_variable.insert(
      pair_variable(0, CompNS::EnergyBoundaryVariable::Temperature));
    boundary_descriptor_energy->boundary_variable.insert(
      pair_variable(1, CompNS::EnergyBoundaryVariable::Temperature));

    boundary_descriptor_energy->dirichlet_bc.insert(
      pair(0, new Functions::ConstantFunction<dim>(T_0, 1)));
    boundary_descriptor_energy->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions(std::shared_ptr<CompNS::FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new InitialSolution<dim>());
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
    pp_data.output_data.write_higher_order   = false;

    std::shared_ptr<CompNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new CompNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace Poiseuille
} // namespace CompNS
} // namespace ExaDG


#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_POISEUILLE_H_ */
