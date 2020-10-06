/*
 * flow_past_cylinder.h
 *
 *  Created on: 2018
 *      Author: fehn
 */

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_

#include "../../../incompressible_navier_stokes/applications/flow_past_cylinder/grid.h"

namespace ExaDG
{
namespace CompNS
{
namespace FlowPastCylinder
{
using namespace dealii;

const double H   = IncNS::FlowPastCylinder::H;
const double D   = IncNS::FlowPastCylinder::D;
const double X_C = IncNS::FlowPastCylinder::X_C;
const double Y_C = IncNS::FlowPastCylinder::Y_C;

template<int dim>
class VelocityBC : public Function<dim>
{
public:
  VelocityBC(const double Um, const double H, const double end_time, const unsigned int test_case)
    : Function<dim>(dim, 0.0), Um(Um), H(H), end_time(end_time), test_case(test_case)
  {
  }

  double
  value(const Point<dim> & p, const unsigned int component = 0) const
  {
    double       t  = this->get_time();
    const double pi = numbers::PI;
    const double T  = 1.0;

    double result = 0.0;

    if(component == 0)
    {
      double coefficient =
        Utilities::fixed_power<dim - 1>(4.) * Um / Utilities::fixed_power<2 * dim - 2>(H);
      if(test_case < 3)
        result =
          coefficient * p[1] * (H - p[1]) * ((t / T) < 1.0 ? std::sin(pi / 2. * t / T) : 1.0);
      else if(test_case == 3)
        result = coefficient * p[1] * (H - p[1]) * std::sin(pi * t / end_time);

      if(dim == 3)
        result *= p[2] * (H - p[2]);
    }

    return result;
  }

private:
  double const       Um, H, end_time;
  unsigned int const test_case;
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);

    // determine parameters that depend on others: test_case has been set in
    // parse_input() so that we can now compute all remaining parameters
    Um             = (dim == 2 ? (test_case == 1 ? 0.3 : 1.5) : (test_case == 1 ? 0.45 : 2.25));
    U_0            = Um;
    SPEED_OF_SOUND = U_0 / MACH;
    T_0            = SPEED_OF_SOUND * SPEED_OF_SOUND / GAMMA / GAS_CONSTANT;
    E_0            = GAS_CONSTANT / (GAMMA - 1.0) * T_0;
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
     prm.add_parameter("TestCase",  test_case, "Number of test case.",
       Patterns::Integer(1,3));
     prm.add_parameter("CylinderType", cylinder_type_string, "Type of cylinder.",
       Patterns::Selection("circular|square"));
    prm.leave_subsection();
    // clang-format on
  }

  // string to read input parameter
  std::string cylinder_type_string = "circular";

  // test case according to benchmark nomenclature
  unsigned int test_case = 3;

  // end time
  double const start_time = 0.0;
  double const end_time   = 8.0;

  // physical quantities
  double const VISCOSITY    = 1.e-3;
  double const GAMMA        = 1.4;
  double const LAMBDA       = 0.0262;
  double const GAS_CONSTANT = 287.058;
  double const MACH         = 0.2;
  double const RHO_0        = 1.0;

  // dependent parameters
  double Um             = 0.0;
  double U_0            = 0.0;
  double SPEED_OF_SOUND = 0.0;
  double T_0            = 0.0;
  double E_0            = 0.0;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.equation_type   = EquationType::NavierStokes;
    param.right_hand_side = false;

    // PHYSICAL QUANTITIES
    param.start_time            = start_time;
    param.end_time              = end_time;
    param.dynamic_viscosity     = VISCOSITY;
    param.reference_density     = RHO_0;
    param.heat_capacity_ratio   = GAMMA;
    param.thermal_conductivity  = LAMBDA;
    param.specific_gas_constant = GAS_CONSTANT;
    param.max_temperature       = T_0;

    // TEMPORAL DISCRETIZATION
    param.temporal_discretization       = TemporalDiscretization::ExplRK3Stage7Reg2;
    param.order_time_integrator         = 3;
    param.stages                        = 7;
    param.calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
    param.time_step_size                = 1.0e-3;
    param.max_velocity                  = U_0;
    param.cfl_number                    = 1.0;
    param.diffusion_number              = 0.1;
    param.exponent_fe_degree_cfl        = 1.5;
    param.exponent_fe_degree_viscous    = 3.0;

    // output of solver information
    param.solver_info_data.interval_time = (param.end_time - param.start_time) / 20;

    // SPATIAL DISCRETIZATION
    param.triangulation_type    = TriangulationType::Distributed;
    param.mapping               = MappingType::Isoparametric;
    param.n_q_points_convective = QuadratureRule::Overintegration32k;
    param.n_q_points_viscous    = QuadratureRule::Overintegration32k;

    // viscous term
    param.IP_factor = 1.0;
  }


  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              unsigned int const                                n_refine_space,
              std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                periodic_faces)
  {
    (void)periodic_faces;


    if(auto tria_fully_dist =
         dynamic_cast<parallel::fullydistributed::Triangulation<dim> *>(&*triangulation))
    {
      const auto construction_data =
        TriangulationDescription::Utilities::create_description_from_triangulation_in_groups<dim,
                                                                                             dim>(
          [&](dealii::Triangulation<dim, dim> & tria) mutable {
            IncNS::FlowPastCylinder::create_cylinder_grid<dim>(tria,
                                                               n_refine_space,
                                                               periodic_faces,
                                                               cylinder_type_string);
          },
          [](dealii::Triangulation<dim, dim> & tria,
             const MPI_Comm                    comm,
             const unsigned int /* group_size */) {
            // metis partitioning
            GridTools::partition_triangulation(Utilities::MPI::n_mpi_processes(comm), tria);
            // p4est partitioning
            //            GridTools::partition_triangulation_zorder(Utilities::MPI::n_mpi_processes(comm),
            //            tria);
          },
          tria_fully_dist->get_communicator(),
          1 /* group size */);
      tria_fully_dist->create_triangulation(construction_data);
    }
    else if(auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> *>(&*triangulation))
    {
      IncNS::FlowPastCylinder::create_cylinder_grid<dim>(*tria,
                                                         n_refine_space,
                                                         periodic_faces,
                                                         cylinder_type_string);
    }
    else
    {
      AssertThrow(false, ExcMessage("Unknown triangulation!"));
    }
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

    // inflow and upper/lower walls: 0, outflow: 1, cylinder: 2

    // density
    // For Neumann boundaries, no value is prescribed (only first derivative of density occurs in
    // equations). Hence the specified function is irrelevant (i.e., it is not used).
    boundary_descriptor_density->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor_density->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor_density->neumann_bc.insert(pair(2, new Functions::ZeroFunction<dim>(1)));

    // velocity
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(0, new VelocityBC<dim>(Um, H, end_time, test_case)));
    boundary_descriptor_velocity->dirichlet_bc.insert(
      pair(2, new Functions::ZeroFunction<dim>(dim)));
    boundary_descriptor_velocity->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(dim)));

    // pressure
    boundary_descriptor_pressure->dirichlet_bc.insert(
      pair(1, new Functions::ConstantFunction<dim>(RHO_0 * GAS_CONSTANT * T_0, 1)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    boundary_descriptor_pressure->neumann_bc.insert(pair(2, new Functions::ZeroFunction<dim>(1)));

    // energy: prescribe temperature
    boundary_descriptor_energy->boundary_variable.insert(
      pair_variable(0, CompNS::EnergyBoundaryVariable::Temperature));
    boundary_descriptor_energy->boundary_variable.insert(
      pair_variable(1, CompNS::EnergyBoundaryVariable::Temperature));
    boundary_descriptor_energy->boundary_variable.insert(
      pair_variable(2, CompNS::EnergyBoundaryVariable::Temperature));

    boundary_descriptor_energy->dirichlet_bc.insert(
      pair(0, new Functions::ConstantFunction<dim>(T_0, 1)));
    boundary_descriptor_energy->dirichlet_bc.insert(
      pair(2, new Functions::ConstantFunction<dim>(T_0, 1)));
    boundary_descriptor_energy->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions(std::shared_ptr<CompNS::FieldFunctions<dim>> field_functions)
  {
    std::vector<double> initial_values = std::vector<double>(dim + 2, 0.0);
    initial_values[0]                  = RHO_0;       // rho
    initial_values[dim + 1]            = RHO_0 * E_0; // rho*E
    field_functions->initial_solution.reset(new Functions::ConstantFunction<dim>(initial_values));
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
    pp_data.calculate_velocity               = true;
    pp_data.calculate_pressure               = true;
    pp_data.output_data.write_pressure       = true;
    pp_data.output_data.write_velocity       = true;
    pp_data.output_data.write_temperature    = true;
    pp_data.output_data.write_vorticity      = true;
    pp_data.output_data.write_divergence     = true;
    pp_data.output_data.output_start_time    = start_time;
    pp_data.output_data.output_interval_time = (end_time - start_time) / 20;
    pp_data.output_data.degree               = degree;
    pp_data.output_data.write_higher_order   = false;

    // lift and drag
    pp_data.lift_and_drag_data.calculate_lift_and_drag = true;
    pp_data.lift_and_drag_data.viscosity               = VISCOSITY;
    const double U                                     = Um * (dim == 2 ? 2. / 3. : 4. / 9.);
    if(dim == 2)
      pp_data.lift_and_drag_data.reference_value = RHO_0 / 2.0 * pow(U, 2.0) * D;
    else if(dim == 3)
      pp_data.lift_and_drag_data.reference_value = RHO_0 / 2.0 * pow(U, 2.0) * D * H;

    // surfaces for calculation of lift and drag coefficients have boundary_ID = 2
    pp_data.lift_and_drag_data.boundary_IDs.insert(2);

    pp_data.lift_and_drag_data.filename_lift = this->output_directory + this->output_name + "_lift";
    pp_data.lift_and_drag_data.filename_drag = this->output_directory + this->output_name + "_drag";

    // pressure difference
    pp_data.pressure_difference_data.calculate_pressure_difference = true;
    if(dim == 2)
    {
      Point<dim> point_1_2D((X_C - D / 2.0), Y_C), point_2_2D((X_C + D / 2.0), Y_C);
      pp_data.pressure_difference_data.point_1 = point_1_2D;
      pp_data.pressure_difference_data.point_2 = point_2_2D;
    }
    else if(dim == 3)
    {
      Point<dim> point_1_3D((X_C - D / 2.0), Y_C, H / 2.0),
        point_2_3D((X_C + D / 2.0), Y_C, H / 2.0);
      pp_data.pressure_difference_data.point_1 = point_1_3D;
      pp_data.pressure_difference_data.point_2 = point_2_3D;
    }

    pp_data.pressure_difference_data.filename =
      this->output_directory + this->output_name + "_pressure_difference";

    std::shared_ptr<CompNS::PostProcessorBase<dim, Number>> pp;
    pp.reset(new CompNS::PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace FlowPastCylinder
} // namespace CompNS
} // namespace ExaDG


#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_FLOW_PAST_CYLINDER_H_ */
