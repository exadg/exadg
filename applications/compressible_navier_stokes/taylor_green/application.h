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

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_

#include <exadg/grid/periodic_box.h>

namespace ExaDG
{
namespace CompNS
{
// set parameters according to Wiart et al. ("Assessment of discontinuous Galerkin method
// for the simulation of vortical flows at high Reynolds number"):

// set Re, rho_0, V_0, L -> calculate viscosity mu
double const Re            = 1600.0;
double const RHO_0         = 1.0;
double const V_0           = 1.0;
double const L             = 1.0;
double const DYN_VISCOSITY = RHO_0 * V_0 * L / Re;

// set R, gamma, Pr -> calculate c_p, lambda
double const R       = 287.0;
double const GAMMA   = 1.4;
double const C_P     = GAMMA / (GAMMA - 1.0) * R;
double const PRANDTL = 0.71; // Pr = mu * c_p / lambda
double const LAMBDA  = DYN_VISCOSITY * C_P / PRANDTL;

// set Ma number -> calculate speed of sound c_0, temperature T_0, pressure p_0
double const MACH           = 0.1;
double const SPEED_OF_SOUND = V_0 / MACH;
double const T_0            = SPEED_OF_SOUND * SPEED_OF_SOUND / GAMMA / R;
double const p_0            = RHO_0 * R * T_0;

double const MAX_VELOCITY        = V_0;
double const CHARACTERISTIC_TIME = L / V_0;

template<int dim>
class InitialSolution : public dealii::Function<dim>
{
public:
  InitialSolution(unsigned int const n_components = dim + 2, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & x, unsigned int const component = 0) const
  {
    double const u1 = V_0 * std::sin(x[0] / L) * std::cos(x[1] / L) * std::cos(x[2] / L);
    double const u2 = -V_0 * std::cos(x[0] / L) * std::sin(x[1] / L) * std::cos(x[2] / L);
    double const p  = p_0 + RHO_0 * V_0 * V_0 / 16.0 *
                             (std::cos(2.0 * x[0] / L) + std::cos(2.0 * x[1] / L)) *
                             (std::cos(2.0 * x[2] / L) + 2.0);
    double const T   = T_0;
    double const rho = p / (R * T);
    double const E   = R / (GAMMA - 1.0) * T /* e = c_v * T */
                     + 0.5 * (u1 * u1 + u2 * u2 /* + u3*u3 with u3=0*/);

    double result = 0.0;

    if(component == 0)
      result = rho;
    else if(component == 1)
      result = rho * u1;
    else if(component == 2)
      result = rho * u2;
    else if(component == 3)
      result = 0.0;
    else if(component == 1 + dim)
      result = rho * E;

    return result;
  }
};

enum class MeshType
{
  Cartesian,
  Curvilinear
};

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
     prm.enter_subsection("Application");
       prm.add_parameter("MeshType", mesh_type_string, "Type of mesh (Cartesian versus curvilinear).", dealii::Patterns::Selection("Cartesian|Curvilinear"));
     prm.leave_subsection();
    // clang-format on
  }

private:
  void
  parse_parameters() final
  {
    ApplicationBase<dim, Number>::parse_parameters();

    Utilities::string_to_enum(mesh_type, mesh_type_string);
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.equation_type   = EquationType::NavierStokes;
    this->param.right_hand_side = false;

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

    // l=4, k=3 (64³), exponent_fe_degree_cfl = 2:
    // Classical RK:
    // classical RK1 (1 stage): CFL_crit = 0.04 - 0.05, costs = stages / CFL = 20 - 25
    // classical RK2 (2 stages):CFL_crit = 0.31  - 0.32, costs = stages / CFL =  6.3 - 6.5
    // classical RK3 (3 stages):CFL_crit = 0.35 - 0.4, costs = stages / CFL =  7.5 - 8.5
    // classical RK4 (4 stages):CFL_crit = 0.4  - 0.45, costs = stages / CFL =  8.9 - 10

    // Kennedy et al.:
    // LowStorageRK3Stage3Reg2: CFL_crit = 0.3 - 0.4, costs = stages / CFL =  7.5 - 10
    // LowStorageRK3Stage4Reg2: CFL_crit = 0.4 - 0.5, costs = stages / CFL =  8 - 10
    // LowStorageRK4Stage5Reg2: CFL_crit = 0.725 - 0.75, costs = stages / CFL =  6.7 - 6.9
    // LowStorageRK4Stage5Reg3: CFL_crit = 0.7 - 0.725, costs = stages / CFL =  6.9 - 7.1
    // LowStorageRK5Stage9Reg2: CFL_crit = 0.8 - 0.9, costs = stages / CFL =  10  - 11.3

    // Toulorge & Desmet: 3rd order scheme with 7 stages currently the most efficient scheme
    // LowStorageRK3Stage7Reg2: CFL_crit = 1.3 - 1.35, costs = stages / CFL =  5.2 - 5.4,
    // computational costs = 0.69 CPUh LowStorageRK4Stage8Reg2: CFL_crit = 1.25 - 1.3, costs =
    // stages / CFL =  6.2 - 6.4

    // Kubatko et al.:
    // SSPRK(7,3): CFL_crit = 1.2 - 1.25, costs = stages / CFL = 5.6 - 5.8
    // SSPRK(8,3): CFL_crit = 1.5 - 1.6, costs = stages / CFL = 5.0 - 5.3, computational costs =
    // 0.77 CPUh SSPRK(8,4): CFL_crit = 1.25 - 1.3, costs = stages / CFL = 6.2 - 6.4

    this->param.temporal_discretization       = TemporalDiscretization::ExplRK3Stage7Reg2;
    this->param.order_time_integrator         = 3;
    this->param.stages                        = 7;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
    this->param.time_step_size                = 1.0e-3;
    this->param.max_velocity                  = MAX_VELOCITY;
    this->param.cfl_number                    = 0.6;
    this->param.diffusion_number              = 0.02;
    this->param.exponent_fe_degree_cfl        = 1.5;
    this->param.exponent_fe_degree_viscous    = 3.0;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 20;

    // restart
    this->param.restart_data.write_restart = false;
    this->param.restart_data.interval_time = 1.0;
    this->param.restart_data.filename =
      this->output_parameters.directory + this->output_parameters.filename + "_restart";

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = 1;
    this->param.n_q_points_convective   = QuadratureRule::Overintegration32k;
    this->param.n_q_points_viscous      = QuadratureRule::Overintegration32k;

    // viscous term
    this->param.IP_factor = 1.0;

    // NUMERICAL PARAMETERS
    this->param.use_combined_operator = true;
  }

  void
  create_grid() final
  {
    double const pi   = dealii::numbers::PI;
    double const left = -pi * L, right = pi * L;
    double const deformation = 0.1;

    bool curvilinear_mesh = false;
    if(mesh_type == MeshType::Cartesian)
    {
      // do nothing
    }
    else if(mesh_type == MeshType::Curvilinear)
    {
      curvilinear_mesh = true;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    create_periodic_box(this->grid->triangulation,
                        this->param.grid.n_refine_global,
                        this->grid->periodic_face_pairs,
                        this->n_subdivisions_1d_hypercube,
                        left,
                        right,
                        curvilinear_mesh,
                        deformation);
  }

  void
  set_boundary_descriptor() final
  {
    // test case with periodic BC -> boundary descriptors remain empty
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new InitialSolution<dim>());
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
    pp_data.output_data.time_control_data.trigger_interval = (end_time - start_time) / 20.0;
    pp_data.output_data.directory         = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename          = this->output_parameters.filename;
    pp_data.output_data.write_pressure    = true;
    pp_data.output_data.write_velocity    = true;
    pp_data.output_data.write_temperature = true;
    pp_data.output_data.write_vorticity   = true;
    pp_data.output_data.write_divergence  = true;
    pp_data.output_data.degree            = this->param.degree;

    // kinetic energy
    pp_data.kinetic_energy_data.time_control_data.is_active                = true;
    pp_data.kinetic_energy_data.time_control_data.trigger_every_time_steps = 1;
    pp_data.kinetic_energy_data.time_control_data.start_time               = start_time;
    pp_data.kinetic_energy_data.viscosity                                  = DYN_VISCOSITY / RHO_0;
    pp_data.kinetic_energy_data.directory = this->output_parameters.directory;
    pp_data.kinetic_energy_data.filename  = this->output_parameters.filename;

    // kinetic energy spectrum
    pp_data.kinetic_energy_spectrum_data.time_control_data.is_active        = true;
    pp_data.kinetic_energy_spectrum_data.time_control_data.trigger_interval = 0.5;
    pp_data.kinetic_energy_spectrum_data.time_control_data.start_time       = start_time;


    pp_data.kinetic_energy_spectrum_data.directory = this->output_parameters.directory;
    pp_data.kinetic_energy_spectrum_data.filename  = this->output_parameters.filename + "_spectrum";
    pp_data.kinetic_energy_spectrum_data.degree    = this->param.degree;
    pp_data.kinetic_energy_spectrum_data.evaluation_points_per_cell = (this->param.degree + 1) * 1;
    pp_data.kinetic_energy_spectrum_data.exploit_symmetry           = false;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  std::string mesh_type_string = "Cartesian";
  MeshType    mesh_type        = MeshType::Cartesian;

  double const start_time = 0.0;
  double const end_time   = 20.0 * CHARACTERISTIC_TIME;
};

} // namespace CompNS

} // namespace ExaDG

#include <exadg/compressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_3D_TAYLOR_GREEN_VORTEX_H_ */
