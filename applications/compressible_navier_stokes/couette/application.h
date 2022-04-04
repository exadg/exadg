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

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COUETTE_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COUETTE_H_

namespace ExaDG
{
namespace CompNS
{
// problem specific parameters
double const DYN_VISCOSITY = 1.0e-2;
double const GAMMA         = 1.4;
double const LAMBDA        = 0.0262;
double const R             = 287.058;
double const U_0           = 1.0;
double const PRESSURE      = 1.0e5;
double const GAS_CONSTANT  = 287.058;
double const T_0           = 273.0;
double const RHO_0         = PRESSURE / (R * T_0);

double const H = 3.0;
double const L = 2.0 * H;

template<int dim>
class Solution : public dealii::Function<dim>
{
public:
  Solution(unsigned int const n_components = dim + 2, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double const T =
      T_0 + DYN_VISCOSITY * U_0 * U_0 / (2.0 * LAMBDA) * (p[1] * p[1] / (H * H) - 1.0);
    double const rho = PRESSURE / (R * T);
    double const u   = U_0 / H * p[1];

    double result = 0.0;

    if(component == 0)
      result = rho;
    else if(component == 1)
      result = rho * u;
    else if(component == 2)
      result = 0.0;
    else if(component == 1 + dim)
      result = rho * (0.5 * u * u + R * T / (GAMMA - 1.0));

    return result;
  }
};

template<int dim>
class VelocityBC : public dealii::Function<dim>
{
public:
  VelocityBC(unsigned int const n_components = dim, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    double result = 0.0;

    if(component == 0)
      result = U_0 * p[1] / H;
    else if(component == 1)
      result = 0.0;

    return result;
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
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const
  {
    (void)component;

    double const T =
      T_0 + DYN_VISCOSITY * U_0 * U_0 / (2.0 * LAMBDA) * (p[1] * p[1] / (H * H) - 1.0);

    double const rho = PRESSURE / (R * T);

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
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree;
    this->param.n_q_points_convective   = QuadratureRule::Standard;
    this->param.n_q_points_viscous      = QuadratureRule::Standard;

    // viscous term
    this->param.IP_factor = 1.0;

    // NUMERICAL PARAMETERS
    this->param.use_combined_operator = false;
  }

  void
  create_grid() final
  {
    std::vector<unsigned int> repetitions({2, 1});
    dealii::Point<dim>        point1(0.0, 0.0), point2(L, H);
    dealii::GridGenerator::subdivided_hyper_rectangle(*this->grid->triangulation,
                                                      repetitions,
                                                      point1,
                                                      point2);

    // indicator
    // fixed wall = 0
    // moving wall = 1
    /*
     *             indicator = 1
     *   ___________________________________
     *   |             --->                 |
     *   |                                  |
     *   | <---- periodic B.C.  --------->  |
     *   |                                  |
     *   |                                  |
     *   |__________________________________|
     *             indicator = 0
     */
    for(auto cell : *this->grid->triangulation)
    {
      for(auto const & face : cell.face_indices())
      {
        if(std::fabs(cell.face(face)->center()(1) - point2[1]) < 1e-12)
        {
          cell.face(face)->set_boundary_id(1);
        }
        else if(std::fabs(cell.face(face)->center()(1) - 0.0) < 1e-12)
        {
          cell.face(face)->set_boundary_id(0);
        }
        else if(std::fabs(cell.face(face)->center()(0) - 0.0) < 1e-12)
        {
          cell.face(face)->set_boundary_id(0 + 10);
        }
        else if(std::fabs(cell.face(face)->center()(0) - point2[0]) < 1e-12)
        {
          cell.face(face)->set_boundary_id(1 + 10);
        }
      }
    }

    dealii::GridTools::collect_periodic_faces(
      *this->grid->triangulation, 0 + 10, 1 + 10, 0, this->grid->periodic_faces);
    this->grid->triangulation->add_periodicity(this->grid->periodic_faces);

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
                                                                                   pair;
    typedef typename std::pair<dealii::types::boundary_id, EnergyBoundaryVariable> pair_variable;

    // density
    this->boundary_descriptor->density.dirichlet_bc.insert(pair(0, new DensityBC<dim>()));
    this->boundary_descriptor->density.dirichlet_bc.insert(pair(1, new DensityBC<dim>()));
    //  boundary_descriptor->density.neumann_bc.insert(pair(0,new
    //  dealii::Functions::ZeroFunction<dim>(1)));
    //  boundary_descriptor->density.neumann_bc.insert(pair(1,new
    //  dealii::Functions::ZeroFunction<dim>(1)));

    // velocity
    this->boundary_descriptor->velocity.dirichlet_bc.insert(pair(0, new VelocityBC<dim>()));
    this->boundary_descriptor->velocity.dirichlet_bc.insert(pair(1, new VelocityBC<dim>()));

    // pressure
    this->boundary_descriptor->pressure.neumann_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(1)));
    this->boundary_descriptor->pressure.neumann_bc.insert(
      pair(1, new dealii::Functions::ZeroFunction<dim>(1)));

    // energy: prescribe temperature
    this->boundary_descriptor->energy.boundary_variable.insert(
      pair_variable(0, EnergyBoundaryVariable::Temperature));
    this->boundary_descriptor->energy.boundary_variable.insert(
      pair_variable(1, EnergyBoundaryVariable::Temperature));

    this->boundary_descriptor->energy.neumann_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(1)));
    this->boundary_descriptor->energy.dirichlet_bc.insert(
      pair(1, new dealii::Functions::ConstantFunction<dim>(T_0, 1)));
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
    pp_data.output_data.write_output      = this->output_parameters.write;
    pp_data.output_data.directory         = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename          = this->output_parameters.filename;
    pp_data.output_data.write_pressure    = true;
    pp_data.output_data.write_velocity    = true;
    pp_data.output_data.write_temperature = true;
    pp_data.output_data.write_vorticity   = true;
    pp_data.output_data.write_divergence  = true;
    pp_data.output_data.start_time        = start_time;
    pp_data.output_data.interval_time     = (end_time - start_time) / 10;
    pp_data.output_data.degree            = this->param.degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>());
    pp_data.error_data.error_calc_start_time    = start_time;
    pp_data.error_data.error_calc_interval_time = (end_time - start_time) / 10;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  double const start_time = 0.0;
  double const end_time   = 10.0;
};

} // namespace CompNS

} // namespace ExaDG

#include <exadg/compressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_COUETTE_H_ */
