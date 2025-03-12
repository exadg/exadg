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

#ifndef APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_MANUFACTURED_SOLUTION_H_
#define APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_MANUFACTURED_SOLUTION_H_

/*
 *  This 2D test case is a quasi one-dimensional problem with periodic boundary
 *  conditions in x_2-direction. The velocity u_2 is zero. The energy is constant.
 *  The density and the velocity u_1 are a function of x_1 and time t.
 */

namespace ExaDG
{
namespace CompNS
{
// problem specific parameters
double const DYN_VISCOSITY = 0.1;
double const GAMMA         = 1.4;
double const LAMBDA        = 0.0;
double const R             = 1.0;
double const U_0           = 1.0;
double const V_0           = 0.0;
double const RHO_0         = 1.0;
double const E_0           = 1.0e5;
double const EPSILON       = 0.1 * RHO_0;
double const T_MAX         = 140.0;

enum class SolutionType
{
  Polynomial,
  SineAndPolynomial
};

SolutionType const SOLUTION_TYPE = SolutionType::Polynomial;

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
    double       t  = this->get_time();
    double const pi = dealii::numbers::PI;

    double result = 0.0;

    double sin_pit = sin(pi * t);
    double cos_pit = cos(pi * t);
    double sin_pix = sin(pi * p[0]);
    double x3      = p[0] * p[0] * p[0];

    double rho = 0.0;

    if(SOLUTION_TYPE == SolutionType::Polynomial)
    {
      rho = RHO_0 + EPSILON * x3 * sin_pit;
    }
    else if(SOLUTION_TYPE == SolutionType::SineAndPolynomial)
    {
      rho = RHO_0 + EPSILON * sin_pix * sin_pit;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

    if(component == 0)
      result = rho;
    else if(component == 1)
      result = rho * U_0 * x3 * sin_pit;
    else if(component == 2)
      result = rho * V_0 * x3 * cos_pit;
    else if(component == 1 + dim)
      result = rho * E_0;

    return result;
  }
};


template<int dim>
class RightHandSideDensity : public dealii::Function<dim>
{
public:
  RightHandSideDensity(unsigned int const n_components = 1, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    (void)component;

    double t       = this->get_time();
    double pi      = dealii::numbers::PI;
    double sin_pix = sin(pi * p[0]);
    double cos_pix = cos(pi * p[0]);
    double sin_pit = sin(pi * t);
    double cos_pit = cos(pi * t);
    double x2      = p[0] * p[0];
    double x3      = p[0] * p[0] * p[0];
    double x5      = x2 * x3;

    double result = 0.0;

    // clang-format off
    if(SOLUTION_TYPE == SolutionType::Polynomial)
    {
      result = EPSILON * pi * x3 * cos_pit // = d(rho)/dt
               + 3.0 * RHO_0 * U_0 * x2 * sin_pit + 6.0 * EPSILON * U_0 * x5 * sin_pit * sin_pit; //d(rho*u1)/dx1
    }
    else if (SOLUTION_TYPE == SolutionType::SineAndPolynomial)
    {
      result = + EPSILON * pi * sin_pix * cos_pit // = d(rho)/dt
               + EPSILON * U_0 * pi * x3 * cos_pix * sin_pit * sin_pit // = u1 * d(rho)/dx1
               + (RHO_0 + EPSILON * sin_pix * sin_pit) * 3.0 * U_0 * x2 * sin_pit;// rho * d(u1)/dx1
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
    // clang-format on

    return result;
  }
};

template<int dim>
class RightHandSideVelocity : public dealii::Function<dim>
{
public:
  RightHandSideVelocity(unsigned int const n_components = dim, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double       t       = this->get_time();
    double const pi      = dealii::numbers::PI;
    double const sin_pix = sin(pi * p[0]);
    double const cos_pix = cos(pi * p[0]);
    double const sin_pit = sin(pi * t);
    double const cos_pit = cos(pi * t);
    double const p1      = p[0];
    double const p2      = p[0] * p[0];
    double const p3      = p[0] * p[0] * p[0];
    double const p5      = p[0] * p[0] * p[0] * p[0] * p[0];
    double const p6      = p5 * p[0];
    double const p8      = p5 * p3;

    double result = 0.0;

    // clang-format off
    if(SOLUTION_TYPE == SolutionType::Polynomial)
    {
      if(component==0)
      {
      result = + RHO_0  * pi * U_0 * p3 * cos_pit + 2.0 * U_0 * EPSILON * pi * p6 * cos_pit * sin_pit // = d(rho u1)/dt
               + (1.0-(GAMMA-1.0)/2.0) * (RHO_0 * 6.0 * U_0*U_0 * p5 * sin_pit*sin_pit + 9.0 * EPSILON * U_0*U_0  * p8  * sin_pit*sin_pit*sin_pit) // =(1 + (gamma-1)/2) d(rho u1^2)/dx1
               + (GAMMA-1.0) * E_0 * 3.0 * EPSILON * p2 * sin_pit // = (gamma-1) d(rhoE)/dx1
               - 8.0 * DYN_VISCOSITY * U_0  * p1 * sin_pit; // viscous term (= - 4/3 mu d²(u1)/dx1²)
      }
    }
    else if (SOLUTION_TYPE == SolutionType::SineAndPolynomial)
    {
      if(component==0)
      {
        result = + U_0 * EPSILON * pi * sin_pix * p3 * cos_pit * sin_pit + (RHO_0 + EPSILON * sin_pix * sin_pit) * pi * U_0 * p3 * cos_pit // = d(rho u1)/dt
                 + (1.0-(GAMMA-1.0)/2.0) * ( RHO_0 * U_0*U_0 * 6.0 * p5 * sin_pit*sin_pit + U_0*U_0 * EPSILON * sin_pit*sin_pit*sin_pit * ( 6.0 * sin_pix * p5 + pi * p6 * cos_pix )) // =(1 + (gamma-1)/2) d(rho u1^2)/dx1
                 + (GAMMA-1.0) * E_0 * EPSILON * pi * cos_pix * sin_pit // = (gamma-1) d(rhoE)/dx1
                 - 8.0 * DYN_VISCOSITY * U_0 * p1 * sin_pit; // viscous term (= - 4/3 mu d²(u1)/dx1²)
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
    // clang-format on

    return result;
  }
};

template<int dim>
class RightHandSideEnergy : public dealii::Function<dim>
{
public:
  RightHandSideEnergy(unsigned int const n_components = 1, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    (void)component;

    double       t             = this->get_time();
    double const pi            = dealii::numbers::PI;
    double const sin_pix       = sin(pi * p[0]);
    double const cos_pix       = cos(pi * p[0]);
    double const sin_pit       = sin(pi * t);
    double const cos_pit       = cos(pi * t);
    double const p2            = p[0] * p[0];
    double const p3            = p2 * p[0];
    double const p4            = p2 * p2;
    double const p5            = p2 * p3;
    double const p8            = p3 * p3 * p2;
    double const p9            = p4 * p5;
    double const p11           = p8 * p3;
    double const dyn_viscosity = DYN_VISCOSITY;

    double result = 0.0;

    // clang-format off
    if(SOLUTION_TYPE == SolutionType::Polynomial)
    {
      result = + E_0 * EPSILON * pi * p3 * cos_pit // = d(rho*E)/dt
               + GAMMA * (3.0 * RHO_0 * E_0 * U_0 * p2 * sin_pit + 6.0 * EPSILON * U_0 * E_0 * p5 * sin_pit*sin_pit) // = d(rho gamma E u1)/dx1
               - (GAMMA-1.0)/2.0 * (9.0 * RHO_0 * U_0*U_0*U_0 * p8 * sin_pit*sin_pit*sin_pit + 12.0 * EPSILON * U_0*U_0*U_0 * p11 * sin_pit*sin_pit*sin_pit*sin_pit)// = -(gamma-1)/2 d(rho u1^3)/dx1
               - dyn_viscosity * 20.0 * U_0*U_0 * p4 * sin_pit*sin_pit; // viscous term (= d(u1*tau11)/dx1
    }
    else if (SOLUTION_TYPE == SolutionType::SineAndPolynomial)
    {
      result = + E_0 * EPSILON * sin_pix * pi * cos_pit // = d(rho*E)/dt
               + GAMMA * (3.0 * RHO_0 * E_0 * U_0 * p2 * sin_pit + EPSILON * E_0 * U_0 * sin_pit*sin_pit * ( 3.0 * p2 * sin_pix + pi * p3 * cos_pix)) // = d(rho gamma E u1)/dx1
               - (GAMMA-1.0)/2.0 * (9.0 * RHO_0 * U_0*U_0*U_0 * p8 * sin_pit*sin_pit*sin_pit + EPSILON * U_0*U_0*U_0 * sin_pit*sin_pit*sin_pit*sin_pit * (9.0 * p8 * sin_pix + pi * p9 * cos_pix))// = -(gamma-1)/2 d(rho u1^3)/dx1
               - dyn_viscosity * 20.0 * U_0*U_0 * p4 * sin_pit*sin_pit; // viscous term (= d(u1*tau11)/dx1
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
    // clang-format on

    double f_times_u = 0;

    RightHandSideVelocity<dim> rhs_velocity(dim, t);
    Solution<dim>              analytical_solution(dim + 2, t);

    for(unsigned int d = 0; d < dim; ++d)
    {
      f_times_u += rhs_velocity.value(p, d) * analytical_solution.value(p, 1 + d) /
                   analytical_solution.value(p, 0);
    }

    return result - f_times_u;
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
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    double       t       = this->get_time();
    double const pi      = dealii::numbers::PI;
    double       x3      = p[0] * p[0] * p[0];
    double       sin_pit = sin(pi * t);
    double       cos_pit = cos(pi * t);

    double result = 0.0;

    if(SOLUTION_TYPE == SolutionType::Polynomial or
       SOLUTION_TYPE == SolutionType::SineAndPolynomial)
    {
      if(component == 0)
        result = U_0 * x3 * sin_pit;
      else if(component == 1)
        result = V_0 * x3 * cos_pit;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

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
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const final
  {
    (void)component;

    double       t       = this->get_time();
    double const pi      = dealii::numbers::PI;
    double       sin_pit = sin(pi * t);
    double       sin_pix = sin(pi * p[0]);
    double       x3      = p[0] * p[0] * p[0];

    double result = 0.0;

    if(SOLUTION_TYPE == SolutionType::Polynomial)
    {
      result = RHO_0 + EPSILON * x3 * sin_pit;
    }
    else if(SOLUTION_TYPE == SolutionType::SineAndPolynomial)
    {
      result = RHO_0 + EPSILON * sin_pix * sin_pit;
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }

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
    this->param.max_temperature       = T_MAX;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization       = TemporalDiscretization::ExplRK3Stage7Reg2;
    this->param.order_time_integrator         = 4;
    this->param.stages                        = 8;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFLAndDiffusion;
    this->param.time_step_size                = 1.0e-3;
    this->param.max_velocity                  = std::sqrt(U_0 * U_0 + V_0 * V_0);
    this->param.cfl_number                    = 0.025;
    this->param.diffusion_number              = 0.1;
    this->param.exponent_fe_degree_cfl        = 2.0;
    this->param.exponent_fe_degree_viscous    = 4.0;

    // restart
    this->param.restarted_simulation       = false;
    this->param.restart_data.write_restart = false;
    this->param.restart_data.interval_time = 0.5;
    this->param.restart_data.filename =
      this->output_parameters.directory + this->output_parameters.filename + "_restart";

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 10;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.mapping_degree          = this->param.degree;
    this->param.n_q_points_convective   = QuadratureRule::Standard;
    this->param.n_q_points_viscous      = QuadratureRule::Standard;

    // viscous term
    this->param.IP_factor = 1.0;

    // NUMERICAL PARAMETERS
    this->param.detect_instabilities  = false;
    this->param.use_combined_operator = false;
  }

  void
  create_grid(Grid<dim> & grid, std::shared_ptr<dealii::Mapping<dim>> & mapping) final
  {
    auto const lambda_create_triangulation = [&](dealii::Triangulation<dim, dim> & tria,
                                                 std::vector<dealii::GridTools::PeriodicFacePair<
                                                   typename dealii::Triangulation<
                                                     dim>::cell_iterator>> & periodic_face_pairs,
                                                 unsigned int const          global_refinements,
                                                 std::vector<unsigned int> const &
                                                   vector_local_refinements) {
      (void)vector_local_refinements;

      // hypercube volume is [left,right]^dim
      double const left = -1.0, right = 0.5;
      dealii::GridGenerator::hyper_cube(tria, left, right);

      AssertThrow(
        this->param.grid.triangulation_type != TriangulationType::FullyDistributed,
        dealii::ExcMessage(
          "Periodic faces might not be applied correctly for TriangulationType::FullyDistributed. "
          "Try to use another triangulation type, or try to fix these limitations in ExaDG or deal.II."));

      for(auto cell : tria)
      {
        for(auto const & face : cell.face_indices())
        {
          if(std::fabs(cell.face(face)->center()(1) - left) < 1e-12)
          {
            cell.face(face)->set_boundary_id(0 + 10);
          }
          else if(std::fabs(cell.face(face)->center()(1) - right) < 1e-12)
          {
            cell.face(face)->set_boundary_id(1 + 10);
          }
        }
      }

      dealii::GridTools::collect_periodic_faces(tria, 0 + 10, 1 + 10, 1, periodic_face_pairs);
      tria.add_periodicity(periodic_face_pairs);

      tria.refine_global(global_refinements);
    };

    GridUtilities::create_triangulation<dim>(grid,
                                             this->mpi_comm,
                                             this->param.grid,
                                             lambda_create_triangulation,
                                             {} /* no local refinements */);

    GridUtilities::create_mapping(mapping,
                                  this->param.grid.element_type,
                                  this->param.mapping_degree);
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
    this->boundary_descriptor->energy.dirichlet_bc.insert(
      pair(0, new dealii::Functions::ConstantFunction<dim>(E_0, 1)));
    // set energy boundary variable
    this->boundary_descriptor->energy.boundary_variable.insert(
      pair_variable(0, EnergyBoundaryVariable::Energy));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new Solution<dim>());
    this->field_functions->right_hand_side_density.reset(new RightHandSideDensity<dim>());
    this->field_functions->right_hand_side_velocity.reset(new RightHandSideVelocity<dim>());
    this->field_functions->right_hand_side_energy.reset(new RightHandSideEnergy<dim>());
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

    pp_data.error_data.time_control_data.is_active        = true;
    pp_data.error_data.time_control_data.start_time       = start_time;
    pp_data.error_data.time_control_data.trigger_interval = (end_time - start_time) / 20.0;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>());

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  double const start_time = 0.0;
  double const end_time   = 0.75;
};

} // namespace CompNS

} // namespace ExaDG

#include <exadg/compressible_navier_stokes/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_COMPRESSIBLE_NAVIER_STOKES_TEST_CASES_TEST_MANUFACTURED_SOLUTION_H_ */
