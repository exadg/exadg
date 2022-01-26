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

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_PROPAGATING_SINE_WAVE_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_PROPAGATING_SINE_WAVE_H_

// test case for a purely convective problem
// sine wave that is advected from left to right by a constant velocity field

#include <exadg/grid/mesh_movement_functions.h>

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution(unsigned int const n_components = 1, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const /*component*/) const
  {
    double t = this->get_time();

    double result = std::sin(numbers::PI * (p[0] - t));

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
  double const end_time   = 8.0;

  double const left  = -1.0;
  double const right = +1.0;

  bool const ale = false;

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type                = ProblemType::Unsteady;
    this->param.equation_type               = EquationType::Convection;
    this->param.analytical_velocity_field   = true;
    this->param.right_hand_side             = false;
    this->param.ale_formulation             = ale;
    this->param.formulation_convective_term = FormulationConvectiveTerm::ConvectiveFormulation;

    // PHYSICAL QUANTITIES
    this->param.start_time  = start_time;
    this->param.end_time    = end_time;
    this->param.diffusivity = 0.0;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization = TemporalDiscretization::ExplRK;
    this->param.time_integrator_rk      = TimeIntegratorRK::ExplRK3Stage7Reg2;
    this->param.treatment_of_convective_term =
      TreatmentOfConvectiveTerm::Explicit; // ExplicitOIF; //Explicit;
    this->param.time_integrator_oif           = TimeIntegratorRK::ExplRK3Stage7Reg2;
    this->param.adaptive_time_stepping        = true;
    this->param.order_time_integrator         = 2;
    this->param.start_with_low_order          = false;
    this->param.calculation_of_time_step_size = TimeStepCalculation::CFL;
    this->param.time_step_size                = 1.0e-1;
    this->param.cfl                           = 0.2;
    this->param.cfl_oif                       = this->param.cfl / 1.0;
    this->param.diffusion_number              = 0.01;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = 1;

    // convective term
    this->param.numerical_flux_convective_operator =
      NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    this->param.IP_factor = 1.0;

    // SOLVER
    this->param.solver         = Solver::GMRES;
    this->param.solver_data    = SolverData(1e4, 1.e-20, 1.e-6, 100);
    this->param.preconditioner = Preconditioner::InverseMassMatrix;
    // use default parameters of multigrid preconditioner

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 20;

    // NUMERICAL PARAMETERS
    this->param.use_combined_operator                   = true;
    this->param.store_analytical_velocity_in_dof_vector = false;
  }

  void
  create_grid() final
  {
    GridGenerator::hyper_cube(*this->grid->triangulation, left, right);

    // set boundary id of 1 at right boundary (outflow)
    for(auto cell : *this->grid->triangulation)
    {
      for(unsigned int f = 0; f < GeometryInfo<dim>::faces_per_cell; ++f)
      {
        if((std::fabs(cell.face(f)->center()(0) - right) < 1e-12))
          cell.face(f)->set_boundary_id(1);
      }
    }

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  std::shared_ptr<Function<dim>>
  create_mesh_movement_function() final
  {
    std::shared_ptr<Function<dim>> mesh_motion;

    MeshMovementData<dim> data;
    data.temporal                       = MeshMovementAdvanceInTime::Sin;
    data.shape                          = MeshMovementShape::SineZeroAtBoundary; // SineAligned;
    data.dimensions[0]                  = std::abs(right - left);
    data.dimensions[1]                  = std::abs(right - left);
    data.amplitude                      = 0.08 * (right - left); // A_max = (right-left)/(2*pi)
    data.period                         = end_time;
    data.t_start                        = 0.0;
    data.t_end                          = end_time;
    data.spatial_number_of_oscillations = 1.0;
    mesh_motion.reset(new CubeMeshMovementFunctions<dim>(data));

    return mesh_motion;
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    this->boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>()));
    this->boundary_descriptor->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
  }


  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new Solution<dim>());
    this->field_functions->right_hand_side.reset(new Functions::ZeroFunction<dim>(1));
    std::vector<double> velocity = std::vector<double>(dim, 0.0);
    velocity[0]                  = 1.0;
    this->field_functions->velocity.reset(new Functions::ConstantFunction<dim>(velocity));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output  = this->write_output;
    pp_data.output_data.directory     = this->output_directory + "vtu/";
    pp_data.output_data.filename      = this->output_name;
    pp_data.output_data.start_time    = start_time;
    pp_data.output_data.interval_time = (end_time - start_time) / 100;
    pp_data.output_data.degree        = this->param.degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>(1));
    pp_data.error_data.error_calc_start_time    = start_time;
    pp_data.error_data.error_calc_interval_time = (end_time - start_time) / 20;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace ConvDiff

} // namespace ExaDG

#include <exadg/convection_diffusion/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_PROPAGATING_SINE_WAVE_H_ */
