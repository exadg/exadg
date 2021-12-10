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

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_CONST_RHS_CONST_AND_CIRCULAR_WIND_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_CONST_RHS_CONST_AND_CIRCULAR_WIND_H_

// constant source term inside rectangular domain
// pure Dirichlet boundary conditions (homogeneous)
// use constant or circular advection velocity

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

enum class VelocityType
{
  Constant,
  Circular,
  CircularZeroAtBoundary
};
VelocityType const VELOCITY_TYPE = VelocityType::CircularZeroAtBoundary;

template<int dim>
class VelocityField : public Function<dim>
{
public:
  VelocityField(unsigned int const n_components = dim, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & point, unsigned int const component = 0) const
  {
    double value = 0.0;

    if(VELOCITY_TYPE == VelocityType::Constant)
    {
      // constant velocity field (u,v) = (1,1)
      value = 1.0;
    }
    else if(VELOCITY_TYPE == VelocityType::Circular)
    {
      // circular velocity field (u,v) = (-y,x)
      if(component == 0)
        value = -point[1];
      else if(component == 1)
        value = point[0];
      else
        AssertThrow(component <= 1,
                    ExcMessage("Velocity field for 3-dimensional problem is not implemented!"));
    }
    else if(VELOCITY_TYPE == VelocityType::CircularZeroAtBoundary)
    {
      double const pi    = numbers::PI;
      double       sinx  = std::sin(pi * point[0]);
      double       siny  = std::sin(pi * point[1]);
      double       sin2x = std::sin(2. * pi * point[0]);
      double       sin2y = std::sin(2. * pi * point[1]);
      if(component == 0)
        value = pi * sin2y * std::pow(sinx, 2.);
      else if(component == 1)
        value = -pi * sin2x * std::pow(siny, 2.);
    }
    else
    {
      AssertThrow(false, ExcMessage("Invalid type of velocity field prescribed for this problem."));
    }

    return value;
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

  double const left = -1.0, right = 1.0;

  double const diffusivity = 1.0;

  double const start_time = 0.0;
  double const end_time   = 1.0;

  void
  set_input_parameters(unsigned int const degree) final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type              = ProblemType::Steady;
    this->param.equation_type             = EquationType::ConvectionDiffusion;
    this->param.analytical_velocity_field = true;
    this->param.right_hand_side           = true;

    // PHYSICAL QUANTITIES
    this->param.start_time  = start_time;
    this->param.end_time    = end_time;
    this->param.diffusivity = diffusivity;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization       = TemporalDiscretization::BDF;
    this->param.time_integrator_rk            = TimeIntegratorRK::ExplRK3Stage7Reg2;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit;
    this->param.order_time_integrator         = 2;
    this->param.start_with_low_order          = true;
    this->param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    this->param.time_step_size                = 1.0e-1;
    this->param.cfl                           = 0.2;
    this->param.diffusion_number              = 0.01;

    // SPATIAL DISCRETIZATION
    this->param.triangulation_type = TriangulationType::Distributed;
    this->param.mapping            = MappingType::Affine;
    this->param.degree             = degree;

    // convective term
    this->param.numerical_flux_convective_operator =
      NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    this->param.IP_factor = 1.0;

    // SOLVER
    this->param.solver         = Solver::GMRES;
    this->param.solver_data    = SolverData(1e4, 1.e-20, 1.e-8, 100);
    this->param.preconditioner = Preconditioner::Multigrid; // PointJacobi; //BlockJacobi;
    this->param.implement_block_diagonal_preconditioner_matrix_free = false;
    this->param.use_cell_based_face_loops                           = false;
    this->param.solver_block_diagonal                               = Elementwise::Solver::GMRES;
    this->param.update_preconditioner                               = true;

    this->param.multigrid_data.type = MultigridType::phMG;
    this->param.mg_operator_type    = MultigridOperatorType::ReactionConvectionDiffusion;
    // MG smoother
    this->param.multigrid_data.smoother_data.smoother = MultigridSmoother::Jacobi;

    // MG smoother data: Chebyshev smoother
    //  this->param.multigrid_data.smoother_data.iterations = 3;

    // MG smoother data: GMRES smoother, CG smoother
    this->param.multigrid_data.smoother_data.preconditioner = PreconditionerSmoother::BlockJacobi;
    this->param.multigrid_data.smoother_data.iterations     = 4;

    // MG smoother data: Jacobi smoother
    //  this->param.multigrid_data.smoother_data.preconditioner =
    //  PreconditionerSmoother::BlockJacobi; this->param.multigrid_data.smoother_data.iterations =
    //  5; this->param.multigrid_data.smoother_data.relaxation_factor = 0.8;

    // MG coarse grid solver
    this->param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::AMG; // GMRES;

    // output of solver information
    this->param.solver_info_data.interval_time = (end_time - start_time) / 10;

    // NUMERICAL PARAMETERS
    this->param.use_combined_operator                   = true;
    this->param.store_analytical_velocity_in_dof_vector = true;
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid(GridData const & grid_data) final
  {
    std::shared_ptr<Grid<dim, Number>> grid =
      std::make_shared<Grid<dim, Number>>(grid_data, this->mpi_comm);

    GridGenerator::hyper_cube(*grid->triangulation, left, right);

    grid->triangulation->refine_global(grid_data.n_refine_global);

    return grid;
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    this->boundary_descriptor->dirichlet_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new Functions::ConstantFunction<dim>(1.0, 1));
    this->field_functions->velocity.reset(new VelocityField<dim>());
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = this->output_name;
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = end_time - start_time;
    pp_data.output_data.write_higher_order = false;
    pp_data.output_data.degree             = this->param.degree;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace ConvDiff

} // namespace ExaDG

#include <exadg/convection_diffusion/user_interface/implement_get_application.h>

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_CONST_RHS_CONST_AND_CIRCULAR_WIND_H_ */
