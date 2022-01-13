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

#ifndef APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DECAYING_HILL_H_
#define APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DECAYING_HILL_H_

namespace ExaDG
{
namespace ConvDiff
{
using namespace dealii;

enum class BoundaryConditionType
{
  HomogeneousDBC,
  HomogeneousNBC,
  HomogeneousNBCWithRHS
};

BoundaryConditionType const BOUNDARY_TYPE = BoundaryConditionType::HomogeneousDBC;

bool const RIGHT_HAND_SIDE =
  (BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS) ? true : false;

template<int dim>
class Solution : public Function<dim>
{
public:
  Solution(double const diffusivity) : Function<dim>(1, 0.0), diffusivity(diffusivity)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const /*component*/) const
  {
    double t      = this->get_time();
    double result = 1.0;

    if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousDBC)
    {
      for(int d = 0; d < dim; d++)
        result *= std::cos(p[d] * numbers::PI / 2.0);
      result *= std::exp(-0.5 * diffusivity * pow(numbers::PI, 2.0) * t);
    }
    else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBC)
    {
      for(int d = 0; d < dim; d++)
        result *= std::cos(p[d] * numbers::PI);
      result *= std::exp(-2.0 * diffusivity * pow(numbers::PI, 2.0) * t);
    }
    else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS)
    {
      for(int d = 0; d < dim; d++)
        result *= std::cos(p[d] * numbers::PI) + 1.0;
      result *= std::exp(-2.0 * diffusivity * pow(numbers::PI, 2.0) * t);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return result;
  }

private:
  double const diffusivity;
};

template<int dim>
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(double const diffusivity) : Function<dim>(1, 0.0), diffusivity(diffusivity)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const /*component*/) const
  {
    double t      = this->get_time();
    double result = 0.0;

    if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousDBC ||
       BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBC)
    {
      // do nothing, rhs=0
    }
    else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS)
    {
      for(int d = 0; d < dim; ++d)
        result += std::cos(p[d] * numbers::PI) + 1;
      result *= -std::pow(numbers::PI, 2.0) * diffusivity *
                std::exp(-2.0 * diffusivity * pow(numbers::PI, 2.0) * t);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    return result;
  }

private:
  double const diffusivity;
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

  double const diffusivity = 1.0e-1;

  double const start_time = 0.0;
  double const end_time   = 1.0;

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.problem_type    = ProblemType::Unsteady;
    this->param.equation_type   = EquationType::Diffusion;
    this->param.right_hand_side = RIGHT_HAND_SIDE;

    // PHYSICAL QUANTITIES
    this->param.start_time  = start_time;
    this->param.end_time    = end_time;
    this->param.diffusivity = diffusivity;

    // TEMPORAL DISCRETIZATION
    this->param.temporal_discretization       = TemporalDiscretization::BDF;
    this->param.time_integrator_rk            = TimeIntegratorRK::ExplRK3Stage7Reg2;
    this->param.treatment_of_convective_term  = TreatmentOfConvectiveTerm::Implicit;
    this->param.order_time_integrator         = 3;
    this->param.start_with_low_order          = false;
    this->param.calculation_of_time_step_size = TimeStepCalculation::UserSpecified;
    this->param.time_step_size                = 1.0e-3;
    this->param.cfl                           = 0.1;
    this->param.diffusion_number              = 0.04; // 0.01;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = 1;

    // convective term
    this->param.numerical_flux_convective_operator =
      NumericalFluxConvectiveOperator::LaxFriedrichsFlux;

    // viscous term
    this->param.IP_factor = 1.0;

    // SOLVER
    this->param.solver =
      Solver::CG; // use FGMRES for elementwise iterative block Jacobi type preconditioners
    this->param.solver_data      = SolverData(1e4, 1.e-20, 1.e-6, 100);
    this->param.preconditioner   = Preconditioner::InverseMassMatrix; // BlockJacobi; //Multigrid;
    this->param.mg_operator_type = MultigridOperatorType::ReactionDiffusion;
    this->param.multigrid_data.smoother_data.smoother = MultigridSmoother::Chebyshev; // GMRES;
    this->param.implement_block_diagonal_preconditioner_matrix_free = true;
    this->param.solver_block_diagonal                               = Elementwise::Solver::CG;
    this->param.preconditioner_block_diagonal = Elementwise::Preconditioner::InverseMassMatrix;
    this->param.solver_data_block_diagonal    = SolverData(1000, 1.e-12, 1.e-2, 1000);
    this->param.use_cell_based_face_loops     = true;
    this->param.update_preconditioner         = false;

    // output of solver information
    this->param.solver_info_data.interval_time = end_time - start_time;

    // NUMERICAL PARAMETERS
    this->param.use_combined_operator = true;
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid() final
  {
    std::shared_ptr<Grid<dim, Number>> grid =
      std::make_shared<Grid<dim, Number>>(this->param.grid, this->mpi_comm);

    // hypercube volume is [left,right]^dim
    GridGenerator::hyper_cube(*grid->triangulation, left, right);

    grid->triangulation->refine_global(this->param.grid.n_refine_global);

    return grid;
  }


  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousDBC)
    {
      this->boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>(diffusivity)));
    }
    else if(BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBC ||
            BOUNDARY_TYPE == BoundaryConditionType::HomogeneousNBCWithRHS)
    {
      this->boundary_descriptor->neumann_bc.insert(pair(0, new Functions::ZeroFunction<dim>(1)));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new Solution<dim>(diffusivity));
    this->field_functions->right_hand_side.reset(new RightHandSide<dim>(diffusivity));
    this->field_functions->velocity.reset(new Functions::ZeroFunction<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = this->output_name;
    pp_data.output_data.start_time         = start_time;
    pp_data.output_data.interval_time      = (end_time - start_time) / 20;
    pp_data.output_data.write_higher_order = false;
    pp_data.output_data.degree             = this->param.degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>(diffusivity));
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

#endif /* APPLICATIONS_CONVECTION_DIFFUSION_TEST_CASES_DECAYING_HILL_H_ */
