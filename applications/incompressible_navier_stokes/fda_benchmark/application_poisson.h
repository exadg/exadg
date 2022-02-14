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

#ifndef APPLICATIONS_POISSON_TEST_CASES_NOZZLE_H_
#define APPLICATIONS_POISSON_TEST_CASES_NOZZLE_H_

#include "include/grid.h"

namespace ExaDG
{
namespace Poisson
{
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
    this->param.right_hand_side = false;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = 3;
    this->param.spatial_discretization  = SpatialDiscretization::DG;
    this->param.IP_factor               = 1.0e0;

    // SOLVER
    this->param.solver                      = Poisson::Solver::CG;
    this->param.solver_data.abs_tol         = 1.e-20;
    this->param.solver_data.rel_tol         = 1.e-10;
    this->param.solver_data.max_iter        = 1e4;
    this->param.compute_performance_metrics = true;
    this->param.preconditioner              = Preconditioner::Multigrid;
    this->param.multigrid_data.type         = MultigridType::cphMG;
    this->param.multigrid_data.p_sequence   = PSequenceType::Bisect;
    // MG smoother
    this->param.multigrid_data.smoother_data.smoother   = MultigridSmoother::Chebyshev;
    this->param.multigrid_data.smoother_data.iterations = 5;
    // MG coarse grid solver
    this->param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    this->param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    this->param.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-3;
  }

  void
  create_grid() final
  {
    FDANozzle::create_grid_and_set_boundary_ids_nozzle(this->grid->triangulation,
                                                       this->param.grid.n_refine_global,
                                                       this->grid->periodic_faces);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // inflow
    this->boundary_descriptor->dirichlet_bc.insert(
      pair(1, new dealii::Functions::ConstantFunction<dim>(1.0, 1)));

    // outflow
    this->boundary_descriptor->dirichlet_bc.insert(
      pair(2, new dealii::Functions::ZeroFunction<dim>(1)));

    // walls
    this->boundary_descriptor->neumann_bc.insert(
      pair(0, new dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new dealii::Functions::ZeroFunction<dim>(1));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = this->output_parameters.write;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

} // namespace Poisson

} // namespace ExaDG

#include <exadg/poisson/user_interface/implement_get_application.h>

#endif
