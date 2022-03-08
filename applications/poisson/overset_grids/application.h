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
#ifndef APPLICATIONS_POISSON_TEST_CASES_TEMPLATE_H_
#define APPLICATIONS_POISSON_TEST_CASES_TEMPLATE_H_

namespace ExaDG
{
namespace Poisson
{
double const FREQUENCY = 1.5 * dealii::numbers::PI;

template<int dim>
class Solution : public dealii::Function<dim>
{
public:
  Solution(unsigned int const n_components) : dealii::Function<dim>(n_components, 0.0)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const
  {
    double result = 1.0;
    for(unsigned int d = 0; d < dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

    return result;
  }
};

template<int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide(unsigned int const n_components) : dealii::Function<dim>(n_components, 0.0 /*time*/)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const component) const
  {
    double result = 0.0;

    if(component > 0)
    {
      result = FREQUENCY * FREQUENCY * dim;
      for(unsigned int d = 0; d < dim; ++d)
        result *= std::sin(FREQUENCY * p[d]);
    }

    return result;
  }
};


template<int dim, int n_components, typename Number>
class Domain1 : public ApplicationBase<dim, n_components, Number>
{
public:
  Domain1(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, n_components, Number>(input_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    Parameters & p = this->param;

    // MATHEMATICAL MODEL
    p.right_hand_side = true;

    // SPATIAL DISCRETIZATION
    p.grid.triangulation_type = TriangulationType::Distributed;
    p.grid.mapping_degree     = 3;
    p.spatial_discretization  = SpatialDiscretization::DG;
    p.IP_factor               = 1.0e0;

    // SOLVER
    p.solver                      = Solver::CG;
    p.solver_data.abs_tol         = 1.e-20;
    p.solver_data.rel_tol         = 1.e-10;
    p.solver_data.max_iter        = 1e4;
    p.compute_performance_metrics = true;
    p.preconditioner              = Preconditioner::Multigrid;
    p.multigrid_data.type         = MultigridType::cphMG;
    p.multigrid_data.p_sequence   = PSequenceType::Bisect;
    // MG smoother
    p.multigrid_data.smoother_data.smoother        = MultigridSmoother::Chebyshev;
    p.multigrid_data.smoother_data.iterations      = 5;
    p.multigrid_data.smoother_data.smoothing_range = 20;
    // MG coarse grid solver
    p.multigrid_data.coarse_problem.solver              = MultigridCoarseGridSolver::CG;
    p.multigrid_data.coarse_problem.preconditioner      = MultigridCoarseGridPreconditioner::AMG;
    p.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-3;
  }

  void
  create_grid() final
  {
    double const length = 1.0;
    double const left = -length, right = length;
    dealii::GridGenerator::subdivided_hyper_cube(*this->grid->triangulation, 1, left, right);

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    // these lines show exemplarily how the boundary descriptors are filled
    this->boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>(dim)));
  }

  void
  set_field_functions() final
  {
    // these lines show exemplarily how the field functions are filled
    this->field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->right_hand_side.reset(new RightHandSide<dim>(dim));
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


template<int dim, int n_components, typename Number>
class Domain2 : public ApplicationBase<dim, n_components, Number>
{
public:
  Domain2(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, n_components, Number>(input_file, comm)
  {
  }

private:
  void
  set_parameters() final
  {
    Parameters & p = this->param;

    // MATHEMATICAL MODEL
    p.right_hand_side = true;

    // SPATIAL DISCRETIZATION
    p.grid.triangulation_type = TriangulationType::Distributed;
    p.grid.mapping_degree     = 3;
    p.spatial_discretization  = SpatialDiscretization::DG;
    p.IP_factor               = 1.0e0;

    // SOLVER
    p.solver                      = Solver::CG;
    p.solver_data.abs_tol         = 1.e-20;
    p.solver_data.rel_tol         = 1.e-10;
    p.solver_data.max_iter        = 1e4;
    p.compute_performance_metrics = true;
    p.preconditioner              = Preconditioner::Multigrid;
    p.multigrid_data.type         = MultigridType::cphMG;
    p.multigrid_data.p_sequence   = PSequenceType::Bisect;
    // MG smoother
    p.multigrid_data.smoother_data.smoother        = MultigridSmoother::Chebyshev;
    p.multigrid_data.smoother_data.iterations      = 5;
    p.multigrid_data.smoother_data.smoothing_range = 20;
    // MG coarse grid solver
    p.multigrid_data.coarse_problem.solver              = MultigridCoarseGridSolver::CG;
    p.multigrid_data.coarse_problem.preconditioner      = MultigridCoarseGridPreconditioner::AMG;
    p.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-3;
  }

  void
  create_grid() final
  {
    // create triangulation
    double const length = 0.5;
    double const left = -length, right = length;
    dealii::GridGenerator::subdivided_hyper_cube(*this->grid->triangulation, 1, left, right);

    this->grid->triangulation->refine_global(2 * this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<FunctionCached<1, dim>>>
      pair_cached;

    // these lines show exemplarily how the boundary descriptors are filled
    this->boundary_descriptor->dirichlet_cached_bc.insert(
      pair_cached(0, new FunctionCached<1, dim>()));
  }

  void
  set_field_functions() final
  {
    // these lines show exemplarily how the field functions are filled
    this->field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(dim));
    this->field_functions->right_hand_side.reset(new RightHandSide<dim>(dim));
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = this->output_parameters.write;
    pp_data.output_data.directory          = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename           = this->output_parameters.filename + "_second";
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = this->param.degree;

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }
};

template<int dim, int n_components, typename Number>
class Application : public ApplicationOversetGridsBase<dim, n_components, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
  {
    this->domain1 = std::make_shared<Domain1<dim, n_components, Number>>(input_file, comm);
    this->domain2 = std::make_shared<Domain2<dim, n_components, Number>>(input_file, comm);
  }
};

} // namespace Poisson

} // namespace ExaDG

#include <exadg/poisson/user_interface/implement_get_application_overset_grids.h>

#endif /* APPLICATIONS_POISSON_TEST_CASES_TEMPLATE_H_ */
