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

#ifndef APPLICATIONS_POISSON_TEST_CASES_GAUSSIAN_H_
#define APPLICATIONS_POISSON_TEST_CASES_GAUSSIAN_H_

// ExaDG
#include <exadg/grid/deformed_cube_manifold.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim>
class CoefficientFunction : public dealii::Function<dim>
{
public:
  CoefficientFunction() : dealii::Function<dim>(1)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const c = 0) const
  {
    (void)c;
    return value<double>(p);
  }

  dealii::Tensor<1, dim>
  gradient(dealii::Point<dim> const & p, unsigned int const c = 0) const
  {
    (void)c;
    (void)p;
    dealii::Tensor<1, dim> grad;

    return grad;
  }

  template<typename Number>
  Number
  value(const dealii::Point<dim, Number> & p) const
  {
    (void)p;
    Number value;
    value = 1;

    return value;
  }
};

template<int dim>
class SolutionBase
{
protected:
  static unsigned int const       n_source_centers = 3;
  static dealii::Point<dim> const source_centers[n_source_centers];
  static double const             width;
};


template<>
const dealii::Point<1> SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] =
  {dealii::Point<1>(-1.0 / 3.0), dealii::Point<1>(0.0), dealii::Point<1>(+1.0 / 3.0)};

template<>
dealii::Point<2> const SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] =
  {dealii::Point<2>(-0.5, +0.5), dealii::Point<2>(-0.5, -0.5), dealii::Point<2>(+0.5, -0.5)};

template<>
dealii::Point<3> const SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers] = {
  dealii::Point<3>(-0.5, +0.5, 0.25),
  dealii::Point<3>(-0.6, -0.5, -0.125),
  dealii::Point<3>(+0.5, -0.5, 0.5)};

template<int dim>
double const SolutionBase<dim>::width = 1. / 5.;

template<int dim>
class Solution : public dealii::Function<dim>, protected SolutionBase<dim>
{
public:
  Solution() : dealii::Function<dim>()
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/ = 0) const
  {
    double return_value = 0;
    for(unsigned int i = 0; i < this->n_source_centers; ++i)
    {
      const dealii::Tensor<1, dim> x_minus_xi = p - this->source_centers[i];
      return_value += std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
    }

    return return_value /
           dealii::Utilities::fixed_power<dim>(std::sqrt(2. * dealii::numbers::PI) * this->width);
  }

  dealii::Tensor<1, dim>
  gradient(dealii::Point<dim> const & p, unsigned int const /*component*/ = 0) const
  {
    dealii::Tensor<1, dim> return_value;

    for(unsigned int i = 0; i < this->n_source_centers; ++i)
    {
      const dealii::Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

      return_value +=
        (-2 / (this->width * this->width) *
         std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) * x_minus_xi);
    }

    return return_value /
           dealii::Utilities::fixed_power<dim>(std::sqrt(2 * dealii::numbers::PI) * this->width);
  }
};

template<int dim>
class RightHandSide : public dealii::Function<dim>, protected SolutionBase<dim>
{
public:
  RightHandSide() : dealii::Function<dim>()
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/ = 0) const
  {
    CoefficientFunction<dim>     coefficient;
    double const                 coef         = coefficient.value(p);
    const dealii::Tensor<1, dim> coef_grad    = coefficient.gradient(p);
    double                       return_value = 0;
    for(unsigned int i = 0; i < this->n_source_centers; ++i)
    {
      const dealii::Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

      return_value += ((2 * dim * coef + 2 * (coef_grad)*x_minus_xi -
                        4 * coef * x_minus_xi.norm_square() / (this->width * this->width)) /
                       (this->width * this->width) *
                       std::exp(-x_minus_xi.norm_square() / (this->width * this->width)));
    }

    return return_value /
           dealii::Utilities::fixed_power<dim>(std::sqrt(2 * dealii::numbers::PI) * this->width);
  }
};

enum class MeshType
{
  Cartesian,
  Curvilinear
};

void
string_to_enum(MeshType & enum_type, std::string const & string_type)
{
  // clang-format off
  if     (string_type == "Cartesian")   enum_type = MeshType::Cartesian;
  else if(string_type == "Curvilinear") enum_type = MeshType::Curvilinear;
  else AssertThrow(false, dealii::ExcMessage("Not implemented."));
  // clang-format on
}

template<int dim, int n_components, typename Number>
class Application : public ApplicationBase<dim, n_components, Number>
{
public:
  Application(std::string input_file, MPI_Comm const & comm)
    : ApplicationBase<dim, n_components, Number>(input_file, comm)
  {
  }

  void
  add_parameters(dealii::ParameterHandler & prm)
  {
    ApplicationBase<dim, n_components, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("MeshType", mesh_type_string, "Type of mesh (Cartesian versus curvilinear).", dealii::Patterns::Selection("Cartesian|Curvilinear"));
      prm.add_parameter("GlobalCoarsening", global_coarsening, "Use Global Coarsening", dealii::Patterns::Bool());
    prm.leave_subsection();
    // clang-format on
  }

private:
  void
  parse_parameters() final
  {
    ApplicationBase<dim, n_components, Number>::parse_parameters();

    string_to_enum(mesh_type, mesh_type_string);
  }

  void
  set_parameters() final
  {
    // MATHEMATICAL MODEL
    this->param.right_hand_side = true;

    // SPATIAL DISCRETIZATION
    this->param.grid.triangulation_type = TriangulationType::Distributed;
    this->param.grid.mapping_degree     = this->param.degree;
    this->param.spatial_discretization  = SpatialDiscretization::DG;
    this->param.IP_factor               = 1.0e0;

    // SOLVER
    this->param.solver                               = Poisson::Solver::CG;
    this->param.solver_data.abs_tol                  = 1.e-20;
    this->param.solver_data.rel_tol                  = 1.e-10;
    this->param.solver_data.max_iter                 = 1e4;
    this->param.compute_performance_metrics          = true;
    this->param.preconditioner                       = Preconditioner::Multigrid;
    this->param.multigrid_data.type                  = MultigridType::cphMG;
    this->param.multigrid_data.p_sequence            = PSequenceType::Bisect;
    this->param.multigrid_data.use_global_coarsening = global_coarsening;
    // MG smoother
    this->param.multigrid_data.smoother_data.smoother   = MultigridSmoother::Chebyshev;
    this->param.multigrid_data.smoother_data.iterations = 5;
    // MG coarse grid solver
    this->param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    this->param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    this->param.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-6;
  }

  void
  create_grid() final
  {
    double const length = 1.0;
    double const left = -length, right = length;
    dealii::GridGenerator::subdivided_hyper_cube(*this->grid->triangulation,
                                                 this->param.grid.n_subdivisions_1d_hypercube,
                                                 left,
                                                 right);

    if(mesh_type == MeshType::Cartesian)
    {
      // do nothing
    }
    else if(mesh_type == MeshType::Curvilinear)
    {
      double const              deformation = 0.1;
      unsigned int const        frequency   = 2;
      DeformedCubeManifold<dim> manifold(left, right, deformation, frequency);
      this->grid->triangulation->set_all_manifold_ids(1);
      this->grid->triangulation->set_manifold(1, manifold);

      std::vector<bool> vertex_touched(this->grid->triangulation->n_vertices(), false);

      for(auto cell : *this->grid->triangulation)
      {
        for(auto const & v : cell->vertex_indices())
        {
          if(vertex_touched[cell.vertex_index(v)] == false)
          {
            dealii::Point<dim> & vertex          = cell.vertex(v);
            dealii::Point<dim>   new_point       = manifold.push_forward(vertex);
            vertex                               = new_point;
            vertex_touched[cell.vertex_index(v)] = true;
          }
        }
      }
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("not implemented."));
    }

    this->grid->triangulation->refine_global(this->param.grid.n_refine_global);
  }

  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    this->boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>()));

    // this->boundary_descriptor->neumann_bc.insert(pair(1, new
    // dealii::Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new RightHandSide<dim>());
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

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>());

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  std::string mesh_type_string = "Cartesian";
  MeshType    mesh_type        = MeshType::Cartesian;

  bool global_coarsening = false;
};

} // namespace Poisson

} // namespace ExaDG

#include <exadg/poisson/user_interface/implement_get_application.h>

#endif
