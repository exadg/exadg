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
using namespace dealii;

template<int dim>
class CoefficientFunction : public Function<dim>
{
public:
  CoefficientFunction() : Function<dim>(1)
  {
  }

  double
  value(const Point<dim> & p, unsigned int const c = 0) const
  {
    (void)c;
    return value<double>(p);
  }

  Tensor<1, dim>
  gradient(const Point<dim> & p, unsigned int const c = 0) const
  {
    (void)c;
    (void)p;
    Tensor<1, dim> grad;

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
  static unsigned int const n_source_centers = 3;
  static const Point<dim>   source_centers[n_source_centers];
  static double const       width;
};


template<>
const Point<1> SolutionBase<1>::source_centers[SolutionBase<1>::n_source_centers] =
  {Point<1>(-1.0 / 3.0), Point<1>(0.0), Point<1>(+1.0 / 3.0)};

template<>
const Point<2> SolutionBase<2>::source_centers[SolutionBase<2>::n_source_centers] =
  {Point<2>(-0.5, +0.5), Point<2>(-0.5, -0.5), Point<2>(+0.5, -0.5)};

template<>
const Point<3> SolutionBase<3>::source_centers[SolutionBase<3>::n_source_centers] =
  {Point<3>(-0.5, +0.5, 0.25), Point<3>(-0.6, -0.5, -0.125), Point<3>(+0.5, -0.5, 0.5)};

template<int dim>
double const SolutionBase<dim>::width = 1. / 5.;

template<int dim>
class Solution : public Function<dim>, protected SolutionBase<dim>
{
public:
  Solution() : Function<dim>()
  {
  }

  double
  value(const Point<dim> & p, unsigned int const /*component*/ = 0) const
  {
    double return_value = 0;
    for(unsigned int i = 0; i < this->n_source_centers; ++i)
    {
      const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];
      return_value += std::exp(-x_minus_xi.norm_square() / (this->width * this->width));
    }

    return return_value / Utilities::fixed_power<dim>(std::sqrt(2. * numbers::PI) * this->width);
  }

  Tensor<1, dim>
  gradient(const Point<dim> & p, unsigned int const /*component*/ = 0) const
  {
    Tensor<1, dim> return_value;

    for(unsigned int i = 0; i < this->n_source_centers; ++i)
    {
      const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

      return_value +=
        (-2 / (this->width * this->width) *
         std::exp(-x_minus_xi.norm_square() / (this->width * this->width)) * x_minus_xi);
    }

    return return_value / Utilities::fixed_power<dim>(std::sqrt(2 * numbers::PI) * this->width);
  }
};

template<int dim>
class RightHandSide : public Function<dim>, protected SolutionBase<dim>
{
public:
  RightHandSide() : Function<dim>()
  {
  }

  double
  value(const Point<dim> & p, unsigned int const /*component*/ = 0) const
  {
    CoefficientFunction<dim> coefficient;
    double const             coef         = coefficient.value(p);
    const Tensor<1, dim>     coef_grad    = coefficient.gradient(p);
    double                   return_value = 0;
    for(unsigned int i = 0; i < this->n_source_centers; ++i)
    {
      const Tensor<1, dim> x_minus_xi = p - this->source_centers[i];

      return_value += ((2 * dim * coef + 2 * (coef_grad)*x_minus_xi -
                        4 * coef * x_minus_xi.norm_square() / (this->width * this->width)) /
                       (this->width * this->width) *
                       std::exp(-x_minus_xi.norm_square() / (this->width * this->width)));
    }

    return return_value / Utilities::fixed_power<dim>(std::sqrt(2 * numbers::PI) * this->width);
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
  else AssertThrow(false, ExcMessage("Not implemented."));
  // clang-format on
}

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
  typedef typename ApplicationBase<dim, Number>::PeriodicFaces PeriodicFaces;

  Application(std::string input_file) : ApplicationBase<dim, Number>(input_file)
  {
    // parse application-specific parameters
    ParameterHandler prm;
    add_parameters(prm);
    prm.parse_input(input_file, "", true, true);

    string_to_enum(mesh_type, mesh_type_string);
  }

  void
  add_parameters(ParameterHandler & prm)
  {
    ApplicationBase<dim, Number>::add_parameters(prm);

    // clang-format off
    prm.enter_subsection("Application");
      prm.add_parameter("MeshType", mesh_type_string, "Type of mesh (Cartesian versus curvilinear).", Patterns::Selection("Cartesian|Curvilinear"));
      prm.add_parameter("GlobalCoarsening", global_coarsening, "Use Global Coarsening", Patterns::Bool());
    prm.leave_subsection();
    // clang-format on
  }

  std::string mesh_type_string = "Cartesian";
  MeshType    mesh_type        = MeshType::Cartesian;

  bool global_coarsening = false;

  void
  set_input_parameters(InputParameters & param)
  {
    // MATHEMATICAL MODEL
    param.right_hand_side = true;

    // SPATIAL DISCRETIZATION
    param.triangulation_type     = TriangulationType::Distributed;
    param.mapping                = MappingType::Isoparametric;
    param.spatial_discretization = SpatialDiscretization::DG;
    param.IP_factor              = 1.0e0;

    // SOLVER
    param.solver                               = Poisson::Solver::CG;
    param.solver_data.abs_tol                  = 1.e-20;
    param.solver_data.rel_tol                  = 1.e-10;
    param.solver_data.max_iter                 = 1e4;
    param.compute_performance_metrics          = true;
    param.preconditioner                       = Preconditioner::Multigrid;
    param.multigrid_data.type                  = MultigridType::cphMG;
    param.multigrid_data.p_sequence            = PSequenceType::Bisect;
    param.multigrid_data.use_global_coarsening = global_coarsening;
    // MG smoother
    param.multigrid_data.smoother_data.smoother   = MultigridSmoother::Chebyshev;
    param.multigrid_data.smoother_data.iterations = 5;
    // MG coarse grid solver
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
    param.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-6;
  }

  void
  create_grid(std::shared_ptr<parallel::TriangulationBase<dim>> triangulation,
              PeriodicFaces &                                   periodic_faces,
              unsigned int const                                n_refine_space,
              std::shared_ptr<Mapping<dim>> &                   mapping,
              unsigned int const                                mapping_degree)
  {
    (void)periodic_faces;

    double const length = 1.0;
    double const left = -length, right = length;
    GridGenerator::subdivided_hyper_cube(*triangulation,
                                         this->n_subdivisions_1d_hypercube,
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
      triangulation->set_all_manifold_ids(1);
      triangulation->set_manifold(1, manifold);

      std::vector<bool> vertex_touched(triangulation->n_vertices(), false);

      for(typename Triangulation<dim>::cell_iterator cell = triangulation->begin();
          cell != triangulation->end();
          ++cell)
      {
        for(unsigned int v = 0; v < GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          if(vertex_touched[cell->vertex_index(v)] == false)
          {
            Point<dim> & vertex                   = cell->vertex(v);
            Point<dim>   new_point                = manifold.push_forward(vertex);
            vertex                                = new_point;
            vertex_touched[cell->vertex_index(v)] = true;
          }
        }
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    triangulation->refine_global(n_refine_space);

    mapping.reset(new MappingQGeneric<dim>(mapping_degree));
  }

  void set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor)
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>()));

    //  boundary_descriptor->neumann_bc.insert(pair(1, new Functions::ZeroFunction<dim>(1)));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions)
  {
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new RightHandSide<dim>());
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  construct_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm)
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.output_folder      = this->output_directory + "vtu/";
    pp_data.output_data.output_name        = this->output_name;
    pp_data.output_data.write_higher_order = true;
    pp_data.output_data.degree             = degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>());

    std::shared_ptr<PostProcessorBase<dim, Number>> pp;
    pp.reset(new PostProcessor<dim, Number>(pp_data, mpi_comm));

    return pp;
  }
};

} // namespace Poisson

template<int dim, typename Number>
std::shared_ptr<Poisson::ApplicationBase<dim, Number>>
get_application(std::string input_file)
{
  return std::make_shared<Poisson::Application<dim, Number>>(input_file);
}

} // namespace ExaDG

#endif
