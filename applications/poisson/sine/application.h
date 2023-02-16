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

#ifndef APPLICATIONS_POISSON_TEST_CASES_SINE_H_
#define APPLICATIONS_POISSON_TEST_CASES_SINE_H_

#include <exadg/grid/deformed_cube_manifold.h>

namespace ExaDG
{
namespace Poisson
{
double const FREQUENCY            = 3.0 * dealii::numbers::PI;
bool const   USE_NEUMANN_BOUNDARY = true;

template<int dim>
class Solution : public dealii::Function<dim>
{
public:
  Solution(unsigned int const n_components = 1, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
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
class NeumannBoundary : public dealii::Function<dim>
{
public:
  NeumannBoundary(unsigned int const n_components = 1, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /*component*/) const
  {
    double result = 1.0;
    for(unsigned int d = 0; d < dim; ++d)
    {
      if(d == 0)
        result *= FREQUENCY * std::cos(FREQUENCY * p[d]);
      else
        result *= std::sin(FREQUENCY * p[d]);
    }

    return result;
  }
};


template<int dim>
class RightHandSide : public dealii::Function<dim>
{
public:
  RightHandSide(unsigned int const n_components = 1, double const time = 0.)
    : dealii::Function<dim>(n_components, time)
  {
  }

  double
  value(dealii::Point<dim> const & p, unsigned int const /* component */) const
  {
    double result = FREQUENCY * FREQUENCY * dim;
    for(unsigned int d = 0; d < dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

    return result;
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
    this->param.grid.element_type = ElementType::Hypercube; // Simplex;
    if(this->param.grid.element_type == ElementType::Simplex)
    {
      this->param.grid.triangulation_type = TriangulationType::FullyDistributed;
      this->param.grid.mapping_degree     = 2;
      this->param.grid.multigrid          = MultigridVariant::GlobalCoarsening;
    }
    else if(this->param.grid.element_type == ElementType::Hypercube)
    {
      this->param.grid.triangulation_type = TriangulationType::Distributed;
      this->param.grid.mapping_degree     = 3;
      this->param.grid.multigrid          = MultigridVariant::LocalSmoothing;
    }
    this->param.grid.file_name = this->grid_parameters.file_name;

    this->param.spatial_discretization = SpatialDiscretization::DG;
    this->param.IP_factor              = 1.0e0;

    // SOLVER
    this->param.solver                      = Solver::CG;
    this->param.solver_data.abs_tol         = 1.e-20;
    this->param.solver_data.rel_tol         = 1.e-10;
    this->param.solver_data.max_iter        = 1e4;
    this->param.compute_performance_metrics = true;
    this->param.preconditioner              = Preconditioner::Multigrid;
    this->param.multigrid_data.type         = MultigridType::cphMG;
    this->param.multigrid_data.p_sequence   = PSequenceType::Bisect;
    // MG smoother
    this->param.multigrid_data.smoother_data.smoother        = MultigridSmoother::Chebyshev;
    this->param.multigrid_data.smoother_data.iterations      = 5;
    this->param.multigrid_data.smoother_data.smoothing_range = 20;
    // MG coarse grid solver
    this->param.multigrid_data.coarse_problem.solver = MultigridCoarseGridSolver::CG;
    this->param.multigrid_data.coarse_problem.preconditioner =
      MultigridCoarseGridPreconditioner::AMG;
    this->param.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-3;
  }

  void
  create_grid() final
  {
    auto const lambda_create_triangulation =
      [&](dealii::Triangulation<dim, dim> & tria,
          unsigned int const                global_refinements,
          std::vector<unsigned int> const & vector_local_refinements) {
        double const length = 1.0;
        double const left = -length, right = length;
        // choose a coarse grid with at least 2^dim elements to obtain a non-trivial coarse grid
        // problem
        unsigned int const n_cells_1d =
          std::max((unsigned int)2, this->n_subdivisions_1d_hypercube);

        if(read_external_grid)
        {
          GridUtilities::read_external_triangulation<dim>(tria, this->param.grid);
        }
        else
        {
          if(this->param.grid.element_type == ElementType::Hypercube)
          {
            dealii::GridGenerator::subdivided_hyper_cube(tria, n_cells_1d, left, right);
          }
          else if(this->param.grid.element_type == ElementType::Simplex)
          {
            dealii::GridGenerator::subdivided_hyper_cube_with_simplices(tria,
                                                                        n_cells_1d,
                                                                        left,
                                                                        right);
          }
          else
          {
            AssertThrow(false, ExcNotImplemented());
          }
        }

        if(USE_NEUMANN_BOUNDARY)
        {
          for(auto cell : tria)
          {
            for(auto const & f : cell.face_indices())
            {
              if(std::fabs(cell.face(f)->center()(0) - right) < 1e-12)
              {
                cell.face(f)->set_boundary_id(1);
              }
            }
          }
        }

        if(vector_local_refinements.size() > 0)
          refine_local(tria, vector_local_refinements);

        if(global_refinements > 0)
          tria.refine_global(global_refinements);

        if(mesh_type == MeshType::Cartesian)
        {
          // do nothing
        }
        else if(mesh_type == MeshType::Curvilinear)
        {
          double const              deformation = 0.15;
          unsigned int const        frequency   = 2;
          DeformedCubeManifold<dim> manifold(left, right, deformation, frequency);
          tria.set_all_manifold_ids(1);
          tria.set_manifold(1, manifold);

          std::vector<bool> vertex_touched(tria.n_vertices(), false);

          for(auto cell : tria)
          {
            for(auto const & v : cell.vertex_indices())
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
      };

    GridUtilities::create_fine_and_coarse_triangulations<dim>(*this->grid,
                                                              this->param.grid,
                                                              this->param.involves_h_multigrid(),
                                                              lambda_create_triangulation);
  }


  void
  set_boundary_descriptor() final
  {
    typedef typename std::pair<dealii::types::boundary_id, std::shared_ptr<dealii::Function<dim>>>
      pair;

    this->boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>()));
    this->boundary_descriptor->neumann_bc.insert(pair(1, new NeumannBoundary<dim>()));
  }

  void
  set_field_functions() final
  {
    this->field_functions->initial_solution.reset(new dealii::Functions::ZeroFunction<dim>(1));
    this->field_functions->right_hand_side.reset(new RightHandSide<dim>());
  }

  std::shared_ptr<PostProcessorBase<dim, n_components, Number>>
  create_postprocessor() final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.time_control_data.is_active = this->output_parameters.write;
    pp_data.output_data.directory                   = this->output_parameters.directory + "vtu/";
    pp_data.output_data.filename                    = this->output_parameters.filename;
    pp_data.output_data.write_higher_order          = true;
    // Currently, we can not use higher order output with 3D simplices.
    if(this->param.grid.element_type == ElementType::Simplex and dim == 3)
      pp_data.output_data.write_higher_order = false;
    pp_data.output_data.degree = this->param.degree;

    pp_data.error_data.time_control_data.is_active = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>());
    pp_data.error_data.calculate_relative_errors = true;

    // these lines show exemplarily how to use the NormalFluxCalculator
    pp_data.normal_flux_data.evaluate = false;
    pp_data.normal_flux_data.boundary_ids.insert(0);
    pp_data.normal_flux_data.boundary_ids.insert(1);
    pp_data.normal_flux_data.directory = this->output_parameters.directory;
    pp_data.normal_flux_data.filename  = this->output_parameters.filename;

    std::shared_ptr<PostProcessorBase<dim, n_components, Number>> pp;
    pp.reset(new PostProcessor<dim, n_components, Number>(pp_data, this->mpi_comm));

    return pp;
  }

  bool const read_external_grid = false;

  std::string mesh_type_string = "Cartesian";
  MeshType    mesh_type        = MeshType::Cartesian;
};

} // namespace Poisson

} // namespace ExaDG

#include <exadg/poisson/user_interface/implement_get_application.h>

#endif
