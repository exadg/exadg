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

#ifndef APPLICATIONS_POISSON_TEST_CASES_SINE_SIMPLEX_H_
#define APPLICATIONS_POISSON_TEST_CASES_SINE_SIMPLEX_H_

#include <deal.II/fe/fe_simplex_p.h>
#include <deal.II/fe/mapping_fe.h>

#include <exadg/grid/deformed_cube_manifold.h>

namespace ExaDG
{
namespace Poisson
{
using namespace dealii;

double const FREQUENCY            = 3.0 * numbers::PI;
bool const   USE_NEUMANN_BOUNDARY = true;

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
    double result = 1.0;
    for(unsigned int d = 0; d < dim; ++d)
      result *= std::sin(FREQUENCY * p[d]);

    return result;
  }
};

template<int dim>
class NeumannBoundary : public Function<dim>
{
public:
  NeumannBoundary(unsigned int const n_components = 1, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const /*component*/) const
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
class RightHandSide : public Function<dim>
{
public:
  RightHandSide(unsigned int const n_components = 1, double const time = 0.)
    : Function<dim>(n_components, time)
  {
  }

  double
  value(Point<dim> const & p, unsigned int const /* component */) const
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
  else AssertThrow(false, ExcMessage("Not implemented."));
  // clang-format on
}

template<int dim, typename Number>
class Application : public ApplicationBase<dim, Number>
{
public:
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
    prm.leave_subsection();
    // clang-format on
  }

  std::string mesh_type_string = "Cartesian";
  MeshType    mesh_type        = MeshType::Cartesian;

  void
  set_input_parameters(InputParameters & param) final
  {
    // MATHEMATICAL MODEL
    param.right_hand_side = true;

    // SPATIAL DISCRETIZATION
    param.triangulation_type     = TriangulationType::Serial;
    param.mapping                = MappingType::Affine;
    param.spatial_discretization = SpatialDiscretization::DG;
    param.IP_factor              = 1.0e0;

    // SOLVER
    param.solver                      = Solver::CG;
    param.solver_data.abs_tol         = 1.e-20;
    param.solver_data.rel_tol         = 1.e-4;
    param.solver_data.max_iter        = 1e4;
    param.compute_performance_metrics = true;
    param.preconditioner              = Preconditioner::PointJacobi;
    param.multigrid_data.type         = MultigridType::cphMG;
    param.multigrid_data.p_sequence   = PSequenceType::Bisect;
    // MG smoother
    param.multigrid_data.smoother_data.smoother        = MultigridSmoother::Chebyshev;
    param.multigrid_data.smoother_data.iterations      = 5;
    param.multigrid_data.smoother_data.smoothing_range = 20;
    // MG coarse grid solver
    param.multigrid_data.coarse_problem.solver         = MultigridCoarseGridSolver::CG;
    param.multigrid_data.coarse_problem.preconditioner = MultigridCoarseGridPreconditioner::AMG;
    param.multigrid_data.coarse_problem.solver_data.rel_tol = 1.e-3;
  }

  std::shared_ptr<Grid<dim, Number>>
  create_grid(GridData const & data, MPI_Comm const & mpi_comm) final
  {
    std::shared_ptr<Grid<dim, Number>> grid = std::make_shared<Grid<dim, Number>>(data, mpi_comm);

    double const length = 1.0;
    double const left = -length, right = length;
    // choose a coarse grid with at least 2^dim elements to obtain a non-trivial coarse grid problem
    unsigned int n_cells_1d = std::max((unsigned int)2, this->n_subdivisions_1d_hypercube);
    GridGenerator::subdivided_hyper_cube_with_simplices(*grid->triangulation,
                                                        n_cells_1d,
                                                        left,
                                                        right);

    grid->mapping = std::make_shared<MappingFE<dim>>(FE_SimplexP<dim>(data.mapping_degree));

    if(mesh_type == MeshType::Cartesian)
    {
      // do nothing
    }
    else if(mesh_type == MeshType::Curvilinear)
    {
      AssertThrow(false, ExcNotImplemented());
    }
    else
    {
      AssertThrow(false, ExcMessage("not implemented."));
    }

    if(USE_NEUMANN_BOUNDARY)
    {
      for(const auto & cell : *grid->triangulation)
      {
        for(const auto f : cell.face_indices())
        {
          if(std::fabs(cell.face(f)->center()(0) - right) < 1e-12)
          {
            cell.face(f)->set_boundary_id(1);
          }
        }
      }
    }

    grid->triangulation->refine_global(data.n_refine_global);

    return grid;
  }


  void
    set_boundary_conditions(std::shared_ptr<BoundaryDescriptor<0, dim>> boundary_descriptor) final
  {
    typedef typename std::pair<types::boundary_id, std::shared_ptr<Function<dim>>> pair;

    boundary_descriptor->dirichlet_bc.insert(pair(0, new Solution<dim>()));
    boundary_descriptor->neumann_bc.insert(pair(1, new NeumannBoundary<dim>()));
  }

  void
  set_field_functions(std::shared_ptr<FieldFunctions<dim>> field_functions) final
  {
    field_functions->initial_solution.reset(new Functions::ZeroFunction<dim>(1));
    field_functions->right_hand_side.reset(new RightHandSide<dim>());
  }

  std::shared_ptr<PostProcessorBase<dim, Number>>
  create_postprocessor(unsigned int const degree, MPI_Comm const & mpi_comm) final
  {
    PostProcessorData<dim> pp_data;
    pp_data.output_data.write_output       = this->write_output;
    pp_data.output_data.directory          = this->output_directory + "vtu/";
    pp_data.output_data.filename           = this->output_name;
    pp_data.output_data.write_higher_order = false;
    pp_data.output_data.degree             = degree;

    pp_data.error_data.analytical_solution_available = true;
    pp_data.error_data.analytical_solution.reset(new Solution<dim>());
    pp_data.error_data.calculate_relative_errors = true;

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
