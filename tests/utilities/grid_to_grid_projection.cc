/*  ______________________________________________________________________
 *
 *  ExaDG - High-Order Discontinuous Galerkin for the Exa-Scale
 *
 *  Copyright (C) 2025 by the ExaDG authors
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

// deal.II
#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>
#include <deal.II/base/index_set.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/fe_point_evaluation.h>
#include <deal.II/numerics/vector_tools.h>

// ExaDG
#include <exadg/grid/grid_data.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/solution_projection_between_triangulations.h>
#include <exadg/postprocessor/error_calculation.h>
#include <exadg/postprocessor/write_output.h>
#include <exadg/utilities/numbers.h>
#include <exadg/operators/solution_projection_between_triangulations.cpp> // temporary

using namespace ExaDG;

template<int dim, int n_components>
class SampleFunction : public dealii::Function<dim>
{
public:
  SampleFunction(double scale) : dealii::Function<dim>(n_components), scale(scale)
  {
  }

  virtual double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const override;

private:
  double scale;
};

template<int dim, int n_components>
double
SampleFunction<dim, n_components>::value(dealii::Point<dim> const & p,
                                         unsigned int const         component) const
{
  if(component == 0)
  {
    return scale * (1.1 + std::sin(0.75 * p[0]));
  }
  else if(component == 1)
  {
    return scale * (1.1 + std::sin(1.0 * p[1]));
  }
  else if(component == 2)
  {
    if constexpr(dim == 3)
    {
      return scale * (1.1 + std::sin(0.5 * p[2]));
    }
  }

  return 0.0;
}

template<int dim, int n_components>
class GridToGridProjector
{
  using VectorType = dealii::LinearAlgebra::distributed::Vector<double>;

public:
  GridToGridProjector(bool const        vector_target_is_RT_else_DGQ,
                      bool const        vector_source_is_RT_else_DGQ,
                      ElementType const element_type);

  void
  run();

  void
  setup();

private:
  void
  project();

  void
  check();

  MPI_Comm                   mpi_comm;
  dealii::ConditionalOStream pcout;

  bool const        vector_target_is_RT_else_DGQ;
  bool const        vector_source_is_RT_else_DGQ;
  ElementType const element_type;

  std::shared_ptr<dealii::FiniteElement<dim>> fe_target;
  std::shared_ptr<dealii::FiniteElement<dim>> fe_source;

  std::shared_ptr<dealii::Mapping<dim> const> mapping;

  dealii::parallel::fullydistributed::Triangulation<dim> tria_source;
  dealii::parallel::fullydistributed::Triangulation<dim> tria_target;

  dealii::DoFHandler<dim> dof_handler_target_continuous;
  dealii::DoFHandler<dim> dof_handler_source_continuous;
  dealii::DoFHandler<dim> dof_handler_target;
  dealii::DoFHandler<dim> dof_handler_source;

  std::vector<VectorType> vectors_target_continuous;
  std::vector<VectorType> vectors_source_continuous;
  std::vector<VectorType> vectors_target;
  std::vector<VectorType> vectors_source;

  static unsigned int constexpr fe_degree_source  = 3;
  static unsigned int constexpr fe_degree_target  = 2;
  static unsigned int constexpr fe_degree_mapping = 2;

  static bool constexpr export_vectors = false;
};

template<int dim, int n_components>
GridToGridProjector<dim, n_components>::GridToGridProjector(bool const vector_target_is_RT_else_DGQ,
                                                            bool const vector_source_is_RT_else_DGQ,
                                                            ElementType const element_type)
  : mpi_comm(MPI_COMM_WORLD),
    pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)),
    vector_target_is_RT_else_DGQ(vector_target_is_RT_else_DGQ),
    vector_source_is_RT_else_DGQ(vector_source_is_RT_else_DGQ),
    element_type(element_type),
    tria_source(mpi_comm),
    tria_target(mpi_comm)
{
}

template<int dim, int n_components>
void
GridToGridProjector<dim, n_components>::setup()
{
  // Initialize finite elements
  pcout << "  element_type == ElementType::Hypercube : " << (element_type == ElementType::Hypercube)
        << "\n"
        << "  dim                          = " << dim << "\n"
        << "  n_components                 = " << n_components << "\n"
        << "  vector_target_is_RT_else_DGQ = " << vector_target_is_RT_else_DGQ << "\n"
        << "  vector_source_is_RT_else_DGQ = " << vector_source_is_RT_else_DGQ << "\n";

  bool const raviart_thomas_used = vector_source_is_RT_else_DGQ or vector_target_is_RT_else_DGQ;
  AssertThrow(element_type == ElementType::Hypercube or not raviart_thomas_used,
              dealii::ExcMessage("Cannot use Raviart-Thomas elements on non-hypercube cells."));
  AssertThrow(not raviart_thomas_used or n_components == dim,
              dealii::ExcMessage(
                "Raviart-Thomas elements can only be used for dim == n_components."));

  if(n_components == 1 or not vector_target_is_RT_else_DGQ)
  {
    fe_target = create_finite_element<dim>(element_type, true, n_components, fe_degree_target);
  }
  else
  {
    fe_target = std::make_shared<dealii::FE_RaviartThomasNodal<dim>>(fe_degree_target - 1);
  }

  if(n_components == 1 or not vector_source_is_RT_else_DGQ)
  {
    fe_source = create_finite_element<dim>(element_type, true, n_components, fe_degree_source);
  }
  else
  {
    fe_source = std::make_shared<dealii::FE_RaviartThomasNodal<dim>>(fe_degree_source - 1);
  }

  // Create grids (dim == 2 and dim == 3) with a manifold attached and no merged patches for
  // Raviart-Thomas. The target grid is inscribed in the source grid, but the inner radii match to
  // test curved boundaries.
  double const radius            = 0.5;
  double const radius_shift      = radius * 0.1;
  double const halve_edge_length = 0.75;
  double const length            = 1.5;
  {
    auto const construction_data = dealii::TriangulationDescription::Utilities::
      create_description_from_triangulation_in_groups<dim, dim>(
        [&](dealii::Triangulation<dim> & tria_serial) {
          if(element_type == ElementType::Hypercube)
          {
            dealii::GridGenerator::hyper_cube_with_cylindrical_hole(
              tria_serial, radius + radius_shift, halve_edge_length, length, 2, true);
            if constexpr(dim == 2)
            {
              dealii::PolarManifold<dim> const manifold;
              tria_source.set_manifold(0, manifold);
              tria_target.set_manifold(0, manifold);
            }
            else if constexpr(dim == 3)
            {
              dealii::CylindricalManifold<dim> const manifold(2 /* z-axis */);
              tria_source.set_manifold(0, manifold);
              tria_target.set_manifold(0, manifold);
            }
            tria_serial.refine_global(2);
          }
          else
          {
            dealii::Triangulation<dim> tria_hypercube;
            dealii::GridGenerator::hyper_cube_with_cylindrical_hole(
              tria_hypercube, radius + radius_shift, halve_edge_length, length, 2, true);
            tria_hypercube.refine_global(2);
            dealii::GridGenerator::convert_hypercube_to_simplex_mesh(tria_hypercube, tria_serial);
          }
        },
        [](dealii::Triangulation<dim> & tria_serial,
           MPI_Comm const               mpi_communicator,
           unsigned int const /* group_size */) {
          dealii::GridTools::partition_triangulation(
            dealii::Utilities::MPI::n_mpi_processes(mpi_communicator), tria_serial);
        },
        mpi_comm,
        1);

    tria_target.create_triangulation(construction_data);
  }
  {
    auto const construction_data = dealii::TriangulationDescription::Utilities::
      create_description_from_triangulation_in_groups<dim, dim>(
        [&](dealii::Triangulation<dim> & tria_serial) {
          if(element_type == ElementType::Hypercube)
          {
            dealii::GridGenerator::hyper_cube_with_cylindrical_hole(
              tria_serial, radius, halve_edge_length * 1.2, length, 2, true);
            if constexpr(dim == 2)
            {
              dealii::PolarManifold<dim> const manifold;
              tria_source.set_manifold(0, manifold);
              tria_target.set_manifold(0, manifold);
            }
            else if constexpr(dim == 3)
            {
              dealii::CylindricalManifold<dim> const manifold(2 /* z-axis */);
              tria_source.set_manifold(0, manifold);
              tria_target.set_manifold(0, manifold);
            }
            tria_serial.refine_global(2);
          }
          else
          {
            dealii::Triangulation<dim> tria_hypercube;
            dealii::GridGenerator::hyper_cube_with_cylindrical_hole(
              tria_hypercube, radius, halve_edge_length * 1.2, length, 2, true);
            tria_hypercube.refine_global(2);
            dealii::GridGenerator::convert_hypercube_to_simplex_mesh(tria_hypercube, tria_serial);
          }
        },
        [](dealii::Triangulation<dim> & tria_serial,
           MPI_Comm const               mpi_communicator,
           unsigned int const /* group_size */) {
          dealii::GridTools::partition_triangulation(
            dealii::Utilities::MPI::n_mpi_processes(mpi_communicator), tria_serial);
        },
        mpi_comm,
        1);

    tria_source.create_triangulation(construction_data);
  }

  // Construction via description necessitates reconstruction of the manifold.
  if constexpr(dim == 2)
  {
    dealii::PolarManifold<dim> const manifold;
    tria_source.set_manifold(0, manifold);
    tria_target.set_manifold(0, manifold);
  }
  else if constexpr(dim == 3)
  {
    dealii::CylindricalManifold<dim> const manifold(2 /* z-axis */);
    tria_source.set_manifold(0, manifold);
    tria_target.set_manifold(0, manifold);
  }

  // Create identity mapping depending on cell type.
  if(element_type == ElementType::Hypercube)
  {
    mapping = std::make_shared<dealii::MappingFE<dim> const>(dealii::FE_Q<dim>(fe_degree_mapping));
  }
  else
  {
    mapping =
      std::make_shared<dealii::MappingFE<dim> const>(dealii::FE_SimplexP<dim>(fe_degree_mapping));
  }

  // Distribute DoFs.
  dof_handler_target.reinit(tria_target);
  dof_handler_target.distribute_dofs(*fe_target);
  dof_handler_source.reinit(tria_source);
  dof_handler_source.distribute_dofs(*fe_source);

  std::shared_ptr<dealii::FiniteElement<dim>> fe_target_continuous =
    create_finite_element<dim>(element_type, false, n_components, fe_degree_target);
  dof_handler_target_continuous.reinit(tria_target);
  dof_handler_target_continuous.distribute_dofs(*fe_target_continuous);

  std::shared_ptr<dealii::FiniteElement<dim>> fe_source_continuous =
    create_finite_element<dim>(element_type, false, n_components, fe_degree_source);
  dof_handler_source_continuous.reinit(tria_source);
  dof_handler_source_continuous.distribute_dofs(*fe_source_continuous);

  pcout << "  Number of degrees of freedom:\n"
        << "    TARGET: " << dof_handler_target.n_dofs() << "\n"
        << "            " << dof_handler_target_continuous.n_dofs() << " (continuous)\n"
        << "    SOURCE: " << dof_handler_source.n_dofs() << "\n"
        << "            " << dof_handler_source_continuous.n_dofs() << " (continuous)\n";

  // Setup/fill vectors.
  {
    dealii::IndexSet const rel_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_target);
    VectorType tmp(dof_handler_target.locally_owned_dofs(), rel_dofs, mpi_comm);
    vectors_target.push_back(tmp);
    vectors_target.push_back(tmp);
    vectors_target.push_back(tmp);
  }
  {
    dealii::IndexSet const rel_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_target_continuous);
    VectorType tmp(dof_handler_target_continuous.locally_owned_dofs(), rel_dofs, mpi_comm);
    vectors_target_continuous.push_back(tmp);
    vectors_target_continuous.push_back(tmp);
  }
  {
    dealii::IndexSet const rel_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_source);
    VectorType tmp(dof_handler_source.locally_owned_dofs(), rel_dofs, mpi_comm);
    dealii::VectorTools::interpolate(dof_handler_source,
                                     SampleFunction<dim, n_components>(1.0 /* scale */),
                                     tmp);
    vectors_source.push_back(tmp);
    tmp *= 1.0e3;
    vectors_source.push_back(tmp);
  }
  {
    dealii::IndexSet const rel_dofs =
      dealii::DoFTools::extract_locally_relevant_dofs(dof_handler_source_continuous);
    VectorType tmp(dof_handler_source_continuous.locally_owned_dofs(), rel_dofs, mpi_comm);
    dealii::VectorTools::interpolate(dof_handler_source_continuous,
                                     SampleFunction<dim, n_components>(1.0 /* scale */),
                                     tmp);
    vectors_source_continuous.push_back(tmp);
    tmp *= 1.0e2;
    vectors_source_continuous.push_back(tmp);
    tmp *= 1.0e-3;
    vectors_source_continuous.push_back(tmp);
  }
}

template<int dim, int n_components>
void
GridToGridProjector<dim, n_components>::project()
{
  // Output vectors in source.
  if constexpr(export_vectors)
  {
    OutputDataBase output_data;
    output_data.directory = "./";
    output_data.filename  = "source_vectors";
    output_data.degree    = element_type == ElementType::Hypercube ? 3 : 1;

    VectorWriter<dim, VectorType::value_type> vector_writer(output_data,
                                                            0 /* output_counter */,
                                                            mpi_comm);
    std::vector<bool>                         component_is_part_of_vector(n_components, true);
    if(n_components == 1)
    {
      component_is_part_of_vector[0] = false;
    }

    std::string elem_str = element_type == ElementType::Hypercube ? "hypercube" : "simplex";
    for(unsigned int i = 0; i < vectors_source.size(); ++i)
    {
      std::vector<std::string> component_names(n_components,
                                               "source_" + std::to_string(i) + "_" + elem_str);
      vector_writer.add_data_vector(vectors_source[i],
                                    dof_handler_source,
                                    component_names,
                                    component_is_part_of_vector);
    }
    for(unsigned int i = 0; i < vectors_source_continuous.size(); ++i)
    {
      std::vector<std::string> component_names(n_components,
                                               "source_continuous" + std::to_string(i) + "_" +
                                                 elem_str);
      vector_writer.add_data_vector(vectors_source_continuous[i],
                                    dof_handler_source_continuous,
                                    component_names,
                                    component_is_part_of_vector);
    }
  }

  // Project source vectors onto target grids:
  // vectors_source_continupus -> vectors_target (3 vectors)
  // vectors_source -> vectors_target_continuous (2 vectors)
  std::vector<dealii::DoFHandler<dim> const *> source_dof_handlers(
    {&dof_handler_source_continuous, &dof_handler_source});
  std::vector<dealii::DoFHandler<dim> const *> target_dof_handlers(
    {&dof_handler_target, &dof_handler_target_continuous});

  std::vector<std::vector<VectorType *>> source_vectors_per_dof_handler;
  {
    std::vector<VectorType *> tmp;
    for(unsigned int i = 0; i < vectors_source_continuous.size(); ++i)
    {
      tmp.push_back(&vectors_source_continuous[i]);
    }
    source_vectors_per_dof_handler.push_back(tmp);
  }
  {
    std::vector<VectorType *> tmp;
    for(unsigned int i = 0; i < vectors_source.size(); ++i)
    {
      tmp.push_back(&vectors_source[i]);
    }
    source_vectors_per_dof_handler.push_back(tmp);
  }

  std::vector<std::vector<VectorType *>> target_vectors_per_dof_handler;
  {
    std::vector<VectorType *> tmp;
    for(unsigned int i = 0; i < vectors_target.size(); ++i)
    {
      tmp.push_back(&vectors_target[i]);
    }
    target_vectors_per_dof_handler.push_back(tmp);
  }
  {
    std::vector<VectorType *> tmp;
    for(unsigned int i = 0; i < vectors_target_continuous.size(); ++i)
    {
      tmp.push_back(&vectors_target_continuous[i]);
    }
    target_vectors_per_dof_handler.push_back(tmp);
  }

  GridToGridProjection::GridToGridProjectionData<dim> data;
  data.solver_data.max_iter = 1000;
  data.solver_data.abs_tol  = 1.0e-12;
  data.solver_data.rel_tol  = 1.0e-8;

  data.rpe_data.tolerance              = 1e-6;
  data.rpe_data.enforce_unique_mapping = false;
  data.rpe_data.rtree_level            = 0;

  data.additional_quadrature_points = 1;

  data.is_test = true;

  bool all_dg = true;
  for(unsigned int i = 0; i < target_dof_handlers.size(); ++i)
  {
    bool is_dg = target_dof_handlers[i]->get_fe().dofs_per_vertex == 0;
    if(not is_dg)
    {
      all_dg = false;
    }
  }

  data.preconditioner = PreconditionerMass::PointJacobi;
  if(all_dg)
  {
    if(element_type == ElementType::Hypercube)
    {
      data.inverse_mass_type = InverseMassType::MatrixfreeOperator;
    }
    else
    {
      data.inverse_mass_type = InverseMassType::ElementwiseKrylovSolver;
    }
  }
  else
  {
    data.inverse_mass_type = InverseMassType::GlobalKrylovSolver;
  }

  GridToGridProjection::grid_to_grid_projection<dim, VectorType::value_type, VectorType>(
    source_vectors_per_dof_handler,
    source_dof_handlers,
    mapping,
    target_vectors_per_dof_handler,
    target_dof_handlers,
    mapping,
    data);
}

template<int dim, int n_components>
void
GridToGridProjector<dim, n_components>::check()
{
  // Output vectors in target.
  if constexpr(export_vectors)
  {
    OutputDataBase output_data;
    output_data.directory = "./";
    output_data.filename  = "target_vectors";
    output_data.degree    = element_type == ElementType::Hypercube ? 3 : 1;

    VectorWriter<dim, VectorType::value_type> vector_writer(output_data,
                                                            0 /* output_counter */,
                                                            mpi_comm);
    std::vector<bool>                         component_is_part_of_vector(n_components, true);
    if(n_components == 1)
    {
      component_is_part_of_vector[0] = false;
    }

    std::string elem_str = element_type == ElementType::Hypercube ? "hypercube" : "simplex";
    for(unsigned int i = 0; i < vectors_target.size(); ++i)
    {
      std::vector<std::string> component_names(n_components,
                                               "target_" + std::to_string(i) + "_" + elem_str);
      vector_writer.add_data_vector(vectors_target[i],
                                    dof_handler_target,
                                    component_names,
                                    component_is_part_of_vector);
    }
    for(unsigned int i = 0; i < vectors_target_continuous.size(); ++i)
    {
      std::vector<std::string> component_names(n_components,
                                               "target_continuous" + std::to_string(i) + "_" +
                                                 elem_str);
      vector_writer.add_data_vector(vectors_target_continuous[i],
                                    dof_handler_target_continuous,
                                    component_names,
                                    component_is_part_of_vector);
    }
  }

  // Compare vectors with exact values.
  std::array<double, 3> constexpr scales = {{1.0, 1.0e2, 1.0e-1}};
  for(unsigned int i = 0; i < vectors_target.size(); ++i)
  {
    ErrorCalculationData<dim> error_data;
    error_data.time_control_data.is_active  = true;
    error_data.calculate_relative_errors    = true;
    error_data.additional_quadrature_points = 1;
    std::shared_ptr<dealii::Function<dim>> analytical_solution;
    analytical_solution            = std::make_shared<SampleFunction<dim, n_components>>(scales[i]);
    error_data.analytical_solution = analytical_solution;

    ErrorCalculator<dim, typename VectorType::value_type> error_calculator(mpi_comm);
    error_calculator.setup(dof_handler_target, *mapping, error_data);
    error_calculator.time_control.needs_evaluation(0.0 /* time */, ExaDG::numbers::steady_timestep);
    error_calculator.evaluate(vectors_target[i], 0.0 /* time */, false /* unsteady */);
  }

  std::array<double, 3> constexpr scales_continuous = {{1.0, 1.0e3}};
  for(unsigned int i = 0; i < vectors_target_continuous.size(); ++i)
  {
    ErrorCalculationData<dim> error_data;
    error_data.time_control_data.is_active  = true;
    error_data.calculate_relative_errors    = true;
    error_data.additional_quadrature_points = 1;
    std::shared_ptr<dealii::Function<dim>> analytical_solution;
    analytical_solution = std::make_shared<SampleFunction<dim, n_components>>(scales_continuous[i]);
    error_data.analytical_solution = analytical_solution;

    ErrorCalculator<dim, typename VectorType::value_type> error_calculator(mpi_comm);
    error_calculator.setup(dof_handler_target_continuous, *mapping, error_data);
    error_calculator.time_control.needs_evaluation(0.0 /* time */, ExaDG::numbers::steady_timestep);
    error_calculator.evaluate(vectors_target_continuous[i], 0.0 /* time */, false /* unsteady */);
  }
}

template<int dim, int n_components>
void
GridToGridProjector<dim, n_components>::run()
{
  setup();
  project();
  check();

  pcout << "\n\n";
}

int
main(int argc, char * argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    // Simplex tests.
    {
      GridToGridProjector<2, 1> grid_to_grid_projector(false, false, ElementType::Simplex);
      grid_to_grid_projector.run();
    }
    {
      GridToGridProjector<2, 2> grid_to_grid_projector(false, false, ElementType::Simplex);
      grid_to_grid_projector.run();
    }

    // The three-dimensional simplex tests are disabled due to testing time constraints and
    // `RemotePointEvaluation` requiring more time here to find the integration points.
    bool constexpr enable_3d_simplex = false;
    if constexpr(enable_3d_simplex)
    {
      GridToGridProjector<3, 1> grid_to_grid_projector(false, false, ElementType::Simplex);
      grid_to_grid_projector.run();
    }
    if constexpr(enable_3d_simplex)
    {
      GridToGridProjector<3, 3> grid_to_grid_projector(false, false, ElementType::Simplex);
      grid_to_grid_projector.run();
    }

    // Vector-valued tests using hypercube including Raviart-Thomas <-> DGQ.
    for(unsigned int j = 0; j < 2; ++j)
    {
      for(unsigned int k = 0; k < 2; ++k)
      {
        bool const vector_target_is_RT_else_DGQ = j == 0;
        bool const vector_source_is_RT_else_DGQ = k == 0;
        {
          GridToGridProjector<2, 2> grid_to_grid_projector(vector_target_is_RT_else_DGQ,
                                                           vector_source_is_RT_else_DGQ,
                                                           ElementType::Hypercube);
          grid_to_grid_projector.run();
        }
        {
          GridToGridProjector<3, 3> grid_to_grid_projector(vector_target_is_RT_else_DGQ,
                                                           vector_source_is_RT_else_DGQ,
                                                           ElementType::Hypercube);
          grid_to_grid_projector.run();
        }
      }
    }

    // Scalar-valued tests using hypercube and DGQ only.
    {
      GridToGridProjector<2, 1> grid_to_grid_projector(false, false, ElementType::Hypercube);
      grid_to_grid_projector.run();
    }
    {
      GridToGridProjector<3, 1> grid_to_grid_projector(false, false, ElementType::Hypercube);
      grid_to_grid_projector.run();
    }
  }
  catch(std::exception & exc)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Exception on processing: " << std::endl
              << exc.what() << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;

    return 1;
  }
  catch(...)
  {
    std::cerr << std::endl
              << std::endl
              << "----------------------------------------------------" << std::endl;
    std::cerr << "Unknown exception!" << std::endl
              << "Aborting!" << std::endl
              << "----------------------------------------------------" << std::endl;
    return 1;
  }

  return 0;
}
