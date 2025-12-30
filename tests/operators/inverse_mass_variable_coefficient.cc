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
 *  along with this program. If not, see <https://www.gnu.org/licenses/>.
 *  ______________________________________________________________________
 */

// deal.II
#include <deal.II/base/function.h>
#include <deal.II/distributed/fully_distributed_tria.h>
#include <deal.II/fe/fe_raviart_thomas.h>
#include <deal.II/fe/mapping_fe.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/lac/la_parallel_vector.h>

// ExaDG
#include <exadg/matrix_free/matrix_free_data.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/inverse_mass_operator.h>
#include <exadg/operators/quadrature.h>

using namespace ExaDG;

template<int dim, int n_components>
class CoefficientFunction : public dealii::Function<dim>
{
public:
  CoefficientFunction() : dealii::Function<dim>(n_components)
  {
  }

  virtual double
  value(dealii::Point<dim> const & p, unsigned int const component = 0) const override;
};

template<int dim, int n_components>
double
CoefficientFunction<dim, n_components>::value(dealii::Point<dim> const & p,
                                              unsigned int const         component) const
{
  // Coefficient function has to be positive in entire domain.
  if(component == 0)
  {
    return (1.5 + std::sin(0.75 * p[0]));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Only scalar coefficient implemented."));
  }

  return 0.0;
}

template<int dim, int n_components>
class Projector
{
  using Number     = double;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;
  using Range      = std::pair<unsigned int, unsigned int>;

public:
  Projector(bool const            is_dg,
            bool const            RT_on_hypercube,
            ElementType const     element_type,
            InverseMassType const inverse_mass_implementation_type);

  void
  run();

  void
  setup();

private:
  void
  compute(bool const consider_inverse_coefficient);

  void
  cell_loop_set_coefficient(dealii::MatrixFree<dim, Number> const & matrix_free,
                            VectorType &                            dst,
                            VectorType const &                      src,
                            Range const &                           range);

  MPI_Comm                   mpi_comm;
  dealii::ConditionalOStream pcout;

  bool const        is_dg;
  bool const        RT_on_hypercube;
  ElementType const element_type;

  InverseMassType inverse_mass_implementation_type;

  static unsigned int constexpr fe_degree = 1;

  dealii::parallel::fullydistributed::Triangulation<dim> tria;

  dealii::DoFHandler<dim> dof_handler;

  std::shared_ptr<dealii::FiniteElement<dim>> fe;

  std::shared_ptr<dealii::Mapping<dim> const> mapping;

  VectorType vector;

  VariableCoefficients<dealii::VectorizedArray<Number>> variable_coefficients;
};

template<int dim, int n_components>
Projector<dim, n_components>::Projector(bool const            is_dg,
                                        bool const            RT_on_hypercube,
                                        ElementType const     element_type,
                                        InverseMassType const inverse_mass_implementation_type)
  : mpi_comm(MPI_COMM_WORLD),
    pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)),
    is_dg(is_dg),
    RT_on_hypercube(RT_on_hypercube),
    element_type(element_type),
    inverse_mass_implementation_type(inverse_mass_implementation_type),
    tria(mpi_comm)
{
}

template<int dim, int n_components>
void
Projector<dim, n_components>::setup()
{
  // Initialize finite elements
  pcout << "  element_type     = " << Utilities::enum_to_string<ElementType>(element_type) << "\n"
        << "  dim              = " << dim << "\n"
        << "  n_components     = " << n_components << "\n"
        << "  is_dg            = " << is_dg << "\n"
        << "  RT_on_hypercube  = " << RT_on_hypercube << "\n"
        << "  InverseMassType  = "
        << Utilities::enum_to_string<InverseMassType>(inverse_mass_implementation_type) << "\n";

  AssertThrow(element_type == ElementType::Hypercube or not RT_on_hypercube,
              dealii::ExcMessage("Cannot use Raviart-Thomas elements on non-hypercube cells."));
  AssertThrow(not RT_on_hypercube or n_components == dim,
              dealii::ExcMessage(
                "Raviart-Thomas elements can only be used for dim == n_components."));
  if(RT_on_hypercube)
  {
    AssertThrow(is_dg, dealii::ExcMessage("Raviart-Thomas elements are discontinuous."));
  }

  if(n_components == 1 or not RT_on_hypercube)
  {
    fe = create_finite_element<dim>(element_type, is_dg, n_components, fe_degree);
  }
  else
  {
    fe = std::make_shared<dealii::FE_RaviartThomasNodal<dim>>(fe_degree - 1);
  }

  // Create grids (dim == 2 and dim == 3) with a manifold attached and no merged patches for
  // Raviart-Thomas.
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
              tria.set_manifold(0, manifold);
            }
            else if constexpr(dim == 3)
            {
              dealii::CylindricalManifold<dim> const manifold(2 /* z-axis */);
              tria.set_manifold(0, manifold);
            }
            // tria_serial.refine_global(1);
          }
          else
          {
            dealii::Triangulation<dim> tria_hypercube;
            dealii::GridGenerator::hyper_cube_with_cylindrical_hole(
              tria_hypercube, radius + radius_shift, halve_edge_length, length, 2, true);
            // tria_hypercube.refine_global(1);
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

    tria.create_triangulation(construction_data);
  }

  // Construction via description necessitates reconstruction of the manifold.
  if constexpr(dim == 2)
  {
    dealii::PolarManifold<dim> const manifold;
    tria.set_manifold(0, manifold);
  }
  else if constexpr(dim == 3)
  {
    dealii::CylindricalManifold<dim> const manifold(2 /* z-axis */);
    tria.set_manifold(0, manifold);
  }

  // Create identity mapping depending on cell type.
  if(element_type == ElementType::Hypercube)
  {
    mapping = std::make_shared<dealii::MappingFE<dim> const>(dealii::FE_Q<dim>(fe_degree));
  }
  else
  {
    mapping = std::make_shared<dealii::MappingFE<dim> const>(dealii::FE_SimplexP<dim>(fe_degree));
  }

  // Distribute DoFs.
  dof_handler.reinit(tria);
  dof_handler.distribute_dofs(*fe);

  pcout << "  Number of degrees of freedom: " << dof_handler.n_dofs() << "\n";

  // Setup vector.
  vector.reinit(dof_handler.locally_owned_dofs(), mpi_comm);
}

template<int dim, int n_components>
void
Projector<dim, n_components>::compute(bool const consider_inverse_coefficient)
{
  // Setup MatrixFree.
  using Number = typename VectorType::value_type;
  MatrixFreeData<dim, Number> matrix_free_data;

  MappingFlags mapping_flags;
  mapping_flags.cells =
    dealii::update_quadrature_points | dealii::update_values | dealii::update_JxW_values;
  matrix_free_data.append_mapping_flags(mapping_flags);

  dealii::AffineConstraints<Number> empty_constraints;
  empty_constraints.close();

  matrix_free_data.insert_dof_handler(&dof_handler, std::to_string(0));
  matrix_free_data.insert_constraint(&empty_constraints, std::to_string(0));

  ElementType element_type = get_element_type(tria);

  std::shared_ptr<dealii::Quadrature<dim>> quadrature =
    create_quadrature<dim>(element_type, fe_degree + 1);

  matrix_free_data.insert_quadrature(*quadrature, std::to_string(0));

  dealii::MatrixFree<dim, Number, dealii::VectorizedArray<Number>> matrix_free;
  matrix_free.reinit(*mapping,
                     matrix_free_data.get_dof_handler_vector(),
                     matrix_free_data.get_constraint_vector(),
                     matrix_free_data.get_quadrature_vector(),
                     matrix_free_data.data);

  // Fill variable coefficients with smooth, positive analytical function.
  variable_coefficients.initialize(matrix_free,
                                   0 /* quad_index */,
                                   false /* store_face_data_in */,
                                   false /* store_cell_based_face_data_in */);

  matrix_free.cell_loop(&Projector<dim, n_components>::cell_loop_set_coefficient,
                        this,
                        vector,
                        vector,
                        false /* no zeroing of dst vector */);

  // Setup a `MassOperator` with a variable coefficient.
  MassOperatorData<dim, Number> mass_operator_data;
  mass_operator_data.coefficient_is_variable      = true;
  mass_operator_data.variable_coefficients        = &variable_coefficients;
  mass_operator_data.consider_inverse_coefficient = consider_inverse_coefficient;

  MassOperator<dim, n_components, Number> mass_operator;
  mass_operator.initialize(matrix_free, empty_constraints, mass_operator_data);

  // Setup an `InverseMassOperator` with a variable coefficient.
  InverseMassOperatorData<Number> inverse_mass_operator_data;
  inverse_mass_operator_data.dof_index                    = 0;
  inverse_mass_operator_data.quad_index                   = 0;
  inverse_mass_operator_data.coefficient_is_variable      = true;
  inverse_mass_operator_data.variable_coefficients        = &variable_coefficients;
  inverse_mass_operator_data.consider_inverse_coefficient = consider_inverse_coefficient;

  inverse_mass_operator_data.parameters.implementation_type  = inverse_mass_implementation_type;
  inverse_mass_operator_data.parameters.preconditioner       = PreconditionerMass::PointJacobi;
  inverse_mass_operator_data.parameters.solver_data.max_iter = 1000;
  inverse_mass_operator_data.parameters.solver_data.abs_tol  = 1e-12;
  inverse_mass_operator_data.parameters.solver_data.rel_tol  = 1e-8;

  InverseMassOperator<dim, n_components, Number> inverse_mass_operator;
  inverse_mass_operator.initialize(matrix_free, inverse_mass_operator_data, &empty_constraints);

  // Setup a vector with random values and copy to reference.
  vector = 0;
  for(unsigned int i = 0; i < vector.locally_owned_size(); ++i)
  {
    vector.local_element(i) = static_cast<Number>(std::rand()) / RAND_MAX;
  }
  VectorType const reference(vector);

  pcout << "\n"
        << "  ||ref||_infty = " << reference.linfty_norm() << "\n";

  // Multiply vector by `MassOperator` and `InverseMassOperator`:
  // M^-1 * M * v = I * v = v
  {
    VectorType tmp;
    // Note that `operator.apply()` zeroes the entries of `dst`.
    tmp.reinit(vector, true /* omit_zeroing_entries */);
    mass_operator.apply(tmp /* dst */, vector /* src */);

    // Perform standard `operator.apply()` zeroing entries.
    inverse_mass_operator.apply(vector, tmp);

    // Check `&src == &dst` case.
    inverse_mass_operator.apply(tmp, tmp);
    tmp -= vector;
    pcout << "  &src == &dst in InverseMassOperator: || vec1 - vec2||_infty = " << tmp.linfty_norm()
          << "\n";
  }

  // Compare to reference vector.
  vector -= reference;
  pcout << "  consider_inverse_coefficient = " << consider_inverse_coefficient
        << "   : || vec  - ref ||_infty = " << vector.linfty_norm() << "\n";
}

template<int dim, int n_components>
void
Projector<dim, n_components>::cell_loop_set_coefficient(
  dealii::MatrixFree<dim, Number> const & matrix_free,
  VectorType &                            dst,
  VectorType const &                      src,
  Range const &                           range)
{
  (void)dst;
  (void)src;

  CellIntegrator<dim, n_components, Number> integrator(matrix_free,
                                                       0 /* dof_index */,
                                                       0 /* quad_index */);

  CoefficientFunction<dim, n_components> coefficient_function;

  for(auto cell = range.first; cell < range.second; ++cell)
  {
    integrator.reinit(cell);

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      dealii::Point<dim, dealii::VectorizedArray<Number>> const & q_point =
        integrator.quadrature_point(q);
      dealii::VectorizedArray<Number> value_vec = dealii::make_vectorized_array<Number>(0.0);

      for(unsigned int i = 0; i < value_vec.size(); ++i)
      {
        dealii::Point<dim> single_q_point;
        for(unsigned int d = 0; d < dim; ++d)
        {
          single_q_point[d] = q_point[d][i];
        }

        Number value = coefficient_function.value(single_q_point, 0 /* component */);
        AssertThrow(value > 0.0,
                    dealii::ExcMessage("Non-positive value encountered in coefficient function."));

        value_vec[i] = value;
      }

      variable_coefficients.set_coefficient_cell(cell, q, value_vec);
    }
  }
}

template<int dim, int n_components>
void
Projector<dim, n_components>::run()
{
  setup();
  compute(false /* consider_inverse_coefficient */);
  compute(true /* consider_inverse_coefficient */);

  pcout << "\n\n";
}

int
main(int argc, char * argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    for(unsigned int i = 0; i < 2; ++i)
    {
      bool const is_dg = i == 0;

      bool RT_on_Hypercube = false;

      // Simplex tests FE_SimplexP or FE_SimplexDGP.
      for(unsigned int j = 0; j < 2; ++j)
      {
        if(j == 0 or is_dg)
        {
          InverseMassType inverse_mass_type_dg =
            j == 0 ? InverseMassType::ElementwiseKrylovSolver : InverseMassType::BlockMatrices;
          InverseMassType inverse_mass_type_simplex =
            is_dg ? inverse_mass_type_dg : InverseMassType::GlobalKrylovSolver;
          {
            Projector<2 /* dim */, 1 /* n_components */> projector(is_dg,
                                                                   RT_on_Hypercube,
                                                                   ElementType::Simplex,
                                                                   inverse_mass_type_simplex);
            projector.run();
          }
          {
            Projector<2 /* dim */, 2 /* n_components */> projector(is_dg,
                                                                   RT_on_Hypercube,
                                                                   ElementType::Simplex,
                                                                   inverse_mass_type_simplex);
            projector.run();
          }
          {
            Projector<3 /* dim */, 1 /* n_components */> projector(is_dg,
                                                                   RT_on_Hypercube,
                                                                   ElementType::Simplex,
                                                                   inverse_mass_type_simplex);
            projector.run();
          }
          {
            Projector<3 /* dim */, 3 /* n_components */> projector(is_dg,
                                                                   RT_on_Hypercube,
                                                                   ElementType::Simplex,
                                                                   inverse_mass_type_simplex);
            projector.run();
          }
        }
      }

      // Hypercube tests FE_Q or FE_DGQ.
      InverseMassType inverse_mass_type_hypercube =
        is_dg ? InverseMassType::MatrixfreeOperator : InverseMassType::GlobalKrylovSolver;
      {
        Projector<2 /* dim */, 1 /* n_components */> projector(is_dg,
                                                               RT_on_Hypercube,
                                                               ElementType::Hypercube,
                                                               inverse_mass_type_hypercube);
        projector.run();
      }
      {
        Projector<2 /* dim */, 2 /* n_components */> projector(is_dg,
                                                               RT_on_Hypercube,
                                                               ElementType::Hypercube,
                                                               inverse_mass_type_hypercube);
        projector.run();
      }
      {
        Projector<3 /* dim */, 1 /* n_components */> projector(is_dg,
                                                               RT_on_Hypercube,
                                                               ElementType::Hypercube,
                                                               inverse_mass_type_hypercube);
        projector.run();
      }
      {
        Projector<3 /* dim */, 3 /* n_components */> projector(is_dg,
                                                               RT_on_Hypercube,
                                                               ElementType::Hypercube,
                                                               inverse_mass_type_hypercube);
        projector.run();
      }

      // Hypercube tests FE_RaviartThomasNodal, dim == n_components.
      if(is_dg)
      {
        RT_on_Hypercube = true;
        {
          Projector<2 /* dim */, 2 /* n_components */> projector(
            is_dg, RT_on_Hypercube, ElementType::Hypercube, InverseMassType::GlobalKrylovSolver);
          projector.run();
        }
        {
          Projector<3 /* dim */, 3 /* n_components */> projector(
            is_dg, RT_on_Hypercube, ElementType::Hypercube, InverseMassType::GlobalKrylovSolver);
          projector.run();
        }
      }
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
