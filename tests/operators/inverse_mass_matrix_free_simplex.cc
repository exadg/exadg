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
class Projector
{
  using Number     = double;
  using VectorType = dealii::LinearAlgebra::distributed::Vector<Number>;
  using Range      = std::pair<unsigned int, unsigned int>;

public:
  Projector(const unsigned int degree);

  void
  run();

  void
  setup();

private:
  void
  compute();

  MPI_Comm                   mpi_comm;
  dealii::ConditionalOStream pcout;

  unsigned int fe_degree;

  dealii::parallel::fullydistributed::Triangulation<dim> tria;

  dealii::DoFHandler<dim> dof_handler;

  std::shared_ptr<dealii::FiniteElement<dim>> fe;

  std::shared_ptr<dealii::Mapping<dim> const> mapping;

  VectorType vector;
};

template<int dim, int n_components>
Projector<dim, n_components>::Projector(const unsigned int degree)
  : mpi_comm(MPI_COMM_WORLD),
    pcout(std::cout, (dealii::Utilities::MPI::this_mpi_process(mpi_comm) == 0)),
    fe_degree(degree),
    tria(mpi_comm)
{
}

template<int dim, int n_components>
void
Projector<dim, n_components>::setup()
{
  // Initialize finite elements
  pcout << "  dim          = " << dim << "\n"
        << "  n_components = " << n_components << "\n"
        << "  fe degree    = " << fe_degree << "\n";

  fe = create_finite_element<dim>(ElementType::Simplex, true, n_components, fe_degree);
  {
    auto const construction_data = dealii::TriangulationDescription::Utilities::
      create_description_from_triangulation_in_groups<dim, dim>(
        [&](dealii::Triangulation<dim> & tria_serial) {
          dealii::GridGenerator::subdivided_hyper_cube_with_simplices(tria_serial, 2);

          tria_serial.refine_global(3);
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


  mapping = std::make_shared<dealii::MappingFE<dim> const>(dealii::FE_SimplexP<dim>(1));

  // Distribute DoFs.
  dof_handler.reinit(tria);
  dof_handler.distribute_dofs(*fe);

  pcout << "  Number of degrees of freedom: " << dof_handler.n_dofs() << "\n";

  // Setup vector.
  vector.reinit(dof_handler.locally_owned_dofs(), mpi_comm);
}

template<int dim, int n_components>
void
Projector<dim, n_components>::compute()
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

  // Setup a `MassOperator` with a variable coefficient.
  MassOperatorData<dim, Number> mass_operator_data;
  mass_operator_data.coefficient_is_variable = false;

  MassOperator<dim, n_components, Number> mass_operator;
  mass_operator.initialize(matrix_free, empty_constraints, mass_operator_data);

  // Setup an `InverseMassOperator` with a variable coefficient.
  InverseMassOperatorData<Number> inverse_mass_operator_data;
  inverse_mass_operator_data.dof_index               = 0;
  inverse_mass_operator_data.quad_index              = 0;
  inverse_mass_operator_data.coefficient_is_variable = false;

  inverse_mass_operator_data.parameters.implementation_type = InverseMassType::MatrixfreeOperator;

  InverseMassOperator<dim, n_components, Number> inverse_mass_operator;
  inverse_mass_operator.initialize(matrix_free, inverse_mass_operator_data, &empty_constraints);

  // Setup a vector with random values and copy to reference.
  vector = 0;
  for(unsigned int i = 0; i < vector.locally_owned_size(); ++i)
  {
    vector.local_element(i) = static_cast<Number>(std::rand()) / RAND_MAX;
  }

  pcout << "  ||ref||_infty = " << vector.linfty_norm() << "\n";

  // Multiply vector by `MassOperator` and `InverseMassOperator`:
  // M^-1 * M * v = I * v = v
  {
    VectorType tmp, tmp2;
    // Note that `operator.apply()` zeroes the entries of `dst`.
    tmp.reinit(vector, true /* omit_zeroing_entries */);
    tmp2.reinit(vector, false /* omit_zeroing_entries */);
    mass_operator.apply(tmp /* dst */, vector /* src */);

    inverse_mass_operator.apply(tmp2, tmp);
    tmp2 -= vector;
    pcout << "  || vec1 - vec2||_infty = " << tmp2.linfty_norm() << "\n";
  }
}


template<int dim, int n_components>
void
Projector<dim, n_components>::run()
{
  setup();
  compute();
  pcout << "\n\n";
}

int
main(int argc, char * argv[])
{
  try
  {
    dealii::Utilities::MPI::MPI_InitFinalize mpi_initialization(argc, argv, 1);

    for(unsigned int i = 1; i < 4; ++i)
    {
      Projector<2 /* dim */, 1 /* n_components */> projector1(i);
      projector1.run();
      Projector<2 /* dim */, 2 /* n_components */> projector2(i);
      projector2.run();

      Projector<3 /* dim */, 1 /* n_components */> projector4(i);
      projector4.run();

      Projector<3 /* dim */, 3 /* n_components */> projector6(i);
      projector6.run();
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
