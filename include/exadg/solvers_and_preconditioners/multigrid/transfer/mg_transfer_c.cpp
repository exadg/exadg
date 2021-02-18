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

// deal.II
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/matrix_free/fe_evaluation.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer_c.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number, typename VectorType, int components>
MGTransferC<dim, Number, VectorType, components>::MGTransferC(
  Mapping<dim> const &              mapping,
  MatrixFree<dim, Number> const &   matrixfree_dg,
  MatrixFree<dim, Number> const &   matrixfree_cg,
  AffineConstraints<Number> const & constraints_dg,
  AffineConstraints<Number> const & constraints_cg,
  unsigned int const                level,
  unsigned int const                fe_degree,
  unsigned int const                dof_handler_index)
  : fe_degree(fe_degree)
{
  std::vector<DoFHandler<dim> const *> dofhandlers = {
    &matrixfree_cg.get_dof_handler(dof_handler_index),
    &matrixfree_dg.get_dof_handler(dof_handler_index)};

  std::vector<AffineConstraints<Number> const *> constraint_matrices = {&constraints_cg,
                                                                        &constraints_dg};
  QGauss<1>                                      quadrature(1);

  typename MatrixFree<dim, Number>::AdditionalData additional_data;
  additional_data.mg_level = level;
  data_composite.reinit(mapping, dofhandlers, constraint_matrices, quadrature, additional_data);
}

template<int dim, typename Number, typename VectorType, int components>
MGTransferC<dim, Number, VectorType, components>::~MGTransferC()
{
}

template<int dim, typename Number, typename VectorType, int components>
template<int degree>
void
MGTransferC<dim, Number, VectorType, components>::do_interpolate(VectorType &       dst,
                                                                 VectorType const & src) const
{
  FEEvaluation<dim, degree, 1, components, Number> fe_eval_cg(data_composite, 0);
  FEEvaluation<dim, degree, 1, components, Number> fe_eval_dg(data_composite, 1);

  VectorType vec_dg;
  data_composite.initialize_dof_vector(vec_dg, 1);
  vec_dg.copy_locally_owned_data_from(src);

  for(unsigned int cell = 0; cell < data_composite.n_cell_batches(); ++cell)
  {
    fe_eval_cg.reinit(cell);
    fe_eval_dg.reinit(cell);

    fe_eval_dg.read_dof_values(vec_dg);

    for(unsigned int i = 0; i < fe_eval_cg.static_dofs_per_cell; i++)
      fe_eval_cg.begin_dof_values()[i] = fe_eval_dg.begin_dof_values()[i];

    fe_eval_cg.set_dof_values(dst);
  }
}

template<int dim, typename Number, typename VectorType, int components>
template<int degree>
void
MGTransferC<dim, Number, VectorType, components>::do_restrict_and_add(VectorType &       dst,
                                                                      VectorType const & src) const
{
  FEEvaluation<dim, degree, 1, components, Number> fe_eval_cg(data_composite, 0);
  FEEvaluation<dim, degree, 1, components, Number> fe_eval_dg(data_composite, 1);

  VectorType vec_dg;
  data_composite.initialize_dof_vector(vec_dg, 1);
  vec_dg.copy_locally_owned_data_from(src);

  for(unsigned int cell = 0; cell < data_composite.n_cell_batches(); ++cell)
  {
    fe_eval_cg.reinit(cell);
    fe_eval_dg.reinit(cell);

    fe_eval_dg.read_dof_values(vec_dg);

    for(unsigned int i = 0; i < fe_eval_cg.static_dofs_per_cell; i++)
      fe_eval_cg.begin_dof_values()[i] = fe_eval_dg.begin_dof_values()[i];

    fe_eval_cg.distribute_local_to_global(dst);
  }

  dst.compress(VectorOperation::add);
}

template<int dim, typename Number, typename VectorType, int components>
template<int degree>
void
MGTransferC<dim, Number, VectorType, components>::do_prolongate(VectorType &       dst,
                                                                VectorType const & src) const
{
  src.update_ghost_values();

  FEEvaluation<dim, degree, 1, components, Number> fe_eval_cg(data_composite, 0);
  FEEvaluation<dim, degree, 1, components, Number> fe_eval_dg(data_composite, 1);

  VectorType vec_dg;
  data_composite.initialize_dof_vector(vec_dg, 1);

  for(unsigned int cell = 0; cell < data_composite.n_cell_batches(); ++cell)
  {
    fe_eval_cg.reinit(cell);
    fe_eval_dg.reinit(cell);

    fe_eval_cg.read_dof_values(src);

    for(unsigned int i = 0; i < fe_eval_cg.static_dofs_per_cell; i++)
      fe_eval_dg.begin_dof_values()[i] = fe_eval_cg.begin_dof_values()[i];

    fe_eval_dg.distribute_local_to_global(vec_dg);
  }
  dst.copy_locally_owned_data_from(vec_dg);
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferC<dim, Number, VectorType, components>::interpolate(unsigned int const level,
                                                              VectorType &       dst,
                                                              VectorType const & src) const
{
  (void)level;

  switch(this->fe_degree)
  {
      // clang-format off
    case  1: do_interpolate< 1>(dst, src); break;
    case  2: do_interpolate< 2>(dst, src); break;
    case  3: do_interpolate< 3>(dst, src); break;
    case  4: do_interpolate< 4>(dst, src); break;
    case  5: do_interpolate< 5>(dst, src); break;
    case  6: do_interpolate< 6>(dst, src); break;
    case  7: do_interpolate< 7>(dst, src); break;
    case  8: do_interpolate< 8>(dst, src); break;
    case  9: do_interpolate< 9>(dst, src); break;
    case 10: do_interpolate<10>(dst, src); break;
    case 11: do_interpolate<11>(dst, src); break;
    case 12: do_interpolate<12>(dst, src); break;
    case 13: do_interpolate<13>(dst, src); break;
    case 14: do_interpolate<14>(dst, src); break;
    case 15: do_interpolate<15>(dst, src); break;
    default:
      AssertThrow(false, ExcMessage("MGTransferC::interpolate() not implemented for this degree!"));
      // clang-format on
  }
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferC<dim, Number, VectorType, components>::restrict_and_add(unsigned int const /*level*/,
                                                                   VectorType &       dst,
                                                                   VectorType const & src) const
{
  switch(this->fe_degree)
  {
      // clang-format off
    case  1: do_restrict_and_add< 1>(dst, src); break;
    case  2: do_restrict_and_add< 2>(dst, src); break;
    case  3: do_restrict_and_add< 3>(dst, src); break;
    case  4: do_restrict_and_add< 4>(dst, src); break;
    case  5: do_restrict_and_add< 5>(dst, src); break;
    case  6: do_restrict_and_add< 6>(dst, src); break;
    case  7: do_restrict_and_add< 7>(dst, src); break;
    case  8: do_restrict_and_add< 8>(dst, src); break;
    case  9: do_restrict_and_add< 9>(dst, src); break;
    case 10: do_restrict_and_add<10>(dst, src); break;
    case 11: do_restrict_and_add<11>(dst, src); break;
    case 12: do_restrict_and_add<12>(dst, src); break;
    case 13: do_restrict_and_add<13>(dst, src); break;
    case 14: do_restrict_and_add<14>(dst, src); break;
    case 15: do_restrict_and_add<15>(dst, src); break;
    default:
      AssertThrow(false, ExcMessage("MGTransferC::restrict_and_add() not implemented for this degree!"));
      // clang-format on
  }
}

template<int dim, typename Number, typename VectorType, int components>
void
MGTransferC<dim, Number, VectorType, components>::prolongate(unsigned int const /*level*/,
                                                             VectorType &       dst,
                                                             VectorType const & src) const
{
  switch(this->fe_degree)
  {
      // clang-format off
    case  1: do_prolongate< 1>(dst, src); break;
    case  2: do_prolongate< 2>(dst, src); break;
    case  3: do_prolongate< 3>(dst, src); break;
    case  4: do_prolongate< 4>(dst, src); break;
    case  5: do_prolongate< 5>(dst, src); break;
    case  6: do_prolongate< 6>(dst, src); break;
    case  7: do_prolongate< 7>(dst, src); break;
    case  8: do_prolongate< 8>(dst, src); break;
    case  9: do_prolongate< 9>(dst, src); break;
    case 10: do_prolongate<10>(dst, src); break;
    case 11: do_prolongate<11>(dst, src); break;
    case 12: do_prolongate<12>(dst, src); break;
    case 13: do_prolongate<13>(dst, src); break;
    case 14: do_prolongate<14>(dst, src); break;
    case 15: do_prolongate<15>(dst, src); break;
    default:
      AssertThrow(false, ExcMessage("MGTransferC::prolongate() not implemented for this degree!"));
      // clang-format on
  }
}

typedef dealii::LinearAlgebra::distributed::Vector<float>  VectorTypeFloat;
typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

template class MGTransferC<2, float, VectorTypeFloat, 1>;
template class MGTransferC<2, float, VectorTypeFloat, 2>;

template class MGTransferC<3, float, VectorTypeFloat, 1>;
template class MGTransferC<3, float, VectorTypeFloat, 3>;

template class MGTransferC<2, double, VectorTypeDouble, 1>;
template class MGTransferC<2, double, VectorTypeDouble, 2>;

template class MGTransferC<3, double, VectorTypeDouble, 1>;
template class MGTransferC<3, double, VectorTypeDouble, 3>;

} // namespace ExaDG
