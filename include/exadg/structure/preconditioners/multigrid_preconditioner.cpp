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

#include <exadg/structure/preconditioners/multigrid_preconditioner.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
MultigridPreconditioner<dim, Number>::MultigridPreconditioner(MPI_Comm const & mpi_comm)
  : Base(mpi_comm), pde_operator(nullptr), nonlinear(true)
{
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::initialize(
  MultigridData const &                                                  mg_data,
  MultigridVariant const &                                               multigrid_variant,
  dealii::Triangulation<dim> const *                                     triangulation,
  PeriodicFacePairs const &                                              periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations,
  std::vector<PeriodicFacePairs> const &                                 coarse_periodic_face_pairs,
  dealii::FiniteElement<dim> const &                                     fe,
  std::shared_ptr<dealii::Mapping<dim> const>                            mapping,
  ElasticityOperatorBase<dim, Number> const &                            pde_operator,
  bool const                                                             nonlinear_operator,
  Map_DBC const &                                                        dirichlet_bc,
  Map_DBC_ComponentMask const & dirichlet_bc_component_mask)
{
  this->pde_operator = &pde_operator;

  this->data = this->pde_operator->get_data();

  this->nonlinear = nonlinear_operator;

  Base::initialize(mg_data,
                   multigrid_variant,
                   triangulation,
                   periodic_face_pairs,
                   coarse_triangulations,
                   coarse_periodic_face_pairs,
                   fe,
                   mapping,
                   false /*operator_is_singular*/,
                   dirichlet_bc,
                   dirichlet_bc_component_mask);
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update()
{
  if(nonlinear)
  {
    update_operators();

    this->update_smoothers();

    // singular operators do not occur for this operator
    this->update_coarse_solver(false /* operator_is_singular */);
  }
  else
  {
    AssertThrow(false,
                dealii::ExcMessage(
                  "Update of multigrid preconditioner is not implemented for linear elasticity."));
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
  unsigned int const                     level,
  unsigned int const                     h_level)
{
  matrix_free_data.data.mg_level = h_level;

  if(nonlinear)
    matrix_free_data.append_mapping_flags(PDEOperatorNonlinear::get_mapping_flags());
  else // linear
    matrix_free_data.append_mapping_flags(PDEOperatorLinear::get_mapping_flags());

  matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "elasticity_dof_handler");
  matrix_free_data.insert_constraint(&(*this->constraints[level]), "elasticity_dof_handler");

  if(this->dof_handlers[level]->get_triangulation().all_reference_cells_are_hyper_cube())
  {
    matrix_free_data.insert_quadrature(dealii::QGauss<1>(this->level_info[level].degree() + 1),
                                       "elasticity_quadrature");
  }
  else if(this->dof_handlers[level]->get_triangulation().all_reference_cells_are_simplex())
  {
    matrix_free_data.insert_quadrature(dealii::QGaussSimplex<dim>(this->level_info[level].degree() +
                                                                  1),
                                       "elasticity_quadrature");
  }
  else
  {
    AssertThrow(
      false,
      dealii::ExcMessage(
        "Only pure hypercube or pure simplex meshes are implemented for Structure::MultigridPreconditioner."));
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update_operators()
{
  PDEOperatorNonlinear const * pde_operator_nonlinear =
    dynamic_cast<PDEOperatorNonlinear const *>(pde_operator);

  VectorType const & vector_linearization = pde_operator_nonlinear->get_solution_linearization();

  // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
  VectorTypeMG         vector_multigrid_type_copy;
  VectorTypeMG const * vector_multigrid_type_ptr;
  if(std::is_same<MultigridNumber, Number>::value)
  {
    vector_multigrid_type_ptr = reinterpret_cast<VectorTypeMG const *>(&vector_linearization);
  }
  else
  {
    vector_multigrid_type_copy = vector_linearization;
    vector_multigrid_type_ptr  = &vector_multigrid_type_copy;
  }

  set_solution_linearization(*vector_multigrid_type_ptr);
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::set_solution_linearization(
  VectorTypeMG const & vector_linearization)
{
  // copy velocity to finest level
  this->get_operator_nonlinear(this->fine_level)->set_solution_linearization(vector_linearization);

  // interpolate velocity from fine to coarse level
  for(unsigned int level = this->fine_level; level > this->coarse_level; --level)
  {
    auto & vector_fine_level =
      this->get_operator_nonlinear(level - 0)->get_solution_linearization();
    auto vector_coarse_level =
      this->get_operator_nonlinear(level - 1)->get_solution_linearization();
    this->transfers->interpolate(level, vector_coarse_level, vector_fine_level);
    this->get_operator_nonlinear(level - 1)->set_solution_linearization(vector_coarse_level);
  }
}

template<int dim, typename Number>
std::shared_ptr<
  NonLinearOperator<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::get_operator_nonlinear(unsigned int level)
{
  std::shared_ptr<MGOperatorNonlinear> mg_operator =
    std::dynamic_pointer_cast<MGOperatorNonlinear>(this->operators[level]);

  return mg_operator->get_pde_operator();
}

template<int dim, typename Number>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::initialize_operator(unsigned int const level)
{
  std::shared_ptr<MGOperatorBase> mg_operator_level;

  data.dof_index  = this->matrix_free_data_objects[level]->get_dof_index("elasticity_dof_handler");
  data.quad_index = this->matrix_free_data_objects[level]->get_quad_index("elasticity_quadrature");

  if(nonlinear)
  {
    std::shared_ptr<PDEOperatorNonlinearMG> pde_operator_level(new PDEOperatorNonlinearMG());
    pde_operator_level->initialize(*this->matrix_free_objects[level],
                                   *this->constraints[level],
                                   data);

    mg_operator_level = std::make_shared<MGOperatorNonlinear>(pde_operator_level);
  }
  else // linear
  {
    std::shared_ptr<PDEOperatorLinearMG> pde_operator_level(new PDEOperatorLinearMG());
    pde_operator_level->initialize(*this->matrix_free_objects[level],
                                   *this->constraints[level],
                                   data);

    mg_operator_level = std::make_shared<MGOperatorLinear>(pde_operator_level);
  }

  return mg_operator_level;
}

template class MultigridPreconditioner<2, float>;
template class MultigridPreconditioner<3, float>;

template class MultigridPreconditioner<2, double>;
template class MultigridPreconditioner<3, double>;

} // namespace Structure
} // namespace ExaDG
