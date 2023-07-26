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

#include <exadg/grid/grid_utilities.h>
#include <exadg/operators/quadrature.h>
#include <exadg/poisson/preconditioners/multigrid_preconditioner.h>

namespace ExaDG
{
namespace Poisson
{
template<int dim, typename Number, int n_components>
MultigridPreconditioner<dim, Number, n_components>::MultigridPreconditioner(
  MPI_Comm const & mpi_comm)
  : Base(mpi_comm), is_dg(true), mesh_is_moving(false)
{
}

template<int dim, typename Number, int n_components>
void
MultigridPreconditioner<dim, Number, n_components>::initialize(
  MultigridData const &                       mg_data,
  std::shared_ptr<Grid<dim> const>            grid,
  std::shared_ptr<dealii::Mapping<dim> const> mapping,
  dealii::FiniteElement<dim> const &          fe,
  LaplaceOperatorData<rank, dim> const &      data_in,
  bool const                                  mesh_is_moving,
  Map_DBC const &                             dirichlet_bc,
  Map_DBC_ComponentMask const &               dirichlet_bc_component_mask)
{
  data = data_in;

  is_dg = (fe.dofs_per_vertex == 0);

  this->mesh_is_moving = mesh_is_moving;

  Base::initialize(mg_data,
                   grid,
                   mapping,
                   fe,
                   data.operator_is_singular,
                   dirichlet_bc,
                   dirichlet_bc_component_mask);
}

template<int dim, typename Number, int n_components>
void
MultigridPreconditioner<dim, Number, n_components>::update()
{
  // update of this multigrid preconditioner is only needed
  // if the mesh is moving
  if(mesh_is_moving)
  {
    this->initialize_mapping();

    this->update_matrix_free_objects();

    this->for_all_levels(
      [&](unsigned int const level) { get_operator(level)->update_penalty_parameter(); });

    // Once the operators are updated, the update of smoothers and the coarse grid solver is generic
    // functionality implemented in the base class.
    this->update_smoothers();
    this->update_coarse_solver();
  }
}

template<int dim, typename Number, int n_components>
void
MultigridPreconditioner<dim, Number, n_components>::fill_matrix_free_data(
  MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
  unsigned int const                     level,
  unsigned int const                     dealii_triangulation_level)
{
  matrix_free_data.data.mg_level = dealii_triangulation_level;

  matrix_free_data.append_mapping_flags(
    Operators::LaplaceKernel<dim, Number>::get_mapping_flags(this->level_info[level].is_dg(),
                                                             this->level_info[level].is_dg()));


  if(data.use_cell_based_loops and this->level_info[level].is_dg())
  {
    auto tria = &this->dof_handlers[level]->get_triangulation();
    Categorization::do_cell_based_loops(*tria, matrix_free_data.data, dealii_triangulation_level);
  }

  matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "laplace_dof_handler");
  matrix_free_data.insert_constraint(&(*this->constraints[level]), "laplace_dof_handler");

  ElementType const element_type = GridUtilities::get_element_type(*this->grid->triangulation);
  std::shared_ptr<dealii::Quadrature<dim>> quadrature =
    create_quadrature<dim>(element_type, this->level_info[level].degree() + 1);
  matrix_free_data.insert_quadrature(*quadrature, "laplace_quadrature");
}

template<int dim, typename Number, int n_components>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number, n_components>::initialize_operator(unsigned int const level)
{
  // initialize pde_operator in a first step
  std::shared_ptr<Laplace> pde_operator(new Laplace());

  data.dof_index  = this->matrix_free_data_objects[level]->get_dof_index("laplace_dof_handler");
  data.quad_index = this->matrix_free_data_objects[level]->get_quad_index("laplace_quadrature");

  pde_operator->initialize(*this->matrix_free_objects[level], *this->constraints[level], data);

  // initialize MGOperator which is a wrapper around the PDEOperator
  std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

  return mg_operator;
}

template<int dim, typename Number, int n_components>
std::shared_ptr<LaplaceOperator<dim,
                                typename MultigridPreconditionerBase<dim, Number>::MultigridNumber,
                                n_components>>
MultigridPreconditioner<dim, Number, n_components>::get_operator(unsigned int level)
{
  std::shared_ptr<MGOperator> mg_operator =
    std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

  return mg_operator->get_pde_operator();
}

template class MultigridPreconditioner<2, float, 1>;
template class MultigridPreconditioner<2, float, 2>;
template class MultigridPreconditioner<3, float, 1>;
template class MultigridPreconditioner<3, float, 3>;

template class MultigridPreconditioner<2, double, 1>;
template class MultigridPreconditioner<2, double, 2>;
template class MultigridPreconditioner<3, double, 1>;
template class MultigridPreconditioner<3, double, 3>;

} // namespace Poisson
} // namespace ExaDG
