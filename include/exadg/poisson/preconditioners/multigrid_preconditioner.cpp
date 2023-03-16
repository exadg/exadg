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
  MultigridData const &                                                  mg_data,
  MultigridVariant const &                                               multigrid_variant,
  dealii::Triangulation<dim> const *                                     triangulation,
  PeriodicFacePairs const &                                              periodic_face_pairs,
  std::vector<std::shared_ptr<dealii::Triangulation<dim> const>> const & coarse_triangulations,
  std::vector<PeriodicFacePairs> const &                                 coarse_periodic_face_pairs,
  dealii::FiniteElement<dim> const &                                     fe,
  std::shared_ptr<dealii::Mapping<dim> const>                            mapping,
  LaplaceOperatorData<rank, dim> const &                                 data_in,
  bool const                                                             mesh_is_moving,
  Map_DBC const &                                                        dirichlet_bc,
  Map_DBC_ComponentMask const & dirichlet_bc_component_mask)
{
  data = data_in;

  is_dg = (fe.dofs_per_vertex == 0);

  this->mesh_is_moving = mesh_is_moving;

  Base::initialize(mg_data,
                   multigrid_variant,
                   triangulation,
                   periodic_face_pairs,
                   coarse_triangulations,
                   coarse_periodic_face_pairs,
                   fe,
                   mapping,
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

    this->update_matrix_free();

    update_operators_after_mesh_movement();

    this->update_smoothers();

    // singular operators do not occur for this operator
    this->update_coarse_solver(data.operator_is_singular);
  }
}

template<int dim, typename Number, int n_components>
void
MultigridPreconditioner<dim, Number, n_components>::fill_matrix_free_data(
  MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
  unsigned int const                     level,
  unsigned int const                     h_level)
{
  matrix_free_data.data.mg_level = h_level;

  matrix_free_data.append_mapping_flags(
    Operators::LaplaceKernel<dim, Number>::get_mapping_flags(this->level_info[level].is_dg(),
                                                             this->level_info[level].is_dg()));


  if(data.use_cell_based_loops && this->level_info[level].is_dg())
  {
    auto tria = dynamic_cast<dealii::parallel::distributed::Triangulation<dim> const *>(
      &this->dof_handlers[level]->get_triangulation());
    Categorization::do_cell_based_loops(*tria, matrix_free_data.data, h_level);
  }

  matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "laplace_dof_handler");
  matrix_free_data.insert_constraint(&(*this->constraints[level]), "laplace_dof_handler");

  if(this->dof_handlers[level]->get_triangulation().all_reference_cells_are_hyper_cube())
  {
    matrix_free_data.insert_quadrature(dealii::QGauss<1>(this->level_info[level].degree() + 1),
                                       "laplace_quadrature");
  }
  else if(this->dof_handlers[level]->get_triangulation().all_reference_cells_are_simplex())
  {
    matrix_free_data.insert_quadrature(dealii::QGaussSimplex<dim>(this->level_info[level].degree() +
                                                                  1),
                                       "laplace_quadrature");
  }
  else
  {
    AssertThrow(
      false,
      dealii::ExcMessage(
        "Only pure hypercube or pure simplex meshes are implemented for Poisson::MultigridPreconditioner."));
  }
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
void
MultigridPreconditioner<dim, Number, n_components>::update_operators_after_mesh_movement()
{
  for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
  {
    get_operator(level)->update_penalty_parameter();
  }
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
