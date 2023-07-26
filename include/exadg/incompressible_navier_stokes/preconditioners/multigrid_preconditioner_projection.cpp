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
#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_projection.h>
#include <exadg/operators/mass_kernel.h>
#include <exadg/operators/quadrature.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
MultigridPreconditionerProjection<dim, Number>::MultigridPreconditionerProjection(
  MPI_Comm const & mpi_comm)
  : Base(mpi_comm), pde_operator(nullptr), mesh_is_moving(false)
{
}

template<int dim, typename Number>
void
MultigridPreconditionerProjection<dim, Number>::initialize(
  MultigridData const &                       mg_data,
  std::shared_ptr<Grid<dim> const>            grid,
  std::shared_ptr<dealii::Mapping<dim> const> mapping,
  dealii::FiniteElement<dim> const &          fe,
  PDEOperator const &                         pde_operator,
  bool const                                  mesh_is_moving,
  Map_DBC const &                             dirichlet_bc,
  Map_DBC_ComponentMask const &               dirichlet_bc_component_mask)
{
  this->pde_operator = &pde_operator;

  data = this->pde_operator->get_data();

  this->mesh_is_moving = mesh_is_moving;

  Base::initialize(mg_data,
                   grid,
                   mapping,
                   fe,
                   false /*operator_is_singular*/,
                   dirichlet_bc,
                   dirichlet_bc_component_mask);
}

template<int dim, typename Number>
void
MultigridPreconditionerProjection<dim, Number>::update()
{
  if(mesh_is_moving)
  {
    this->initialize_mapping();

    this->update_matrix_free_objects();
  }

  // update operators for all levels
  double const time_step_size = pde_operator->get_time_step_size();

  VectorType const & velocity = pde_operator->get_velocity();

  // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
  VectorTypeMG         velocity_multigrid_type_copy;
  VectorTypeMG const * velocity_multigrid_type_ptr;
  if(std::is_same<MultigridNumber, Number>::value)
  {
    velocity_multigrid_type_ptr = reinterpret_cast<VectorTypeMG const *>(&velocity);
  }
  else
  {
    velocity_multigrid_type_copy = velocity;
    velocity_multigrid_type_ptr  = &velocity_multigrid_type_copy;
  }

  // update operator on fine level
  this->get_operator(this->get_number_of_levels() - 1)
    ->update(*velocity_multigrid_type_ptr, time_step_size);

  // we store only two vectors since the velocity is no longer needed after having updated the
  // operators
  VectorTypeMG velocity_fine_level = *velocity_multigrid_type_ptr;
  VectorTypeMG velocity_coarse_level;

  this->transfer_from_fine_to_coarse_levels(
    [&](unsigned int const fine_level, unsigned int const coarse_level) {
      // interpolate velocity from fine to coarse level
      this->get_operator(coarse_level)->initialize_dof_vector(velocity_coarse_level);
      this->transfers->interpolate(fine_level, velocity_coarse_level, velocity_fine_level);

      // update operator
      this->get_operator(coarse_level)->update(velocity_coarse_level, time_step_size);

      // current coarse level becomes the fine level in the next iteration
      this->get_operator(coarse_level)->initialize_dof_vector(velocity_fine_level);
      velocity_fine_level.copy_locally_owned_data_from(velocity_coarse_level);
    });

  // Once the operators are updated, the update of smoothers and the coarse grid solver is generic
  // functionality implemented in the base class.
  this->update_smoothers();
  this->update_coarse_solver();
}

template<int dim, typename Number>
void
MultigridPreconditionerProjection<dim, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
  unsigned int const                     level,
  unsigned int const                     dealii_tria_level)
{
  matrix_free_data.data.mg_level = dealii_tria_level;

  MappingFlags flags;
  matrix_free_data.append_mapping_flags(MassKernel<dim, Number>::get_mapping_flags());
  if(data.use_divergence_penalty)
    matrix_free_data.append_mapping_flags(
      Operators::DivergencePenaltyKernel<dim, Number>::get_mapping_flags());
  if(data.use_continuity_penalty and this->level_info[level].is_dg())
    matrix_free_data.append_mapping_flags(
      Operators::ContinuityPenaltyKernel<dim, Number>::get_mapping_flags());

  if(data.use_cell_based_loops and this->level_info[level].is_dg())
  {
    auto tria = &this->dof_handlers[level]->get_triangulation();
    Categorization::do_cell_based_loops(*tria, matrix_free_data.data, dealii_tria_level);
  }

  matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "std_dof_handler");
  matrix_free_data.insert_constraint(&(*this->constraints[level]), "std_dof_handler");

  ElementType const element_type = GridUtilities::get_element_type(*this->grid->triangulation);
  std::shared_ptr<dealii::Quadrature<dim>> quadrature =
    create_quadrature<dim>(element_type, this->level_info[level].degree() + 1);
  matrix_free_data.insert_quadrature(*quadrature, "std_quadrature");
}

template<int dim, typename Number>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditionerProjection<dim, Number>::initialize_operator(unsigned int const level)
{
  // initialize pde_operator in a first step
  std::shared_ptr<PDEOperatorMG> pde_operator_level(new PDEOperatorMG());

  data.dof_index  = this->matrix_free_data_objects[level]->get_dof_index("std_dof_handler");
  data.quad_index = this->matrix_free_data_objects[level]->get_quad_index("std_quadrature");

  // The polynomial degree changes in case of p-multigrid, so we have to adapt the kernel_data
  // objects.
  Operators::DivergencePenaltyKernelData div_kernel_data =
    this->pde_operator->get_divergence_kernel_data();
  div_kernel_data.degree = this->level_info[level].degree();

  Operators::ContinuityPenaltyKernelData conti_kernel_data =
    this->pde_operator->get_continuity_kernel_data();
  conti_kernel_data.degree = this->level_info[level].degree();

  pde_operator_level->initialize(*this->matrix_free_objects[level],
                                 *this->constraints[level],
                                 data,
                                 div_kernel_data,
                                 conti_kernel_data);

  // initialize MGOperator which is a wrapper around the PDEOperatorMG
  std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator_level));

  return mg_operator;
}

template<int dim, typename Number>
std::shared_ptr<
  ProjectionOperator<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditionerProjection<dim, Number>::get_operator(unsigned int level)
{
  std::shared_ptr<MGOperator> mg_operator =
    std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

  return mg_operator->get_pde_operator();
}

template class MultigridPreconditionerProjection<2, float>;
template class MultigridPreconditionerProjection<3, float>;

template class MultigridPreconditionerProjection<2, double>;
template class MultigridPreconditionerProjection<3, double>;

} // namespace IncNS
} // namespace ExaDG
