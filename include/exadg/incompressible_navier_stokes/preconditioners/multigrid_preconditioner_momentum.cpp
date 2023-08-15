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

#include <exadg/incompressible_navier_stokes/preconditioners/multigrid_preconditioner_momentum.h>
#include <exadg/operators/quadrature.h>

namespace ExaDG
{
namespace IncNS
{
template<int dim, typename Number>
MultigridPreconditioner<dim, Number>::MultigridPreconditioner(MPI_Comm const & comm)
  : Base(comm),
    pde_operator(nullptr),
    mg_operator_type(MultigridOperatorType::ReactionDiffusion),
    mesh_is_moving(false)
{
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::initialize(
  MultigridData const &                       mg_data,
  std::shared_ptr<Grid<dim> const>            grid,
  std::shared_ptr<dealii::Mapping<dim> const> mapping,
  dealii::FiniteElement<dim> const &          fe,
  PDEOperator const &                         pde_operator,
  MultigridOperatorType const &               mg_operator_type,
  bool const                                  mesh_is_moving,
  Map_DBC const &                             dirichlet_bc,
  Map_DBC_ComponentMask const &               dirichlet_bc_component_mask)
{
  this->pde_operator = &pde_operator;

  this->mg_operator_type = mg_operator_type;

  this->mesh_is_moving = mesh_is_moving;

  data = this->pde_operator->get_data();

  // When solving the reaction-convection-diffusion problem, it might be possible
  // that one wants to apply the multigrid preconditioner only to the reaction-diffusion
  // operator (which is symmetric, Chebyshev smoother, etc.) instead of the non-symmetric
  // reaction-convection-diffusion operator. Accordingly, we have to reset which
  // operators should be "active" for the multigrid preconditioner, independently of
  // the actual equation type that is solved.
  AssertThrow(this->mg_operator_type != MultigridOperatorType::Undefined,
              dealii::ExcMessage("Invalid parameter mg_operator_type."));

  if(this->mg_operator_type == MultigridOperatorType::ReactionDiffusion)
  {
    // deactivate convective term for multigrid preconditioner
    data.convective_problem = false;
  }
  else if(this->mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
  {
    AssertThrow(data.convective_problem == true, dealii::ExcMessage("Invalid parameter."));
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

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
MultigridPreconditioner<dim, Number>::update()
{
  // Update matrix-free objects and operators
  if(mesh_is_moving)
  {
    this->initialize_mapping();

    this->update_matrix_free_objects();

    this->for_all_levels(
      [&](unsigned int const level) { this->get_operator(level)->update_after_grid_motion(); });
  }

  if(data.unsteady_problem)
  {
    this->for_all_levels([&](unsigned int const level) {
      // The operator also depends on the time. This is due to the fact that the linearized
      // convective term does not only depend on the linearized velocity field but also on Dirichlet
      // boundary data which itself depends on the current time.
      this->get_operator(level)->set_time(pde_operator->get_time());
      // In case of adaptive time stepping, the scaling factor of the time derivative term changes
      // over time.
      this->get_operator(level)->set_scaling_factor_mass_operator(
        pde_operator->get_scaling_factor_mass_operator());
    });
  }

  if(mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
  {
    VectorType const & vector_linearization = pde_operator->get_velocity();

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

    // copy velocity to finest level
    this->get_operator(this->get_number_of_levels() - 1)
      ->set_velocity_copy(*vector_multigrid_type_ptr);

    // interpolate velocity from fine to coarse level
    this->transfer_from_fine_to_coarse_levels(
      [&](unsigned int const fine_level, unsigned int const coarse_level) {
        auto & vector_fine_level   = this->get_operator(fine_level)->get_velocity();
        auto   vector_coarse_level = this->get_operator(coarse_level)->get_velocity();
        this->transfers->interpolate(fine_level, vector_coarse_level, vector_fine_level);
        this->get_operator(coarse_level)->set_velocity_copy(vector_coarse_level);
      });
  }

  // In case the operators have been updated, we also need to update the smoothers and the coarse
  // grid solver. This is generic functionality implemented in the base class.
  if(mesh_is_moving or data.unsteady_problem or
     (mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion))
  {
    this->update_smoothers();
    this->update_coarse_solver();
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
  unsigned int const                     level,
  unsigned int const                     dealii_tria_level)
{
  matrix_free_data.data.mg_level = dealii_tria_level;

  if(data.unsteady_problem)
    matrix_free_data.append_mapping_flags(MassKernel<dim, Number>::get_mapping_flags());
  if(data.convective_problem)
    matrix_free_data.append_mapping_flags(
      Operators::ConvectiveKernel<dim, Number>::get_mapping_flags());
  if(data.viscous_problem)
    matrix_free_data.append_mapping_flags(
      Operators::ViscousKernel<dim, Number>::get_mapping_flags(this->level_info[level].is_dg(),
                                                               this->level_info[level].is_dg()));

  if(data.use_cell_based_loops and this->level_info[level].is_dg())
  {
    auto tria = &this->dof_handlers[level]->get_triangulation();
    Categorization::do_cell_based_loops(*tria, matrix_free_data.data, dealii_tria_level);
  }

  matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "std_dof_handler");
  matrix_free_data.insert_constraint(&(*this->constraints[level]), "std_dof_handler");

  ElementType const element_type = get_element_type(*this->grid->triangulation);
  std::shared_ptr<dealii::Quadrature<dim>> quadrature =
    create_quadrature<dim>(element_type, this->level_info[level].degree() + 1);
  matrix_free_data.insert_quadrature(*quadrature, "std_quadrature");
}

template<int dim, typename Number>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::initialize_operator(unsigned int const level)
{
  // initialize pde_operator in a first step
  std::shared_ptr<PDEOperatorMG> pde_operator_level(new PDEOperatorMG());

  data.dof_index  = this->matrix_free_data_objects[level]->get_dof_index("std_dof_handler");
  data.quad_index = this->matrix_free_data_objects[level]->get_quad_index("std_quadrature");

  pde_operator_level->initialize(*this->matrix_free_objects[level],
                                 *this->constraints[level],
                                 data);

  // make sure that scaling factor of time derivative term has been set before the smoothers are
  // initialized
  pde_operator_level->set_scaling_factor_mass_operator(
    pde_operator->get_scaling_factor_mass_operator());

  // initialize MGOperator which is a wrapper around the PDEOperatorMG
  std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator_level));

  return mg_operator;
}

template<int dim, typename Number>
std::shared_ptr<
  MomentumOperator<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::get_operator(unsigned int level)
{
  std::shared_ptr<MGOperator> mg_operator =
    std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

  return mg_operator->get_pde_operator();
}

template class MultigridPreconditioner<2, float>;
template class MultigridPreconditioner<3, float>;

template class MultigridPreconditioner<2, double>;
template class MultigridPreconditioner<3, double>;

} // namespace IncNS
} // namespace ExaDG
