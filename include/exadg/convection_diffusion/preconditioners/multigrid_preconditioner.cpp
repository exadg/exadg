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
#include <deal.II/fe/fe_system.h>

// ExaDG
#include <exadg/convection_diffusion/preconditioners/multigrid_preconditioner.h>
#include <exadg/operators/finite_element.h>
#include <exadg/operators/mapping_flags.h>
#include <exadg/operators/quadrature.h>

namespace ExaDG
{
namespace ConvDiff
{
template<int dim, typename Number>
MultigridPreconditioner<dim, Number>::MultigridPreconditioner(MPI_Comm const & mpi_comm)
  : Base(mpi_comm),
    degree_velocity(1),
    pde_operator(nullptr),
    mg_operator_type(MultigridOperatorType::Undefined),
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
  this->degree_velocity = fe.degree;

  this->pde_operator     = &pde_operator;
  this->mg_operator_type = mg_operator_type;
  this->mesh_is_moving   = mesh_is_moving;

  data = this->pde_operator->get_data();

  // When solving the reaction-convection-diffusion equations, it might be possible
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
    data.diffusive_problem  = true;
  }
  else if(this->mg_operator_type == MultigridOperatorType::ReactionConvection)
  {
    data.convective_problem = true;
    // deactivate viscous term for multigrid preconditioner
    data.diffusive_problem = false;
  }
  else if(this->mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
  {
    data.convective_problem = true;
    data.diffusive_problem  = true;
  }
  else
  {
    AssertThrow(false, dealii::ExcMessage("Not implemented."));
  }

  Base::initialize(mg_data,
                   grid,
                   mapping,
                   fe,
                   data.operator_is_singular,
                   dirichlet_bc,
                   dirichlet_bc_component_mask,
                   false /* initialize_preconditioners */);
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

  this->for_all_levels([&](unsigned int const level) {
    // the velocity field of the convective term is a function of the time
    this->get_operator(level)->set_time(pde_operator->get_time());
    // in case of adaptive time stepping, the scaling factor of the time derivative term changes
    // over time
    this->get_operator(level)->set_scaling_factor_mass_operator(
      pde_operator->get_scaling_factor_mass_operator());
  });

  if(data.convective_problem and
     data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
  {
    // If necessary, interpolate fine level velocity field to coarser levels and set velocity for
    // all operators

    VectorType const & velocity = pde_operator->get_velocity();

    // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
    VectorTypeMG         vector_multigrid_type_copy;
    VectorTypeMG const * vector_multigrid_type_ptr;
    if(std::is_same<MultigridNumber, Number>::value)
    {
      vector_multigrid_type_ptr = reinterpret_cast<VectorTypeMG const *>(&velocity);
    }
    else
    {
      vector_multigrid_type_copy = velocity;
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
        transfers_velocity->interpolate(fine_level, vector_coarse_level, vector_fine_level);
        this->get_operator(coarse_level)->set_velocity_copy(vector_coarse_level);
      });
  }

  // Once the operators are updated, the update of smoothers and the coarse grid solver is generic
  // functionality implemented in the base class.
  this->update_smoothers();
  this->update_coarse_solver();

  this->update_needed = false;
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::fill_matrix_free_data(
  MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
  unsigned int const                     level,
  unsigned int const                     dealii_tria_level)
{
  matrix_free_data.data.mg_level = dealii_tria_level;

  MappingFlags flags;
  if(data.unsteady_problem)
    matrix_free_data.append_mapping_flags(MassKernel<dim, Number>::get_mapping_flags());
  if(data.convective_problem)
    matrix_free_data.append_mapping_flags(
      Operators::ConvectiveKernel<dim, Number>::get_mapping_flags());
  if(data.diffusive_problem)
    matrix_free_data.append_mapping_flags(
      Operators::DiffusiveKernel<dim, Number>::get_mapping_flags(this->level_info[level].is_dg(),
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

  if(data.convective_problem)
  {
    if(data.convective_kernel_data.velocity_type == TypeVelocityField::Function)
    {
      // do nothing
    }
    else if(data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
    {
      matrix_free_data.insert_dof_handler(&(*dof_handlers_velocity[level]), "velocity_dof_handler");
      matrix_free_data.insert_constraint(&(*constraints_velocity[level]), "velocity_dof_handler");
    }
    else
    {
      AssertThrow(false, dealii::ExcMessage("Not implemented."));
    }
  }
}

template<int dim, typename Number>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::initialize_operator(unsigned int const level)
{
  // initialize pde_operator in a first step
  std::shared_ptr<PDEOperatorMG> pde_operator_level(new PDEOperatorMG());

  // set dof and quad indices after matrix_free_data has been filled
  data.dof_index = this->matrix_free_data_objects[level]->get_dof_index("std_dof_handler");
  if(data.convective_problem and
     data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
  {
    data.convective_kernel_data.dof_index_velocity =
      this->matrix_free_data_objects[level]->get_dof_index("velocity_dof_handler");
  }
  data.quad_index = this->matrix_free_data_objects[level]->get_quad_index("std_quadrature");

  pde_operator_level->initialize(*this->matrix_free_objects[level],
                                 *this->constraints[level],
                                 data);

  // make sure that scaling factor of time derivative term has been set before the smoothers are
  // initialized
  pde_operator_level->set_scaling_factor_mass_operator(
    pde_operator->get_scaling_factor_mass_operator());

  // initialize MGOperator which is a wrapper around the PDEOperatorMG
  std::shared_ptr<MGOperator> mg_operator_level(new MGOperator(pde_operator_level));

  return mg_operator_level;
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::initialize_dof_handler_and_constraints(
  bool const                    operator_is_singular,
  unsigned int const            n_components,
  Map_DBC const &               dirichlet_bc,
  Map_DBC_ComponentMask const & dirichlet_bc_component_mask)
{
  Base::initialize_dof_handler_and_constraints(operator_is_singular,
                                               n_components,
                                               dirichlet_bc,
                                               dirichlet_bc_component_mask);

  if(data.convective_problem and
     data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
  {
    std::shared_ptr<dealii::FiniteElement<dim>> fe_velocity = create_finite_element<dim>(
      get_element_type(*this->grid->triangulation), true, dim, degree_velocity);

    Map_DBC               dirichlet_bc_velocity;
    Map_DBC_ComponentMask dirichlet_bc_velocity_component_mask;
    this->do_initialize_dof_handler_and_constraints(false,
                                                    fe_velocity->n_components(),
                                                    dirichlet_bc_velocity,
                                                    dirichlet_bc_velocity_component_mask,
                                                    dof_handlers_velocity,
                                                    constraints_velocity);
  }
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::initialize_transfer_operators()
{
  Base::initialize_transfer_operators();

  if(data.convective_problem and
     data.convective_kernel_data.velocity_type == TypeVelocityField::DoFVector)
  {
    unsigned int const dof_index = 1;
    this->do_initialize_transfer_operators(transfers_velocity, dof_index);
  }
}

template<int dim, typename Number>
std::shared_ptr<
  CombinedOperator<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::get_operator(unsigned int level) const
{
  std::shared_ptr<MGOperator> mg_operator =
    std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

  return mg_operator->get_pde_operator();
}

template class MultigridPreconditioner<2, float>;
template class MultigridPreconditioner<3, float>;

template class MultigridPreconditioner<2, double>;
template class MultigridPreconditioner<3, double>;

} // namespace ConvDiff
} // namespace ExaDG
