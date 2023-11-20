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

#include <exadg/grid/grid_data.h>
#include <exadg/operators/quadrature.h>
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
  MultigridData const &                       mg_data,
  std::shared_ptr<Grid<dim> const>            grid,
  std::shared_ptr<dealii::Mapping<dim> const> mapping,
  dealii::FiniteElement<dim> const &          fe,
  ElasticityOperatorBase<dim, Number> const & pde_operator_in,
  bool const                                  nonlinear_in,
  Map_DBC const &                             dirichlet_bc,
  Map_DBC_ComponentMask const &               dirichlet_bc_component_mask)
{
  pde_operator = &pde_operator_in;

  data = this->pde_operator->get_data();

  nonlinear = nonlinear_in;

  Base::initialize(mg_data,
                   grid,
                   mapping,
                   fe,
                   false /*operator_is_singular*/,
                   dirichlet_bc,
                   dirichlet_bc_component_mask,
                   false /* initialize_preconditioners */);
}

template<int dim, typename Number>
void
MultigridPreconditioner<dim, Number>::update()
{
  // update operators for all levels
  if(data.unsteady)
  {
    this->for_all_levels([&](unsigned int const level) {
      if(nonlinear)
      {
        this->get_operator_nonlinear(level)->set_time(pde_operator->get_time());
        this->get_operator_nonlinear(level)->set_scaling_factor_mass_operator(
          pde_operator->get_scaling_factor_mass_operator());
      }
      else
      {
        this->get_operator_linear(level)->set_time(pde_operator->get_time());
        this->get_operator_linear(level)->set_scaling_factor_mass_operator(
          pde_operator->get_scaling_factor_mass_operator());
      }
    });
  }

  if(nonlinear)
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

    // copy velocity to finest level
    this->get_operator_nonlinear(this->get_number_of_levels() - 1)
      ->set_solution_linearization(*vector_multigrid_type_ptr);

    // interpolate velocity from fine to coarse level
    this->transfer_from_fine_to_coarse_levels(
      [&](unsigned int const fine_level, unsigned int const coarse_level) {
        auto const & vector_fine_level =
          this->get_operator_nonlinear(fine_level)->get_solution_linearization();
        auto vector_coarse_level =
          this->get_operator_nonlinear(coarse_level)->get_solution_linearization();
        this->transfers->interpolate(fine_level, vector_coarse_level, vector_fine_level);
        this->get_operator_nonlinear(coarse_level)->set_solution_linearization(vector_coarse_level);
      });
  }

  // In case that the operators have been updated, we also need to update the smoothers and the
  // coarse grid solver. This is generic functionality implemented in the base class.
  if(nonlinear or data.unsteady)
  {
    this->update_smoothers();
    this->update_coarse_solver();
  }

  this->update_needed = false;
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

  // additional constraints without Dirichlet degrees of freedom:
  // Due to the interface of the base class MultigridPreconditionerBase, we also need to set up
  // additional DoFHandler objects.
  if(nonlinear)
  {
    Map_DBC               dirichlet_bc_empty;
    Map_DBC_ComponentMask dirichlet_bc_empty_component_mask;
    this->do_initialize_dof_handler_and_constraints(false,
                                                    n_components,
                                                    dirichlet_bc_empty,
                                                    dirichlet_bc_empty_component_mask,
                                                    dof_handlers_inhomogeneous,
                                                    constraints_inhomogeneous);
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

  if(nonlinear)
    matrix_free_data.append_mapping_flags(PDEOperatorNonlinear::get_mapping_flags());
  else // linear
    matrix_free_data.append_mapping_flags(PDEOperatorLinear::get_mapping_flags());

  matrix_free_data.insert_dof_handler(&(*this->dof_handlers[level]), "elasticity_dof_index");
  matrix_free_data.insert_constraint(&(*this->constraints[level]), "elasticity_dof_index");

  // additional constraints without Dirichlet degrees of freedom
  if(nonlinear)
  {
    matrix_free_data.insert_dof_handler(&(*dof_handlers_inhomogeneous[level]),
                                        "elasticity_dof_index_inhomogeneous");
    matrix_free_data.insert_constraint(&(*constraints_inhomogeneous[level]),
                                       "elasticity_dof_index_inhomogeneous");
  }

  ElementType const element_type = get_element_type(*this->grid->triangulation);
  std::shared_ptr<dealii::Quadrature<dim>> quadrature =
    create_quadrature<dim>(element_type, this->level_info[level].degree() + 1);
  matrix_free_data.insert_quadrature(*quadrature, "elasticity_quadrature");
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
  LinearOperator<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::get_operator_linear(unsigned int level)
{
  std::shared_ptr<MGOperatorLinear> mg_operator =
    std::dynamic_pointer_cast<MGOperatorLinear>(this->operators[level]);

  return mg_operator->get_pde_operator();
}

template<int dim, typename Number>
std::shared_ptr<
  MultigridOperatorBase<dim, typename MultigridPreconditionerBase<dim, Number>::MultigridNumber>>
MultigridPreconditioner<dim, Number>::initialize_operator(unsigned int const level)
{
  std::shared_ptr<MGOperatorBase> mg_operator_level;

  data.dof_index = this->matrix_free_data_objects[level]->get_dof_index("elasticity_dof_index");
  if(nonlinear)
  {
    data.dof_index_inhomogeneous =
      this->matrix_free_data_objects[level]->get_dof_index("elasticity_dof_index_inhomogeneous");
  }
  data.quad_index = this->matrix_free_data_objects[level]->get_quad_index("elasticity_quadrature");

  if(nonlinear)
  {
    std::shared_ptr<PDEOperatorNonlinearMG> pde_operator_level(new PDEOperatorNonlinearMG());

    if(data.unsteady)
    {
      pde_operator_level->set_time(pde_operator->get_time());
      pde_operator_level->set_scaling_factor_mass_operator(
        pde_operator->get_scaling_factor_mass_operator());
    }

    pde_operator_level->initialize(*this->matrix_free_objects[level],
                                   *this->constraints[level],
                                   data);

    mg_operator_level = std::make_shared<MGOperatorNonlinear>(pde_operator_level);
  }
  else // linear
  {
    std::shared_ptr<PDEOperatorLinearMG> pde_operator_level(new PDEOperatorLinearMG());

    if(data.unsteady)
    {
      pde_operator_level->set_time(pde_operator->get_time());
      pde_operator_level->set_scaling_factor_mass_operator(
        pde_operator->get_scaling_factor_mass_operator());
    }

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
