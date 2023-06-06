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

#ifndef INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_

#include <exadg/convection_diffusion/spatial_discretization/operators/combined_operator.h>
#include <exadg/operators/multigrid_operator.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h>

namespace ExaDG
{
namespace ConvDiff
{
/*
 *  Multigrid preconditioner for scalar convection-diffusion equation.
 */
template<int dim, typename Number>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number>
{
private:
  typedef MultigridPreconditionerBase<dim, Number> Base;

public:
  typedef typename Base::MultigridNumber MultigridNumber;

private:
  typedef CombinedOperator<dim, Number>          PDEOperator;
  typedef CombinedOperator<dim, MultigridNumber> PDEOperatorMG;

  typedef MultigridOperatorBase<dim, MultigridNumber>            MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorMG> MGOperator;

  typedef typename Base::Map_DBC               Map_DBC;
  typedef typename Base::Map_DBC_ComponentMask Map_DBC_ComponentMask;
  typedef typename Base::PeriodicFacePairs     PeriodicFacePairs;
  typedef typename Base::VectorType            VectorType;
  typedef typename Base::VectorTypeMG          VectorTypeMG;

public:
  MultigridPreconditioner(MPI_Comm const & mpi_comm);

  virtual ~MultigridPreconditioner(){};

  /**
   *  This function initializes the multigrid preconditioner.
   */
  void
  initialize(MultigridData const &                       mg_data,
             MultigridVariant const &                    multigrid_variant,
             std::shared_ptr<Grid<dim> const>            grid,
             std::shared_ptr<dealii::Mapping<dim> const> mapping,
             dealii::FiniteElement<dim> const &          fe,
             PDEOperator const &                         pde_operator,
             MultigridOperatorType const &               mg_operator_type,
             bool const                                  mesh_is_moving,
             Map_DBC const &                             dirichlet_bc,
             Map_DBC_ComponentMask const &               dirichlet_bc_component_mask);

  /**
   *  This function updates the multigrid preconditioner.
   */
  void
  update() final;

private:
  void
  fill_matrix_free_data(MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
                        unsigned int const                     level,
                        unsigned int const                     h_level) final;

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level) final;

  void
  initialize_dof_handler_and_constraints(
    bool const                         operator_is_singular,
    dealii::FiniteElement<dim> const & fe,
    Map_DBC const &                    dirichlet_bc,
    Map_DBC_ComponentMask const &      dirichlet_bc_component_mask) final;

  void
  initialize_transfer_operators() final;

  std::shared_ptr<PDEOperatorMG>
  get_operator(unsigned int level) const;

  std::shared_ptr<MGTransfer<VectorTypeMG>> transfers_velocity;

  dealii::MGLevelObject<std::shared_ptr<dealii::DoFHandler<dim> const>> dof_handlers_velocity;
  dealii::MGLevelObject<std::shared_ptr<dealii::MGConstrainedDoFs>>     constrained_dofs_velocity;
  dealii::MGLevelObject<std::shared_ptr<dealii::AffineConstraints<MultigridNumber>>>
    constraints_velocity;

  CombinedOperatorData<dim> data;

  PDEOperator const * pde_operator;

  MultigridOperatorType mg_operator_type;

  bool mesh_is_moving;
};

} // namespace ConvDiff
} // namespace ExaDG

#endif /* INCLUDE_CONVECTION_DIFFUSION_MULTIGRID_PRECONDITIONER_H_ */
