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

#ifndef INCLUDE_EXADG_STRUCTURE_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_EXADG_STRUCTURE_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_

#include <exadg/operators/multigrid_operator.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h>
#include <exadg/structure/spatial_discretization/operators/elasticity_operator_base.h>
#include <exadg/structure/spatial_discretization/operators/linear_operator.h>
#include <exadg/structure/spatial_discretization/operators/nonlinear_operator.h>

namespace ExaDG
{
namespace Structure
{
template<int dim, typename Number>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number>
{
private:
  typedef MultigridPreconditionerBase<dim, Number> Base;

public:
  typedef typename Base::MultigridNumber MultigridNumber;

public:
  typedef LinearOperator<dim, Number>          PDEOperatorLinear;
  typedef LinearOperator<dim, MultigridNumber> PDEOperatorLinearMG;

  typedef NonLinearOperator<dim, Number>          PDEOperatorNonlinear;
  typedef NonLinearOperator<dim, MultigridNumber> PDEOperatorNonlinearMG;

  typedef MultigridOperatorBase<dim, MultigridNumber>                     MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorLinearMG>    MGOperatorLinear;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorNonlinearMG> MGOperatorNonlinear;

  typedef typename Base::Map_DBC               Map_DBC;
  typedef typename Base::Map_DBC_ComponentMask Map_DBC_ComponentMask;
  typedef typename Base::PeriodicFacePairs     PeriodicFacePairs;
  typedef typename Base::VectorType            VectorType;
  typedef typename Base::VectorTypeMG          VectorTypeMG;

  MultigridPreconditioner(MPI_Comm const & mpi_comm);

  void
  initialize(MultigridData const &                       mg_data,
             std::shared_ptr<Grid<dim> const>            grid,
             std::shared_ptr<dealii::Mapping<dim> const> mapping,
             dealii::FiniteElement<dim> const &          fe,
             ElasticityOperatorBase<dim, Number> const & pde_operator,
             bool const                                  nonlinear_operator,
             Map_DBC const &                             dirichlet_bc,
             Map_DBC_ComponentMask const &               dirichlet_bc_component_mask);

  /*
   * This function updates the multigrid preconditioner.
   */
  void
  update() final;

private:
  void
  initialize_dof_handler_and_constraints(
    bool const                    operator_is_singular,
    unsigned int const            n_components,
    Map_DBC const &               dirichlet_bc,
    Map_DBC_ComponentMask const & dirichlet_bc_component_mask) final;

  void
  fill_matrix_free_data(MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
                        unsigned int const                     level,
                        unsigned int const                     dealii_tria_level) final;

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level) final;

  /*
   * This function updates the multigrid operators for all levels
   */
  void
  update_operators();

  std::shared_ptr<PDEOperatorNonlinearMG>
  get_operator_nonlinear(unsigned int level);

  std::shared_ptr<PDEOperatorLinearMG>
  get_operator_linear(unsigned int level);

private:
  OperatorData<dim> data;

  ElasticityOperatorBase<dim, Number> const * pde_operator;

  // additional constraints without Dirichlet degrees of freedom
  dealii::MGLevelObject<std::shared_ptr<dealii::DoFHandler<dim> const>> dof_handlers_inhomogeneous;
  dealii::MGLevelObject<std::shared_ptr<dealii::AffineConstraints<MultigridNumber>>>
    constraints_inhomogeneous;

  bool nonlinear;
};

} // namespace Structure
} // namespace ExaDG

#endif
