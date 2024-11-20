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

#ifndef INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_

#include <exadg/operators/multigrid_operator.h>
#include <exadg/poisson/spatial_discretization/laplace_operator.h>
#include <exadg/solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h>

namespace ExaDG
{
namespace Poisson
{
/*
 *  Multigrid preconditioner for Laplace operator.
 */
template<int dim, typename Number, int n_components>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number>
{
private:
  typedef MultigridPreconditionerBase<dim, Number> Base;

public:
  typedef typename Base::MultigridNumber MultigridNumber;

private:
  static unsigned int const rank =
    (n_components == 1) ? 0 : ((n_components == dim) ? 1 : dealii::numbers::invalid_unsigned_int);

  typedef typename Base::Map_DBC               Map_DBC;
  typedef typename Base::Map_DBC_ComponentMask Map_DBC_ComponentMask;
  typedef typename Base::PeriodicFacePairs     PeriodicFacePairs;

  typedef LaplaceOperator<dim, MultigridNumber, n_components> Laplace;

  typedef MultigridOperatorBase<dim, MultigridNumber>      MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, Laplace> MGOperator;

public:
  MultigridPreconditioner(MPI_Comm const & mpi_comm);

  void
  initialize(MultigridData const &                                 mg_data,
             std::shared_ptr<Grid<dim> const>                      grid,
             std::shared_ptr<MultigridMappings<dim, Number>> const multigrid_mappings,
             dealii::FiniteElement<dim> const &                    fe,
             LaplaceOperatorData<rank, dim> const &                data,
             bool const                                            mesh_is_moving,
             Map_DBC const &                                       dirichlet_bc,
             Map_DBC_ComponentMask const &                         dirichlet_bc_component_mask);

  void
  update() final;

private:
  void
  fill_matrix_free_data(MatrixFreeData<dim, MultigridNumber> & matrix_free_data,
                        unsigned int const                     level,
                        unsigned int const                     dealii_triangulation_level) final;

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level) final;

  std::shared_ptr<Laplace>
  get_operator(unsigned int level);

  LaplaceOperatorData<rank, dim> data;

  bool is_dg;

  bool mesh_is_moving;
};

} // namespace Poisson
} // namespace ExaDG

#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_ */
