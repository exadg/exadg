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

#ifndef PRECONDITIONER_AMG
#define PRECONDITIONER_AMG

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <exadg/solvers_and_preconditioners/multigrid/multigrid_input_parameters.h>
#include <exadg/solvers_and_preconditioners/preconditioner/preconditioner_base.h>
#include <exadg/utilities/print_functions.h>

namespace ExaDG
{
using namespace dealii;

template<typename Operator, typename TrilinosNumber>
class PreconditionerAMG : public PreconditionerBase<TrilinosNumber>
{
private:
  typedef LinearAlgebra::distributed::Vector<TrilinosNumber> VectorTypeTrilinos;

#ifdef DEAL_II_WITH_TRILINOS
public:
  // distributed sparse system matrix
  TrilinosWrappers::SparseMatrix system_matrix;

private:
  TrilinosWrappers::PreconditionAMG amg;
#endif

public:
  PreconditionerAMG(Operator const & op, AMGData data = AMGData())
    : pde_operator(op), amg_data(data)
  {
#ifdef DEAL_II_WITH_TRILINOS
    // initialize system matrix
    pde_operator.init_system_matrix(system_matrix);

    // calculate_matrix
    pde_operator.calculate_system_matrix(system_matrix);

    // initialize Trilinos' AMG
    amg.initialize(system_matrix, amg_data.data);
#else
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
  }

#ifdef DEAL_II_WITH_TRILINOS
  TrilinosWrappers::SparseMatrix const &
  get_system_matrix()
  {
    return system_matrix;
  }
#endif

  void
  update()
  {
#ifdef DEAL_II_WITH_TRILINOS
    // clear content of matrix since the next calculate_system_matrix-commands add their result
    system_matrix *= 0.0;

    // re-calculate matrix
    pde_operator.calculate_system_matrix(system_matrix);

    // initialize Trilinos' AMG
    amg.initialize(system_matrix, amg_data.data);
#else
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
  }

  void
  vmult(VectorTypeTrilinos & dst, VectorTypeTrilinos const & src) const
  {
#ifdef DEAL_II_WITH_TRILINOS
    amg.vmult(dst, src);
#else
    (void)dst;
    (void)src;
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
  }

private:
  // reference to matrix-free operator
  Operator const & pde_operator;

  AMGData amg_data;
};
} // namespace ExaDG

#endif
