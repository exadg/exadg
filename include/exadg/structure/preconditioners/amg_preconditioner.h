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

#ifndef INCLUDE_EXADG_STRUCTURE_PRECONDITIONERS_AMG_PRECONDITIONER_H_
#define INCLUDE_EXADG_STRUCTURE_PRECONDITIONERS_AMG_PRECONDITIONER_H_

#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_amg.h>
#include <exadg/solvers_and_preconditioners/preconditioners/preconditioner_base.h>

namespace ExaDG
{
namespace Structure
{
using namespace dealii;

template<typename Operator, typename Number>
class AlgebraicMultigridPreconditioner : public PreconditionerBase<Number>
{
private:
  typedef typename PreconditionerBase<Number>::VectorType VectorType;
  typedef LinearAlgebra::distributed::Vector<double>      VectorTypeTrilinos;

public:
  AlgebraicMultigridPreconditioner(Operator const & pde_operator, AMGData data)
    : preconditioner_amg(pde_operator, data)
  {
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    // create temporal vectors of type double
    VectorTypeTrilinos dst_trilinos;
    dst_trilinos.reinit(dst, false);
    VectorTypeTrilinos src_trilinos;
    src_trilinos.reinit(src, true);
    src_trilinos = src;

    preconditioner_amg.vmult(dst_trilinos, src_trilinos);

    // convert: double -> Number
    dst.copy_locally_owned_data_from(dst_trilinos);
  }

  void
  update()
  {
    preconditioner_amg.update();
  }

private:
  PreconditionerAMG<Operator, double> preconditioner_amg;
};

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_PRECONDITIONERS_AMG_PRECONDITIONER_H_ */
