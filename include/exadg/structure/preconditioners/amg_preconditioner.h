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
class PreconditionerAMG : public PreconditionerBase<Number>
{
private:
  typedef double                                          NumberAMG;
  typedef typename PreconditionerBase<Number>::VectorType VectorType;
  typedef LinearAlgebra::distributed::Vector<NumberAMG>   VectorTypeAMG;

public:
  PreconditionerAMG(Operator const & pde_operator, bool const use_boomer_amg, AMGData data)
  {
    if(use_boomer_amg)
    {
      preconditioner_amg =
        std::make_shared<PreconditionerBoomerAMG<Operator, double>>(pde_operator, data);
    }
    else
    {
      preconditioner_amg =
        std::make_shared<PreconditionerTrilinosAMG<Operator, double>>(pde_operator, data);
    }
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    // create temporal vectors of type double
    VectorTypeAMG dst_amg;
    dst_amg.reinit(dst, false);
    VectorTypeAMG src_amg;
    src_amg.reinit(src, true);
    src_amg = src;

    preconditioner_amg->vmult(dst_amg, src_amg);

    // convert: double -> Number
    dst.copy_locally_owned_data_from(dst_amg);
  }

  void
  update()
  {
    preconditioner_amg->update();
  }

private:
  std::shared_ptr<PreconditionerBase<NumberAMG>> preconditioner_amg;
};

} // namespace Structure
} // namespace ExaDG


#endif /* INCLUDE_EXADG_STRUCTURE_PRECONDITIONERS_AMG_PRECONDITIONER_H_ */
