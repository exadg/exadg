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
template<typename Operator, typename Number>
class PreconditionerAMG : public PreconditionerBase<Number>
{
private:
  typedef double                                                NumberAMG;
  typedef typename PreconditionerBase<Number>::VectorType       VectorType;
  typedef dealii::LinearAlgebra::distributed::Vector<NumberAMG> VectorTypeAMG;

public:
  PreconditionerAMG(Operator const & pde_operator, AMGData const & data)
  {
    (void)pde_operator;
    (void)data;

    if(data.amg_type == AMGType::BoomerAMG)
    {
#ifdef DEAL_II_WITH_PETSC
      preconditioner_amg =
        std::make_shared<PreconditionerBoomerAMG<Operator, double>>(pde_operator, data.boomer_data);
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with PETSc!"));
#endif
    }
    else if(data.amg_type == AMGType::ML)
    {
#ifdef DEAL_II_WITH_TRILINOS
      preconditioner_amg =
        std::make_shared<PreconditionerML<Operator, double>>(pde_operator, data.ml_data);
#else
      AssertThrow(false, dealii::ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
    }
    else
    {
      AssertThrow(false, dealii::ExcNotImplemented());
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
