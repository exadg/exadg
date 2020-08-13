/*
 * amg_preconditioner.h
 *
 *  Created on: 17.04.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_PRECONDITIONERS_AMG_PRECONDITIONER_H_
#define INCLUDE_STRUCTURE_PRECONDITIONERS_AMG_PRECONDITIONER_H_

#include "../../solvers_and_preconditioners/preconditioner/preconditioner_amg.h"
#include "../../solvers_and_preconditioners/preconditioner/preconditioner_base.h"

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


#endif /* INCLUDE_STRUCTURE_PRECONDITIONERS_AMG_PRECONDITIONER_H_ */
