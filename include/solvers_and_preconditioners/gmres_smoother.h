/*
 * GMRESSmoother.h
 *
 *  Created on: Nov 16, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_GMRESSMOOTHER_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_GMRESSMOOTHER_H_

#include <deal.II/lac/solver_gmres.h>

#include "smoother_base.h"
#include "solvers_and_preconditioners/multigrid_input_parameters.h"
#include "solvers_and_preconditioners/jacobi_preconditioner.h"
#include "solvers_and_preconditioners/block_jacobi_preconditioner.h"

template<int dim, typename Operator, typename VectorType>
class GMRESSmoother : public SmootherBase<VectorType>
{
public:
  GMRESSmoother()
    :
    underlying_operator(nullptr),
    preconditioner(nullptr)
  {}

  ~GMRESSmoother()
  {
    delete preconditioner;
    preconditioner = nullptr;
  }

  GMRESSmoother(GMRESSmoother const &) = delete;
  GMRESSmoother & operator=(GMRESSmoother const &) = delete;

  struct AdditionalData
  {
    /**
     * Constructor.
     */
    AdditionalData()
     :
     preconditioner(PreconditionerGMRESSmoother::None),
     number_of_iterations(5)
    {}

    // preconditioner
    PreconditionerGMRESSmoother preconditioner;

    // number of GMRES iterations per smoothing step
    unsigned int number_of_iterations;
  };

  void initialize(Operator &operator_in, AdditionalData const &additional_data_in)
  {
    underlying_operator = &operator_in;
    data = additional_data_in;

    if(data.preconditioner == PreconditionerGMRESSmoother::PointJacobi)
    {
      preconditioner = new JacobiPreconditioner<typename Operator::value_type,Operator>(*underlying_operator);
    }
    else if(data.preconditioner == PreconditionerGMRESSmoother::BlockJacobi)
    {
      preconditioner = new BlockJacobiPreconditioner<dim,typename Operator::value_type,Operator>(*underlying_operator);
    }
    else
    {
      AssertThrow(data.preconditioner == PreconditionerGMRESSmoother::None,
          ExcMessage("Specified preconditioner not implemented for GMRES smoother"));
    }
  }

  void update()
  {
    if(preconditioner != nullptr)
      preconditioner->update(underlying_operator);
  }

  void vmult(VectorType       &dst,
             VectorType const &src) const
  {
    IterationNumberControl control (data.number_of_iterations,1.e-20,1.e-10);

    typename SolverGMRES<VectorType>::AdditionalData additional_data;
    additional_data.right_preconditioning = true;
    SolverGMRES<VectorType> solver (control,additional_data);

    dst = 0.0;
    if(preconditioner != nullptr)
      solver.solve(*underlying_operator,dst,src,*preconditioner);
    else
      solver.solve(*underlying_operator,dst,src,PreconditionIdentity());
  }

private:
  Operator *underlying_operator;
  AdditionalData data;

  PreconditionerBase<typename Operator::value_type> *preconditioner;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_GMRESSMOOTHER_H_ */
