/*
 * GMRESSmoother.h
 *
 *  Created on: Nov 16, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_GMRESSMOOTHER_H_
#define INCLUDE_GMRESSMOOTHER_H_


template<typename Operator, typename VectorType>
class GMRESSmoother
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

  void initialize(Operator &operator_in)
  {
    underlying_operator = &operator_in;
    preconditioner = new JacobiPreconditioner<typename Operator::value_type,Operator>(*underlying_operator);
  }

  void update()
  {
    preconditioner->update(underlying_operator);
  }

  void vmult(VectorType       &dst,
             VectorType const &src) const
  {
    unsigned int max_iter = 5;
    IterationNumberControl control (max_iter,1.e-20,1.e-10);

    typename SolverGMRES<parallel::distributed::Vector<typename Operator::value_type> >::AdditionalData additional_data;
    additional_data.right_preconditioning = true;

    SolverGMRES<VectorType> solver (control,additional_data);

    dst = 0.0;
    bool use_preconditioner = true; // TODO
    if(use_preconditioner == true)
      solver.solve(*underlying_operator,dst,src,*preconditioner);
    else
      solver.solve(*underlying_operator,dst,src,PreconditionIdentity());
  }

private:
  Operator *underlying_operator;
  JacobiPreconditioner<typename Operator::value_type,Operator> *preconditioner;

};


#endif /* INCLUDE_GMRESSMOOTHER_H_ */
