#include "mg_coarse_ml.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#ifdef DEAL_II_WITH_TRILINOS

template<typename Operator, typename Number>
MGCoarseML<Operator, Number>::MGCoarseML(Operator const & operator_dg,
                                         bool             setup,
                                         int              level,
                                         MGCoarseMLData   data)
  : operator_dg(operator_dg)
{
  if(setup)
    this->reinit(level, data);
}

template<typename Operator, typename Number>
MGCoarseML<Operator, Number>::~MGCoarseML()
{
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::reinit(int level, MGCoarseMLData data_in)
{
  AssertThrow(level >= 0, ExcMessage("Invalid level specified!"));

  // save additional_data locally: we need it later
  this->additional_data = data_in;

  // initialize system matrix
  operator_dg.init_system_matrix(system_matrix);
  operator_dg.calculate_system_matrix(system_matrix);

  // intialize Trilinos' AMG
  pamg.initialize(system_matrix, this->additional_data.amg_data);
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::update(LinearOperatorBase const * /*linear_operator*/)
{
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::
operator()(unsigned int const /*level*/, VectorTypeMG & dst, VectorTypeMG const & src) const
{
  // create temporal vectors of type TrilinosScalar (unfortunately: double)
  VectorTypeTrilinos dst_ml;
  dst_ml.reinit(dst, false);
  VectorTypeTrilinos src_ml;
  src_ml.reinit(src, true);

  // convert: MultigridNumber -> TrilinosScalar
  src_ml.copy_locally_owned_data_from(src);

  if(additional_data.use_conjugate_gradient_solver)
  {
    // use PCG with Trilinos to perform AMG
    ReductionControl solver_control(additional_data.max_iter,
                                    additional_data.solver_tolerance_abs,
                                    additional_data.solver_tolerance_rel);
    solver_control.set_failure_criterion(additional_data.pcg_failure_criterion);
    SolverCG<VectorTypeTrilinos> solver(solver_control);
    solver.solve(system_matrix, dst_ml, src_ml, pamg);
    // std::cout << "   " << solver_control.last_step() << "   "
    //          << solver_control.last_value() << "   " << std::endl;
  }
  else
  {
    // use Trilinos to perform AMG
    pamg.vmult(dst_ml, src_ml);
  }

  // convert: TrilinosScalar -> MultigridNumber
  dst.copy_locally_owned_data_from(dst_ml);
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::vmult(VectorType &, VectorType const &) const
{
  AssertThrow(false, ExcMessage("MGCoarseML::vmult not implemented yet!"));
}

#else


template<typename Operator, typename Number>
MGCoarseML<Operator, Number>::MGCoarseML(Operator const &,
                                         bool,
                                         int,
                                         MGCoarseMLData)
{
}

template<typename Operator, typename Number>
MGCoarseML<Operator, Number>::~MGCoarseML()
{
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::reinit(int, MGCoarseMLData)
{
  AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::update(LinearOperatorBase const *)
{
  AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::
operator()(unsigned int const, VectorTypeMG &, VectorTypeMG const &) const
{
  AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

template<typename Operator, typename Number>
void
MGCoarseML<Operator, Number>::vmult(VectorType &, VectorType const &) const
{
  AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
}

#endif

#include "mg_coarse_ml.hpp"
