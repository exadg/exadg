#include "mg_coarse_ml.h"

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_control.h>

#ifdef DEAL_II_WITH_TRILINOS

template<typename Operator, typename Number>
MGCoarseML<Operator, Number>::MGCoarseML(Operator const & operator_dg,
                                         Operator const & operator_cg,
                                         bool             setup,
                                         int              level,
                                         MGCoarseMLData   data)
  : operator_dg(operator_dg), operator_cg(operator_cg)
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

  // create wrapper
  if(this->additional_data.transfer_to_continuous_galerkin)
  {
    const unsigned int degree = operator_dg.get_data().get_dof_handler().get_fe().degree;
    this->transfer.reset(new CGToDGTransfer<Operator::DIM, NumberMG>(
      operator_dg.get_data(), operator_cg.get_data(), level, degree));
  }

  // initialize system matrix
  auto & matrix_temp =
    this->additional_data.transfer_to_continuous_galerkin ? operator_cg : operator_dg;
  matrix_temp.init_system_matrix(system_matrix);
  matrix_temp.calculate_system_matrix(system_matrix);

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
  // TODO: remove unnecessary moves...
  VectorTypeMG src_0, dst_0;
  src_0.reinit(src, false);
  src_0.copy_locally_owned_data_from(src);
  dst_0.reinit(dst, false);

  VectorTypeMG  src_cg, dst_cg;
  VectorTypeMG *ptr_src, *ptr_dst;

  // DG (NumberMG) -> CG (NumberMG)
  if(this->additional_data.transfer_to_continuous_galerkin)
  {
    this->operator_cg.initialize_dof_vector(src_cg);
    this->operator_cg.initialize_dof_vector(dst_cg);
    ptr_src = &src_cg;
    ptr_dst = &dst_cg;
    transfer->toCG(*ptr_src, src_0);
  }
  else
  {
    ptr_src = &src_0;
    ptr_dst = &dst_0;
  }

  // create temporal vectors of type TrilinosScalar
  VectorTypeTrilinos dst_trilinos;
  dst_trilinos.reinit(*ptr_dst, false);
  VectorTypeTrilinos src_trilinos;
  src_trilinos.reinit(*ptr_src, false);

  // convert NumberMG to TrilinosScalar
  // (TrilinosScalar is double, we generally use float as NumberMG, so an explicit conversion
  // is needed)
  src_trilinos.copy_locally_owned_data_from(*ptr_src);

  if(additional_data.use_conjugate_gradient_solver)
  {
    // use PCG with Trilinos to perform AMG
    ReductionControl solver_control(additional_data.max_iter,
                                    additional_data.solver_tolerance_abs,
                                    additional_data.solver_tolerance_rel);
    solver_control.set_failure_criterion(additional_data.pcg_failure_criterion);
    SolverCG<VectorTypeTrilinos> solver(solver_control);

    solver.solve(system_matrix, dst_trilinos, src_trilinos, pamg);

    // std::cout << "   " << solver_control.last_step() << "   "
    //          << solver_control.last_value() << "   " << std::endl;
  }
  else
  {
    // use Trilinos to perform AMG
    pamg.vmult(dst_trilinos, src_trilinos);
  }

  // convert TrilinosScalar to NumberMG
  ptr_dst->copy_locally_owned_data_from(dst_trilinos);
  ptr_dst->update_ghost_values();
  // CG (NumberMG) -> DG (NumberMG)
  if(this->additional_data.transfer_to_continuous_galerkin)
    transfer->toDG(dst_0, *ptr_dst);
  dst.copy_locally_owned_data_from(dst_0);
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
                                         Operator const &,
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
