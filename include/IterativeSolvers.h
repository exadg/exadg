/*
 * IterativeSolvers.h
 *
 *  Created on: Aug 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_ITERATIVESOLVERS_H_
#define INCLUDE_ITERATIVESOLVERS_H_

template<typename VectorType>
class IterativeSolverBase
{
public:
  virtual unsigned int solve(VectorType       &dst,
                             VectorType const &rhs) const = 0;

  virtual ~IterativeSolverBase(){}
};

struct CGSolverData
{
  CGSolverData()
    :
    max_iter(1e4),
    solver_tolerance_abs(1.e-20),
    solver_tolerance_rel(1.e-6),
    use_preconditioner(false)
  {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  bool use_preconditioner;
};

template<typename Operator, typename Preconditioner, typename VectorType>
class CGSolver : public IterativeSolverBase<VectorType>
{
public:
  CGSolver(Operator const       &underlying_operator_in,
           Preconditioner const &preconditioner_in,
           CGSolverData const   &solver_data_in)
    :
    underlying_operator(underlying_operator_in),
    preconditioner(preconditioner_in),
    solver_data(solver_data_in)
  {}

  unsigned int solve(VectorType       &dst,
                     VectorType const &rhs) const
  {
    ReductionControl solver_control (solver_data.max_iter, solver_data.solver_tolerance_abs, solver_data.solver_tolerance_rel);
    SolverCG<VectorType> solver (solver_control);

    if(solver_data.use_preconditioner == false)
    {
      solver.solve(underlying_operator, dst, rhs, PreconditionIdentity());
    }
    else
    {
      solver.solve(underlying_operator, dst, rhs, preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                ExcMessage("Solver contained NaN of Inf values"));

    return solver_control.last_step();
  }

private:
  Operator const &underlying_operator;
  Preconditioner const &preconditioner;
  CGSolverData const solver_data;
};

template<class NUMBER>
void output_eigenvalues(const std::vector<NUMBER> &eigenvalues,const std::string &text)
{
//    deallog << text << std::endl;
//    for (unsigned int j = 0; j < eigenvalues.size(); ++j)
//      {
//        deallog << ' ' << eigenvalues.at(j) << std::endl;
//      }
//    deallog << std::endl;

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << text << std::endl;
    for (unsigned int j = 0; j < eigenvalues.size(); ++j)
    {
      std::cout << ' ' << eigenvalues.at(j) << std::endl;
    }
    std::cout << std::endl;
  }
}

struct GMRESSolverData
{
  GMRESSolverData()
    :
    max_iter(1e4),
    solver_tolerance_abs(1.e-20),
    solver_tolerance_rel(1.e-6),
    use_preconditioner(false),
    right_preconditioning(true),
    max_n_tmp_vectors(30),
    compute_eigenvalues(false)
  {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  bool use_preconditioner;
  bool right_preconditioning;
  unsigned int max_n_tmp_vectors;
  bool compute_eigenvalues;
};

template<typename Operator, typename Preconditioner, typename VectorType>
class GMRESSolver : public IterativeSolverBase<VectorType>
{
public:
  GMRESSolver(Operator const        &underlying_operator_in,
              Preconditioner const  &preconditioner_in,
              GMRESSolverData const &solver_data_in)
    :
    underlying_operator(underlying_operator_in),
    preconditioner(preconditioner_in),
    solver_data(solver_data_in)
  {}

  unsigned int solve(VectorType       &dst,
                     VectorType const &rhs) const
  {
    ReductionControl solver_control (solver_data.max_iter,
                                     solver_data.solver_tolerance_abs,
                                     solver_data.solver_tolerance_rel);

    typename SolverGMRES<VectorType>::AdditionalData additional_data;
    additional_data.max_n_tmp_vectors = solver_data.max_n_tmp_vectors;
    additional_data.right_preconditioning = solver_data.right_preconditioning;
    SolverGMRES<VectorType> solver (solver_control, additional_data);

    if(solver_data.compute_eigenvalues == true)
    {
      solver.connect_eigenvalues_slot(std_cxx11::bind(
          output_eigenvalues<std::complex<double> >,std_cxx11::_1,"Eigenvalues: "),true);
    }

    if(solver_data.use_preconditioner == false)
    {
      solver.solve(underlying_operator, dst, rhs, PreconditionIdentity());
    }
    else
    {
      solver.solve(underlying_operator, dst, rhs, preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                ExcMessage("Solver contained NaN of Inf values"));

    return solver_control.last_step();
  }

private:
  Operator const & underlying_operator;
  Preconditioner const & preconditioner;
  GMRESSolverData const solver_data;
};

struct FGMRESSolverData
{
  FGMRESSolverData()
    :
    max_iter(1e4),
    solver_tolerance_abs(1.e-20),
    solver_tolerance_rel(1.e-6),
    use_preconditioner(false)
  {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  bool use_preconditioner;
};

template<typename Operator, typename Preconditioner, typename VectorType>
class FGMRESSolver : public IterativeSolverBase<VectorType>
{
public:
  FGMRESSolver(Operator const         &underlying_operator_in,
               Preconditioner const   &preconditioner_in,
               FGMRESSolverData const &solver_data_in)
    :
    underlying_operator(underlying_operator_in),
    preconditioner(preconditioner_in),
    solver_data(solver_data_in)
  {}

  unsigned int solve(VectorType       &dst,
                     VectorType const &rhs) const
  {
    ReductionControl solver_control (solver_data.max_iter,
                                     solver_data.solver_tolerance_abs,
                                     solver_data.solver_tolerance_rel);

    SolverFGMRES<VectorType> solver (solver_control);

    if(solver_data.use_preconditioner == false)
    {
      solver.solve(underlying_operator, dst, rhs, PreconditionIdentity());
    }
    else
    {
      solver.solve(underlying_operator, dst, rhs, preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                ExcMessage("Solver contained NaN of Inf values"));

    return solver_control.last_step();
  }

private:
  Operator const & underlying_operator;
  Preconditioner const & preconditioner;
  FGMRESSolverData const solver_data;
};


#endif /* INCLUDE_ITERATIVESOLVERS_H_ */
