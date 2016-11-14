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
    use_preconditioner(false),
    update_preconditioner(false)
  {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  bool use_preconditioner;
  bool update_preconditioner;
};

template<typename Operator, typename Preconditioner, typename VectorType>
class CGSolver : public IterativeSolverBase<VectorType>
{
public:
  CGSolver(Operator const       &underlying_operator_in,
           Preconditioner       &preconditioner_in,
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
      if(solver_data.update_preconditioner == true)
        preconditioner.update(&underlying_operator);

      solver.solve(underlying_operator, dst, rhs, preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                ExcMessage("Solver contained NaN of Inf values"));

    return solver_control.last_step();
  }

private:
  Operator const &underlying_operator;
  Preconditioner &preconditioner;
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
    update_preconditioner(false),
    right_preconditioning(true),
    max_n_tmp_vectors(30),
    compute_eigenvalues(false)
  {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  bool use_preconditioner;
  bool update_preconditioner;
  bool right_preconditioning;
  unsigned int max_n_tmp_vectors;
  bool compute_eigenvalues;
};

template<typename Operator, typename Preconditioner, typename VectorType>
class GMRESSolver : public IterativeSolverBase<VectorType>
{
public:
  GMRESSolver(Operator const        &underlying_operator_in,
              Preconditioner        &preconditioner_in,
              GMRESSolverData const &solver_data_in)
    :
    underlying_operator(underlying_operator_in),
    preconditioner(preconditioner_in),
    solver_data(solver_data_in)
  {}

  virtual ~GMRESSolver(){}

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
      if(solver_data.update_preconditioner == true)
        preconditioner.update(&this->underlying_operator);

      solver.solve(this->underlying_operator, dst, rhs, this->preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                ExcMessage("Solver contained NaN of Inf values"));

    return solver_control.last_step();
  }

//  virtual void do_solve(SolverGMRES<VectorType> &solver,
//                        VectorType              &dst,
//                        VectorType const        &rhs) const
//  {
//    solver.solve(underlying_operator, dst, rhs, preconditioner);
//  }

protected:
  Operator const & underlying_operator;
  Preconditioner & preconditioner;
  GMRESSolverData const solver_data;
};

struct FGMRESSolverData
{
  FGMRESSolverData()
    :
    max_iter(1e4),
    solver_tolerance_abs(1.e-20),
    solver_tolerance_rel(1.e-6),
    use_preconditioner(false),
    update_preconditioner(false),
    max_n_tmp_vectors(30)
  {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  bool use_preconditioner;
  bool update_preconditioner;
  unsigned int max_n_tmp_vectors;
};

template<typename Operator, typename Preconditioner, typename VectorType>
class FGMRESSolver : public IterativeSolverBase<VectorType>
{
public:
  FGMRESSolver(Operator const         &underlying_operator_in,
               Preconditioner         &preconditioner_in,
               FGMRESSolverData const &solver_data_in)
    :
    underlying_operator(underlying_operator_in),
    preconditioner(preconditioner_in),
    solver_data(solver_data_in)
  {}

  virtual ~FGMRESSolver(){}

  unsigned int solve(VectorType       &dst,
                     VectorType const &rhs) const
  {
    ReductionControl solver_control (solver_data.max_iter,
                                     solver_data.solver_tolerance_abs,
                                     solver_data.solver_tolerance_rel);

    typename SolverFGMRES<VectorType>::AdditionalData additional_data;
    additional_data.max_basis_size = solver_data.max_n_tmp_vectors;
    // FGMRES always uses right preconditioning

    SolverFGMRES<VectorType> solver (solver_control,additional_data);

    if(solver_data.use_preconditioner == false)
    {
      solver.solve(underlying_operator, dst, rhs, PreconditionIdentity());
    }
    else
    {
      if(solver_data.update_preconditioner == true)
        preconditioner.update(&underlying_operator);

      solver.solve(underlying_operator, dst, rhs, preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                ExcMessage("Solver contained NaN of Inf values"));

    return solver_control.last_step();
  }

//  virtual void do_solve(SolverFGMRES<VectorType> &solver,
//                        VectorType               &dst,
//                        VectorType const         &rhs) const
//  {
//    solver.solve(underlying_operator, dst, rhs, preconditioner);
//  }

protected:
  Operator const & underlying_operator;
  Preconditioner & preconditioner;
  FGMRESSolverData const solver_data;
};

//template<typename Operator, typename Preconditioner, typename VectorType>
//class GMRESSolverNavierStokes : public GMRESSolver<Operator,Preconditioner,VectorType>
//{
//public:
//  GMRESSolverNavierStokes(Operator const        &underlying_operator_in,
//                          Preconditioner        &preconditioner_in,
//                          GMRESSolverData const &solver_data_in)
//    :
//    GMRESSolver<Operator,Preconditioner,VectorType>(underlying_operator_in,preconditioner_in,solver_data_in)
//  {}
//
//  void do_solve(SolverGMRES<VectorType> &solver,
//                VectorType              &dst,
//                VectorType const        &rhs) const
//  {
//    this->preconditioner.update(&this->underlying_operator);
//    solver.solve(this->underlying_operator, dst, rhs, this->preconditioner);
//  }
//};
//
//
//template<typename Operator, typename Preconditioner, typename VectorType>
//class FGMRESSolverNavierStokes : public FGMRESSolver<Operator,Preconditioner,VectorType>
//{
//public:
//  FGMRESSolverNavierStokes(Operator const         &underlying_operator_in,
//                           Preconditioner         &preconditioner_in,
//                           FGMRESSolverData const &solver_data_in)
//    :
//    FGMRESSolver<Operator,Preconditioner,VectorType>(underlying_operator_in,preconditioner_in,solver_data_in)
//  {}
//
//  void do_solve(SolverFGMRES<VectorType> &solver,
//                VectorType               &dst,
//                VectorType const         &rhs) const
//  {
//    this->preconditioner.update(&this->underlying_operator);
//    solver.solve(this->underlying_operator, dst, rhs, this->preconditioner);
//  }
//};


#endif /* INCLUDE_ITERATIVESOLVERS_H_ */
