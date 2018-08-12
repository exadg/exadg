/*
 * IterativeSolvers.h
 *
 *  Created on: Aug 9, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_SOLVERS_AND_PRECONDITIONERS_ITERATIVESOLVERS_H_
#define INCLUDE_SOLVERS_AND_PRECONDITIONERS_ITERATIVESOLVERS_H_

#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include "solver_cg_wrapper.h"

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
    solver_control.enable_history_data();
    
    SolverCGWrapper<Operator::DIM, VectorType> solver (solver_control);

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
    
    solver.print_vectors(0,dst,dst,dst);
    
//    auto & conv = solver_control.get_history_data();
//    auto & l2 = solver.get_history_data();
//    
//    printf("\n");
//    printf("---------------------------------------------------------------\n");
//    double rho = std::pow(conv.back()/conv.front(), 1.0/(conv.size()-1));
//    double r   = -log(rho)/std::log(10.0);
//    int n10    = std::ceil(-10.0*std::log(10.0)/log(rho));
//    printf("rho=%.3e r=%.3e n10=%d \n", rho, r, n10);
//    printf("---------------------------------------------------------------\n");
//    
//    for(unsigned int i = 0; i < conv.size(); i++)
//        printf("%3d %.10e %.10e\n",i, conv[i], l2[i]);
//    printf("---------------------------------------------------------------\n");
//    printf("\n\n");

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
      solver.connect_eigenvalues_slot(std::bind(
        output_eigenvalues<std::complex<double> >,std::placeholders::_1,"Eigenvalues: "),true);
    }

    if(solver_data.use_preconditioner == false)
    {
      solver.solve(underlying_operator, dst, rhs, PreconditionIdentity());
    }
    else
    {
      if(solver_data.update_preconditioner == true)
      {
        preconditioner.update(&this->underlying_operator);
      }

      solver.solve(this->underlying_operator, dst, rhs, this->preconditioner);
    }

    AssertThrow(std::isfinite(solver_control.last_value()),
                ExcMessage("Solver contained NaN of Inf values"));

    return solver_control.last_step();
  }

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

protected:
  Operator const & underlying_operator;
  Preconditioner & preconditioner;
  FGMRESSolverData const solver_data;
};

struct GMGSolverData
{
  GMGSolverData()
    :
    max_iter(1e4),
    solver_tolerance_abs(1.e-20),
    solver_tolerance_rel(1.e-6)
  {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
};

template <int dim>
class SolutionMode : public Function<dim> {
public:

    SolutionMode(const double wave_number = 1.0) : Function<dim>(),
    wave_number(wave_number) {
    }

    virtual double value(const Point<dim> &p,
            const unsigned int ) const {
        double temp = 1;
        for (int i = 0; i < dim; i++)
            temp *= std::cos(p[i] * wave_number);
        return temp;
    }

    const double wave_number;
};

template<int dim, typename Operator, typename Preconditioner, typename VectorType>
class GMGSolver : public IterativeSolverBase<VectorType>
{
public:
  GMGSolver(Operator         const &underlying_operator_in,
            Preconditioner         &preconditioner_in,
            GMGSolverData    const &solver_data_in,
            DoFHandler<dim>& dof_handler)
    :
    underlying_operator(underlying_operator_in),
    preconditioner(preconditioner_in),
    solver_data(solver_data_in), dof_handler(dof_handler)
  {}

  virtual ~GMGSolver(){}

  /**
   * Solve Ax=b with initial guess x=0
   * @param dst x
   * @param src b
   * @return 
   */
  unsigned int solve(VectorType       &dst,
                     VectorType const &src) const
  {
    VectorType defect;
    VectorType temp;
    
    // initial condition: initial guess x0=0
    dst = 0; temp = dst;
    // ... r0=b-A*x=b
    defect = src;
    // ... reset counter
    int counter = 0;
    // ... reset norm
    double norm_0 = defect.norm_sqr();

    // loop until (not) converged...
    while(true){
      // perform v-cycle on: 
      //        A*y=r
      preconditioner.vmult(temp, defect);
      
      // update solution vector:
      //        x=x+y
      dst+=temp;
      
      // update residuum in two steps:
      //            A*x
      underlying_operator.vmult(defect,dst);
      //        r=b-A*x
      defect.sadd(-1.0, 1.0, src);

      // calculate norms
      double norm = defect.norm_sqr();
      double norm_rel = norm / norm_0;
      
      std::cout << norm << std::endl;
      
//        {
//            DataOut<dim> data_out;
//        
//            data_out.attach_dof_handler(dof_handler);
//            data_out.add_data_vector(dst, "solution");
//            data_out.build_patches(5);
//        
//            const std::string filename = "solution";
//            std::ofstream output_pressure("output/" + filename + ".vtu");
//            data_out.write_vtu(output_pressure);
//            exit(0);
//        }
        
      // absolute tolerance reached -> success
      if(norm     < solver_data.solver_tolerance_abs) break;
      // relative tolerance reached -> success
      if(norm_rel < solver_data.solver_tolerance_rel) break;
      // maximum number of iterations reached
      if(counter++>1000) throw std::runtime_error("Not converged!");
    }

//        {
//            std::cout << "sf" << std::endl;
//            SolverControl control(10000, 1e-5);
//            internal::PreconditionChebyshevImplementation::EigenvalueTracker eigenvalue_tracker;
//            SolverCG<VectorType > solver(control);
//            solver.connect_eigenvalues_slot(std::bind(&internal::PreconditionChebyshevImplementation::EigenvalueTracker::slot,
//                    &eigenvalue_tracker,
//                    std::placeholders::_1));
//
//            VectorType right = src;
//            //srand(1);
//            //for (unsigned int i=0; i<right.local_size(); ++i)
//            //    right.local_element(i) = (double)rand()/RAND_MAX;
//            //underlying_operator.set_zero_mean_value(right);
//
//            
//            dst = 0;
//            //solver.solve(underlying_operator, dst, right/*, preconditioner*/);
//            solver.solve(preconditioner, dst, right, PreconditionIdentity());
//            
//            std::cout << "sf" << std::endl;
//
//            std::pair<double, double> eigenvalues;
//            if (eigenvalue_tracker.values.empty()) {
//                eigenvalues.first = eigenvalues.second = 1.;
//            } else {
//                eigenvalues.first = eigenvalue_tracker.values.front();
//                eigenvalues.second = eigenvalue_tracker.values.back();
//            }
//
//            for(unsigned int i = 0; i < eigenvalue_tracker.values.size(); i++)
//                std::cout << i << " " << eigenvalue_tracker.values[i] << std::endl;
//
//        }
//    
//    for (int j = 0; j<256; j++){
//        VectorType temp1 = src; 
//        
//        VectorTools::interpolate(dof_handler, SolutionMode<dim>(1.0*j),
//                             temp1);
//        
//        //temp1 = 1;
//        double norm_0 = temp1.norm_sqr(); 
//        double norm_n = 0.0; 
//        std::cout << temp1.norm_sqr() << std::endl;
//        VectorType temp2 = src; temp2 = 0;
//        VectorType temp3 = src; temp3 = 0;
//        
//        int count = 10;
//        for(int i = 0; i<=count; i++){
//            underlying_operator.vmult(temp2, temp1);
//            preconditioner.vmult(temp3, temp2);
//            temp1.sadd(1.0, -1.0, temp3);
//            norm_n = temp1.norm_sqr();
//            std::cout << temp1.norm_sqr() << std::endl;
//        }
//        printf(" @@@ %d %12.7e %12.7e %12.7e", j, norm_0, norm_n, 
//                std::pow(norm_n/norm_0,1.0/count));
//        std::cout << std::endl;
//        
//        
//    }
    
    return counter;
  }

protected:
  Operator const & underlying_operator;
  Preconditioner & preconditioner;
  GMGSolverData const solver_data;
  DoFHandler<dim>& dof_handler;
};


#endif /* INCLUDE_SOLVERS_AND_PRECONDITIONERS_ITERATIVESOLVERS_H_ */
