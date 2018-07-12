/*
 * DGLaplaceOperation.h
 *
 *  Created on: 
 *      Author: 
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_
#define INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "../../operators/inverse_mass_matrix.h"
#include "../../operators/matrix_operator_base.h"
#include "../../solvers_and_preconditioners/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"

#include "../user_interface/boundary_descriptor.h"
#include "../user_interface/field_functions.h"
#include "../user_interface/input_parameters.h"
#include "../preconditioners/multigrid_preconditioner.h"

#include "laplace_operator.h"
#include "rhs_operator.h"

namespace Laplace
{

template<int dim, int fe_degree, typename value_type>
class DGOperation : public MatrixOperatorBase
{
public:

  DGOperation(parallel::distributed::Triangulation<dim> const &triangulation,
              Laplace::InputParameters const                 &param_in)
    :
    fe(fe_degree),
    mapping(fe_degree),
    dof_handler(triangulation),
    param(param_in)
  {}

  void setup(const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
                                                                 periodic_face_pairs,
             std::shared_ptr<Laplace::BoundaryDescriptor<dim> > boundary_descriptor_in,
             std::shared_ptr<Laplace::FieldFunctions<dim> >     field_functions_in)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Setup convection-diffusion operation ..." << std::endl;

    this->periodic_face_pairs = periodic_face_pairs;
    boundary_descriptor = boundary_descriptor_in;
    field_functions = field_functions_in;

    create_dofs();

    initialize_matrix_free();

    setup_operators();

    pcout << std::endl << "... done!" << std::endl;
  }

  void setup_solver()
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Setup solver ..." << std::endl;

    // initialize preconditioner
    if(param.preconditioner == Laplace::Preconditioner::InverseMassMatrix)
    {
      preconditioner.reset(new InverseMassMatrixPreconditioner<dim, fe_degree, value_type, 1>(data,0,0));
    }
    else if(param.preconditioner == Laplace::Preconditioner::PointJacobi)
    {
      preconditioner.reset(new JacobiPreconditioner<
          Laplace::LaplaceOperator<dim,fe_degree,value_type> >(laplace_operator));
    }
    else if(param.preconditioner == Laplace::Preconditioner::BlockJacobi)
    {
      preconditioner.reset(new BlockJacobiPreconditioner<
          Laplace::LaplaceOperator<dim,fe_degree,value_type> >(laplace_operator));
    }
    else if(param.preconditioner == Laplace::Preconditioner::Multigrid)
    {
      MultigridData mg_data;
      mg_data = param.multigrid_data;

      typedef float Number;

       typedef Laplace::MultigridPreconditioner<dim,value_type,
           Laplace::LaplaceOperator<dim,fe_degree,Number>,
           Laplace::LaplaceOperator<dim,fe_degree,value_type> > MULTIGRID;

      preconditioner.reset(new MULTIGRID());
      std::shared_ptr<MULTIGRID> mg_preconditioner = std::dynamic_pointer_cast<MULTIGRID>(preconditioner);
      mg_preconditioner->initialize(mg_data,dof_handler,mapping,laplace_operator,this->periodic_face_pairs);
    }
    else
    {
      AssertThrow(param.preconditioner == Laplace::Preconditioner::None ||
                  param.preconditioner == Laplace::Preconditioner::InverseMassMatrix ||
                  param.preconditioner == Laplace::Preconditioner::PointJacobi ||
                  param.preconditioner == Laplace::Preconditioner::BlockJacobi ||
                  param.preconditioner == Laplace::Preconditioner::Multigrid,
                  ExcMessage("Specified preconditioner is not implemented!"));
    }


    if(param.solver == Laplace::Solver::PCG)
    {
      // initialize solver_data
      CGSolverData solver_data;
      solver_data.solver_tolerance_abs = param.abs_tol;
      solver_data.solver_tolerance_rel = param.rel_tol;
      solver_data.max_iter = param.max_iter;

      if(param.preconditioner != Laplace::Preconditioner::None)
        solver_data.use_preconditioner = true;

      // initialize solver
      iterative_solver.reset(new CGSolver<Laplace::LaplaceOperator<dim,fe_degree,value_type>,
                                          PreconditionerBase<value_type>,
                                          parallel::distributed::Vector<value_type> >
                                 (laplace_operator,*preconditioner,solver_data));
    }
    else if(param.solver == Laplace::Solver::GMRES)
    {
      // initialize solver_data
      GMRESSolverData solver_data;
      solver_data.solver_tolerance_abs = param.abs_tol;
      solver_data.solver_tolerance_rel = param.rel_tol;
      solver_data.max_iter = param.max_iter;
      solver_data.right_preconditioning = param.use_right_preconditioner;
      solver_data.max_n_tmp_vectors = param.max_n_tmp_vectors;

      if(param.preconditioner != Laplace::Preconditioner::None)
        solver_data.use_preconditioner = true;

      // initialize solver
      iterative_solver.reset(new GMRESSolver<Laplace::LaplaceOperator<dim,fe_degree,value_type>,
                                             PreconditionerBase<value_type>,
                                             parallel::distributed::Vector<value_type> >
                                 (laplace_operator,*preconditioner,solver_data));
    }
    else
    {
      AssertThrow(param.solver == Laplace::Solver::PCG ||
                  param.solver == Laplace::Solver::GMRES,
                  ExcMessage("Specified solver is not implemented!"));
    }

    pcout << std::endl << "... done!" << std::endl;
  }

  void initialize_dof_vector(parallel::distributed::Vector<value_type> &src) const
  {
    data.initialize_dof_vector(src);
  }

  /*
   *  This function calculates the inhomogeneous parts of all operators
   *  arising e.g. from inhomogeneous boundary conditions or the solution
   *  at previous instants of time occuring in the discrete time derivate
   *  term.
   *  Note that the convective operator only has a contribution if it is
   *  treated implicitly. In case of an explicit treatment the whole
   *  convective operator (function evaluate() instead of rhs()) has to be
   *  added to the right-hand side of the equations.
   */
  void rhs(parallel::distributed::Vector<value_type>       &dst,
           double const                                    evaluation_time = 0.0) const
  {
    dst = 0;
    if(param.right_hand_side == true)
      rhs_operator.evaluate_add(dst,evaluation_time);
  }

  unsigned int solve(parallel::distributed::Vector<value_type>       &sol,
                     parallel::distributed::Vector<value_type> const &rhs,
                     double const                                    scaling_factor_time_derivative_term_in = -1.0,
                     double const                                    evaluation_time_in = -1.0)
  {
    unsigned int iterations = iterative_solver->solve(sol,rhs);

    return iterations;
  }

  MatrixFree<dim,value_type> const & get_data() const { return data; }

  Mapping<dim> const & get_mapping() const { return mapping; }

  DoFHandler<dim> const & get_dof_handler() const { return dof_handler; }


private:
  void create_dofs()
  {
    // enumerate degrees of freedom
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    unsigned int ndofs_per_cell = Utilities::pow(fe_degree+1,dim);

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    pcout << std::endl
          << "Discontinuous Galerkin finite element discretization:" << std::endl << std::endl;

    print_parameter(pcout,"degree of 1D polynomials",fe_degree);
    print_parameter(pcout,"number of dofs per cell",ndofs_per_cell);
    print_parameter(pcout,"number of dofs (total)",dof_handler.n_dofs());
  }

  void initialize_matrix_free()
  {
    // quadrature formula used to perform integrals
    QGauss<1> quadrature (fe_degree+1);

    // initialize matrix_free_data
    typename MatrixFree<dim,value_type>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim,value_type>::AdditionalData::partition_partition;
    additional_data.build_face_info = true;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points | update_normal_vectors |
                                            update_values);

    additional_data.mapping_update_flags_inner_faces = (update_gradients | update_JxW_values |
                                                        update_quadrature_points | update_normal_vectors |
                                                        update_values);

    additional_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values |
                                                           update_quadrature_points | update_normal_vectors |
                                                           update_values);

    ConstraintMatrix dummy;
    dummy.close();
    data.reinit (mapping, dof_handler, dummy, quadrature, additional_data);
  }

  void setup_operators(){
      // TODO
  }


  FE_DGQ<dim> fe;
  MappingQGeneric<dim> mapping;
  DoFHandler<dim> dof_handler;

  MatrixFree<dim,value_type> data;

  Laplace::InputParameters const &param;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs;

  std::shared_ptr<Laplace::BoundaryDescriptor<dim> > boundary_descriptor;
  std::shared_ptr<Laplace::FieldFunctions<dim> > field_functions;

  Laplace::RHSOperator<dim, fe_degree, value_type> rhs_operator;

  //Laplace::ConvectiveOperatorData<dim> convective_operator_data;
  Laplace::LaplaceOperator<dim, fe_degree, value_type> laplace_operator;

  std::shared_ptr<PreconditionerBase<value_type> > preconditioner;
  std::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > iterative_solver;
};

} // namespace Laplace

#endif /* INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_ */
