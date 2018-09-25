/*
 * DGConvDiffOperation.h
 *
 *  Created on: Aug 2, 2016
 *      Author: fehn
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
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers.h"

#include "../../convection_diffusion/preconditioners/multigrid_preconditioner.h"
#include "../../convection_diffusion/spatial_discretization/convection_diffusion_operators.h"
#include "../../convection_diffusion/user_interface/boundary_descriptor.h"
#include "../../convection_diffusion/user_interface/field_functions.h"
#include "../../convection_diffusion/user_interface/input_parameters.h"

namespace ConvDiff
{
template<int dim, int fe_degree, typename value_type>
class DGOperation : public MatrixOperatorBase
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  DGOperation(parallel::distributed::Triangulation<dim> const & triangulation,
              ConvDiff::InputParameters const &                 param_in)
    : fe(fe_degree), mapping(fe_degree), dof_handler(triangulation), param(param_in)
  {
  }

  void
  setup(const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
                                                           periodic_face_pairs,
        std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor_in,
        std::shared_ptr<ConvDiff::FieldFunctions<dim>>     field_functions_in)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Setup convection-diffusion operation ..." << std::endl;

    this->periodic_face_pairs = periodic_face_pairs;
    boundary_descriptor       = boundary_descriptor_in;
    field_functions           = field_functions_in;

    create_dofs();

    initialize_matrix_free();

    setup_operators();

    pcout << std::endl << "... done!" << std::endl;
  }

  void
  setup_solver(double const scaling_factor_time_derivative_term_in = -1.0)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Setup solver ..." << std::endl;

    // convection-diffusion operator
    ConvDiff::ConvectionDiffusionOperatorData<dim> conv_diff_operator_data;
    conv_diff_operator_data.mass_matrix_operator_data = mass_matrix_operator.get_operator_data();
    conv_diff_operator_data.convective_operator_data  = convective_operator.get_operator_data();
    conv_diff_operator_data.diffusive_operator_data   = diffusive_operator.get_operator_data();
    conv_diff_operator_data.scaling_factor_time_derivative_term =
      scaling_factor_time_derivative_term_in;
    conv_diff_operator_data.use_cell_based_loops = param.enable_cell_based_face_loops;

    if(this->param.problem_type == ConvDiff::ProblemType::Unsteady)
    {
      conv_diff_operator_data.unsteady_problem = true;
    }
    else
    {
      conv_diff_operator_data.unsteady_problem = false;
    }

    if(this->param.equation_type == ConvDiff::EquationType::Diffusion ||
       this->param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
    {
      conv_diff_operator_data.diffusive_problem = true;
    }
    else
    {
      conv_diff_operator_data.diffusive_problem = false;
    }

    if((this->param.equation_type == ConvDiff::EquationType::Convection ||
        this->param.equation_type == ConvDiff::EquationType::ConvectionDiffusion) &&
       this->param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Implicit)
    {
      conv_diff_operator_data.convective_problem = true;
    }
    else
    {
      conv_diff_operator_data.convective_problem = false;
    }

    conv_diff_operator_data.update_mapping_update_flags();

    conv_diff_operator_data.dof_index = 0;

    conv_diff_operator_data.mg_operator_type = param.mg_operator_type;
    conv_diff_operator_data.bc               = boundary_descriptor;

    conv_diff_operator.initialize(
      data, conv_diff_operator_data, mass_matrix_operator, convective_operator, diffusive_operator);

    // initialize preconditioner
    if(param.preconditioner == ConvDiff::Preconditioner::InverseMassMatrix)
    {
      preconditioner.reset(
        new InverseMassMatrixPreconditioner<dim, fe_degree, value_type, 1>(data, 0, 0));
    }
    else if(param.preconditioner == ConvDiff::Preconditioner::PointJacobi)
    {
      preconditioner.reset(
        new JacobiPreconditioner<ConvDiff::ConvectionDiffusionOperator<dim, fe_degree, value_type>>(
          conv_diff_operator));
    }
    else if(param.preconditioner == ConvDiff::Preconditioner::BlockJacobi)
    {
      preconditioner.reset(
        new BlockJacobiPreconditioner<
          ConvDiff::ConvectionDiffusionOperator<dim, fe_degree, value_type>>(conv_diff_operator));
    }
    else if(param.preconditioner == ConvDiff::Preconditioner::Multigrid)
    {
      MultigridData mg_data;
      mg_data = param.multigrid_data;

      typedef float Number;

      typedef ConvDiff::MultigridPreconditioner<
        dim,
        value_type,
        ConvDiff::ConvectionDiffusionOperator<dim, fe_degree, Number>,
        ConvDiff::ConvectionDiffusionOperator<dim, fe_degree, value_type>>
        MULTIGRID;

      preconditioner.reset(new MULTIGRID());
      std::shared_ptr<MULTIGRID> mg_preconditioner =
        std::dynamic_pointer_cast<MULTIGRID>(preconditioner);
      mg_preconditioner->initialize(mg_data,
                                    dof_handler,
                                    mapping,
                                    conv_diff_operator.get_operator_data().bc->dirichlet_bc,
                                    (void *)&conv_diff_operator.get_operator_data());
    }
    else
    {
      AssertThrow(param.preconditioner == ConvDiff::Preconditioner::None ||
                    param.preconditioner == ConvDiff::Preconditioner::InverseMassMatrix ||
                    param.preconditioner == ConvDiff::Preconditioner::PointJacobi ||
                    param.preconditioner == ConvDiff::Preconditioner::BlockJacobi ||
                    param.preconditioner == ConvDiff::Preconditioner::Multigrid,
                  ExcMessage("Specified preconditioner is not implemented!"));
    }


    if(param.solver == ConvDiff::Solver::PCG)
    {
      // initialize solver_data
      CGSolverData solver_data;
      solver_data.solver_tolerance_abs  = param.abs_tol;
      solver_data.solver_tolerance_rel  = param.rel_tol;
      solver_data.max_iter              = param.max_iter;
      solver_data.update_preconditioner = param.update_preconditioner;

      if(param.preconditioner != ConvDiff::Preconditioner::None)
        solver_data.use_preconditioner = true;

      // initialize solver
      iterative_solver.reset(
        new CGSolver<ConvDiff::ConvectionDiffusionOperator<dim, fe_degree, value_type>,
                     PreconditionerBase<value_type>,
                     VectorType>(conv_diff_operator, *preconditioner, solver_data));
    }
    else if(param.solver == ConvDiff::Solver::GMRES)
    {
      // initialize solver_data
      GMRESSolverData solver_data;
      solver_data.solver_tolerance_abs  = param.abs_tol;
      solver_data.solver_tolerance_rel  = param.rel_tol;
      solver_data.max_iter              = param.max_iter;
      solver_data.right_preconditioning = param.use_right_preconditioner;
      solver_data.max_n_tmp_vectors     = param.max_n_tmp_vectors;
      solver_data.update_preconditioner = param.update_preconditioner;

      if(param.preconditioner != ConvDiff::Preconditioner::None)
        solver_data.use_preconditioner = true;

      // initialize solver
      iterative_solver.reset(
        new GMRESSolver<ConvDiff::ConvectionDiffusionOperator<dim, fe_degree, value_type>,
                        PreconditionerBase<value_type>,
                        VectorType>(conv_diff_operator, *preconditioner, solver_data));
    }
    else
    {
      AssertThrow(param.solver == ConvDiff::Solver::PCG || param.solver == ConvDiff::Solver::GMRES,
                  ExcMessage("Specified solver is not implemented!"));
    }

    pcout << std::endl << "... done!" << std::endl;
  }

  void
  initialize_dof_vector(VectorType & src) const
  {
    data.initialize_dof_vector(src);
  }

  void
  prescribe_initial_conditions(VectorType & src, double const evaluation_time) const
  {
    field_functions->analytical_solution->set_time(evaluation_time);
    VectorTools::interpolate(dof_handler, *(field_functions->analytical_solution), src);
  }

  /*
   *  This function is used in case of explicit time integration:
   *  This function evaluates the right-hand side operator, the
   *  convective and diffusive term (subsequently multiplied by -1.0 in order
   *  to shift these terms to the right-hand side of the equations)
   *  and finally applies the inverse mass matrix operator.
   */
  void
  evaluate(VectorType & dst, VectorType const & src, double const evaluation_time) const
  {
    if(param.runtime_optimization ==
       false) // apply volume and surface integrals for each operator separately
    {
      // set dst to zero
      dst = 0.0;

      // diffusive operator
      if(param.equation_type == ConvDiff::EquationType::Diffusion ||
         param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
      {
        diffusive_operator.evaluate_add(dst, src, evaluation_time);
      }

      // convective operator
      if(param.equation_type == ConvDiff::EquationType::Convection ||
         param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
      {
        convective_operator.evaluate_add(dst, src, evaluation_time);
      }

      // shift diffusive and convective term to the rhs of the equation
      dst *= -1.0;

      if(param.right_hand_side == true)
      {
        rhs_operator.evaluate_add(dst, evaluation_time);
      }
    }
    else // param.runtime_optimization == true
    {
      convection_diffusion_operator_efficiency.evaluate(dst, src, evaluation_time);
    }

    // apply inverse mass matrix
    inverse_mass_matrix_operator.apply(dst, dst);
  }

  void
  evaluate_convective_term(VectorType &       dst,
                           VectorType const & src,
                           double const       evaluation_time) const
  {
    convective_operator.evaluate(dst, src, evaluation_time);
  }

  void
  evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
    VectorType &       dst,
    VectorType const & src,
    double const       evaluation_time) const
  {
    convective_operator.evaluate(dst, src, evaluation_time);

    // shift convective term to the rhs of the equation
    dst *= -1.0;

    inverse_mass_matrix_operator.apply(dst, dst);
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
  void
  rhs(VectorType & dst, double const evaluation_time = 0.0) const
  {
    // set dst to zero since we call functions of type ..._add()
    dst = 0;

    // diffusive operator
    if(param.equation_type == ConvDiff::EquationType::Diffusion ||
       param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
    {
      diffusive_operator.rhs_add(dst, evaluation_time);
    }

    // convective operator
    if(param.equation_type == ConvDiff::EquationType::Convection ||
       param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
    {
      if(param.problem_type == ConvDiff::ProblemType::Steady ||
         (param.problem_type == ConvDiff::ProblemType::Unsteady &&
          param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Implicit))
      {
        convective_operator.rhs_add(dst, evaluation_time);
      }
    }

    // rhs operator f(t)
    if(param.right_hand_side == true)
    {
      rhs_operator.evaluate_add(dst, evaluation_time);
    }
  }

  /*
   *  This function applies the mass matrix operator to the src-vector
   *  and adds the result to the dst-vector.
   */
  void
  apply_mass_matrix_add(VectorType & dst, VectorType const & src) const
  {
    mass_matrix_operator.apply_add(dst, src);
  }

  unsigned int
  solve(VectorType &       sol,
        VectorType const & rhs,
        double const       scaling_factor_time_derivative_term_in = -1.0,
        double const       evaluation_time_in                     = -1.0)
  {
    conv_diff_operator.set_scaling_factor_time_derivative_term(
      scaling_factor_time_derivative_term_in);
    conv_diff_operator.set_evaluation_time(evaluation_time_in);

    unsigned int iterations = iterative_solver->solve(sol, rhs);

    return iterations;
  }

  // getters
  MatrixFree<dim, value_type> const &
  get_data() const
  {
    return data;
  }

  Mapping<dim> const &
  get_mapping() const
  {
    return mapping;
  }

  DoFHandler<dim> const &
  get_dof_handler() const
  {
    return dof_handler;
  }


private:
  void
  create_dofs()
  {
    // enumerate degrees of freedom
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    constexpr int ndofs_per_cell = Utilities::pow(fe_degree + 1, dim);

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    pcout << std::endl
          << "Discontinuous Galerkin finite element discretization:" << std::endl
          << std::endl;

    print_parameter(pcout, "degree of 1D polynomials", fe_degree);
    print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
    print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
  }

  void
  initialize_matrix_free()
  {
    // quadrature formula used to perform integrals
    QGauss<1> quadrature(fe_degree + 1);

    // initialize matrix_free_data
    typename MatrixFree<dim, value_type>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, value_type>::AdditionalData::partition_partition;
    additional_data.build_face_info = true;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    additional_data.mapping_update_flags_inner_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    additional_data.mapping_update_flags_boundary_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    if(param.enable_cell_based_face_loops)
    {
      auto tria = dynamic_cast<const parallel::distributed::Triangulation<dim> *>(
        &dof_handler.get_triangulation());
      Categorization::do_cell_based_loops(*tria, additional_data);
    }

    ConstraintMatrix dummy;
    dummy.close();
    data.reinit(mapping, dof_handler, dummy, quadrature, additional_data);
  }

  void
  setup_operators()
  {
    // mass matrix operator
    MassMatrixOperatorData<dim> mass_matrix_operator_data;
    mass_matrix_operator_data.dof_index            = 0;
    mass_matrix_operator_data.quad_index           = 0;
    mass_matrix_operator_data.use_cell_based_loops = param.enable_cell_based_face_loops;
    mass_matrix_operator.initialize(data, mass_matrix_operator_data);

    // inverse mass matrix operator
    // dof_index = 0, quad_index = 0
    inverse_mass_matrix_operator.initialize(data, 0, 0);

    // convective operator
    ConvectiveOperatorData<dim> convective_operator_data;
    convective_operator_data.dof_index                  = 0;
    convective_operator_data.quad_index                 = 0;
    convective_operator_data.numerical_flux_formulation = param.numerical_flux_convective_operator;
    convective_operator_data.bc                         = boundary_descriptor;
    convective_operator_data.velocity                   = field_functions->velocity;
    convective_operator_data.use_cell_based_loops       = param.enable_cell_based_face_loops;
    convective_operator.initialize(data, convective_operator_data);

    // diffusive operator
    DiffusiveOperatorData<dim> diffusive_operator_data;
    diffusive_operator_data.dof_index            = 0;
    diffusive_operator_data.quad_index           = 0;
    diffusive_operator_data.IP_factor            = param.IP_factor;
    diffusive_operator_data.diffusivity          = param.diffusivity;
    diffusive_operator_data.bc                   = boundary_descriptor;
    diffusive_operator_data.use_cell_based_loops = param.enable_cell_based_face_loops;
    diffusive_operator.initialize(mapping, data, diffusive_operator_data);

    // rhs operator
    ConvDiff::RHSOperatorData<dim> rhs_operator_data;
    rhs_operator_data.dof_index  = 0;
    rhs_operator_data.quad_index = 0;
    rhs_operator_data.rhs        = field_functions->right_hand_side;
    rhs_operator.initialize(data, rhs_operator_data);

    // convection-diffusion operator (efficient implementation, only for explicit time integration,
    // includes also rhs operator)
    ConvDiff::ConvectionDiffusionOperatorDataEfficiency<dim, value_type>
      conv_diff_operator_data_eff;
    conv_diff_operator_data_eff.conv_data = convective_operator_data;
    conv_diff_operator_data_eff.diff_data = diffusive_operator_data;
    conv_diff_operator_data_eff.rhs_data  = rhs_operator_data;
    convection_diffusion_operator_efficiency.initialize(mapping, data, conv_diff_operator_data_eff);
  }


  FE_DGQ<dim>          fe;
  MappingQGeneric<dim> mapping;
  DoFHandler<dim>      dof_handler;

  MatrixFree<dim, value_type> data;

  ConvDiff::InputParameters const & param;

  // TODO This variable is only needed when using the GeometricMultigrid preconditioner
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<ConvDiff::FieldFunctions<dim>>     field_functions;

  ConvDiff::MassMatrixOperator<dim, fe_degree, value_type> mass_matrix_operator;
  InverseMassMatrixOperator<dim, fe_degree, value_type, 1> inverse_mass_matrix_operator;
  ConvDiff::ConvectiveOperator<dim, fe_degree, value_type> convective_operator;
  ConvDiff::DiffusiveOperator<dim, fe_degree, value_type>  diffusive_operator;
  ConvDiff::RHSOperator<dim, fe_degree, value_type>        rhs_operator;

  ConvDiff::ConvectionDiffusionOperator<dim, fe_degree, value_type> conv_diff_operator;

  // convection-diffusion operator for runtime optimization (also includes rhs operator)
  ConvDiff::ConvectionDiffusionOperatorEfficiency<dim, fe_degree, value_type>
    convection_diffusion_operator_efficiency;

  std::shared_ptr<PreconditionerBase<value_type>> preconditioner;

  std::shared_ptr<IterativeSolverBase<VectorType>> iterative_solver;
};

template<int dim, int fe_degree, typename value_type>
class ConvectiveOperatorOIFSplitting
{
public:
  typedef LinearAlgebra::distributed::Vector<value_type> VectorType;

  ConvectiveOperatorOIFSplitting(
    std::shared_ptr<ConvDiff::DGOperation<dim, fe_degree, value_type>> conv_diff_operation_in)
    : conv_diff_operation(conv_diff_operation_in)
  {
  }

  void
  evaluate(VectorType & dst, VectorType const & src, double const evaluation_time) const
  {
    conv_diff_operation->evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
      dst, src, evaluation_time);
  }

  void
  initialize_dof_vector(VectorType & src) const
  {
    conv_diff_operation->initialize_dof_vector(src);
  }

private:
  std::shared_ptr<ConvDiff::DGOperation<dim, fe_degree, value_type>> conv_diff_operation;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_ */
