/*
 * dg_convection_diffusion_operation.h
 *
 *  Created on: Aug 2, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_
#define INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/numerics/vector_tools.h>

#include "../../operators/inverse_mass_matrix.h"
#include "../../operators/linear_operator_base.h"
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/preconditioner/jacobi_preconditioner.h"
#include "../../solvers_and_preconditioners/solvers/iterative_solvers_dealii_wrapper.h"

#include "../preconditioners/multigrid_preconditioner.h"
#include "../user_interface/boundary_descriptor.h"
#include "../user_interface/field_functions.h"
#include "../user_interface/input_parameters.h"
#include "operators/convection_diffusion_operator.h"
#include "operators/convection_diffusion_operator_efficiency.h"
#include "operators/convective_operator_discontinuous_velocity.h"

#include "time_integration/interpolate.h"
#include "time_integration/time_step_calculation.h"

#include "../interface_space_time/operator.h"

namespace ConvDiff
{
template<int dim, int degree, typename Number>
class DGOperation : public dealii::Subscriptor, public Interface::Operator<Number>
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  DGOperation(parallel::distributed::Triangulation<dim> const & triangulation,
              InputParameters const &                           param_in,
              std::shared_ptr<PostProcessor<dim, degree>>       postprocessor_in)
    : dealii::Subscriptor(),
      fe(degree),
      mapping(param_in.degree_mapping),
      dof_handler(triangulation),
      param(param_in),
      postprocessor(postprocessor_in)
  {
    if(param.type_velocity_field == TypeVelocityField::Numerical)
    {
      fe_velocity.reset(new FESystem<dim>(FE_DGQ<dim>(degree), dim));
      dof_handler_velocity.reset(new DoFHandler<dim>(triangulation));
    }
  }

  void
  setup(const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
                                                 periodic_face_pairs,
        std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor_in,
        std::shared_ptr<FieldFunctions<dim>>     field_functions_in,
        std::shared_ptr<AnalyticalSolution<dim>> analytical_solution_in)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Setup convection-diffusion operation ..." << std::endl;

    this->periodic_face_pairs = periodic_face_pairs;
    boundary_descriptor       = boundary_descriptor_in;
    field_functions           = field_functions_in;

    create_dofs();

    initialize_matrix_free();

    setup_operators();

    setup_postprocessor(analytical_solution_in);

    pcout << std::endl << "... done!" << std::endl;
  }

  void
  initialize_convection_diffusion_operator(
    double const scaling_factor_time_derivative_term_in = -1.0)
  {
    // convection-diffusion operator
    ConvectionDiffusionOperatorData<dim> conv_diff_operator_data;
    conv_diff_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
    conv_diff_operator_data.implement_block_diagonal_preconditioner_matrix_free =
      param.implement_block_diagonal_preconditioner_matrix_free;
    conv_diff_operator_data.preconditioner_block_jacobi = param.preconditioner_block_diagonal;
    conv_diff_operator_data.block_jacobi_solver_data    = param.block_jacobi_solver_data;
    conv_diff_operator_data.mass_matrix_operator_data   = mass_matrix_operator.get_operator_data();
    conv_diff_operator_data.convective_operator_data    = convective_operator.get_operator_data();
    conv_diff_operator_data.diffusive_operator_data     = diffusive_operator.get_operator_data();
    conv_diff_operator_data.scaling_factor_time_derivative_term =
      scaling_factor_time_derivative_term_in;

    if(this->param.problem_type == ProblemType::Unsteady)
    {
      conv_diff_operator_data.unsteady_problem = true;
    }
    else
    {
      conv_diff_operator_data.unsteady_problem = false;
    }

    if(this->param.equation_type == EquationType::Diffusion ||
       this->param.equation_type == EquationType::ConvectionDiffusion)
    {
      conv_diff_operator_data.diffusive_problem = true;
    }
    else
    {
      conv_diff_operator_data.diffusive_problem = false;
    }

    if((this->param.equation_type == EquationType::Convection ||
        this->param.equation_type == EquationType::ConvectionDiffusion) &&
       this->param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit)
    {
      conv_diff_operator_data.convective_problem = true;

      AssertThrow(param.type_velocity_field == TypeVelocityField::Analytical,
                  ExcMessage("Not implemented."));
    }
    else
    {
      conv_diff_operator_data.convective_problem = false;
    }

    conv_diff_operator_data.update_mapping_update_flags();

    conv_diff_operator_data.dof_index        = 0;
    conv_diff_operator_data.mg_operator_type = param.mg_operator_type;

    conv_diff_operator.reinit(
      data, conv_diff_operator_data, mass_matrix_operator, convective_operator, diffusive_operator);
  }

  void
  initialize_preconditioner()
  {
    if(param.preconditioner == Preconditioner::InverseMassMatrix)
    {
      preconditioner.reset(new InverseMassMatrixPreconditioner<dim, degree, Number, 1>(data, 0, 0));
    }
    else if(param.preconditioner == Preconditioner::PointJacobi)
    {
      preconditioner.reset(
        new JacobiPreconditioner<ConvectionDiffusionOperator<dim, degree, Number>>(
          conv_diff_operator));
    }
    else if(param.preconditioner == Preconditioner::BlockJacobi)
    {
      preconditioner.reset(
        new BlockJacobiPreconditioner<ConvectionDiffusionOperator<dim, degree, Number>>(
          conv_diff_operator));
    }
    else if(param.preconditioner == Preconditioner::Multigrid)
    {
      MultigridData mg_data;
      mg_data = param.multigrid_data;

      typedef float MultigridNumber;

      typedef MultigridPreconditioner<dim, degree, Number, MultigridNumber> MULTIGRID;

      preconditioner.reset(new MULTIGRID());
      std::shared_ptr<MULTIGRID> mg_preconditioner =
        std::dynamic_pointer_cast<MULTIGRID>(preconditioner);
      mg_preconditioner->initialize(mg_data,
                                    dof_handler,
                                    mapping,
                                    conv_diff_operator.get_boundary_descriptor()->dirichlet_bc,
                                    (void *)&conv_diff_operator.get_operator_data());
    }
    else
    {
      AssertThrow(param.preconditioner == Preconditioner::None ||
                    param.preconditioner == Preconditioner::InverseMassMatrix ||
                    param.preconditioner == Preconditioner::PointJacobi ||
                    param.preconditioner == Preconditioner::BlockJacobi ||
                    param.preconditioner == Preconditioner::Multigrid,
                  ExcMessage("Specified preconditioner is not implemented!"));
    }
  }

  void
  initialize_solver()
  {
    if(param.solver == Solver::PCG)
    {
      // initialize solver_data
      CGSolverData solver_data;
      solver_data.solver_tolerance_abs  = param.abs_tol;
      solver_data.solver_tolerance_rel  = param.rel_tol;
      solver_data.max_iter              = param.max_iter;
      solver_data.update_preconditioner = param.update_preconditioner;

      if(param.preconditioner != Preconditioner::None)
        solver_data.use_preconditioner = true;

      // initialize solver
      iterative_solver.reset(
        new CGSolver<ConvectionDiffusionOperator<dim, degree, Number>,
                     PreconditionerBase<Number>,
                     VectorType>(conv_diff_operator, *preconditioner, solver_data));
    }
    else if(param.solver == Solver::GMRES)
    {
      // initialize solver_data
      GMRESSolverData solver_data;
      solver_data.solver_tolerance_abs  = param.abs_tol;
      solver_data.solver_tolerance_rel  = param.rel_tol;
      solver_data.max_iter              = param.max_iter;
      solver_data.right_preconditioning = param.use_right_preconditioner;
      solver_data.max_n_tmp_vectors     = param.max_n_tmp_vectors;
      solver_data.update_preconditioner = param.update_preconditioner;

      if(param.preconditioner != Preconditioner::None)
        solver_data.use_preconditioner = true;

      // initialize solver
      iterative_solver.reset(
        new GMRESSolver<ConvectionDiffusionOperator<dim, degree, Number>,
                        PreconditionerBase<Number>,
                        VectorType>(conv_diff_operator, *preconditioner, solver_data));
    }
    else
    {
      AssertThrow(param.solver == Solver::PCG || param.solver == Solver::GMRES,
                  ExcMessage("Specified solver is not implemented!"));
    }
  }

  void
  setup_solver(double const scaling_factor_time_derivative_term_in = -1.0)
  {
    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
    pcout << std::endl << "Setup solver ..." << std::endl;

    initialize_convection_diffusion_operator(scaling_factor_time_derivative_term_in);

    initialize_preconditioner();

    initialize_solver();

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
    // apply volume and surface integrals for each operator separately
    if(param.runtime_optimization == false)
    {
      // set dst to zero
      dst = 0.0;

      // diffusive operator
      if(param.equation_type == EquationType::Diffusion ||
         param.equation_type == EquationType::ConvectionDiffusion)
      {
        diffusive_operator.evaluate_add(dst, src, evaluation_time);
      }

      // convective operator
      if(param.equation_type == EquationType::Convection ||
         param.equation_type == EquationType::ConvectionDiffusion)
      {
        if(param.type_velocity_field == TypeVelocityField::Analytical)
        {
          convective_operator.evaluate_add(dst, src, evaluation_time);
        }
        else if(param.type_velocity_field == TypeVelocityField::Numerical)
        {
          // We first have to interpolate the velocity field so that it is evaluated at the correct
          // time.
          interpolate(velocity, evaluation_time, velocities, times);
          convective_operator_discontinuous.set_velocity(velocity);
          convective_operator_discontinuous.evaluate_add(dst, src, evaluation_time);
        }
        else
        {
          AssertThrow(false, ExcMessage("Not implemented."));
        }
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
  set_velocity(VectorType const & velocity) const
  {
    convective_operator_discontinuous.set_velocity(velocity);
  }

  void
  set_velocities_and_times(std::vector<VectorType const *> & velocities_in,
                           std::vector<double> &             times_in) const
  {
    velocities = velocities_in;
    times      = times_in;
  }

  void
  evaluate_convective_term(VectorType &       dst,
                           VectorType const & src,
                           double const       evaluation_time) const
  {
    if(param.type_velocity_field == TypeVelocityField::Analytical)
      convective_operator.evaluate(dst, src, evaluation_time);
    else if(param.type_velocity_field == TypeVelocityField::Numerical)
      convective_operator_discontinuous.evaluate(dst, src, evaluation_time);
    else
      AssertThrow(false, ExcMessage("Not implemented."));
  }

  /*
   * This function is called by OIF sub-stepping algorithm.
   */
  void
  evaluate_negative_convective_term_and_apply_inverse_mass_matrix(
    VectorType &       dst,
    VectorType const & src,
    double const       evaluation_time) const
  {
    if(param.type_velocity_field == TypeVelocityField::Analytical)
    {
      convective_operator.evaluate(dst, src, evaluation_time);
    }
    else if(param.type_velocity_field == TypeVelocityField::Numerical)
    {
      // We first have to interpolate the velocity field so that it is evaluated at the correct
      // time.
      interpolate(velocity, evaluation_time, velocities, times);
      convective_operator_discontinuous.set_velocity(velocity);
      convective_operator_discontinuous.evaluate(dst, src, evaluation_time);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

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
    if(param.equation_type == EquationType::Diffusion ||
       param.equation_type == EquationType::ConvectionDiffusion)
    {
      diffusive_operator.rhs_add(dst, evaluation_time);
    }

    // convective operator
    if(param.equation_type == EquationType::Convection ||
       param.equation_type == EquationType::ConvectionDiffusion)
    {
      if(param.problem_type == ProblemType::Steady ||
         (param.problem_type == ProblemType::Unsteady &&
          param.treatment_of_convective_term == TreatmentOfConvectiveTerm::Implicit))
      {
        if(param.type_velocity_field == TypeVelocityField::Analytical)
          convective_operator.rhs_add(dst, evaluation_time);
        else if(param.type_velocity_field == TypeVelocityField::Numerical)
          convective_operator_discontinuous.rhs_add(dst, evaluation_time);
        else
          AssertThrow(false, ExcMessage("Not implemented."));
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
  MatrixFree<dim, Number> const &
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

  void
  do_postprocessing(VectorType const & solution,
                    double const       time             = 0.0,
                    int const          time_step_number = -1) const
  {
    postprocessor->do_postprocessing(solution, time, time_step_number);
  }

  // Calculate time step size according to local CFL criterion

  // use numerical velocity field
  double
  calculate_time_step_cfl(double const cfl, double const exponent_degree) const
  {
    return calculate_time_step_cfl_local<dim, degree /* = degree_velocity */, Number>(
      data, /*dof_index_velocity = */ 1, /*quad_index = */ 0, velocity, cfl, exponent_degree);
  }

  // use analytical velocity field
  double
  calculate_time_step_cfl(double const time, double const cfl, double const exponent_degree) const
  {
    return calculate_time_step_cfl_local<dim, degree, Number>(data,
                                                              0 /*dof_index*/,
                                                              0 /*quad_index*/,
                                                              field_functions->velocity,
                                                              time,
                                                              cfl,
                                                              exponent_degree);
  }

  double
  calculate_maximum_velocity(double const time) const
  {
    return calculate_max_velocity(dof_handler.get_triangulation(), field_functions->velocity, time);
  }

  double
  calculate_minimum_element_length() const
  {
    return calculate_minimum_vertex_distance(dof_handler.get_triangulation());
  }

  unsigned int
  get_polynomial_degree() const
  {
    return degree;
  }

private:
  void
  create_dofs()
  {
    // enumerate degrees of freedom
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs();

    if(param.type_velocity_field == TypeVelocityField::Numerical)
    {
      dof_handler_velocity->distribute_dofs(*fe_velocity);
      dof_handler_velocity->distribute_mg_dofs();
    }

    constexpr int ndofs_per_cell = Utilities::pow(degree + 1, dim);

    ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);

    pcout << std::endl
          << "Discontinuous Galerkin finite element discretization:" << std::endl
          << std::endl;

    print_parameter(pcout, "degree of 1D polynomials", degree);
    print_parameter(pcout, "number of dofs per cell", ndofs_per_cell);
    print_parameter(pcout, "number of dofs (total)", dof_handler.n_dofs());
  }

  void
  initialize_matrix_free()
  {
    // initialize matrix_free_data
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::partition_partition;
    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    additional_data.mapping_update_flags_inner_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    additional_data.mapping_update_flags_boundary_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    if(param.use_cell_based_face_loops)
    {
      auto tria = dynamic_cast<const parallel::distributed::Triangulation<dim> *>(
        &dof_handler.get_triangulation());
      Categorization::do_cell_based_loops(*tria, additional_data);
    }

    if(param.type_velocity_field == TypeVelocityField::Analytical)
    {
      AffineConstraints<double> constraint_dummy;
      constraint_dummy.close();

      // quadrature formula used to perform integrals
      QGauss<1> quadrature(degree + 1);

      data.reinit(mapping, dof_handler, constraint_dummy, quadrature, additional_data);
    }
    // we need two dof-handlers in case the velocity field comes from the fluid solver.
    else if(param.type_velocity_field == TypeVelocityField::Numerical)
    {
      std::vector<const DoFHandler<dim> *> dof_handler_vec;
      dof_handler_vec.resize(2);
      dof_handler_vec[0] = &dof_handler;
      dof_handler_vec[1] = &(*dof_handler_velocity);

      std::vector<const AffineConstraints<double> *> constraint_vec;
      constraint_vec.resize(2);
      AffineConstraints<double> constraint_dummy;
      constraint_dummy.close();
      constraint_vec[0] = &constraint_dummy;
      constraint_vec[1] = &constraint_dummy;

      std::vector<Quadrature<1>> quadrature_vec;
      quadrature_vec.resize(1);
      quadrature_vec[0] = QGauss<1>(degree + 1);

      data.reinit(mapping, dof_handler_vec, constraint_vec, quadrature_vec, additional_data);
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
  }

  void
  setup_operators()
  {
    // mass matrix operator
    MassMatrixOperatorData<dim> mass_matrix_operator_data;
    mass_matrix_operator_data.dof_index            = 0;
    mass_matrix_operator_data.quad_index           = 0;
    mass_matrix_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
    mass_matrix_operator_data.implement_block_diagonal_preconditioner_matrix_free =
      param.implement_block_diagonal_preconditioner_matrix_free;
    mass_matrix_operator.reinit(data, mass_matrix_operator_data);

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
    convective_operator_data.use_cell_based_loops       = param.use_cell_based_face_loops;
    convective_operator_data.implement_block_diagonal_preconditioner_matrix_free =
      param.implement_block_diagonal_preconditioner_matrix_free;
    convective_operator.reinit(data, convective_operator_data);

    if(param.type_velocity_field == TypeVelocityField::Numerical)
    {
      ConvectiveOperatorDisVelData<dim> operator_data;

      operator_data.dof_index                  = 0;
      operator_data.dof_index_velocity         = 1;
      operator_data.quad_index                 = 0;
      operator_data.numerical_flux_formulation = param.numerical_flux_convective_operator;
      operator_data.bc                         = boundary_descriptor;

      convective_operator_discontinuous.initialize(data, operator_data);

      data.initialize_dof_vector(velocity, operator_data.dof_index_velocity);
    }


    // diffusive operator
    DiffusiveOperatorData<dim> diffusive_operator_data;
    diffusive_operator_data.dof_index            = 0;
    diffusive_operator_data.quad_index           = 0;
    diffusive_operator_data.IP_factor            = param.IP_factor;
    diffusive_operator_data.diffusivity          = param.diffusivity;
    diffusive_operator_data.bc                   = boundary_descriptor;
    diffusive_operator_data.use_cell_based_loops = param.use_cell_based_face_loops;
    diffusive_operator_data.implement_block_diagonal_preconditioner_matrix_free =
      param.implement_block_diagonal_preconditioner_matrix_free;
    diffusive_operator.reinit(mapping, data, diffusive_operator_data);

    // rhs operator
    RHSOperatorData<dim> rhs_operator_data;
    rhs_operator_data.dof_index  = 0;
    rhs_operator_data.quad_index = 0;
    rhs_operator_data.rhs        = field_functions->right_hand_side;
    rhs_operator.reinit(data, rhs_operator_data);

    // convection-diffusion operator (efficient implementation, only for explicit time integration,
    // includes also rhs operator)
    ConvectionDiffusionOperatorDataEfficiency<dim, Number> conv_diff_operator_data_eff;
    conv_diff_operator_data_eff.conv_data = convective_operator_data;
    conv_diff_operator_data_eff.diff_data = diffusive_operator_data;
    conv_diff_operator_data_eff.rhs_data  = rhs_operator_data;
    convection_diffusion_operator_efficiency.initialize(mapping, data, conv_diff_operator_data_eff);
  }

  void
  setup_postprocessor(std::shared_ptr<AnalyticalSolution<dim>> analytical_solution)
  {
    PostProcessorData pp_data;
    pp_data.output_data = param.output_data;
    pp_data.error_data  = param.error_data;

    postprocessor->setup(pp_data, dof_handler, mapping, data, analytical_solution);
  }


  FE_DGQ<dim>          fe;
  MappingQGeneric<dim> mapping;
  DoFHandler<dim>      dof_handler;

  std::shared_ptr<FESystem<dim>>   fe_velocity;
  std::shared_ptr<DoFHandler<dim>> dof_handler_velocity;

  MatrixFree<dim, Number> data;

  InputParameters const & param;

  // TODO This variable is only needed when using the GeometricMultigrid preconditioner
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>>
    periodic_face_pairs;

  std::shared_ptr<BoundaryDescriptor<dim>> boundary_descriptor;
  std::shared_ptr<FieldFunctions<dim>>     field_functions;

  MassMatrixOperator<dim, degree, Number>           mass_matrix_operator;
  InverseMassMatrixOperator<dim, degree, Number, 1> inverse_mass_matrix_operator;
  ConvectiveOperator<dim, degree, Number>           convective_operator;
  DiffusiveOperator<dim, degree, Number>            diffusive_operator;
  RHSOperator<dim, degree, Number>                  rhs_operator;

  ConvectiveOperatorDisVel<dim, degree, degree, Number> convective_operator_discontinuous;

  mutable std::vector<VectorType const *> velocities;
  mutable std::vector<double>             times;

  mutable VectorType velocity;

  ConvectionDiffusionOperator<dim, degree, Number> conv_diff_operator;

  // convection-diffusion operator for runtime optimization (also includes rhs operator)
  ConvectionDiffusionOperatorEfficiency<dim, degree, Number>
    convection_diffusion_operator_efficiency;

  std::shared_ptr<PreconditionerBase<Number>> preconditioner;

  std::shared_ptr<IterativeSolverBase<VectorType>> iterative_solver;

  // postprocessor
  std::shared_ptr<PostProcessor<dim, degree>> postprocessor;
};

} // namespace ConvDiff

#endif /* INCLUDE_CONVECTION_DIFFUSION_DG_CONVECTION_DIFFUSION_OPERATION_H_ */
