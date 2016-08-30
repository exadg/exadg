/*
 * DGConvDiffOperation.h
 *
 *  Created on: Aug 2, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_DGCONVDIFFOPERATION_H_
#define INCLUDE_DGCONVDIFFOPERATION_H_

using namespace dealii;

// operators
#include "../include/InverseMassMatrix.h"
#include "../include/ScalarConvectionDiffusionOperators.h"

// preconditioner, solver
#include "../include/Preconditioner.h"
#include "../include/IterativeSolvers.h"


#include "../include/BoundaryDescriptorConvDiff.h"
#include "../include/FieldFunctionsConvDiff.h"

#include "InputParametersConvDiff.h"

template<int dim, int fe_degree, typename value_type>
class DGConvDiffOperation
{
public:

  DGConvDiffOperation(parallel::distributed::Triangulation<dim> const &triangulation,
                      ConvDiff::InputParametersConvDiff const         &param_in)
    :
    fe(QGaussLobatto<1>(fe_degree+1)),
    mapping(fe_degree),
    dof_handler(triangulation),
    param(param_in),
    scaling_factor_time_derivative_term(-1.0),
    evaluation_time(0.0)
  {}

  void setup(const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> >
                                                                     periodic_face_pairs,
             std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor_in,
             std_cxx11::shared_ptr<FieldFunctionsConvDiff<dim> >     field_functions_in)
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
    // initialize preconditioner
    if(param.preconditioner == ConvDiff::Preconditioner::InverseMassMatrix)
    {
      preconditioner.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type,1>(data,0,0));
    }
    else if(param.preconditioner == ConvDiff::Preconditioner::Jacobi)
    {
      preconditioner.reset(new JacobiPreconditioner<value_type,
                                 DGConvDiffOperation<dim,fe_degree,value_type> >
                               (*this));
    }
    else if(param.preconditioner == ConvDiff::Preconditioner::GeometricMultigrid)
    {
      MultigridData mg_data;
      mg_data = param.multigrid_data;

      ScalarConvDiffOperators::HelmholtzOperatorData<dim> helmholtz_operator_data;
      helmholtz_operator_data.dof_index = 0;
      helmholtz_operator_data.mass_matrix_coefficient = this->scaling_factor_time_derivative_term;
      helmholtz_operator_data.mass_matrix_operator_data = mass_matrix_operator_data;
      helmholtz_operator_data.diffusive_operator_data = diffusive_operator_data;
      helmholtz_operator_data.periodic_face_pairs_level0 = periodic_face_pairs;

      // use single precision for multigrid
      typedef float Number;

      // fill dirichlet_boundary set
      fill_dbc_set(boundary_descriptor);

      preconditioner.reset(new MyMultigridPreconditioner<dim,value_type,
                                ScalarConvDiffOperators::HelmholtzOperator<dim,fe_degree,Number>,
                                ScalarConvDiffOperators::HelmholtzOperatorData<dim> >
                               (mg_data,
                                dof_handler,
                                mapping,
                                helmholtz_operator_data,
                                dirichlet_boundary));
    }
    else
    {
      AssertThrow(param.preconditioner == ConvDiff::Preconditioner::None ||
                  param.preconditioner == ConvDiff::Preconditioner::InverseMassMatrix ||
                  param.preconditioner == ConvDiff::Preconditioner::Jacobi ||
                  param.preconditioner == ConvDiff::Preconditioner::GeometricMultigrid,
                  ExcMessage("Specified preconditioner is not implemented!"));
    }


    if(param.solver == ConvDiff::Solver::PCG)
    {
      // initialize solver_data
      CGSolverData solver_data;
      solver_data.solver_tolerance_abs = param.abs_tol;
      solver_data.solver_tolerance_rel = param.rel_tol;
      solver_data.max_iter = param.max_iter;

      if(param.preconditioner != ConvDiff::Preconditioner::None)
        solver_data.use_preconditioner = true;

      // initialize solver
      iterative_solver.reset(new CGSolver<DGConvDiffOperation<dim,fe_degree,value_type>,
                                          PreconditionerBase<value_type>,
                                          parallel::distributed::Vector<value_type> >
                                 (*this,*preconditioner,solver_data));
    }
    else if(param.solver == ConvDiff::Solver::GMRES)
    {
      // initialize solver_data
      GMRESSolverData solver_data;
      solver_data.solver_tolerance_abs = param.abs_tol;
      solver_data.solver_tolerance_rel = param.rel_tol;
      solver_data.max_iter = param.max_iter;
      solver_data.right_preconditioning = param.use_right_preconditioner;
      solver_data.max_n_tmp_vectors = param.max_n_tmp_vectors;

      if(param.preconditioner != ConvDiff::Preconditioner::None)
        solver_data.use_preconditioner = true;

      // initialize solver
      iterative_solver.reset(new GMRESSolver<DGConvDiffOperation<dim,fe_degree,value_type>,
                                             PreconditionerBase<value_type>,
                                             parallel::distributed::Vector<value_type> >
                                 (*this,*preconditioner,solver_data));
    }
    else
    {
      AssertThrow(param.solver == ConvDiff::Solver::PCG ||
                  param.solver == ConvDiff::Solver::GMRES,
                  ExcMessage("Specified solver is not implemented!"));
    }
  }

  void initialize_dof_vector(parallel::distributed::Vector<value_type> &src) const
  {
    data.initialize_dof_vector(src);
  }

  void prescribe_initial_conditions(parallel::distributed::Vector<value_type> &src,
                                    const double                              evaluation_time) const
  {
    field_functions->analytical_solution->set_time(evaluation_time);
    VectorTools::interpolate(dof_handler, *(field_functions->analytical_solution), src);
  }

  // getters
  MatrixFree<dim,value_type> const & get_data() const
  {
    return data;
  }

  void evaluate(parallel::distributed::Vector<value_type>       &dst,
                parallel::distributed::Vector<value_type> const &src,
                const value_type                                evaluation_time) const
  {
    if(param.runtime_optimization == false) //apply volume and surface integrals for each operator separately
    {
      // set dst to zero
      dst = 0.0;

      // diffusive operator
      if(param.equation_type == ConvDiff::EquationType::Diffusion ||
         param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
      {
        diffusive_operator.evaluate_add(dst,src,evaluation_time);
      }

      // convective operator
      if(param.equation_type == ConvDiff::EquationType::Convection ||
         param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
      {
        convective_operator.evaluate_add(dst,src,evaluation_time);
      }

      // shift diffusive and convective term to the rhs of the equation
      dst *= -1.0;

      if(param.right_hand_side == true)
      {
        rhs_operator.evaluate_add(dst,evaluation_time);
      }
    }
    else // param.runtime_optimization == true
    {
      convection_diffusion_operator.evaluate(dst,src,evaluation_time);
    }

    // apply inverse mass matrix
    inverse_mass_matrix_operator.apply_inverse_mass_matrix(dst,dst);
  }

  void evaluate_convective_term(parallel::distributed::Vector<value_type>       &dst,
                                parallel::distributed::Vector<value_type> const &src,
                                const value_type                                evaluation_time) const
  {
    convective_operator.evaluate(dst,src,evaluation_time);
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
           parallel::distributed::Vector<value_type> const *src,
           double const evaluation_time) const
  {
    // mass matrix operator
    if(param.problem_type == ConvDiff::ProblemType::Steady)
    {
      dst = 0;
    }
    else if(param.problem_type == ConvDiff::ProblemType::Unsteady)
    {
      mass_matrix_operator.apply(dst,*src);
    }
    else
    {
      AssertThrow(param.problem_type == ConvDiff::ProblemType::Steady ||
                  param.problem_type == ConvDiff::ProblemType::Unsteady,
                  ExcMessage("Specified problem type for convection-diffusion equation not implemented."));
    }

    // diffusive operator
    if(param.equation_type == ConvDiff::EquationType::Diffusion ||
       param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
    {
      diffusive_operator.rhs_add(dst,evaluation_time);
    }

    // convective operator
    if((param.equation_type == ConvDiff::EquationType::Convection ||
        param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
        &&
       param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Implicit)
    {
      convective_operator.rhs_add(dst,evaluation_time);
    }

    if(param.right_hand_side == true)
    {
      rhs_operator.evaluate_add(dst,evaluation_time);
    }
  }

  void vmult(parallel::distributed::Vector<value_type>       &dst,
             parallel::distributed::Vector<value_type> const &src) const
  {
    apply(dst,src);
  }

  unsigned int solve(parallel::distributed::Vector<value_type>       &sol,
                     parallel::distributed::Vector<value_type> const &rhs,
                     double const scaling_factor_time_derivative_term_in,
                     double const evaluation_time_in)
  {
    this->scaling_factor_time_derivative_term = scaling_factor_time_derivative_term_in;
    this->evaluation_time = evaluation_time_in;

    unsigned int iterations = iterative_solver->solve(sol,rhs);

    return iterations;
  }

  void set_scaling_factor_time_derivative_term(double const value)
  {
    scaling_factor_time_derivative_term = value;
  }

  /*
   *  This function is called by the Jacobi preconditioner to calculate the diagonal.
   *  Note that the convective term is currently neglected.
   */
  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    if(param.problem_type == ConvDiff::ProblemType::Steady)
    {
      diagonal = 0;
    }
    else if(param.problem_type == ConvDiff::ProblemType::Unsteady)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
                  ExcMessage("Scaling factor of time derivative term has not been initialized!"));

      mass_matrix_operator.calculate_diagonal(diagonal);
      diagonal *= scaling_factor_time_derivative_term;
    }

    if(param.equation_type == ConvDiff::EquationType::Diffusion ||
       param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
    {
      diffusive_operator.add_diagonal(diagonal);
    }
  }

  void invert_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      if( std::abs(diagonal.local_element(i)) > 1.0e-10 )
        diagonal.local_element(i) = 1.0/diagonal.local_element(i);
      else
        diagonal.local_element(i) = 1.0;
    }
  }

  void calculate_inverse_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    calculate_diagonal(diagonal);

    invert_diagonal(diagonal);
  }

  Mapping<dim> const & get_mapping() const
  {
    return mapping;
  }

private:
  void create_dofs()
  {
    // enumerate degrees of freedom
    dof_handler.distribute_dofs(fe);
    dof_handler.distribute_mg_dofs(fe);

    unsigned int ndofs_per_cell = Utilities::fixed_int_power<fe_degree+1,dim>::value;

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
    additional_data.mpi_communicator = MPI_COMM_WORLD;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim,value_type>::AdditionalData::partition_partition;
    additional_data.build_face_info = true;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                        update_quadrature_points | update_normal_vectors |
                        update_values);

    ConstraintMatrix dummy;
    dummy.close();
    data.reinit (mapping, dof_handler, dummy, quadrature, additional_data);
  }

  void setup_operators()
  {
    // mass matrix operator
    mass_matrix_operator_data.dof_index = 0;
    mass_matrix_operator_data.quad_index = 0;
    mass_matrix_operator.initialize(data,mass_matrix_operator_data);

    // inverse mass matrix operator
    // dof_index = 0, quad_index = 0
    inverse_mass_matrix_operator.initialize(data,0,0);

    // convective operator
    ScalarConvDiffOperators::ConvectiveOperatorData<dim> convective_operator_data;
    convective_operator_data.dof_index = 0;
    convective_operator_data.quad_index = 0;
    convective_operator_data.numerical_flux_formulation = param.numerical_flux_convective_operator;
    convective_operator_data.bc = boundary_descriptor;
    convective_operator_data.velocity = field_functions->velocity;
    convective_operator.initialize(data,convective_operator_data);

    // diffusive operator
    diffusive_operator_data.dof_index = 0;
    diffusive_operator_data.quad_index = 0;
    diffusive_operator_data.IP_factor = param.IP_factor;
    diffusive_operator_data.diffusivity = param.diffusivity;
    diffusive_operator_data.bc = boundary_descriptor;
    diffusive_operator.initialize(mapping,data,diffusive_operator_data);

    // rhs operator
    ScalarConvDiffOperators::RHSOperatorData<dim> rhs_operator_data;
    rhs_operator_data.dof_index = 0;
    rhs_operator_data.quad_index = 0;
    rhs_operator_data.rhs = field_functions->right_hand_side;
    rhs_operator.initialize(data,rhs_operator_data);

    // convection-diffusion operator (also includes rhs operator)
    ScalarConvDiffOperators::ConvectionDiffusionOperatorData<dim> conv_diff_operator_data;
    conv_diff_operator_data.conv_data = convective_operator_data;
    conv_diff_operator_data.diff_data = diffusive_operator_data;
    conv_diff_operator_data.rhs_data = rhs_operator_data;
    convection_diffusion_operator.initialize(mapping, data, conv_diff_operator_data);

  }

  // TODO This function is only needed when using the GeometricMultigrid preconditioner
  // which expects the set 'dirichlet_boundary' as input parameter
  void fill_dbc_set(std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor)
  {
    // Dirichlet boundary conditions: copy Dirichlet boundary ID's from
    // boundary_descriptor.dirichlet_bc (map) to dirichlet_boundary (set)
    for (typename std::map<types::boundary_id,std_cxx11::shared_ptr<Function<dim> > >::
         const_iterator it = boundary_descriptor->dirichlet_bc.begin();
         it != boundary_descriptor->dirichlet_bc.end(); ++it)
    {
      dirichlet_boundary.insert(it->first);
    }
  }

  /*
   *  This function implements the matrix-vector product for the
   *  convection-diffusion equation. Hence, only the homogeneous parts
   *  of the operators are evaluated.
   *  Note that the convective operator only has a contribution if
   *  it is treated implicitly. In case of an explicit treatment the
   *  convective term occurs on the right-hand side of the equation
   *  but not in the matrix-vector product.
   */
  void apply(parallel::distributed::Vector<value_type>       &dst,
             parallel::distributed::Vector<value_type> const &src) const
  {
    // mass matrix operator
    if(param.problem_type == ConvDiff::ProblemType::Steady)
    {
      dst = 0;
    }
    else if(param.problem_type == ConvDiff::ProblemType::Unsteady)
    {
      mass_matrix_operator.apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else
    {
      AssertThrow(param.problem_type == ConvDiff::ProblemType::Steady ||
                  param.problem_type == ConvDiff::ProblemType::Unsteady,
                  ExcMessage("Specified problem type for convection-diffusion equation not implemented."));
    }

    // diffusive and convective operator
    if(param.equation_type == ConvDiff::EquationType::Diffusion ||
       param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
    {
      diffusive_operator.apply_add(dst,src);
    }

    if((param.equation_type == ConvDiff::EquationType::Convection ||
        param.equation_type == ConvDiff::EquationType::ConvectionDiffusion)
       &&
       param.treatment_of_convective_term == ConvDiff::TreatmentOfConvectiveTerm::Implicit)
    {
      convective_operator.apply_add(dst,src,evaluation_time);
    }
  }


  FE_DGQArbitraryNodes<dim> fe;
  MappingQGeneric<dim> mapping;
  DoFHandler<dim> dof_handler;

  MatrixFree<dim,value_type> data;

  ConvDiff::InputParametersConvDiff const &param;

  double scaling_factor_time_derivative_term;
  double evaluation_time;

  // TODO This variable is only needed when using the GeometricMultigrid preconditioner
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs;

  std_cxx11::shared_ptr<BoundaryDescriptorConvDiff<dim> > boundary_descriptor;
  std_cxx11::shared_ptr<FieldFunctionsConvDiff<dim> > field_functions;

  // TODO This variable is only needed when using the GeometricMultigrid preconditioner
  // which expects the set 'dirichlet_boundary' as input parameter
  std::set<types::boundary_id> dirichlet_boundary;

  ScalarConvDiffOperators::MassMatrixOperatorData mass_matrix_operator_data;
  ScalarConvDiffOperators::MassMatrixOperator<dim, fe_degree, value_type> mass_matrix_operator;
  InverseMassMatrixOperator<dim,fe_degree,value_type> inverse_mass_matrix_operator;

  ScalarConvDiffOperators::ConvectiveOperator<dim, fe_degree, value_type> convective_operator;

  ScalarConvDiffOperators::DiffusiveOperatorData<dim> diffusive_operator_data;
  ScalarConvDiffOperators::DiffusiveOperator<dim, fe_degree, value_type> diffusive_operator;
  ScalarConvDiffOperators::RHSOperator<dim, fe_degree, value_type> rhs_operator;

  // convection-diffusion operator for runtime optimization (also includes rhs operator)
  ScalarConvDiffOperators::ConvectionDiffusionOperator<dim, fe_degree, value_type> convection_diffusion_operator;

  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner;
  std_cxx11::shared_ptr<IterativeSolverBase<parallel::distributed::Vector<value_type> > > iterative_solver;

};


#endif /* INCLUDE_DGCONVDIFFOPERATION_H_ */
