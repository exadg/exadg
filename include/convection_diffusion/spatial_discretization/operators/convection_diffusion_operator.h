#ifndef CONV_DIFF_CONVECTION_DIFFUSION_OPERATOR
#define CONV_DIFF_CONVECTION_DIFFUSION_OPERATOR

#include "../../../operators/operation_base.h"

namespace ConvDiff {
    
    
template<int dim>
struct ConvectionDiffusionOperatorData
    : public OperatorBaseData<dim, BoundaryType, OperatorType,
                              ConvDiff::BoundaryDescriptor<dim>> {
  ConvectionDiffusionOperatorData()
      : OperatorBaseData<dim, BoundaryType, OperatorType,
                         ConvDiff::BoundaryDescriptor<dim>>(
            0, 0),
    unsteady_problem(true),
    convective_problem(true),
    diffusive_problem(true),
    mg_operator_type(MultigridOperatorType::Undefined) {}

  bool unsteady_problem;
  bool convective_problem;
  bool diffusive_problem;
  MultigridOperatorType mg_operator_type;
};

template <int dim, int fe_degree, typename Number = double>
class ConvectionDiffusionOperator : public MatrixOperatorBaseNew<dim, Number>
{
public:
  // TODO: Issue#2
  typedef Number value_type;
  typedef ConvectionDiffusionOperator<dim,fe_degree,Number> This;
  static const int DIM = dim;

  ConvectionDiffusionOperator()
    :
    block_jacobi_matrices_have_been_initialized(false),
    data(nullptr),
    mass_matrix_operator(nullptr),
    convective_operator(nullptr),
    diffusive_operator(nullptr),
    scaling_factor_time_derivative_term(-1.0),
    evaluation_time(0.0)
  {}

  void initialize(MatrixFree<dim,Number> const                     &mf_data_in,
                  ConvectionDiffusionOperatorData<dim> const       &operator_data_in,
                  MassMatrixOperator<dim, fe_degree, Number> const &mass_matrix_operator_in,
                  ConvectiveOperator<dim, fe_degree, Number> const &convective_operator_in,
                  DiffusiveOperator<dim, fe_degree, Number> const  &diffusive_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->operator_data = operator_data_in;
    this->mass_matrix_operator = &mass_matrix_operator_in;
    this->convective_operator = &convective_operator_in;
    this->diffusive_operator = &diffusive_operator_in;
  }


  /*
   *  This function is called by the multigrid algorithm to initialize the
   *  matrices on all levels. To construct the matrices, and object of
   *  type UnderlyingOperator is used that provides all the information for
   *  the setup, i.e., the information that is needed to call the
   *  member function initialize(...).
   */
  template<typename UnderlyingOperator>
  void initialize_mg_matrix (unsigned int const                               level,
                             DoFHandler<dim> const                            &dof_handler,
                             Mapping<dim> const                               &mapping,
                             UnderlyingOperator const                         &underlying_operator,
                             std::vector<GridTools::PeriodicFacePair<typename
                             Triangulation<dim>::cell_iterator> > const       &/*periodic_face_pairs_level0*/)
  {
    // setup own matrix free object
    QGauss<1> const quad(dof_handler.get_fe().degree+1);
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    if (dof_handler.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;

    addit_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                       update_quadrature_points | update_normal_vectors |
                                       update_values);

    addit_data.mapping_update_flags_inner_faces = (update_gradients | update_JxW_values |
                                                   update_quadrature_points | update_normal_vectors |
                                                   update_values);

    addit_data.mapping_update_flags_boundary_faces = (update_gradients | update_JxW_values |
                                                      update_quadrature_points | update_normal_vectors |
                                                      update_values);

    addit_data.level_mg_handler = level;

    ConstraintMatrix constraints;
    // reinit
    own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

    // setup own mass matrix operator
    MassMatrixOperatorData<dim> mass_matrix_operator_data = underlying_operator.get_mass_matrix_operator_data();
    mass_matrix_operator_data.dof_index = 0;
    mass_matrix_operator_data.quad_index = 0;
    own_mass_matrix_operator_storage.initialize(own_matrix_free_storage,mass_matrix_operator_data);

    // setup own convective operator
    ConvectiveOperatorData<dim> convective_operator_data = underlying_operator.get_convective_operator_data();
    convective_operator_data.dof_index = 0;
    convective_operator_data.quad_index = 0;
    own_convective_operator_storage.initialize(own_matrix_free_storage,convective_operator_data);

    // setup own viscous operator
    DiffusiveOperatorData<dim> diffusive_operator_data = underlying_operator.get_diffusive_operator_data();
    diffusive_operator_data.dof_index = 0;
    diffusive_operator_data.quad_index = 0;
    own_diffusive_operator_storage.initialize(mapping,own_matrix_free_storage,diffusive_operator_data);

    // setup convection-diffusion operator
    ConvectionDiffusionOperatorData<dim> my_operator_data = underlying_operator.get_operator_data();

    // When solving the reaction-convection-diffusion equations, it might be possible
    // that one wants to apply the multigrid preconditioner only to the reaction-diffusion
    // operator (which is symmetric, Chebyshev smoother, etc.) instead of the non-symmetric
    // reaction-convection-diffusion operator. Accordingly, we have to reset which
    // operators should be "active" for the multigrid preconditioner, independently of
    // the actual equation type that is solved.
    AssertThrow(my_operator_data.mg_operator_type != MultigridOperatorType::Undefined,
        ExcMessage("Invalid parameter mg_operator_type."));

    if(my_operator_data.mg_operator_type == MultigridOperatorType::ReactionDiffusion)
    {
      my_operator_data.convective_problem = false; // deactivate convective term for multigrid preconditioner
      my_operator_data.diffusive_problem = true;
    }
    else if(my_operator_data.mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      my_operator_data.convective_problem = true;
      my_operator_data.diffusive_problem = true;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    initialize(own_matrix_free_storage,
               my_operator_data,
               own_mass_matrix_operator_storage,
               own_convective_operator_storage,
               own_diffusive_operator_storage);

    // Initialize other variables:

    // mass matrix term: set scaling factor time derivative term
    set_scaling_factor_time_derivative_term(underlying_operator.get_scaling_factor_time_derivative_term());

    // convective term: evaluation_time
    // This variables is not set here. If the convective term
    // is considered, this variables has to be updated anyway,
    // which is done somewhere else.

    // viscous term: nothing to do



    // initialize temp vector: this is done in this function because
    // the vector temp is only used in the function vmult_add(), i.e.,
    // when using the multigrid preconditioner
    initialize_dof_vector(temp);
  }

  /*
   *  Scaling factor of time derivative term (mass matrix term)
   */
  void set_scaling_factor_time_derivative_term(double const &factor)
  {
    scaling_factor_time_derivative_term = factor;
  }

  double get_scaling_factor_time_derivative_term() const
  {
    return scaling_factor_time_derivative_term;
  }

  /*
   *  Evaluation time that is needed for evaluation of convective operator.
   */
  void set_evaluation_time(double const &evaluation_time_in)
  {
    evaluation_time = evaluation_time_in;
  }

  double get_evaluation_time() const
  {
    return evaluation_time;
  }

  /*
   *  Operator data
   */
  ConvectionDiffusionOperatorData<dim> const & get_operator_data() const
  {
    return this->operator_data;
  }

  /*
   *  Operator data of basic operators: mass matrix, convective operator, diffusive operator
   */
  MassMatrixOperatorData<dim> const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator->get_operator_data();
  }

  ConvectiveOperatorData<dim> const & get_convective_operator_data() const
  {
    return convective_operator->get_operator_data();
  }

  DiffusiveOperatorData<dim> const & get_diffusive_operator_data() const
  {
    return diffusive_operator->get_operator_data();
  }

  /*
   *  MatrixFree data
   */
  MatrixFree<dim,value_type> const & get_data() const
  {
    return *data;
  }

  /*
   *  This function does nothing in case of the ConvectionDiffusionOperator.
   *  It is only necessary due to the interface of the multigrid preconditioner
   *  and especially the coarse grid solver that calls this function.
   */
  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const {}

  // Apply matrix-vector multiplication.
  void vmult (parallel::distributed::Vector<Number>       &dst,
              parallel::distributed::Vector<Number> const &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized!"));

      mass_matrix_operator->apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else
    {
      dst = 0.0;
    }

    if(operator_data.diffusive_problem == true)
    {
      diffusive_operator->apply_add(dst,src);
    }

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_add(dst,src/*TODO: ,evaluation_time*/);
    }
  }

  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 parallel::distributed::Vector<Number> const &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

      mass_matrix_operator->apply(temp,src);
      temp *= scaling_factor_time_derivative_term;
      dst += temp;
    }

    if(operator_data.diffusive_problem == true)
    {
      diffusive_operator->apply_add(dst,src);
    }

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_add(dst,src/*TODO: ,evaluation_time*/);
    }
  }

  // Apply matrix-vector multiplication (matrix-free) for global block Jacobi system.
  // Do that sequentially for the different operators.
  // This function is only needed for testing in order to make sure that the block Jacobi
  // matrices are calculated correctly.
  void vmult_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized!"));

      // mass matrix operator has already "block Jacobi form" in DG
      mass_matrix_operator->apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else
    {
      dst = 0.0;
    }

    if(operator_data.diffusive_problem == true)
    {
      diffusive_operator->apply_block_jacobi_add(dst,src);
    }

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_block_jacobi_add(dst,src,evaluation_time);
    }
  }

  void vmult_interface_down(parallel::distributed::Vector<Number>       &dst,
                            parallel::distributed::Vector<Number> const &src) const
  {
    vmult(dst,src);
  }

  void vmult_add_interface_up(parallel::distributed::Vector<Number>       &dst,
                              parallel::distributed::Vector<Number> const &src) const
  {
    vmult_add(dst,src);
  }

  types::global_dof_index m() const
  {
    return data->get_vector_partitioner(operator_data.dof_index)->size();
  }

  Number el (const unsigned int,  const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  unsigned int get_dof_index() const
  {
    return operator_data.dof_index;
  }

  unsigned int get_quad_index() const
  {
    // Operator data does not contain quad_index. Hence,
    // ask one of the basic operators (here we choose the mass matrix operator)
    // for the quadrature index.
    return get_mass_matrix_operator_data().quad_index;
  }

  /*
   *  This function initializes a global dof-vector.
   */
  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,operator_data.dof_index);
  }

  /*
   *  Calculation of inverse diagonal (needed for smoothers and preconditioners)
   */
  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    calculate_diagonal(diagonal);

    // test correctness of diagonal computation
//    verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  /*
   *  Apply block Jacobi preconditioner.
   */
  void apply_block_jacobi (parallel::distributed::Vector<Number>       &dst,
                           parallel::distributed::Vector<Number> const &src) const
  {
    /*
    // VARIANT 1: solve global system of equations using an iterative solver
    IterationNumberControl control (30,1.e-20,1.e-3);
    typename SolverGMRES<parallel::distributed::Vector<Number> >::AdditionalData additional_data;
    additional_data.right_preconditioning = true;
    additional_data.max_n_tmp_vectors = 100;
    SolverGMRES<parallel::distributed::Vector<Number> > solver (control,additional_data);

    typedef ConvectionDiffusionOperator<dim,fe_degree,Number> MY_TYPE;
    ConvectionDiffusionBlockJacobiOperator<MY_TYPE, Number> block_jacobi_operator(*this);

    dst = 0.0;
    solver.solve(block_jacobi_operator,dst,src,PreconditionIdentity());
    //std::cout<<"Number of iterations block Jacobi solve = "<<control.last_step()<<std::endl;
    */

    // VARIANT 2: calculate block jacobi matrices and solve block Jacobi problem
    // elementwise using a direct solver

    // apply_inverse_matrices
    data->cell_loop(&This::cell_loop_apply_inverse_block_jacobi_matrices, this, dst, src);
  }

  /*
   *  This function updates the block Jacobi preconditioner.
   *  Since this function also initializes the block Jacobi preconditioner,
   *  make sure that the block Jacobi matrices are allocated before calculating
   *  the matrices and the LU factorization.
   */
  void update_block_jacobi () const
  {
    if(block_jacobi_matrices_have_been_initialized == false)
    {
      matrices.resize(data->n_macro_cells()*VectorizedArray<Number>::n_array_elements,
        LAPACKFullMatrix<Number>(data->get_shape_info().dofs_per_component_on_cell, data->get_shape_info().dofs_per_component_on_cell));

      block_jacobi_matrices_have_been_initialized = true;
    }

    calculate_block_jacobi_matrices();
    calculate_lu_factorization_block_jacobi(matrices);
  }

private:
  /*
   *  This function calculates the diagonal of the scalar reaction-convection-diffusion operator.
   */
  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

      mass_matrix_operator->calculate_diagonal(diagonal);
      diagonal *= scaling_factor_time_derivative_term;
    }
    else
    {
      diagonal = 0.0;
    }

    if(operator_data.diffusive_problem == true)
    {
      diffusive_operator->add_diagonal(diagonal);
    }

    if(operator_data.convective_problem == true)
    {
      convective_operator->add_diagonal(diagonal/*TODO: ,evaluation_time*/);
    }
  }

  /*
   * This function calculates the block Jacobi matrices.
   * This is done sequentially for the different operators.
   */
  void calculate_block_jacobi_matrices() const
  {
    // initialize block Jacobi matrices with zeros
    initialize_block_jacobi_matrices_with_zero(matrices);

    // calculate block Jacobi matrices
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized!"));

      mass_matrix_operator->add_block_jacobi_matrices(matrices);

      for(typename std::vector<LAPACKFullMatrix<Number> >::iterator
          it = matrices.begin(); it != matrices.end(); ++it)
      {
        (*it) *= scaling_factor_time_derivative_term;
      }
    }

    if(operator_data.diffusive_problem == true)
    {
      diffusive_operator->add_block_jacobi_matrices(matrices);
    }

    if(operator_data.convective_problem == true)
    {
      convective_operator->add_block_jacobi_matrices(matrices/*TODO: ,evaluation_time*/);
    }

    // test correctness of block Jacobi matrices
//    check_block_jacobi_matrices();
  }

  /*
   *  Apply matrix-vector multiplication (matrix-based) for global block Jacobi system
   *  by looping over all cells and applying the matrix-based matrix-vector product cellwise.
   *  This function is only needed for testing.
   */
  void vmult_block_jacobi_test (parallel::distributed::Vector<Number>       &dst,
                                parallel::distributed::Vector<Number> const &src) const
  {
    data->cell_loop(&This::cell_loop_apply_block_jacobi_matrices_test, this, dst, src);
  }

  /*
   *  This function is only needed for testing.
   */
  void cell_loop_apply_block_jacobi_matrices_test (MatrixFree<dim,Number> const                &data,
                                                   parallel::distributed::Vector<Number>       &dst,
                                                   parallel::distributed::Vector<Number> const &src,
                                                   std::pair<unsigned int,unsigned int> const  &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,Number> fe_eval(data,
                                                             mass_matrix_operator->get_operator_data().dof_index,
                                                             mass_matrix_operator->get_operator_data().quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<Number> src_vector(fe_eval.dofs_per_cell);
        Vector<Number> dst_vector(fe_eval.dofs_per_cell);
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          src_vector(j) = fe_eval.begin_dof_values()[j][v];

        // apply matrix-vector product
        matrices[cell*VectorizedArray<Number>::n_array_elements+v].vmult(dst_vector,src_vector,false);

        // write solution to dst-vector
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j][v] = dst_vector(j);
      }

      fe_eval.set_dof_values (dst);
    }
  }

  /*
   * Verify computation of block Jacobi matrices.
   */
   void check_block_jacobi_matrices() const
   {
     parallel::distributed::Vector<Number> src;
     initialize_dof_vector(src);
     src = 1.0;

     // test matrix-vector product for block Jacobi problem by comparing
     // matrix-free matrix-vector product and matrix-based matrix-vector product
     // (where the matrices are generated using the matrix-free implementation)
     parallel::distributed::Vector<Number> tmp1(src), tmp2(src), diff(src);
     tmp1 = 0.0;
     tmp2 = 0.0;

     // variant 1 (matrix-free)
     vmult_block_jacobi(tmp1,src);

     // variant 2 (matrix-based)
     vmult_block_jacobi_test(tmp2,src);

     diff = tmp2;
     diff.add(-1.0,tmp1);

     std::cout << "L2 norm variant 1 = " << tmp1.l2_norm() << std::endl
               << "L2 norm variant 2 = " << tmp2.l2_norm() << std::endl
               << "L2 norm v2 - v1   = " << diff.l2_norm() << std::endl << std::endl;
   }

  /*
   *  This function loops over all cells and applies the inverse block Jacobi matrices elementwise.
   */
  void cell_loop_apply_inverse_block_jacobi_matrices (MatrixFree<dim,Number> const                &data,
                                                      parallel::distributed::Vector<Number>       &dst,
                                                      parallel::distributed::Vector<Number> const &src,
                                                      std::pair<unsigned int,unsigned int> const  &cell_range) const
  {
    // apply inverse block matrices
    FEEvaluation<dim,fe_degree,fe_degree+1,1,Number> fe_eval(data,
                                                             mass_matrix_operator->get_operator_data().dof_index,
                                                             mass_matrix_operator->get_operator_data().quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      for (unsigned int v=0; v<VectorizedArray<Number>::n_array_elements; ++v)
      {
        // fill source vector
        Vector<Number> src_vector(fe_eval.dofs_per_cell);
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          src_vector(j) = fe_eval.begin_dof_values()[j][v];

        // apply inverse matrix
        matrices[cell*VectorizedArray<Number>::n_array_elements+v].solve(src_vector,false);

        // write solution to dst-vector
        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
          fe_eval.begin_dof_values()[j][v] = src_vector(j);
      }

      fe_eval.set_dof_values (dst);
    }
  }
  
  MatrixOperatorBaseNew<dim, Number>* get_new(unsigned int deg) const {
    switch (deg) {
    case 1:
      return new ConvectionDiffusionOperator<dim, 1, Number>();
    case 2:
      return new ConvectionDiffusionOperator<dim, 2, Number>();
    case 3:
      return new ConvectionDiffusionOperator<dim, 3, Number>();
    case 4:
      return new ConvectionDiffusionOperator<dim, 4, Number>();
    case 5:
      return new ConvectionDiffusionOperator<dim, 5, Number>();
    case 6:
      return new ConvectionDiffusionOperator<dim, 6, Number>();
    case 7:
      return new ConvectionDiffusionOperator<dim, 7, Number>();
    default:
      AssertThrow(false,
                  ExcMessage("LaplaceOperator not implemented for this degree!"));
      return new ConvectionDiffusionOperator<dim, 1, Number>(); 
          // dummy return (statement not reached)
    }
  }

  mutable std::vector<LAPACKFullMatrix<Number> > matrices;
  mutable bool block_jacobi_matrices_have_been_initialized;

  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, Number>  const *mass_matrix_operator;
  ConvectiveOperator<dim, fe_degree, Number> const *convective_operator;
  DiffusiveOperator<dim, fe_degree, Number>  const *diffusive_operator;
  ConvectionDiffusionOperatorData<dim> operator_data;
  parallel::distributed::Vector<Number> mutable temp;
  double scaling_factor_time_derivative_term;
  double evaluation_time;

  /*
   * The following variables are necessary when applying the multigrid
   * preconditioner to the convection-diffusion operator. In that case, the
   * Helmholtz has to be generated for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of
   * MatrixFree, MassMatrixOperator, DiffusiveOperator,
   *   e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the convection-diffusion operator with these
   * ojects by setting the above pointers to the own_objects_storage,
   *   e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, Number> own_mass_matrix_operator_storage;
  ConvectiveOperator<dim, fe_degree, Number> own_convective_operator_storage;
  DiffusiveOperator<dim, fe_degree, Number> own_diffusive_operator_storage;
};
    
}

#endif