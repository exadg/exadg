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
class ConvectionDiffusionOperator : public OperatorBase<dim, fe_degree, Number,
                                               ConvectionDiffusionOperatorData<dim>> {
public:
  // TODO: Issue#2
  typedef Number value_type;
  typedef ConvectionDiffusionOperator<dim,fe_degree,Number> This;
  static const int DIM = dim;
  typedef OperatorBase<dim, fe_degree, value_type, ConvectionDiffusionOperatorData<dim>>
      Parent;
  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;
  typedef typename Parent::BMatrix BMatrix;

  ConvectionDiffusionOperator()
    :
    mass_matrix_operator(nullptr),
    convective_operator(nullptr),
    diffusive_operator(nullptr),
    scaling_factor_time_derivative_term(-1.0)
  {}

  void initialize(MatrixFree<dim,Number> const                     &mf_data_in,
                  ConvectionDiffusionOperatorData<dim> const       &operator_data_in,
                  MassMatrixOperator<dim, fe_degree, Number> const &mass_matrix_operator_in,
                  ConvectiveOperator<dim, fe_degree, Number> const &convective_operator_in,
                  DiffusiveOperator<dim, fe_degree, Number> const  &diffusive_operator_in)
  {
    // copy parameters into element variables
//    this->data = &mf_data_in;
//    this->ad = operator_data_in;
      ConstraintMatrix cm;
      Parent::reinit(mf_data_in, cm, operator_data_in);
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
    this->initialize_dof_vector(temp);
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
   *  This function does nothing in case of the ConvectionDiffusionOperator.
   *  It is only necessary due to the interface of the multigrid preconditioner
   *  and especially the coarse grid solver that calls this function.
   */
  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const {}

  // Apply matrix-vector multiplication.
  void vmult (parallel::distributed::Vector<Number>       &dst,
              parallel::distributed::Vector<Number> const &src) const
  {
    if(this->ad.unsteady_problem == true)
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

    if(this->ad.diffusive_problem == true)
    {
      diffusive_operator->apply_add(dst,src);
    }

    if(this->ad.convective_problem == true)
    {
      convective_operator->apply_add(dst,src/*TODO: ,evaluation_time*/);
    }
  }

  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 parallel::distributed::Vector<Number> const &src) const
  {
    if(this->ad.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

      mass_matrix_operator->apply(temp,src);
      temp *= scaling_factor_time_derivative_term;
      dst += temp;
    }

    if(this->ad.diffusive_problem == true)
    {
      diffusive_operator->apply_add(dst,src);
    }

    if(this->ad.convective_problem == true)
    {
      convective_operator->apply_add(dst,src/*TODO: ,evaluation_time*/);
    }
  }

private:
  /*
   *  This function calculates the diagonal of the scalar reaction-convection-diffusion operator.
   */
  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    if(this->ad.unsteady_problem == true)
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

    if(this->ad.diffusive_problem == true)
    {
      diffusive_operator->add_diagonal(diagonal);
    }

    if(this->ad.convective_problem == true)
    {
      convective_operator->add_diagonal(diagonal/*TODO: ,evaluation_time*/);
    }
  }

  /*
   * This function calculates the block Jacobi matrices.
   * This is done sequentially for the different operators.
   */
  void add_block_jacobi_matrices(BMatrix &matrices) const
  {
    // calculate block Jacobi matrices
    if(this->ad.unsteady_problem == true)
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

    if(this->ad.diffusive_problem == true)
    {
      diffusive_operator->add_block_jacobi_matrices(matrices);
    }

    if(this->ad.convective_problem == true)
    {
      convective_operator->add_block_jacobi_matrices(matrices/*TODO: ,evaluation_time*/);
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
  
  private:
  void do_cell_integral(FEEvalCell &) const{}

  void do_face_integral(FEEvalFace &, FEEvalFace &) const {}

  void do_face_int_integral(FEEvalFace &, FEEvalFace &) const {}

  void do_face_ext_integral(FEEvalFace &, FEEvalFace &) const {}

  void do_boundary_integral(FEEvalFace &, OperatorType const &,
                            types::boundary_id const &) const {}

  MassMatrixOperator<dim, fe_degree, Number>  const *mass_matrix_operator;
  ConvectiveOperator<dim, fe_degree, Number> const *convective_operator;
  DiffusiveOperator<dim, fe_degree, Number>  const *diffusive_operator;
  parallel::distributed::Vector<Number> mutable temp;
  double scaling_factor_time_derivative_term;

  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, Number> own_mass_matrix_operator_storage;
  ConvectiveOperator<dim, fe_degree, Number> own_convective_operator_storage;
  DiffusiveOperator<dim, fe_degree, Number> own_diffusive_operator_storage;
};
    
}

#endif