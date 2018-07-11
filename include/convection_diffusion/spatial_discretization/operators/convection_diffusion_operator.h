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
    mg_operator_type(MultigridOperatorType::Undefined),
    scaling_factor_time_derivative_term(-1.0) {}

  bool unsteady_problem;
  bool convective_problem;
  bool diffusive_problem;
  MultigridOperatorType mg_operator_type;
  
  double scaling_factor_time_derivative_term;
  
  MassMatrixOperatorData<dim> mass_matrix_operator_data;
  ConvectiveOperatorData<dim> convective_operator_data;
  DiffusiveOperatorData<dim> diffusive_operator_data;
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
    diffusive_operator(nullptr)
  {}

  void initialize(MatrixFree<dim,Number> const                     &mf_data_in,
                  ConvectionDiffusionOperatorData<dim> const       &operator_data_in,
                  MassMatrixOperator<dim, fe_degree, Number> const &mass_matrix_operator_in,
                  ConvectiveOperator<dim, fe_degree, Number> const &convective_operator_in,
                  DiffusiveOperator<dim, fe_degree, Number> const  &diffusive_operator_in)
  {
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
  void reinit (
          const DoFHandler<dim> &dof_handler, 
          const Mapping<dim> &mapping,
          void* od, 
          const MGConstrainedDoFs &mg_constrained_dofs, 
          const unsigned int level) {
      
    Parent::reinit(dof_handler, mapping, od, mg_constrained_dofs, level);
      
    // setup own mass matrix operator
    auto & mass_matrix_operator_data = this->ad.mass_matrix_operator_data;
    mass_matrix_operator_data.dof_index = 0;
    mass_matrix_operator_data.quad_index = 0;
    own_mass_matrix_operator_storage.initialize(
        this->get_data(), mass_matrix_operator_data);

    // setup own convective operator
    auto & convective_operator_data = this->ad.convective_operator_data;
    convective_operator_data.dof_index = 0;
    convective_operator_data.quad_index = 0;
    own_convective_operator_storage.initialize(
        this->get_data(), convective_operator_data);

    // setup own viscous operator
    auto & diffusive_operator_data = this->ad.diffusive_operator_data;
    diffusive_operator_data.dof_index = 0;
    diffusive_operator_data.quad_index = 0;
    own_diffusive_operator_storage.initialize(mapping,
            this->get_data() ,diffusive_operator_data);

    // When solving the reaction-convection-diffusion equations, it might be possible
    // that one wants to apply the multigrid preconditioner only to the reaction-diffusion
    // operator (which is symmetric, Chebyshev smoother, etc.) instead of the non-symmetric
    // reaction-convection-diffusion operator. Accordingly, we have to reset which
    // operators should be "active" for the multigrid preconditioner, independently of
    // the actual equation type that is solved.
    AssertThrow(this->ad.mg_operator_type != MultigridOperatorType::Undefined,
        ExcMessage("Invalid parameter mg_operator_type."));

    if(this->ad.mg_operator_type == MultigridOperatorType::ReactionDiffusion)
    {
      this->ad.convective_problem = false; // deactivate convective term for multigrid preconditioner
      this->ad.diffusive_problem = true;
    }
    else if(this->ad.mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      this->ad.convective_problem = true;
      this->ad.diffusive_problem = true;
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }
    
    this->mass_matrix_operator = &own_mass_matrix_operator_storage;
    this->convective_operator = &own_convective_operator_storage;
    this->diffusive_operator = &own_diffusive_operator_storage;

    // Initialize other variables:

    // mass matrix term: set scaling factor time derivative term
    //set_scaling_factor_time_derivative_term(underlying_operator.get_scaling_factor_time_derivative_term());

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
    this->ad.scaling_factor_time_derivative_term = factor;
  }

  double get_scaling_factor_time_derivative_term() const
  {
    return this->ad.scaling_factor_time_derivative_term;
  }

  /*
   *  Operator data of basic operators: mass matrix, convective operator, diffusive operator
   */
  MassMatrixOperatorData<dim> const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator->get_operator_data(); // TODO: get it from data
  }

  ConvectiveOperatorData<dim> const & get_convective_operator_data() const
  {
    return convective_operator->get_operator_data(); // TODO: get it from data
  }

  DiffusiveOperatorData<dim> const & get_diffusive_operator_data() const
  {
    return diffusive_operator->get_operator_data(); // TODO: get it from data
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
      AssertThrow(this->ad.scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized!"));

      mass_matrix_operator->apply(dst,src);
      dst *= this->ad.scaling_factor_time_derivative_term;
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
      AssertThrow(this->ad.scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

      mass_matrix_operator->apply(temp,src);
      temp *= this->ad.scaling_factor_time_derivative_term;
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
      AssertThrow(this->ad.scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for convection-diffusion operator!"));

      mass_matrix_operator->calculate_diagonal(diagonal);
      diagonal *= this->ad.scaling_factor_time_derivative_term;
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
      AssertThrow(this->ad.scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized!"));

      mass_matrix_operator->add_block_jacobi_matrices(matrices);

      for(typename std::vector<LAPACKFullMatrix<Number> >::iterator
          it = matrices.begin(); it != matrices.end(); ++it)
      {
        (*it) *= this->ad.scaling_factor_time_derivative_term;
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
//    case 2:
//      return new ConvectionDiffusionOperator<dim, 2, Number>();
    case 3:
      return new ConvectionDiffusionOperator<dim, 3, Number>();
//    case 4:
//      return new ConvectionDiffusionOperator<dim, 4, Number>();
//    case 5:
//      return new ConvectionDiffusionOperator<dim, 5, Number>();
//    case 6:
//      return new ConvectionDiffusionOperator<dim, 6, Number>();
//    case 7:
//      return new ConvectionDiffusionOperator<dim, 7, Number>();
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

  MassMatrixOperator<dim, fe_degree, Number> own_mass_matrix_operator_storage;
  ConvectiveOperator<dim, fe_degree, Number> own_convective_operator_storage;
  DiffusiveOperator<dim, fe_degree, Number> own_diffusive_operator_storage;
};
    
}

#endif