/*
 * VelocityConvDiffOperator.h
 *
 *  Created on: Aug 8, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_VELOCITYCONVDIFFOPERATOR_H_
#define INCLUDE_VELOCITYCONVDIFFOPERATOR_H_

#include "MatrixOperatorBase.h"
#include "NavierStokesOperators.h"

//template<int dim>
//struct VelocityConvDiffOperatorData
//{
//  VelocityConvDiffOperatorData ()
//    :
//    unsteady_problem(true),
//    convective_problem(true)
//  {}
//
//  bool unsteady_problem;
//  bool convective_problem;
//};
//
//template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule,typename Number = double>
//class VelocityConvDiffOperator : public Subscriptor
//{
//public:
//  typedef Number value_type;
//
//  VelocityConvDiffOperator()
//    :
//    data(nullptr),
//    mass_matrix_operator(nullptr),
//    viscous_operator(nullptr),
//    convective_operator(nullptr),
//    scaling_factor_time_derivative_term(nullptr),
//    velocity_linearization(nullptr),
//    evaluation_time(0.0)
//  {}
//
//  void initialize(MatrixFree<dim,Number> const                                                            &mf_data_in,
//                  VelocityConvDiffOperatorData<dim> const                                                 &operator_data_in,
//                  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const &mass_matrix_operator_in,
//                  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const     &viscous_operator_in,
//                  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const  &convective_operator_in)
//  {
//    // copy parameters into element variables
//    this->data = &mf_data_in;
//    this->operator_data = operator_data_in;
//    this->mass_matrix_operator = &mass_matrix_operator_in;
//    this->viscous_operator = &viscous_operator_in;
//    this->convective_operator = &convective_operator_in;
//  }
//
//  void set_scaling_factor_time_derivative_term(Number const *factor)
//  {
//    scaling_factor_time_derivative_term = factor;
//  }
//
//  void set_velocity_linearization(parallel::distributed::Vector<Number> const *velocity_linearization_in)
//  {
//    velocity_linearization = velocity_linearization_in;
//  }
//
//  void set_evaluation_time(Number const &evaluation_time_in)
//  {
//    evaluation_time = evaluation_time_in;
//  }
//
//  // apply matrix vector multiplication
//  void vmult (parallel::distributed::Vector<Number>       &dst,
//              const parallel::distributed::Vector<Number> &src) const
//  {
//    AssertThrow(scaling_factor_time_derivative_term != nullptr,
//      ExcMessage("Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));
//
//    if(operator_data.unsteady_problem == true)
//    {
//      mass_matrix_operator->apply(dst,src);
//      dst *= (*scaling_factor_time_derivative_term);
//    }
//    else
//    {
//      dst = 0.0;
//    }
//
//    viscous_operator->apply_add(dst,src);
//
//    if(operator_data.convective_problem == true)
//    {
//      convective_operator->apply_linearized_add(dst,src,velocity_linearization,evaluation_time);
//    }
//  }
//
//private:
//  MatrixFree<dim,Number> const * data;
//  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const * mass_matrix_operator;
//  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const * viscous_operator;
//  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const * convective_operator;
//  VelocityConvDiffOperatorData<dim> operator_data;
//  Number const * scaling_factor_time_derivative_term;
//  parallel::distributed::Vector<Number> const * velocity_linearization;
//  Number evaluation_time;
//};



template<int dim>
struct VelocityConvDiffOperatorData
{
  VelocityConvDiffOperatorData ()
    :
    unsteady_problem(true),
    convective_problem(true),
    dof_index(0)
  {}

  bool unsteady_problem;
  bool convective_problem;
  unsigned int dof_index;
};

template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule,typename Number = double>
class VelocityConvDiffOperator : public MatrixOperatorBase
{
public:
  typedef Number value_type;

  VelocityConvDiffOperator()
    :
    data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    convective_operator(nullptr),
    scaling_factor_time_derivative_term(-1.0),
    evaluation_time(0.0)
  {}

  void initialize(MatrixFree<dim,Number> const                                                        &mf_data_in,
                  VelocityConvDiffOperatorData<dim> const                                             &operator_data_in,
                  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const &mass_matrix_operator_in,
                  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const     &viscous_operator_in,
                  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const  &convective_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->operator_data = operator_data_in;
    this->mass_matrix_operator = &mass_matrix_operator_in;
    this->viscous_operator = &viscous_operator_in;
    this->convective_operator = &convective_operator_in;

    if(operator_data.convective_problem == true)
    {
      this->initialize_dof_vector(velocity_linearization);
    }
  }

  template<typename UnderlyingOperator>
  void initialize_mg_matrix (unsigned int const       level,
                             DoFHandler<dim> const    &dof_handler,
                             Mapping<dim> const       &mapping,
                             UnderlyingOperator const &underlying_operator,
                             const std::vector<GridTools::PeriodicFacePair<typename
                               Triangulation<dim>::cell_iterator> > &periodic_face_pairs_level0)
  {
    // setup own matrix free object

    // dof_handler
    std::vector<const DoFHandler<dim> * >  dof_handler_vec;
    dof_handler_vec.resize(1);
    dof_handler_vec[0] = &dof_handler;
    // constraint matrix
    std::vector<const ConstraintMatrix *> constraint_matrix_vec;
    constraint_matrix_vec.resize(1);
    ConstraintMatrix constraints;
    constraints.close();
    constraint_matrix_vec[0] = &constraints;
    // quadratures
    std::vector<Quadrature<1> > quadrature_vec;
    quadrature_vec.resize(2);
    quadrature_vec[0] = QGauss<1>(dof_handler.get_fe().degree+1);
    quadrature_vec[1] = QGauss<1>(dof_handler.get_fe().degree+(dof_handler.get_fe().degree+2)/2);
    // additional data
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    if (dof_handler.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;

    addit_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                       update_quadrature_points | update_normal_vectors |
                                       update_values);
    addit_data.level_mg_handler = level;
    addit_data.mpi_communicator =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()) ?
      (dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()))->get_communicator() : MPI_COMM_SELF;
    addit_data.periodic_face_pairs_level_0 = periodic_face_pairs_level0;

//    ConstraintMatrix constraints;
//    const QGauss<1> quad(dof_handler.get_fe().degree+1);
//    own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

    // reinit
    own_matrix_free_storage.reinit(mapping, dof_handler_vec, constraint_matrix_vec, quadrature_vec, addit_data);


    // setup own mass matrix operator
    MassMatrixOperatorData mass_matrix_operator_data = underlying_operator.get_mass_matrix_operator_data();
    // set dof index to zero since matrix free object only contains one dof-handler
    mass_matrix_operator_data.dof_index = 0;
    own_mass_matrix_operator_storage.initialize(own_matrix_free_storage, mass_matrix_operator_data);


    // setup own viscous operator
    ViscousOperatorData<dim> viscous_operator_data = underlying_operator.get_viscous_operator_data();
    // set dof index to zero since matrix free object only contains one dof-handler
    viscous_operator_data.dof_index = 0;
    own_viscous_operator_storage.initialize(mapping,own_matrix_free_storage, viscous_operator_data);


    // setup own convective operator
    ConvectiveOperatorData<dim> convective_operator_data = underlying_operator.get_convective_operator_data();
    // set dof index to zero since matrix free object only contains one dof-handler
    convective_operator_data.dof_index = 0;
    own_convective_operator_storage.initialize(own_matrix_free_storage, convective_operator_data);

    // setup velocity convection diffusion operator
    VelocityConvDiffOperatorData<dim> operator_data = underlying_operator.get_velocity_conv_diff_operator_data();
    initialize(own_matrix_free_storage,
               operator_data,
               own_mass_matrix_operator_storage,
               own_viscous_operator_storage,
               own_convective_operator_storage);

    // Initialize other variables:

    // mass matrix term: set scaling factor time derivative term
    set_scaling_factor_time_derivative_term(underlying_operator.get_scaling_factor_time_derivative_term());

    // convective term: evaluation_time and velocity_linearization
    // Note that velocity_linearization has already
    // been initialized in function initialize().
    // These variables are not set here. If the convective term
    // is considered, these variables have to be updated anyway,
    // which is done somewhere else.

    // viscous term:

    // initialize temp_vector: this is done in this function because
    // temp_vector is only used in the function vmult_add(), i.e.,
    // when using the multigrid preconditioner
    initialize_dof_vector(temp_vector);
  }

  void set_scaling_factor_time_derivative_term(double const &factor)
  {
    scaling_factor_time_derivative_term = factor;
  }

  double get_scaling_factor_time_derivative_term() const
  {
    return scaling_factor_time_derivative_term;
  }

  void set_solution_linearization(parallel::distributed::Vector<Number> const &solution_linearization)
  {
    velocity_linearization = solution_linearization;
  }

  parallel::distributed::Vector<value_type> & get_solution_linearization() const
  {
    return velocity_linearization;
  }

  void set_evaluation_time(double const &evaluation_time_in)
  {
    evaluation_time = evaluation_time_in;
  }

  double get_evaluation_time() const
  {
    return evaluation_time;
  }

  VelocityConvDiffOperatorData<dim> const & get_velocity_conv_diff_operator_data() const
  {
    return this->operator_data;
  }

  MassMatrixOperatorData const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator->get_operator_data();
  }

  ConvectiveOperatorData<dim> const & get_convective_operator_data() const
  {
    return convective_operator->get_operator_data();
  }

  ViscousOperatorData<dim> const & get_viscous_operator_data() const
  {
    return viscous_operator->get_operator_data();
  }

  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const
  {
    // does nothing in case of the velocity conv diff operator
    // this function is only necessary due to the interface of the multigrid preconditioner
    // and especially the coarse grid solver that calls this function
  }

  // apply matrix vector multiplication
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

      mass_matrix_operator->apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else
    {
      dst = 0.0;
    }

    viscous_operator->apply_add(dst,src);

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_linearized_add(dst,src,&velocity_linearization,evaluation_time);
    }
  }

  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const
  {
    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

      mass_matrix_operator->apply(temp_vector,src);
      temp_vector *= scaling_factor_time_derivative_term;
      dst += temp_vector;
    }

    viscous_operator->apply_add(dst,src);

    if(operator_data.convective_problem == true)
    {
      convective_operator->apply_linearized_add(dst,src,&velocity_linearization,evaluation_time);
    }
  }

  void vmult_interface_down(parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const
  {
    vmult(dst,src);
  }

  void vmult_add_interface_up(parallel::distributed::Vector<Number>       &dst,
                              const parallel::distributed::Vector<Number> &src) const
  {
    vmult_add(dst,src);
  }

  types::global_dof_index m() const
  {
    return data->get_vector_partitioner(get_dof_index())->size();
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

  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,get_dof_index());
  }

  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {

    if(operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been set for velocity convection-diffusion operator!"));

      mass_matrix_operator->calculate_diagonal(diagonal);
      diagonal *= scaling_factor_time_derivative_term;
    }
    else
    {
      diagonal = 0.0;
    }

    viscous_operator->add_diagonal(diagonal);

    if(operator_data.convective_problem == true)
    {
      convective_operator->add_diagonal(diagonal,&velocity_linearization,evaluation_time);
    }
  }

  void verify_calculation_of_diagonal(parallel::distributed::Vector<Number> const &diagonal) const
  {
    parallel::distributed::Vector<Number>  diagonal2(diagonal);
    diagonal2 = 0.0;
    parallel::distributed::Vector<Number>  src(diagonal2);
    parallel::distributed::Vector<Number>  dst(diagonal2);
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      src.local_element(i) = 1.0;
      vmult(dst,src);
      diagonal2.local_element(i) = dst.local_element(i);
      src.local_element(i) = 0.0;
    }

    std::cout<<"L2 norm diagonal - Variant 1: "<<diagonal.l2_norm()<<std::endl;
    std::cout<<"L2 norm diagonal - Variant 2: "<<diagonal2.l2_norm()<<std::endl;
    diagonal2.add(-1.0,diagonal);
    std::cout<<"L2 error diagonal: "<<diagonal2.l2_norm()<<std::endl;
  }

  void invert_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      if( std::abs(diagonal.local_element(i)) > 1.0e-10 )
        diagonal.local_element(i) = 1.0/diagonal.local_element(i);
      else
        diagonal.local_element(i) = 1.0;
    }
  }

  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    calculate_diagonal(diagonal);

    //verify_calculation_of_diagonal(diagonal);

    invert_diagonal(diagonal);
  }

  // getters
  MatrixFree<dim,value_type> const & get_data() const
  {
    return *data;
  }

private:
  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const * mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const * viscous_operator;
  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const * convective_operator;
  VelocityConvDiffOperatorData<dim> operator_data;
  parallel::distributed::Vector<Number> mutable temp_vector;
  double scaling_factor_time_derivative_term;
  parallel::distributed::Vector<Number> mutable velocity_linearization;
  double evaluation_time;

  /*
   * The following variables are necessary when applying the multigrid
   * preconditioner to the velocity convection-diffusion operator.
   * In that case, the VelocityConvDiffOperator has to be generated
   * for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of
   * MatrixFree, MassMatrixOperator, ViscousOperator,
   *   e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the VelocityConvDiffOperator with these
   * ojects by setting the above pointers to the own_objects_storage,
   *   e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> own_mass_matrix_operator_storage;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> own_viscous_operator_storage;
  ConvectiveOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> own_convective_operator_storage;
};

#endif /* INCLUDE_VELOCITYCONVDIFFOPERATOR_H_ */
