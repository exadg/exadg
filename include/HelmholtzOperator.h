/*
 * HelmholtzSolver.h
 *
 *  Created on: May 11, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_HELMHOLTZOPERATOR_H_
#define INCLUDE_HELMHOLTZOPERATOR_H_

#include "NavierStokesOperators.h"

template<int dim>
struct HelmholtzOperatorData
{
  HelmholtzOperatorData ()
    :
    dof_index(0),
    scaling_factor_time_derivative_term(-1.0)
  {}

  unsigned int dof_index;

  MassMatrixOperatorData mass_matrix_operator_data;
  ViscousOperatorData<dim> viscous_operator_data;

  /*
   * This variable 'scaling_factor_time_derivative_term' is only used when initializing the HelmholtzOperator.
   * In order to change/update this coefficient during the simulation (e.g., varying time step sizes)
   * use the element variable 'scaling_factor_time_derivative_term' of HelmholtzOperator and the corresponding setter
   * set_scaling_factor_time_derivative_term().
   */
  double scaling_factor_time_derivative_term;

  // current interface of multigrid implementation needs this variable
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs_level0;

  std::set<types::boundary_id> const & get_dirichlet_boundaries() const
  {
    return viscous_operator_data.get_dirichlet_boundaries();
  }

  void set_dof_index(unsigned int dof_index_in)
  {
    this->dof_index = dof_index_in;

    // don't forget to set the dof_indices of the mass_matrix_operator_data and viscous_operator_data
    mass_matrix_operator_data.dof_index = dof_index_in;
    viscous_operator_data.dof_index = dof_index_in;
  }
};

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d_xwall,typename Number = double>
class HelmholtzOperator : public Subscriptor
{
public:
  typedef Number value_type;

  HelmholtzOperator()
    :
    data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    scaling_factor_time_derivative_term(-1.0)
  {}

  void initialize(MatrixFree<dim,Number> const                                                            &mf_data_in,
                  HelmholtzOperatorData<dim> const                                                        &helmholtz_operator_data_in,
                  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>  const &mass_matrix_operator_in,
                  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> const     &viscous_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->helmholtz_operator_data = helmholtz_operator_data_in;
    this->mass_matrix_operator = &mass_matrix_operator_in;
    this->viscous_operator = &viscous_operator_in;

    // set mass matrix coefficient!
    AssertThrow(helmholtz_operator_data.scaling_factor_time_derivative_term > 0.0,
                ExcMessage("Scaling factor of time derivative term of HelmholtzOperatorData has not been initialized!"));

    this->scaling_factor_time_derivative_term = helmholtz_operator_data.scaling_factor_time_derivative_term;
  }

  void reinit (const DoFHandler<dim>            &dof_handler,
               const Mapping<dim>               &mapping,
               const HelmholtzOperatorData<dim> &operator_data,
               const MGConstrainedDoFs          &/*mg_constrained_dofs*/,
               const unsigned int               level = numbers::invalid_unsigned_int,
               FEParameters<dim> const          &fe_param = FEParameters<dim>())
  {
    // set the dof index to zero (for the HelmholtzOperator and also
    // for the basic Operators (MassMatrixOperator and ViscousOperator))
    HelmholtzOperatorData<dim> my_operator_data = operator_data;
    my_operator_data.set_dof_index(0);

    // setup own matrix free object
    const QGauss<1> quad(dof_handler.get_fe().degree+1);
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    if (dof_handler.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;
    addit_data.level_mg_handler = level;
    addit_data.mpi_communicator =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()) ?
      (dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()))->get_communicator() : MPI_COMM_SELF;
    addit_data.periodic_face_pairs_level_0 = operator_data.periodic_face_pairs_level0;

    ConstraintMatrix constraints;
    own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

    // setup own mass matrix operator
    MassMatrixOperatorData mass_matrix_operator_data = my_operator_data.mass_matrix_operator_data;
    own_mass_matrix_operator_storage.initialize(own_matrix_free_storage,fe_param,mass_matrix_operator_data);

    // setup own viscous operator
    ViscousOperatorData<dim> viscous_operator_data = my_operator_data.viscous_operator_data;
    own_viscous_operator_storage.initialize(mapping,own_matrix_free_storage,fe_param,viscous_operator_data);

    // setup Helmholtz operator
    initialize(own_matrix_free_storage, my_operator_data, own_mass_matrix_operator_storage, own_viscous_operator_storage);

    // initialize temp vector: this is done in this function because
    // the vector temp is only used in the function vmult_add(), i.e.,
    // when using the multigrid preconditioner
    initialize_dof_vector(temp);
  }

  void set_scaling_factor_time_derivative_term(Number const coefficient_in)
  {
    scaling_factor_time_derivative_term = coefficient_in;
  }

  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const
  {
    // does nothing in case of the Helmholtz equation
    // this function is only necessary due to the interface of the multigrid preconditioner
    // and especially the coarse grid solver that calls this function
  }

  // apply matrix vector multiplication
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    mass_matrix_operator->apply(dst,src);
    dst *= scaling_factor_time_derivative_term;

    viscous_operator->apply_add(dst,src);
  }

//  void Tvmult(parallel::distributed::Vector<Number>       &dst,
//              const parallel::distributed::Vector<Number> &src) const
//  {
//    vmult(dst,src);
//  }
//
//  void Tvmult_add(parallel::distributed::Vector<Number>       &dst,
//                  const parallel::distributed::Vector<Number> &src) const
//  {
//    vmult_add(dst,src);
//  }

  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    mass_matrix_operator->apply(temp,src);
    temp *= scaling_factor_time_derivative_term;
    dst += temp;

    viscous_operator->apply_add(dst,src);
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

//  types::global_dof_index n() const
//  {
//    return data->get_vector_partitioner(helmholtz_operator_data.dof_index)->size();
//  }

  Number el (const unsigned int,  const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  unsigned int get_dof_index() const
  {
    return helmholtz_operator_data.dof_index;
  }

  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    mass_matrix_operator->calculate_diagonal(diagonal);
    diagonal *= scaling_factor_time_derivative_term;

    viscous_operator->add_diagonal(diagonal);

    // verify_calculation_of_diagonal(diagonal);
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

    invert_diagonal(diagonal);
  }

  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,get_dof_index());
  }

private:
  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>  const *mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>  const *viscous_operator;
  HelmholtzOperatorData<dim> helmholtz_operator_data;
  parallel::distributed::Vector<Number> mutable temp;
  double scaling_factor_time_derivative_term;

  /*
   * The following variables are necessary when applying the multigrid preconditioner to the Helmholtz equation:
   * In that case, the HelmholtzOperator has to be generated for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of MatrixFree, MassMatrixOperator, ViscousOperator,
   *  e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the HelmholtzOperator with these ojects by setting the above pointers to the own_objects_storage,
   *  e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> own_mass_matrix_operator_storage;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> own_viscous_operator_storage;

};

#endif /* INCLUDE_HELMHOLTZOPERATOR_H_ */
