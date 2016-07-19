/*
 * CompatibleLaplaceOperator.h
 *
 *  Created on: Jul 18, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_COMPATIBLELAPLACEOPERATOR_H_
#define INCLUDE_COMPATIBLELAPLACEOPERATOR_H_

#include "NavierStokesOperators.h"

template<int dim>
struct CompatibleLaplaceOperatorData
{
  CompatibleLaplaceOperatorData ()
    :
    dof_index_velocity(0),
    dof_index_pressure(1)
  {}

  unsigned int dof_index_velocity;
  unsigned int dof_index_pressure;

  GradientOperatorData gradient_operator_data;
  DivergenceOperatorData divergence_operator_data;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs_level0;

  std::set<types::boundary_id> const & get_dirichlet_boundaries() const
  {
    return gradient_operator_data.get_dirichlet_boundaries();
  }
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall,typename Number = double>
class CompatibleLaplaceOperator : public Subscriptor
{
public:
  typedef Number value_type;

  CompatibleLaplaceOperator()
    :
    data(nullptr),
    gradient_operator(nullptr),
    divergence_operator(nullptr),
    inv_mass_matrix_operator(nullptr)
  {}

  void initialize(MatrixFree<dim,Number> const                                                                        &mf_data_in,
                  CompatibleLaplaceOperatorData<dim> const                                                            &compatible_laplace_operator_data_in,
                  GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number>  const  &gradient_operator_in,
                  DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number> const &divergence_operator_in,
                  InverseMassMatrixOperator<dim,fe_degree, Number> const                                              &inv_mass_matrix_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->compatible_laplace_operator_data = compatible_laplace_operator_data_in;
    this->gradient_operator = &gradient_operator_in;
    this->divergence_operator = &divergence_operator_in;
    this->inv_mass_matrix_operator = &inv_mass_matrix_operator_in;

    // initialize temp vector
    initialize_dof_vector_velocity(temp);
  }

  void reinit (const DoFHandler<dim>                    &dof_handler_p,
               const DoFHandler<dim>                    &dof_handler_u,
               const Mapping<dim>                       &mapping,
               const CompatibleLaplaceOperatorData<dim> &operator_data,
               const MGConstrainedDoFs                  &/*mg_constrained_dofs*/,
               const unsigned int                       level = numbers::invalid_unsigned_int,
               FEParameters const                       &fe_param = FEParameters())
  {
    CompatibleLaplaceOperatorData<dim> my_operator_data = operator_data;

    // setup own matrix free object
    std::vector<const DoFHandler<dim> * >  dof_handler_vec;
    // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
    dof_handler_vec.resize(2);
    dof_handler_vec[my_operator_data.dof_index_velocity] = &dof_handler_u;
    dof_handler_vec[my_operator_data.dof_index_pressure] = &dof_handler_p;

    // quadrature formula with (fe_degree_velocity+1) quadrature points: this is the quadrature formula that is used for
    // the gradient operator and the divergence operator
    const QGauss<1> quad(dof_handler_u.get_fe().degree+1);
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    // continuous or discontinuous elements: discontinuous == 0
    if (dof_handler_p.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;
    addit_data.level_mg_handler = level;
    addit_data.mpi_communicator =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler_p.get_triangulation()) ?
      (dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler_p.get_triangulation()))->get_communicator() : MPI_COMM_SELF;
    addit_data.periodic_face_pairs_level_0 = operator_data.periodic_face_pairs_level0;

    std::vector<ConstraintMatrix const *> constraint_matrix_vec;
    // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
    constraint_matrix_vec.resize(2);
    ConstraintMatrix constraint_u, constraint_p;
    constraint_u.close();
    constraint_p.close();
    constraint_matrix_vec[my_operator_data.dof_index_velocity] = &constraint_u;
    constraint_matrix_vec[my_operator_data.dof_index_pressure] = &constraint_p;

    own_matrix_free_storage.reinit(mapping, dof_handler_vec, constraint_matrix_vec, quad, addit_data);

    // setup own gradient operator
    GradientOperatorData gradient_operator_data = my_operator_data.gradient_operator_data;
    own_gradient_operator_storage.initialize(own_matrix_free_storage,fe_param,gradient_operator_data);

    // setup own divergence operator
    DivergenceOperatorData divergence_operator_data = my_operator_data.divergence_operator_data;
    own_divergence_operator_storage.initialize(own_matrix_free_storage,fe_param,divergence_operator_data);

    // setup own inverse mass matrix operator
    // NOTE: use quad_index = 0 since matrix_free contains only one quadrature formula (also if quad_index_velocity would be 1 !)
    unsigned int quad_index = 0;
    own_inv_mass_matrix_operator_storage.initialize(own_matrix_free_storage,
                                                    my_operator_data.dof_index_velocity,
                                                    quad_index);

    // setup compatible Laplace operator
    initialize(own_matrix_free_storage,
              my_operator_data,
              own_gradient_operator_storage,
              own_divergence_operator_storage,
              own_inv_mass_matrix_operator_storage);
  }

  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const
  {
    // this function is necessary due to the interface of the multigrid preconditioner
    // and especially the coarse grid solver that calls this function
  }

  // apply matrix vector multiplication
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    dst = 0;
    vmult_add(dst,src);
  }

  void Tvmult(parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    vmult(dst,src);
  }

  void Tvmult_add(parallel::distributed::Vector<Number>       &dst,
                  const parallel::distributed::Vector<Number> &src) const
  {
    vmult_add(dst,src);
  }

  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const
  {
    // compatible Laplace operator = B * M^{-1} * B^{T} = (-div) * M^{-1} * grad
    gradient_operator->apply(temp,src);
    inv_mass_matrix_operator->apply_inverse_mass_matrix(temp,temp);
    // NEGATIVE divergence operator
    temp *= -1.0;
    divergence_operator->apply_add(dst,temp);
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
    return data->get_vector_partitioner(compatible_laplace_operator_data.dof_index_pressure)->size();
  }

  types::global_dof_index n() const
  {
    return data->get_vector_partitioner(compatible_laplace_operator_data.dof_index_pressure)->size();
  }

  Number el (const unsigned int,  const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  MatrixFree<dim,Number> const & get_data() const
  {
    return *data;
  }

  unsigned int get_dof_index_pressure() const
  {
    return compatible_laplace_operator_data.dof_index_pressure;
  }

  unsigned int get_dof_index_velocity() const
  {
    return compatible_laplace_operator_data.dof_index_velocity;
  }

  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    // naive implementation of calculation of diagonal (TODO)
    diagonal = 0.0;
    parallel::distributed::Vector<Number>  src(diagonal), dst(diagonal);
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      src.local_element(i) = 1.0;
      vmult(dst,src);
      diagonal.local_element(i) = dst.local_element(i);
      src.local_element(i) = 0.0;
    }
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
    initialize_dof_vector_pressure(vector);
  }

  void initialize_dof_vector_pressure(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,get_dof_index_pressure());
  }

  void initialize_dof_vector_velocity(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,get_dof_index_velocity());
  }

private:
  MatrixFree<dim,Number> const * data;
  GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number>  const *gradient_operator;
  DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number>  const *divergence_operator;
  InverseMassMatrixOperator<dim,fe_degree, Number> const *inv_mass_matrix_operator;
  CompatibleLaplaceOperatorData<dim> compatible_laplace_operator_data;
  parallel::distributed::Vector<Number> mutable temp;

  /*
   * The following variables are necessary when applying the multigrid preconditioner to the compatible Laplace operator
   * In that case, the CompatibleLaplaceOperator has to be generated for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of MatrixFree, GradientOperator, DivergenceOperator,
   *  e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the CompatibleLaplaceOperator with these ojects by setting the above pointers to the own_objects_storage,
   *  e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number> own_gradient_operator_storage;
  DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, Number> own_divergence_operator_storage;
  InverseMassMatrixOperator<dim,fe_degree, Number> own_inv_mass_matrix_operator_storage;
};


#endif /* INCLUDE_COMPATIBLELAPLACEOPERATOR_H_ */
