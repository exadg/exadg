/*
 * HelmholtzOperator.h
 *
 *  Created on: May 11, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_HELMHOLTZ_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_HELMHOLTZ_OPERATOR_H_

#include "../../incompressible_navier_stokes/spatial_discretization/navier_stokes_operators.h"
#include "solvers_and_preconditioners/invert_diagonal.h"

#include "solvers_and_preconditioners/verify_calculation_of_diagonal.h"

template<int dim>
struct HelmholtzOperatorData
{
  HelmholtzOperatorData ()
    :
    unsteady_problem(true),
    dof_index(0)
  {}

  bool unsteady_problem;

  unsigned int dof_index;
};

template <int dim, int fe_degree, int fe_degree_xwall, int xwall_quad_rule, typename Number = double>
class HelmholtzOperator : public MatrixOperatorBase
{
public:
  typedef Number value_type;

  HelmholtzOperator()
    :
    // TODO Benjamin: remove this from Helmholtz operator
    strong_homogeneous_dirichlet_bc(false),
    // TODO Benjamin: remove this from Helmholtz operator
    data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    scaling_factor_time_derivative_term(-1.0)
  {}

  void initialize(MatrixFree<dim,Number> const                                                        &mf_data_in,
                  HelmholtzOperatorData<dim> const                                                    &helmholtz_operator_data_in,
                  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const &mass_matrix_operator_in,
                  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> const     &viscous_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->helmholtz_operator_data = helmholtz_operator_data_in;
    this->mass_matrix_operator = &mass_matrix_operator_in;
    this->viscous_operator = &viscous_operator_in;
  }

  /*
   *  This function is called by the multigrid algorithm to initialize the
   *  matrices on all levels. To construct the matrices, and object of
   *  type UnderlyingOperator is used that provides all the information for
   *  the setup, i.e., the information that is needed to call the
   *  member function initialize(...).
   */
  template<typename UnderlyingOperator>
  void initialize_mg_matrix (unsigned int const                              level,
                             DoFHandler<dim> const                           &dof_handler,
                             Mapping<dim> const                              &mapping,
                             UnderlyingOperator const                        &underlying_operator,
                             const std::vector<GridTools::PeriodicFacePair<
                               typename Triangulation<dim>::cell_iterator> > &periodic_face_pairs_level0)
  {
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
    addit_data.periodic_face_pairs_level_0 = periodic_face_pairs_level0;

    ConstraintMatrix constraints;

    // reinit
    own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

    // setup own mass matrix operator
    MassMatrixOperatorData mass_matrix_operator_data = underlying_operator.get_mass_matrix_operator_data();
    mass_matrix_operator_data.dof_index = 0;
    own_mass_matrix_operator_storage.initialize(own_matrix_free_storage,mass_matrix_operator_data);

    // setup own viscous operator
    ViscousOperatorData<dim> viscous_operator_data = underlying_operator.get_viscous_operator_data();
    // set dof index to zero since matrix free object only contains one dof-handler
    viscous_operator_data.dof_index = 0;
    own_viscous_operator_storage.initialize(mapping,own_matrix_free_storage, viscous_operator_data);

    // setup Helmholtz operator
    HelmholtzOperatorData<dim> operator_data = underlying_operator.get_helmholtz_operator_data();
    initialize(own_matrix_free_storage, operator_data, own_mass_matrix_operator_storage, own_viscous_operator_storage);

    // Initialize other variables:

    // mass matrix term: set scaling factor time derivative term
    set_scaling_factor_time_derivative_term(underlying_operator.get_scaling_factor_time_derivative_term());

    // viscous term:


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
   *  Operator data
   */
  HelmholtzOperatorData<dim> const & get_helmholtz_operator_data() const
  {
    return this->helmholtz_operator_data;
  }

  /*
   *  Operator data of basic operators: mass matrix, viscous operator
   */
  MassMatrixOperatorData const & get_mass_matrix_operator_data() const
  {
    return mass_matrix_operator->get_operator_data();
  }

  ViscousOperatorData<dim> const & get_viscous_operator_data() const
  {
    return viscous_operator->get_operator_data();
  }

  // TODO Benjamin: remove this from Helmholtz operator
  void initialize_strong_homogeneous_dirichlet_boundary_conditions()
  {
    strong_homogeneous_dirichlet_bc = true;
    std::vector<types::global_dof_index> dof_indices(data->get_dof_handler(0).get_fe().dofs_per_cell);
    for (typename DoFHandler<dim>::active_cell_iterator cell = data->get_dof_handler(0).begin_active();
        cell != data->get_dof_handler(0).end(); ++cell)
    if (cell->is_locally_owned())
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
        if (cell->at_boundary(f) && cell->face(f)->boundary_id() == 0)
        {
          cell->get_dof_indices(dof_indices);
          for (unsigned int i=0; i<data->get_dof_handler(0).get_fe().dofs_per_cell; ++i)
            if (data->get_dof_handler(0).get_fe().has_support_on_face(i,f))
            {
              const std::pair<unsigned int,unsigned int> comp =
              data->get_dof_handler(0).get_fe().system_to_component_index(i);
              if (comp.first < dim)
                dbc_indices.push_back(dof_indices[i]);
            }
        }
  }
  // TODO Benjamin: remove this from Helmholtz operator

  /*
   *  This function does nothing in case of the velocity conv diff operator.
   *  IT is only necessary due to the interface of the multigrid preconditioner
   *  and especially the coarse grid solver that calls this function.
   */
  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const {}

  /*
   *  Other function needed in order to apply geometric multigrid to this operator
   */
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

  MatrixFree<dim,value_type> const & get_data() const
  {
    return *data;
  }

  /*
   *  This function applies the matrix vector multiplication.
   */
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    if(helmholtz_operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->apply(dst,src);
      dst *= scaling_factor_time_derivative_term;
    }
    else
    {
      dst = 0.0;
    }

    std::vector<std::pair<Number,Number> > dbc_values;
    strong_homogeneous_dirichlet_pre(src,dst,dbc_values);

    viscous_operator->apply_add(dst,src);

    strong_homogeneous_dirichlet_post(src,dst,dbc_values);
  }

  /*
   *  This function applies matrix vector product and adds the result
   *  to the dst-vector.
   */
  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator

    if(helmholtz_operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->apply(temp,src);
      temp *= scaling_factor_time_derivative_term;
      dst += temp;
    }

    std::vector<std::pair<Number,Number> > dbc_values;
    strong_homogeneous_dirichlet_pre(src,dst,dbc_values);
    viscous_operator->apply_add(dst,src);

    strong_homogeneous_dirichlet_post(src,dst,dbc_values);
  }

  unsigned int get_dof_index() const
  {
    return helmholtz_operator_data.dof_index;
  }

  /*
   *  This function initializes a global dof-vector.
   */
  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,get_dof_index());
  }

  /*
   *  Calculation of inverse diagonal (needed for smoothers and preconditioners)
   */
  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    calculate_diagonal(diagonal);

    // verify_calculation_of_diagonal(*this,diagonal);

    invert_diagonal(diagonal);
  }

  /*
   *  Apply block Jacobi preconditioner
   */
  void apply_block_jacobi (parallel::distributed::Vector<Number>       &/*dst*/,
                           parallel::distributed::Vector<Number> const &/*src*/) const
  {
    AssertThrow(false,ExcMessage("Block Jacobi preconditioner not implemented for velocity reaction-diffusion operator."));
  }

  /*
   *  Update block Jacobi preconditioner
   */
  void update_block_jacobi () const
  {
    AssertThrow(false,ExcMessage("Function update_block_jacobi() has not been implemented."));
  }

private:
  /*
   *  This function calculates the diagonal of the discrete operator representing the
   *  velocity convection-diffusion operator.
   */
  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    if(helmholtz_operator_data.unsteady_problem == true)
    {
      AssertThrow(scaling_factor_time_derivative_term > 0.0,
        ExcMessage("Scaling factor of time derivative term has not been initialized for Helmholtz operator!"));

      mass_matrix_operator->calculate_diagonal(diagonal);
      diagonal *= scaling_factor_time_derivative_term;
    }
    else
    {
      diagonal = 0.0;
    }

    viscous_operator->add_diagonal(diagonal);
  }


  // TODO Benjamin: remove this from Helmholtz operator
  void strong_homogeneous_dirichlet_pre(const parallel::distributed::Vector<Number> & src,
                                        parallel::distributed::Vector<Number> &       dst,
                                        std::vector<std::pair<Number,Number> > &      dbc_values) const
  {
    if(strong_homogeneous_dirichlet_bc)
    {
      // save source and set DBC to zero
      dbc_values.resize(dbc_indices.size());
      for (unsigned int i=0; i<dbc_indices.size(); ++i)
      {
        dbc_values[i] =
          std::pair<Number,Number>(src(dbc_indices[i]),
                                   dst(dbc_indices[i]));
        const_cast<parallel::distributed::Vector<Number>&>(src)(dbc_indices[i]) = 0.;
      }
    }
  }
  // TODO Benjamin: remove this from Helmholtz operator

  // TODO Benjamin: remove this from Helmholtz operator
  void strong_homogeneous_dirichlet_post(const parallel::distributed::Vector<Number> &  src,
                                         parallel::distributed::Vector<Number> &        dst,
                                         std::vector<std::pair<Number,Number> > const & dbc_values) const
  {
    if(strong_homogeneous_dirichlet_bc)
    {
      // reset edge constrained values, multiply by unit matrix and add into
      // destination
      for (unsigned int i=0; i<dbc_indices.size(); ++i)
      {
        const_cast<parallel::distributed::Vector<Number>&>(src)(dbc_indices[i]) = dbc_values[i].first;
        dst(dbc_indices[i]) = dbc_values[i].second + dbc_values[i].first;
      }
    }
  }
  // TODO Benjamin: remove this from Helmholtz operator

  // TODO Benjamin: remove this from Helmholtz operator
  bool strong_homogeneous_dirichlet_bc;
  std::vector<unsigned int> dbc_indices;
  std::vector<std::pair<Number,Number> > dbc_values;
  // TODO Benjamin: remove this from Helmholtz operator

  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const *mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number>  const *viscous_operator;
  HelmholtzOperatorData<dim> helmholtz_operator_data;
  parallel::distributed::Vector<Number> mutable temp;
  double scaling_factor_time_derivative_term;

  /*
   * The following variables are necessary when applying the multigrid
   * preconditioner to the Helmholtz operator. In that case, the
   * Helmholtz has to be generated for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of
   * MatrixFree, MassMatrixOperator, ViscousOperator,
   *   e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the HelmholtzOperator with these
   * ojects by setting the above pointers to the own_objects_storage,
   *   e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> own_mass_matrix_operator_storage;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, xwall_quad_rule, Number> own_viscous_operator_storage;
};

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_SPATIAL_DISCRETIZATION_HELMHOLTZ_OPERATOR_H_ */
