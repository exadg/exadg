/*
 * CompatibleLaplaceOperator.h
 *
 *  Created on: Jul 18, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_OPERATOR_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_OPERATOR_H_

#include "../spatial_discretization/navier_stokes_operators.h"
#include "../../operators/matrix_operator_base_new.h"
#include "../../solvers_and_preconditioners/preconditioner/inverse_mass_matrix_preconditioner.h"
#include "../../solvers_and_preconditioners/util/invert_diagonal.h"

namespace IncNS
{

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
};

template <int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int xwall_quad_rule,typename Number = double>
class CompatibleLaplaceOperator : public MultigridOperatorBase<dim, Number>
{
public:
  static const int DIM = dim;
  typedef Number value_type;

  CompatibleLaplaceOperator()
    :
    data(nullptr),
    gradient_operator(nullptr),
    divergence_operator(nullptr),
    inv_mass_matrix_operator(nullptr),
    needs_mean_value_constraint (false),
    apply_mean_value_constraint_in_matvec (false)
  {}

  void initialize(MatrixFree<dim,Number> const                                                                    &mf_data_in,
                  CompatibleLaplaceOperatorData<dim> const                                                        &compatible_laplace_operator_data_in,
                  GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>  const  &gradient_operator_in,
                  DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> const &divergence_operator_in,
                  InverseMassMatrixOperator<dim,fe_degree, Number> const                                          &inv_mass_matrix_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->compatible_laplace_operator_data = compatible_laplace_operator_data_in;
    this->gradient_operator = &gradient_operator_in;
    this->divergence_operator = &divergence_operator_in;
    this->inv_mass_matrix_operator = &inv_mass_matrix_operator_in;

    // initialize tmp vector
    initialize_dof_vector_velocity(tmp);


    // Check whether the Laplace matrix is singular when applied to a vector
    // consisting of only ones (except for constrained entries)
    parallel::distributed::Vector<Number> in_vec, out_vec;
    initialize_dof_vector(in_vec);
    initialize_dof_vector(out_vec);
    in_vec = 1;
    vmult_add(out_vec, in_vec);
    const double linfty_norm = out_vec.linfty_norm();

    // since we cannot know the magnitude of the entries at this point (the
    // diagonal entries would be a guideline but they are not available here),
    // we instead multiply by a random vector
    for (unsigned int i=0; i<in_vec.local_size(); ++i)
      in_vec.local_element(i) = (double)rand()/RAND_MAX;
    vmult(out_vec, in_vec);
    const double linfty_norm_compare = out_vec.linfty_norm();

    // use mean value constraint if the infty norm with the one vector is very small
    needs_mean_value_constraint = linfty_norm / linfty_norm_compare < std::pow(std::numeric_limits<Number>::epsilon(), 2./3.);
    apply_mean_value_constraint_in_matvec = needs_mean_value_constraint;
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
                             DoFHandler<dim> const                            &dof_handler_p,
                             Mapping<dim> const                               &mapping,
                             UnderlyingOperator const                         &underlying_operator,
                             std::vector<GridTools::PeriodicFacePair<typename
                               Triangulation<dim>::cell_iterator> > const     &/*periodic_face_pairs_level0*/)
  {
    // get compatible Laplace operator data
    CompatibleLaplaceOperatorData<dim> comp_laplace_operator_data = underlying_operator.get_compatible_laplace_operator_data();

    unsigned int dof_index_velocity = comp_laplace_operator_data.dof_index_velocity;
    unsigned int dof_index_pressure = comp_laplace_operator_data.dof_index_pressure;

    const DoFHandler<dim> &dof_handler_u = underlying_operator.get_dof_handler_u();

    AssertThrow(dof_index_velocity == 0, ExcMessage("Expected that dof_index_velocity is 0. Fix implementation of CompatibleLaplaceOperator!"));
    AssertThrow(dof_index_pressure == 1, ExcMessage("Expected that dof_index_pressure is 1. Fix implementation of CompatibleLaplaceOperator!"));

    // setup own matrix free object

    // dof_handler
    std::vector<const DoFHandler<dim> * >  dof_handler_vec;
    // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
    dof_handler_vec.resize(2);
    dof_handler_vec[dof_index_velocity] = &dof_handler_u;
    dof_handler_vec[dof_index_pressure] = &dof_handler_p;

    // constraint matrix
    std::vector<ConstraintMatrix const *> constraint_matrix_vec;
    // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
    constraint_matrix_vec.resize(2);
    ConstraintMatrix constraint_u, constraint_p;
    constraint_u.close();
    constraint_p.close();
    constraint_matrix_vec[dof_index_velocity] = &constraint_u;
    constraint_matrix_vec[dof_index_pressure] = &constraint_p;

    // quadratures
    // quadrature formula with (fe_degree_velocity+1) quadrature points: this is the quadrature formula that is used for
    // the gradient operator and the divergence operator (and the inverse velocity mass matrix operator)
    const QGauss<1> quad(dof_handler_u.get_fe().degree+1);

    // additional data
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    // continuous or discontinuous elements: discontinuous == 0
    if (dof_handler_p.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;
    addit_data.level_mg_handler = level;

    // reinit
    own_matrix_free_storage.reinit(mapping, dof_handler_vec, constraint_matrix_vec, quad, addit_data);

    // setup own gradient operator
    GradientOperatorData<dim> gradient_operator_data = underlying_operator.get_gradient_operator_data();
    own_gradient_operator_storage.initialize(own_matrix_free_storage,gradient_operator_data);

    // setup own divergence operator
    DivergenceOperatorData<dim> divergence_operator_data = underlying_operator.get_divergence_operator_data();
    own_divergence_operator_storage.initialize(own_matrix_free_storage,divergence_operator_data);

    // setup own inverse mass matrix operator
    // NOTE: use quad_index = 0 since own_matrix_free_storage contains only one quadrature formula
    // (i.e. on would use quad_index = 0 also if quad_index_velocity would be 1 !)
    unsigned int quad_index = 0;
    own_inv_mass_matrix_operator_storage.initialize(own_matrix_free_storage,
                                                    underlying_operator.get_dof_index_velocity(),
                                                    quad_index);

    // setup compatible Laplace operator
    initialize(own_matrix_free_storage,
               comp_laplace_operator_data,
               own_gradient_operator_storage,
               own_divergence_operator_storage,
               own_inv_mass_matrix_operator_storage);

    // we do not need the mean value constraint for smoothers on the
    // multigrid levels, so we can disable it
    disable_mean_value_constraint();
  }

  void apply_nullspace_projection(parallel::distributed::Vector<Number> &vec) const
  {
    if (needs_mean_value_constraint)
    {
      const Number mean_val = vec.mean_value();
      vec.add(-mean_val);
    }
  }

  void disable_mean_value_constraint()
  {
    this->apply_mean_value_constraint_in_matvec = false;
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
    const parallel::distributed::Vector<Number> *actual_src = &src;
    if(apply_mean_value_constraint_in_matvec)
    {
      tmp_projection_vector = src;
      apply_nullspace_projection(tmp_projection_vector);
      actual_src = &tmp_projection_vector;
    }

    // compatible Laplace operator = B * M^{-1} * B^{T} = (-div) * M^{-1} * grad
    gradient_operator->apply(tmp,*actual_src);
    inv_mass_matrix_operator->apply(tmp,tmp);
    // NEGATIVE divergence operator
    tmp *= -1.0;
    divergence_operator->apply_add(dst,tmp);

    if(apply_mean_value_constraint_in_matvec)
      apply_nullspace_projection(dst);
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

    if(apply_mean_value_constraint_in_matvec)
    {
      parallel::distributed::Vector<Number> vec1;
      vec1.reinit(diagonal, true);
      for(unsigned int i=0;i<vec1.local_size();++i)
        vec1.local_element(i) = 1.;
      parallel::distributed::Vector<Number> d;
      d.reinit(diagonal, true);
      vmult(d,vec1);
      double length = vec1*vec1;
      double factor = vec1*d;
      diagonal.add(-2./length,d,factor/pow(length,2.),vec1);
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
    data->initialize_dof_vector(vector,compatible_laplace_operator_data.dof_index_pressure);
  }

  void initialize_dof_vector_velocity(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,compatible_laplace_operator_data.dof_index_velocity);
  }

  /*
   *  Apply block Jacobi preconditioner
   */
  void apply_block_jacobi (parallel::distributed::Vector<Number>       &/*dst*/,
                           parallel::distributed::Vector<Number> const &/*src*/) const
  {
    AssertThrow(false,ExcMessage("Block Jacobi preconditioner not implemented for compatible Laplace operator."));
  }

  /*
   *  Update block Jacobi preconditioner
   */
  void update_block_jacobi () const
  {
    AssertThrow(false,ExcMessage("Function update_block_jacobi() has not been implemented."));
  }

private:
  MatrixFree<dim,Number> const * data;
  GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>  const *gradient_operator;
  DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number>  const *divergence_operator;
  InverseMassMatrixOperator<dim,fe_degree, Number> const *inv_mass_matrix_operator;
  CompatibleLaplaceOperatorData<dim> compatible_laplace_operator_data;
  parallel::distributed::Vector<Number> mutable tmp;

  /*
   * The following variables are necessary when applying the multigrid preconditioner to the compatible Laplace operator
   * In that case, the CompatibleLaplaceOperator has to be generated for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of MatrixFree, GradientOperator, DivergenceOperator,
   *  e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the CompatibleLaplaceOperator with these ojects by setting the above pointers to the own_objects_storage,
   *  e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  GradientOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> own_gradient_operator_storage;
  DivergenceOperator<dim, fe_degree, fe_degree_p, fe_degree_xwall, xwall_quad_rule, Number> own_divergence_operator_storage;
  InverseMassMatrixOperator<dim,fe_degree, Number> own_inv_mass_matrix_operator_storage;

  bool needs_mean_value_constraint;
  bool apply_mean_value_constraint_in_matvec;
  mutable parallel::distributed::Vector<Number> tmp_projection_vector;
};


}

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_OPERATOR_H_ */
