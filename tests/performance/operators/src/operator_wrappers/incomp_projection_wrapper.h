#ifndef OPERATOR_WRAPPERS_INCOMP_PROJECTION
#define OPERATOR_WRAPPERS_INCOMP_PROJECTION

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include "operator_wrapper.h"

#include "../../../../../include/incompressible_navier_stokes/spatial_discretization/projection_operator.h"

template<int dim, int degree_u, int degree_p, typename Number>
class OperatorWrapperProjection : public OperatorWrapper
{
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  
public:
  OperatorWrapperProjection(parallel::distributed::Triangulation<dim> const & triangulation)
    : fe_u(new FESystem<dim>(FE_DGQ<dim>(degree_u), dim)),
      fe_p(degree_p),
      fe_u_scalar(degree_u),
      mapping(1 /*TODO*/),
      dof_handler_u(triangulation),
      dof_handler_p(triangulation),
      dof_handler_u_scalar(triangulation)
  {
    this->create_dofs();
    this->initialize_matrix_free();

    IncNS::ProjectionOperatorData laplace_additional_data;
    laplace.reset(new IncNS::ProjectionOperator<dim, degree_u, Number>(this->data, 0, 0, laplace_additional_data));
    
    // initialize vectors
    laplace->initialize_dof_vector(src);
    laplace->initialize_dof_vector(dst);
  }

  void
  create_dofs()
  {
  // enumerate degrees of freedom
  dof_handler_u.distribute_dofs(*fe_u);
  dof_handler_p.distribute_dofs(fe_p);
  dof_handler_u_scalar.distribute_dofs(fe_u_scalar);
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

  // dofhandler
  std::vector<const DoFHandler<dim> *> dof_handler_vec;

  dof_handler_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
    DofHandlerSelector::n_variants));
  dof_handler_vec[dof_index_u]        = &dof_handler_u;
  dof_handler_vec[dof_index_p]        = &dof_handler_p;
  dof_handler_vec[dof_index_u_scalar] = &dof_handler_u_scalar;

  // constraint
  std::vector<const AffineConstraints<double> *> constraint_matrix_vec;
  constraint_matrix_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
    DofHandlerSelector::n_variants));
  AffineConstraints<double> constraint_u, constraint_p, constraint_u_scalar;
  constraint_u.close();
  constraint_p.close();
  constraint_u_scalar.close();
  constraint_matrix_vec[dof_index_u]        = &constraint_u;
  constraint_matrix_vec[dof_index_p]        = &constraint_p;
  constraint_matrix_vec[dof_index_u_scalar] = &constraint_u_scalar;

  // quadrature
  std::vector<Quadrature<1>> quadratures;

  // resize quadratures
  quadratures.resize(static_cast<typename std::underlying_type<QuadratureSelector>::type>(
    QuadratureSelector::n_variants));
  // velocity
  quadratures[quad_index_u] = QGauss<1>(degree_u + 1);
  // pressure
  quadratures[quad_index_p] = QGauss<1>(degree_p + 1);
  // exact integration of nonlinear convective term
  quadratures[quad_index_u_nonlinear] = QGauss<1>(degree_u + (degree_u + 2) / 2);

  // reinit
  data.reinit(mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
  }

  void
  run()
  {
    laplace->apply(dst, src);
  }
  
  

  enum class DofHandlerSelector
  {
    velocity        = 0,
    pressure        = 1,
    velocity_scalar = 2,
    n_variants      = velocity_scalar + 1
  };

  enum class QuadratureSelector
  {
    velocity           = 0,
    pressure           = 1,
    velocity_nonlinear = 2,
    n_variants         = velocity_nonlinear + 1
  };

  static const unsigned int number_vorticity_components = (dim == 2) ? 1 : dim;

  static const unsigned int dof_index_u =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::velocity);
  static const unsigned int dof_index_p =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::pressure);
  static const unsigned int dof_index_u_scalar =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::velocity_scalar);

  static const unsigned int quad_index_u =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::velocity);
  static const unsigned int quad_index_p =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::pressure);
  static const unsigned int quad_index_u_nonlinear =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::velocity_nonlinear);


  std::shared_ptr<FESystem<dim>> fe_u;
  FE_DGQ<dim>                    fe_p;
  FE_DGQ<dim>                    fe_u_scalar;

  MappingQGeneric<dim> mapping;

  DoFHandler<dim> dof_handler_u;
  DoFHandler<dim> dof_handler_p;
  DoFHandler<dim> dof_handler_u_scalar;

  MatrixFree<dim, Number> data;

  std::shared_ptr<IncNS::ProjectionOperator<dim, degree_u, Number>> laplace;

  VectorType dst;
  VectorType src;
};

#endif