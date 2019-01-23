#ifndef OPERATOR_WRAPPERS_COMP_NAVIER_STOKES
#define OPERATOR_WRAPPERS_COMP_NAVIER_STOKES

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include "operator_wrapper.h"

#include "../../../../../include/compressible_navier_stokes/spatial_discretization/comp_navier_stokes_operators.h"

namespace CompNS
{
template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
class OperatorWrapper : public OperatorWrapperBase
{
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  OperatorWrapper(parallel::distributed::Triangulation<dim> const & triangulation)
    : fe(new FESystem<dim>(FE_DGQ<dim>(degree), dim + 2)),
      fe_vector(new FESystem<dim>(FE_DGQ<dim>(degree), dim)),
      fe_scalar(degree),
      mapping(1 /*TODO*/),
      dof_handler(triangulation),
      dof_handler_vector(triangulation),
      dof_handler_scalar(triangulation)
  {
    this->create_dofs();
    this->initialize_matrix_free();

    // initialize vectors
    this->initialize_dof_vector(src);
    this->initialize_dof_vector(dst);
  }

  void
  create_dofs()
  {
    dof_handler.distribute_dofs(*fe);
    dof_handler_vector.distribute_dofs(*fe_vector);
    dof_handler_scalar.distribute_dofs(fe_scalar);
  }

  void
  initialize_dof_vector(VectorType & vct) const
  {
    data.initialize_dof_vector(vct, dof_index_all);
  }

  void
  initialize_matrix_free()
  {
    // quadratures used to perform integrals
    std::vector<Quadrature<1>> quadratures;
    quadratures.resize(static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::n_variants));
    quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::standard)]             = QGauss<1>(degree + 1);
    quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::overintegration_conv)] = QGauss<1>(n_q_points_conv);
    quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::overintegration_vis)]  = QGauss<1>(n_q_points_vis);

    // dof handler
    std::vector<const DoFHandler<dim> *> dof_handler_vec;
    dof_handler_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::n_variants));
    dof_handler_vec[dof_index_all]    = &dof_handler;
    dof_handler_vec[dof_index_vector] = &dof_handler_vector;
    dof_handler_vec[dof_index_scalar] = &dof_handler_scalar;

    // initialize matrix_free_data
    typename MatrixFree<dim, Number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme =
      MatrixFree<dim, Number>::AdditionalData::partition_partition;

    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);
    additional_data.mapping_update_flags_boundary_faces |= update_quadrature_points;
    additional_data.mapping_update_flags_inner_faces |= update_quadrature_points;

    // constraints
    std::vector<const AffineConstraints<double> *> constraint_matrix_vec;
    constraint_matrix_vec.resize(
      static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
        DofHandlerSelector::n_variants));
    AffineConstraints<double> constraint;
    constraint.close();
    constraint_matrix_vec[dof_index_all]    = &constraint;
    constraint_matrix_vec[dof_index_vector] = &constraint;
    constraint_matrix_vec[dof_index_scalar] = &constraint;

    data.reinit(mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
  }

  enum class DofHandlerSelector
  {
    all_components = 0,
    vector         = 1,
    scalar         = 2,
    n_variants     = scalar + 1
  };

  enum class QuadratureSelector
  {
    overintegration_conv = 0,
    overintegration_vis  = 1,
    standard             = 2,
    n_variants           = standard + 1
  };

  static const unsigned int dof_index_all =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::all_components);
  static const unsigned int dof_index_vector =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::vector);
  static const unsigned int dof_index_scalar =
    static_cast<typename std::underlying_type<DofHandlerSelector>::type>(
      DofHandlerSelector::scalar);

  static const unsigned int quad_index_standard =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::standard);
  static const unsigned int quad_index_overintegration_conv =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::overintegration_conv);
  static const unsigned int quad_index_overintegration_vis =
    static_cast<typename std::underlying_type<QuadratureSelector>::type>(
      QuadratureSelector::overintegration_vis);

  // fe
  std::shared_ptr<FESystem<dim>> fe;        // all (dim+2) components: (rho, rho u, rho E)
  std::shared_ptr<FESystem<dim>> fe_vector; // e.g. velocity
  FE_DGQ<dim>                    fe_scalar; // scalar quantity, e.g, pressure

  // mapping
  MappingQGeneric<dim> mapping;

  // DoFHandler for all (dim+2) components: (rho, rho u, rho E)
  DoFHandler<dim> dof_handler;
  // DoFHandler for vectorial quantities such as the velocity
  DoFHandler<dim> dof_handler_vector;
  // DoFHandler for scalar quantities such as pressure, temperature
  DoFHandler<dim> dof_handler_scalar;

  MatrixFree<dim, Number> data;

  VectorType dst;
  VectorType src;
};

template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
class CombinedWrapper : public OperatorWrapper<dim, degree, n_q_points_conv, n_q_points_vis, Number>
{

public:
  CombinedWrapper(parallel::distributed::Triangulation<dim> const & triangulation)
    : OperatorWrapper<dim, degree, n_q_points_conv, n_q_points_vis, Number>(triangulation)
  {
    CombinedOperatorData<dim> operator_data;
    op.initialize(this->mapping, this->data, operator_data);
  }

  void
  run()
  {
    op.evaluate_add(this->dst, this->src, 0.0);
  }

  CombinedOperator<dim, degree, n_q_points_vis, Number> op;

};

template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
class ConvectiveWrapper : public OperatorWrapper<dim, degree, n_q_points_conv, n_q_points_vis, Number>
{

public:
  ConvectiveWrapper(parallel::distributed::Triangulation<dim> const & triangulation)
    : OperatorWrapper<dim, degree, n_q_points_conv, n_q_points_vis, Number>(triangulation)
  {
    ConvectiveOperatorData<dim> operator_data;
    op.initialize(this->data, operator_data);
  }

  void
  run()
  {
    op.evaluate_add(this->dst, this->src, 0.0);
  }

  ConvectiveOperator<dim, degree, n_q_points_conv, Number> op;

};

template<int dim, int degree, int n_q_points_conv, int n_q_points_vis, typename Number>
class ViscousWrapper : public OperatorWrapper<dim, degree, n_q_points_conv, n_q_points_vis, Number>
{

public:
  ViscousWrapper(parallel::distributed::Triangulation<dim> const & triangulation)
    : OperatorWrapper<dim, degree, n_q_points_conv, n_q_points_vis, Number>(triangulation)
  {
    ViscousOperatorData<dim> operator_data;
    op.initialize(this->mapping, this->data, operator_data);
  }

  void
  run()
  {
    op.evaluate_add(this->dst, this->src, 0.0);
  }

  ViscousOperator<dim, degree, n_q_points_vis, Number> op;

};

} // namespace CompNS

#endif