#ifndef OPERATOR_WRAPPERS_LAPLACE
#define OPERATOR_WRAPPERS_LAPLACE

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include "operator_wrapper.h"

#include "../../../../../include/poisson/spatial_discretization/laplace_operator.h"

namespace Poisson
{
template<int dim, int degree, typename Number>
class OperatorWrapper : public OperatorWrapperBase
{
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

public:
  OperatorWrapper(parallel::distributed::Triangulation<dim> const & triangulation)
    : fe(degree), mapping(1 /*TODO*/), dof_handler(triangulation)
  {
    this->create_dofs();
    this->initialize_matrix_free();

    LaplaceOperatorData<dim> laplace_additional_data;
    laplace.reinit(/*mapping,*/ data, dummy, laplace_additional_data);

    // initialize vectors
    laplace.initialize_dof_vector(src);
    laplace.initialize_dof_vector(dst);
  }

  void
  create_dofs()
  {
    dof_handler.distribute_dofs(fe);
  }

  void
  initialize_matrix_free()
  {
    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    LaplaceOperatorData<dim> laplace_additional_data;
    additional_data.mapping_update_flags = laplace_additional_data.mapping_update_flags;
    additional_data.mapping_update_flags_inner_faces =
      laplace_additional_data.mapping_update_flags_inner_faces;
    additional_data.mapping_update_flags_boundary_faces =
      laplace_additional_data.mapping_update_flags_boundary_faces;

    QGauss<1> quadrature(degree + 1);

    data.reinit(mapping, dof_handler, dummy, quadrature, additional_data);
  }

  void
  run()
  {
    laplace.apply(dst, src);
  }

  FE_DGQ<dim> fe;

  MappingQGeneric<dim>      mapping;
  AffineConstraints<double> dummy;

  DoFHandler<dim> dof_handler;

  MatrixFree<dim, Number> data;

  LaplaceOperator<dim, degree, Number> laplace;

  VectorType dst;
  VectorType src;
};

} // namespace Poisson

#endif