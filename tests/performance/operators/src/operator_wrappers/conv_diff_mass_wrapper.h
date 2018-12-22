#ifndef OPERATOR_WRAPPERS_CONV_DIFF_MASS
#define OPERATOR_WRAPPERS_CONV_DIFF_MASS

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include "operator_wrapper.h"

#include "../../../../include/convection_diffusion/spatial_discretization/operators/mass_operator.h"

template<int dim, int degree, typename Number>
class OperatorWrapperMassMatrix : public OperatorWrapper
{
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  
public:
  OperatorWrapperMassMatrix(parallel::distributed::Triangulation<dim> const & triangulation)
    : fe(degree),
      mapping(1 /*TODO*/),
      dof_handler(triangulation)
  {
    this->create_dofs();
    this->initialize_matrix_free();

    ConvDiff::MassMatrixOperatorData<dim> laplace_additional_data;
    laplace.reinit(data, dummy, laplace_additional_data);
    
    // initialize vectors
    laplace.do_initialize_dof_vector(src);
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

    ConvDiff::MassMatrixOperatorData<dim> laplace_additional_data;
    additional_data.mapping_update_flags = laplace_additional_data.mapping_update_flags;

    QGauss<1> quadrature(degree+1);
    
    data.reinit(mapping, dof_handler, dummy, quadrature, additional_data);
  }

  void
  run()
  {
    laplace.apply(src, src); // inplace
  }

  FE_DGQ<dim> fe;

  MappingQGeneric<dim> mapping;
  AffineConstraints<double> dummy;

  DoFHandler<dim> dof_handler;

  MatrixFree<dim, Number> data;

  ConvDiff::MassMatrixOperator<dim, degree, Number> laplace;

  VectorType src;
};

#endif
