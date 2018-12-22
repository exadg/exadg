#ifndef OPERATOR_WRAPPERS_CONV_DIFF_CONV
#define OPERATOR_WRAPPERS_CONV_DIFF_CONV

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_system.h>

#include "operator_wrapper.h"

#include "../../../../include/convection_diffusion/spatial_discretization/operators/convective_operator.h"

template<int dim, int degree, int degree_velocity, typename Number>
class OperatorWrapperConvectiveOperator : public OperatorWrapper
{
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  
public:
  OperatorWrapperConvectiveOperator(parallel::distributed::Triangulation<dim> const & triangulation)
    : fe(degree),
      mapping(1 /*TODO*/),
      dof_handler(triangulation)
  {
    this->create_dofs();
    this->initialize_matrix_free();

    ConvDiff::ConvectiveOperatorData<dim> laplace_additional_data;
    laplace_additional_data.velocity.reset(new Functions::ZeroFunction<dim>(dim));
    
    laplace.reinit(data, dummy, laplace_additional_data);
    
    // initialize vectors
    laplace.do_initialize_dof_vector(src);
    laplace.do_initialize_dof_vector(dst);
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

    ConvDiff::ConvectiveOperatorData<dim> laplace_additional_data;
    additional_data.mapping_update_flags                = laplace_additional_data.mapping_update_flags;
    additional_data.mapping_update_flags_inner_faces    = laplace_additional_data.mapping_update_flags_inner_faces;
    additional_data.mapping_update_flags_boundary_faces = laplace_additional_data.mapping_update_flags_boundary_faces;

    QGauss<1> quadrature(degree+1);
    
    data.reinit(mapping, dof_handler, dummy, quadrature, additional_data);
  }

  void
  run()
  {
    laplace.apply(dst, src); // inplace
  }

  FE_DGQ<dim> fe;

  MappingQGeneric<dim> mapping;
  AffineConstraints<double> dummy;

  DoFHandler<dim> dof_handler;

  MatrixFree<dim, Number> data;

  ConvDiff::ConvectiveOperator<dim, degree, degree_velocity, Number> laplace;

  VectorType src;
  VectorType dst;
};

#endif
