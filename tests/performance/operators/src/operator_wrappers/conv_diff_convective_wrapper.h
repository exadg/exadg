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
  OperatorWrapperConvectiveOperator(parallel::distributed::Triangulation<dim> const & triangulation,
          ConvDiff::TypeVelocityField type_velocity_field)
    : fe(degree),
      mapping(1 /*TODO*/),
      dof_handler(triangulation),
      type_velocity_field(type_velocity_field)
  {
        

  if(this->type_velocity_field == ConvDiff::TypeVelocityField::Numerical)
  {
    fe_velocity.reset(new FESystem<dim>(FE_DGQ<dim>(degree), dim));
    dof_handler_velocity.reset(new DoFHandler<dim>(triangulation));
  }
        
    this->create_dofs();
    this->initialize_matrix_free();

    ConvDiff::ConvectiveOperatorData<dim> laplace_additional_data;
    if(this->type_velocity_field == ConvDiff::TypeVelocityField::Analytical)
      laplace_additional_data.velocity.reset(new Functions::ZeroFunction<dim>(dim));
    
    laplace_additional_data.type_velocity_field = this->type_velocity_field;
    
    laplace.reinit(data, dummy, laplace_additional_data);

  if(this->type_velocity_field == ConvDiff::TypeVelocityField::Numerical)
  {
    data.initialize_dof_vector(velocity, laplace_additional_data.dof_index_velocity);
  }
    
    // initialize vectors
    laplace.do_initialize_dof_vector(src);
    laplace.do_initialize_dof_vector(dst);
  }

  void
  create_dofs()
  {
    dof_handler.distribute_dofs(fe);

  if(this->type_velocity_field == ConvDiff::TypeVelocityField::Numerical)
    dof_handler_velocity->distribute_dofs(*fe_velocity);
  }

  void
  initialize_matrix_free()
  {
    typename MatrixFree<dim, Number>::AdditionalData additional_data;

    ConvDiff::ConvectiveOperatorData<dim> laplace_additional_data;
    additional_data.mapping_update_flags                = laplace_additional_data.mapping_update_flags;
    additional_data.mapping_update_flags_inner_faces    = laplace_additional_data.mapping_update_flags_inner_faces;
    additional_data.mapping_update_flags_boundary_faces = laplace_additional_data.mapping_update_flags_boundary_faces;

  if(this->type_velocity_field == ConvDiff::TypeVelocityField::Analytical)
  {
    QGauss<1> quadrature(degree+1);
    data.reinit(mapping, dof_handler, dummy, quadrature, additional_data);
  } 
  else
  {
    std::vector<const DoFHandler<dim> *> dof_handler_vec;
    dof_handler_vec.resize(2);
    dof_handler_vec[0] = &dof_handler;
    dof_handler_vec[1] = &(*dof_handler_velocity);

    std::vector<const AffineConstraints<double> *> constraint_vec;
    constraint_vec.resize(2);
    AffineConstraints<double> constraint_dummy;
    constraint_dummy.close();
    constraint_vec[0] = &constraint_dummy;
    constraint_vec[1] = &constraint_dummy;

    std::vector<Quadrature<1>> quadrature_vec;
    quadrature_vec.resize(1);
    quadrature_vec[0] = QGauss<1>(degree + 1);

    data.reinit(mapping, dof_handler_vec, constraint_vec, quadrature_vec, additional_data);
  }
    
  }

  void
  run()
  {
    laplace.apply(dst, src);
  }

  FE_DGQ<dim> fe;

  MappingQGeneric<dim> mapping;
  AffineConstraints<double> dummy;

  DoFHandler<dim> dof_handler;

  MatrixFree<dim, Number> data;

  ConvDiff::ConvectiveOperator<dim, degree, degree_velocity, Number> laplace;
  
  ConvDiff::TypeVelocityField type_velocity_field;
  
  /*
   * Numerical velocity field.
   */
  std::shared_ptr<FESystem<dim>>   fe_velocity;
  std::shared_ptr<DoFHandler<dim>> dof_handler_velocity;

  VectorType src;
  VectorType dst;
  VectorType velocity;
};

#endif
