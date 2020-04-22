/*
 * elasticity_operator_base.h
 *
 *  Created on: 16.04.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_ELASTICITY_OPERATOR_BASE_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_ELASTICITY_OPERATOR_BASE_H_

#include "../../../operators/operator_base.h"

#include "../../material/material_handler.h"

#include "../../user_interface/boundary_descriptor.h"
#include "../../user_interface/material_descriptor.h"

#include "../../../functionalities/evaluate_functions.h"
#include "../../../functionalities/tensor_util.h"

namespace Structure
{
template<int dim>
struct Info
{
  static constexpr unsigned int n_stress_components = dim == 1 ? 1 : (dim == 2 ? 3 : 6);
};

template<int dim, typename Number>
Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>>
  tensor_to_vector(Tensor<2, dim, VectorizedArray<Number>> gradient_in)
{
  if(dim == 2)
  {
    Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> vector_in;
    vector_in[0] = gradient_in[0][0];
    vector_in[1] = gradient_in[1][1];
    vector_in[2] = gradient_in[1][0] + gradient_in[0][1];
    return vector_in;
  }
  else // dim==3
  {
    Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> vector_in;
    vector_in[0] = gradient_in[0][0];
    vector_in[1] = gradient_in[1][1];
    vector_in[2] = gradient_in[2][2];
    vector_in[3] = gradient_in[0][1] + gradient_in[1][0];
    vector_in[4] = gradient_in[1][2] + gradient_in[2][1];
    vector_in[5] = gradient_in[0][2] + gradient_in[2][0];
    return vector_in;
  }
}

template<int dim, typename Number>
Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>>
  symmetrize(Tensor<2, dim, VectorizedArray<Number>> gradient_in)
{
  if(dim == 2)
  {
    Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> vector_in;
    vector_in[0] = gradient_in[0][0];
    vector_in[1] = gradient_in[1][1];
    vector_in[2] = (gradient_in[1][0] + gradient_in[0][1]) * 0.5;
    return vector_in;
  }
  else // dim==3
  {
    Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> vector_in;
    vector_in[0] = gradient_in[0][0];
    vector_in[1] = gradient_in[1][1];
    vector_in[2] = gradient_in[2][2];
    vector_in[3] = (gradient_in[0][1] + gradient_in[1][0]) * 0.5;
    vector_in[4] = (gradient_in[1][2] + gradient_in[2][1]) * 0.5;
    vector_in[5] = (gradient_in[0][2] + gradient_in[2][0]) * 0.5;
    return vector_in;
  }
}

template<int dim, typename Number>
Tensor<2, dim, VectorizedArray<Number>>
  vector_to_tensor(Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> vector_out)
{
  if(dim == 2)
  {
    Tensor<2, dim, VectorizedArray<Number>> gradient_out;
    gradient_out[0][0] = vector_out[0];
    gradient_out[1][1] = vector_out[1];

    gradient_out[0][1] = vector_out[2];
    gradient_out[1][0] = vector_out[2];
    return gradient_out;
  }
  else // dim==3
  {
    Tensor<2, dim, VectorizedArray<Number>> gradient_out;
    gradient_out[0][0] = vector_out[0];
    gradient_out[1][1] = vector_out[1];
    gradient_out[2][2] = vector_out[2];

    gradient_out[0][1] = vector_out[3];
    gradient_out[1][0] = vector_out[3];

    gradient_out[1][2] = vector_out[4];
    gradient_out[2][1] = vector_out[4];

    gradient_out[0][2] = vector_out[5];
    gradient_out[2][0] = vector_out[5];
    return gradient_out;
  }
}

/**
 * This function computes the Cauchy stress tensor sigma from 2nd Piola-Kirchhoff stress S and the
 * deformation gradient F. This function is required for an updated Lagrangian formulation.
 */
template<int dim, typename Number>
inline const Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>>
get_sigma(const Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>> & S,
          const Tensor<2, dim, VectorizedArray<Number>> &                            F)
{
  // compute the determinant of the deformation gradient
  auto const det_F = determinant(F);

  // redo Voigt notation of S (now S is normal 2nd order tensor)
  auto const S_tensor = vector_to_tensor<dim, Number>(S);

  // compute Cauchy stresses
  auto const sigma_tensor = (F * S_tensor * transpose(F)) / det_F;

  // use Voigt notation on sigma (now sigma is a 1st order tensor)
  auto const sigma = tensor_to_vector<dim, Number>(sigma_tensor);

  // return sigma in Voigt notation
  return sigma;
}

template<int dim, typename Number>
Tensor<2, dim, VectorizedArray<Number>>
get_F(const Tensor<2, dim, VectorizedArray<Number>> & H)
{
  return add_identity(H);
}

template<int dim, typename Number>
Tensor<1, Info<dim>::n_stress_components, VectorizedArray<Number>>
get_E(const Tensor<2, dim, VectorizedArray<Number>> & F)
{
  return 0.5 * tensor_to_vector<dim, Number>(subtract_identity(transpose(F) * F));
}

/*
 * This function calculates the Neumann boundary value.
 */
template<int dim, typename Number>
inline DEAL_II_ALWAYS_INLINE //
  Tensor<1, dim, VectorizedArray<Number>>
  calculate_neumann_value(unsigned int const                             q,
                          FaceIntegrator<dim, dim, Number> const &       integrator,
                          BoundaryType const &                           boundary_type,
                          types::boundary_id const                       boundary_id,
                          std::shared_ptr<BoundaryDescriptor<dim>> const boundary_descriptor,
                          double const &                                 time)
{
  Tensor<1, dim, VectorizedArray<Number>> normal_gradient;

  if(boundary_type == BoundaryType::Neumann)
  {
    auto bc       = boundary_descriptor->neumann_bc.find(boundary_id)->second;
    auto q_points = integrator.quadrature_point(q);

    normal_gradient = FunctionEvaluator<1, dim, Number>::value(bc, q_points, time);
  }
  else
  {
    // do nothing

    AssertThrow(boundary_type == BoundaryType::Dirichlet,
                ExcMessage("Boundary type of face is invalid or not implemented."));
  }

  return normal_gradient;
}

template<int dim>
struct OperatorData : public OperatorBaseData
{
  OperatorData()
    : OperatorBaseData(0 /* dof_index */, 0 /* quad_index */),
      pull_back_traction(false),
      unsteady(false),
      density(1.0)
  {
  }

  std::shared_ptr<BoundaryDescriptor<dim>> bc;
  std::shared_ptr<MaterialDescriptor>      material_descriptor;

  // This parameter is only relevant for nonlinear operator
  // with large deformations. When set to true, the traction t
  // is pulled back to the reference configuration, t_0 = da/dA t.
  bool pull_back_traction;

  // activates mass matrix operator in operator evaluation for unsteady problems
  bool unsteady;

  // density
  double density;
};

template<int dim, typename Number>
class ElasticityOperatorBase : public OperatorBase<dim, Number, OperatorData<dim>, dim>
{
public:
  typedef Number value_type;

protected:
  typedef OperatorBase<dim, Number, OperatorData<dim>, dim> Base;
  typedef typename Base::VectorType                         VectorType;

public:
  ElasticityOperatorBase() : scaling_factor_mass(1.0)
  {
  }

  virtual ~ElasticityOperatorBase()
  {
  }

  IntegratorFlags
  get_integrator_flags(bool const unsteady) const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(unsteady, true, false);
    flags.cell_integrate = CellFlags(unsteady, true, false);

    // evaluation of Neumann BCs
    flags.face_evaluate  = FaceFlags(false, false);
    flags.face_integrate = FaceFlags(true, false);

    return flags;
  }

  static MappingFlags
  get_mapping_flags()
  {
    MappingFlags flags;

    flags.cells = update_gradients | update_JxW_values;

    flags.boundary_faces =
      update_gradients | update_JxW_values | update_normal_vectors | update_quadrature_points;

    return flags;
  }

  virtual void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         OperatorData<dim> const &         operator_data)
  {
    Base::reinit(matrix_free, constraint_matrix, operator_data);

    this->integrator_flags = this->get_integrator_flags(operator_data.unsteady);

    material_handler.initialize(this->get_data().material_descriptor);
  }

  void
  set_scaling_factor_mass(double const factor) const
  {
    scaling_factor_mass = factor;
  }

  void
  set_dirichlet_values_continuous(VectorType & dst, double const time) const
  {
    std::map<types::global_dof_index, double> boundary_values;
    fill_dirichlet_values_continuous(boundary_values, time);

    // set Dirichlet values in solution vector
    for(auto m : boundary_values)
      if(dst.get_partitioner()->in_local_range(m.first))
        dst[m.first] = m.second;
  }

  void
  set_dirichlet_values_continuous_hom(VectorType & dst) const
  {
    std::map<types::global_dof_index, double> boundary_values;
    fill_dirichlet_values_continuous_hom(boundary_values);

    // set Dirichlet values in solution vector
    for(auto m : boundary_values)
      if(dst.get_partitioner()->in_local_range(m.first))
        dst[m.first] = m.second;
  }

protected:
  virtual void
  reinit_cell(unsigned int const cell) const
  {
    Base::reinit_cell(cell);

    this->material_handler.reinit(*this->matrix_free, cell);
  }

  mutable MaterialHandler<dim, Number> material_handler;

  mutable double scaling_factor_mass;

private:
  void
  fill_dirichlet_values_continuous(std::map<types::global_dof_index, double> & boundary_values,
                                   double const                                time) const
  {
    for(auto dbc : this->data.bc->dirichlet_bc)
    {
      dbc.second->set_time(time);
      ComponentMask mask = this->data.bc->dirichlet_bc_component_mask.find(dbc.first)->second;

      VectorTools::interpolate_boundary_values(*this->matrix_free->get_mapping_info().mapping,
                                               this->matrix_free->get_dof_handler(
                                                 this->data.dof_index),
                                               dbc.first,
                                               *dbc.second,
                                               boundary_values,
                                               mask);
    }
  }

  void
  fill_dirichlet_values_continuous_hom(
    std::map<types::global_dof_index, double> & boundary_values) const
  {
    for(auto dbc : this->data.bc->dirichlet_bc)
    {
      Functions::ZeroFunction<dim> zero_function = Functions::ZeroFunction<dim>(dim);

      ComponentMask mask = this->data.bc->dirichlet_bc_component_mask.find(dbc.first)->second;

      VectorTools::interpolate_boundary_values(*this->matrix_free->get_mapping_info().mapping,
                                               this->matrix_free->get_dof_handler(
                                                 this->data.dof_index),
                                               dbc.first,
                                               zero_function,
                                               boundary_values,
                                               mask);
    }
  }
};

} // namespace Structure



#endif /* INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_OPERATORS_ELASTICITY_OPERATOR_BASE_H_ */
