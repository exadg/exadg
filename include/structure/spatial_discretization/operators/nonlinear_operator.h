/*
 * nonlinear_operator.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_NONLINEAR_OPERATOR_H_
#define INCLUDE_STRUCTURE_SPATIAL_DISCRETIZATION_NONLINEAR_OPERATOR_H_

#include "../../material/material_handler.h"

#include "continuum_mechanics_util.h"
#include "nonlinear_operator_base.h"
#include "operator_data.h"
#include "tensor_util.h"

namespace Structure
{
template<int dim, typename Number>
class NonLinearOperator : public OperatorBase<dim, Number, OperatorData<dim>, dim>,
                          public NonLinearOperatorBase<dim, Number, dim>
{
public:
  typedef NonLinearOperator<dim, Number>                    This;
  typedef OperatorBase<dim, Number, OperatorData<dim>, dim> Base;
  typedef NonLinearOperatorBase<dim, Number, dim>           NonLinearBase;
  typedef CellIntegrator<dim, dim, Number>                  IntegratorCell;

  void
  reinit(MatrixFree<dim, Number> const &   matrix_free,
         AffineConstraints<double> const & constraint_matrix,
         OperatorData<dim> const &         operator_data)
  {
    Base::reinit(matrix_free, constraint_matrix, operator_data);
    NonLinearBase::reinit(matrix_free);

    this->integrator_flags = this->get_integrator_flags();

    material_handler.initialize(this->get_data().material_descriptor);
  }

  IntegratorFlags
  get_integrator_flags() const
  {
    IntegratorFlags flags;

    flags.cell_evaluate  = CellFlags(false, true, false);
    flags.cell_integrate = CellFlags(false, true, false);

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

  void
  do_cell_integral(IntegratorCell & integrator) const
  {
    (void)integrator;

    AssertThrow(false, ExcMessage("not implemented."));
  }

  void
  do_cell_integral(IntegratorCell & integrator, unsigned int const cell) const
  {
    IntegratorCell integrator_lin(this->get_linerization_point(cell));

    // TODO this->matrix_free is ambiguous due to multiple inheritance
    this->material_handler.reinit(*Base::matrix_free, cell);

    std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      auto const H = integrator.get_gradient(q);

      // ... and for the linearization point
      auto const F = get_F<dim, Number>(integrator_lin.get_gradient(q));
      auto const E = get_E<dim, Number>(F);

      // update material
      material->reinit(E);
      auto const S = material->get_S();
      auto const C = material->get_dSdE();

      // step 1: keu * u
      auto const v_eu =
        F * apply_l_transposed<dim, Number>(C * apply_l<dim, Number>(transpose(F) * H));

      // step 2: kg * u
      // distinguish between total Lagrangian formulation and updated Lagrangian formulation
      auto const v_g = false ? H * apply_l_transposed<dim, Number>(S) :
                               H * apply_l_transposed<dim, Number>(get_sigma<dim, Number>(S, F));

      // test with gradients
      integrator.submit_gradient(v_eu + v_g, q);
    }
  }

  void
  do_cell_residuum_integral(IntegratorCell & integrator, unsigned int const cell) const
  {
    // TODO this->matrix_free is ambiguous due to multiple inheritance
    this->material_handler.reinit(*Base::matrix_free, cell);

    std::shared_ptr<Material<dim, Number>> material = this->material_handler.get_material();

    // loop over all quadrature points
    for(unsigned int q = 0; q < integrator.n_q_points; ++q)
    {
      // update geometrical information
      auto const grad = integrator.get_gradient(q);
      auto const F    = get_F<dim, Number>(grad);
      auto const E    = get_E<dim, Number>(F);

      // update material
      material->reinit(E);
      auto const S = material->get_S();

      // compute J^-1 * |J| * w * F * S //compute internal Forces of cell
      // distinguish between total Lagrangian formulation and updated Lagrangian formulation
      if(this->data.updated_formulation)
        integrator.submit_gradient(F * apply_l_transposed<dim, Number>(S), q);
      else
        integrator.submit_gradient(apply_l_transposed<dim, Number>(get_sigma<dim, Number>(S, F)),
                                   q);
    }
  }

private:
  mutable MaterialHandler<dim, Number> material_handler;
};

} // namespace Structure

#endif
