#ifndef CONV_DIFF_CONVECTION_OPERATOR
#define CONV_DIFF_CONVECTION_OPERATOR

#include "../../../operators/operation_base.h"

namespace ConvDiff {

template<int dim>
struct ConvectiveOperatorData
    : public OperatorBaseData<dim, BoundaryType, OperatorType,
                              ConvDiff::BoundaryDescriptor<dim>> {
    
  ConvectiveOperatorData()
      : OperatorBaseData<dim, BoundaryType, OperatorType,
                         ConvDiff::BoundaryDescriptor<dim>>(
            0, 0, true, false, false, false, true, false, // cell
            true, false, true, false, // face
            true, false, true, false  // boundary
          ),
    numerical_flux_formulation(NumericalFluxConvectiveOperator::Undefined){}

  NumericalFluxConvectiveOperator numerical_flux_formulation;
  std::shared_ptr<Function<dim> > velocity;
};

template <int dim, int fe_degree, typename value_type>
class ConvectiveOperator : public OperatorBase<dim, fe_degree, value_type,
                                              ConvectiveOperatorData<dim>> {
public:
  typedef ConvectiveOperator<dim,fe_degree, value_type> This;
  typedef OperatorBase<dim, fe_degree, value_type, ConvectiveOperatorData<dim>>
      Parent;
  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;
  typedef typename Parent::VNumber VNumber;
  
    void initialize(MatrixFree<dim,value_type> const  &mf_data,
                  ConvectiveOperatorData<dim> const &operator_data_in)
  {
  ConstraintMatrix cm;
  Parent::reinit(mf_data, cm, operator_data_in);
  }

  /*
   *  This function calculates the numerical flux for interior faces
   *  using the central flux.
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_central_flux(VectorizedArray<value_type> &value_m,
                         VectorizedArray<value_type> &value_p,
                         VectorizedArray<value_type> &normal_velocity) const
  {
    VectorizedArray<value_type> average_value = 0.5*(value_m + value_p);
    return normal_velocity*average_value;
  }

  /*
   *  This function calculates the numerical flux for interior faces
   *  using the Lax-Friedrichs flux.
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_lax_friedrichs_flux(VectorizedArray<value_type> &value_m,
                                VectorizedArray<value_type> &value_p,
                                VectorizedArray<value_type> &normal_velocity) const
  {
    VectorizedArray<value_type> average_value = 0.5*(value_m + value_p);
    VectorizedArray<value_type> jump_value = value_m - value_p;
    VectorizedArray<value_type> lambda = std::abs(normal_velocity);
    return normal_velocity*average_value + 0.5*lambda*jump_value;
  }

  /*
   *  This function calculates the numerical flux for interior faces where
   *  the type of the numerical flux depends on the specified input parameter.
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_flux(unsigned int const          q,
                 FEEvalFace                &fe_eval,
                 VectorizedArray<value_type> &value_m,
                 VectorizedArray<value_type> &value_p) const
  {
    VectorizedArray<value_type> flux = make_vectorized_array<value_type>(0.0);

    Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
    Tensor<1,dim,VectorizedArray<value_type> > velocity;

    evaluate_vectorial_function(velocity,this->ad.velocity,q_points,this->eval_time);

    Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
    VectorizedArray<value_type> normal_velocity = velocity*normal;

    if(this->ad.numerical_flux_formulation
        == NumericalFluxConvectiveOperator::CentralFlux)
    {
      flux = calculate_central_flux(value_m,value_p,normal_velocity);
    }
    else if(this->ad.numerical_flux_formulation
        == NumericalFluxConvectiveOperator::LaxFriedrichsFlux)
    {
      flux = calculate_lax_friedrichs_flux(value_m,value_p,normal_velocity);
    }

    return flux;
  }

  /*
   *  The following two functions calculate the interior_value/exterior_value
   *  depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *                            +----------------------+--------------------+
   *                            | Dirichlet boundaries | Neumann boundaries |
   *  +-------------------------+----------------------+--------------------+
   *  | full operator           | phi⁺ = -phi⁻ + 2g    | phi⁺ = phi⁻        |
   *  +-------------------------+----------------------+--------------------+
   *  | homogeneous operator    | phi⁺ = -phi⁻         | phi⁺ = phi⁻        |
   *  +-------------------------+----------------------+--------------------+
   *  | inhomogeneous operator  | phi⁻ = 0, phi⁺ = 2g  | phi⁻ = 0, phi⁺ = 0 |
   *  +-------------------------+----------------------+--------------------+
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_interior_value(unsigned int const q,
                           FEEvalFace const &fe_eval,
                           OperatorType const &op_type) const
  {

    if(op_type == OperatorType::full || op_type == OperatorType::homogeneous)
      return fe_eval.get_value(q);
    else if(op_type == OperatorType::inhomogeneous)
      return make_vectorized_array<value_type>(0.0);
    else
      AssertThrow(false, ExcMessage("Specified OpType is not implemented!"));

    return make_vectorized_array<value_type>(0.0);
  }

  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_exterior_value(VectorizedArray<value_type> const &value_m,
                           unsigned int const                q,
                           FEEvalFace const                &fe_eval,
                           OperatorType const                &operator_type,
                           BoundaryType const                &boundary_type,
                           types::boundary_id const          boundary_id = types::boundary_id()) const
  {
    VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);

    if(boundary_type == BoundaryType::dirichlet)
    {
      if(operator_type == OperatorType::full ||
         operator_type == OperatorType::inhomogeneous)
      {
        VectorizedArray<value_type> g = make_vectorized_array<value_type>(0.0);
        typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
        it = this->ad.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        evaluate_scalar_function(g, it->second, q_points, this->eval_time);

        value_p = - value_m + 2.0*g;
      }
      else if(operator_type == OperatorType::homogeneous)
      {
        value_p = - value_m;
      }
      else
      {
        AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
      }
    }
    else if(boundary_type == BoundaryType::neumann)
    {
      value_p = value_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return value_p;
  }

  void do_cell_integral(FEEvalCell &fe_eval) const
  {
    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
      Tensor<1,dim,VectorizedArray<value_type> > velocity;

      evaluate_vectorial_function(velocity,this->ad.velocity,q_points,this->eval_time);

      fe_eval.submit_gradient(-fe_eval.get_value(q)*velocity,q);
    }
  }

  void do_face_integral(FEEvalFace &fe_eval,
                                        FEEvalFace &fe_eval_neighbor) const
  {
    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      VectorizedArray<value_type> value_m = fe_eval.get_value(q);
      VectorizedArray<value_type> value_p = fe_eval_neighbor.get_value(q);
      VectorizedArray<value_type> numerical_flux = calculate_flux(q, fe_eval, value_m, value_p);

      fe_eval.submit_value(numerical_flux,q);
      fe_eval_neighbor.submit_value(-numerical_flux,q);
    }
  }

  void do_face_int_integral(FEEvalFace &fe_eval, FEEvalFace &/*fe_eval_neighbor*/) const
  {
    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      VectorizedArray<value_type> value_m = fe_eval.get_value(q);
      // set value_p to zero
      VectorizedArray<value_type> value_p = make_vectorized_array<value_type>(0.0);
      VectorizedArray<value_type> numerical_flux = calculate_flux(q, fe_eval, value_m, value_p);

      fe_eval.submit_value(numerical_flux,q);
    }
  }

  /*
   *  When performing face integrals in a cell-based manner, i.e., looping over all
   *  faces of a cell instead of all interior or boundary faces, this function will
   *  not be necessary anymore.
   */
  void do_face_ext_integral(FEEvalFace &/*fe_eval*/, FEEvalFace &fe_eval_neighbor) const
  {
    for(unsigned int q=0;q<fe_eval_neighbor.n_q_points;++q)
    {
      // set value_m to zero
      VectorizedArray<value_type> value_m = make_vectorized_array<value_type>(0.0);
      VectorizedArray<value_type> value_p = fe_eval_neighbor.get_value(q);
      VectorizedArray<value_type> numerical_flux = calculate_flux(q, fe_eval_neighbor, value_m, value_p);

      // hack (minus sign) since n⁺ = -n⁻
      fe_eval_neighbor.submit_value(-numerical_flux,q);
    }
  }

  void do_boundary_integral(FEEvalFace             &fe_eval,
                                        OperatorType const       &operator_type,
                                        types::boundary_id const &boundary_id) const
  {
    BoundaryType boundary_type = this->ad.get_boundary_type(boundary_id);

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      VectorizedArray<value_type> value_m = calculate_interior_value(q, fe_eval, operator_type);
      VectorizedArray<value_type> value_p = calculate_exterior_value(value_m, q, fe_eval, operator_type, boundary_type, boundary_id);
      VectorizedArray<value_type> numerical_flux = calculate_flux(q, fe_eval, value_m, value_p);

      fe_eval.submit_value(numerical_flux,q);
    }
  }

};

}

#endif