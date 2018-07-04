#ifndef CONV_DIFF_DIFFUSIVE_OPERATOR
#define CONV_DIFF_DIFFUSIVE_OPERATOR

#include "../../../operators/operation_base.h"

namespace ConvDiff
{
template<int dim>
struct DiffusiveOperatorData : public OperatorBaseData<dim, BoundaryType, OperatorType, ConvDiff::BoundaryDescriptor<dim>> 
{
  DiffusiveOperatorData ()
    : OperatorBaseData<dim, BoundaryType, OperatorType, ConvDiff::BoundaryDescriptor<dim>>(
        0, 0, false, true, false, false, true, false,
                              true, true, true, true, // face
                              true, true, true, true  // boundary
                              ), IP_factor(1.0), diffusivity(1.0) {}

  double IP_factor;
  double diffusivity;
};



template <int dim, int fe_degree, typename value_type>
class DiffusiveOperator 
    : public OperatorBase<dim, fe_degree, value_type, DiffusiveOperatorData<dim>>
{
public:
  typedef DiffusiveOperator<dim, fe_degree, value_type> This;
  typedef OperatorBase<dim, fe_degree, value_type, DiffusiveOperatorData<dim>> Parent;
  typedef typename Parent::FEEvalCell FEEvalCell;
  typedef typename Parent::FEEvalFace FEEvalFace;

  DiffusiveOperator() : /*data(nullptr),*/ diffusivity(-1.0) {}

  void initialize(Mapping<dim> const               &mapping,
                  MatrixFree<dim,value_type> const &mf_data,
                  DiffusiveOperatorData<dim> const &operator_data_in)
  {
      ConstraintMatrix cm;
      Parent::reinit(mf_data, cm, operator_data_in);

    IP::calculate_penalty_parameter<dim, fe_degree, value_type>(array_penalty_parameter,
                                                                *this->data,
                                                                mapping,
                                                                this->ad.dof_index);

    diffusivity = this->ad.diffusivity;
  }

  /*
   *  Calculation of "value_flux".
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_value_flux (VectorizedArray<value_type> const &jump_value) const
  {
    return -0.5 * diffusivity * jump_value;
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
  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_interior_value(unsigned int const q,
                           FEEvaluation const &fe_eval,
                           OperatorType const &operator_type) const
  {
    VectorizedArray<value_type> value_m = make_vectorized_array<value_type>(0.0);

    if(operator_type == OperatorType::full ||
       operator_type == OperatorType::homogeneous)
    {
      value_m = fe_eval.get_value(q);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      value_m = make_vectorized_array<value_type>(0.0);
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    return value_m;
  }

  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_exterior_value(VectorizedArray<value_type> const &value_m,
                           unsigned int const                q,
                           FEEvaluation const                &fe_eval,
                           OperatorType const                &operator_type,
                           BoundaryType const                &boundary_type,
                           types::boundary_id const          boundary_id) const
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

  /*
   *  Calculation of gradient flux. Strictly speaking, this value is not a numerical flux since
   *  the flux is multiplied by the normal vector, i.e., "gradient_flux" = numerical_flux * normal,
   *  where normal denotes the normal vector of element e⁻.
   */
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_gradient_flux (VectorizedArray<value_type> const &normal_gradient_m,
                           VectorizedArray<value_type> const &normal_gradient_p,
                           VectorizedArray<value_type> const &jump_value,
                           VectorizedArray<value_type> const &penalty_parameter) const
  {
    return diffusivity * 0.5 * (normal_gradient_m + normal_gradient_p) - diffusivity * penalty_parameter * jump_value;
  }

  /*
   *  The following two functions calculate the interior/exterior velocity gradient
   *  in normal direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *                            +-------------------------------------+---------------------------------------+
   *                            | Dirichlet boundaries                | Neumann boundaries                    |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | full operator           | grad(phi⁺)*n = grad(phi⁻)*n         | grad(phi⁺)*n = -grad(phi⁻)*n + 2h     |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | homogeneous operator    | grad(phi⁺)*n = grad(phi⁻)*n         | grad(phi⁺)*n = -grad(phi⁻)*n          |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | inhomogeneous operator  | grad(phi⁻)*n  = 0, grad(phi⁺)*n = 0 | grad(phi⁻)*n  = 0, grad(phi⁺)*n  = 2h |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *
   *                            +-------------------------------------+---------------------------------------+
   *                            | Dirichlet boundaries                | Neumann boundaries                    |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | full operator           | {{grad(phi)}}*n = grad(phi⁻)*n      | {{grad(phi)}}*n = h                   |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | homogeneous operator    | {{grad(phi)}}*n = grad(phi⁻)*n      | {{grad(phi)}}*n = 0                   |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   *  | inhomogeneous operator  | {{grad(phi)}}*n = 0                 | {{grad(phi)}}*n = h                   |
   *  +-------------------------+-------------------------------------+---------------------------------------+
   */
  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_interior_normal_gradient(unsigned int const q,
                                     FEEvaluation const &fe_eval,
                                     OperatorType const &operator_type) const
  {
    VectorizedArray<value_type> normal_gradient_m = make_vectorized_array<value_type>(0.0);

    if(operator_type == OperatorType::full ||
       operator_type == OperatorType::homogeneous)
    {
      normal_gradient_m = fe_eval.get_normal_gradient(q);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      normal_gradient_m = make_vectorized_array<value_type>(0.0);
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    return normal_gradient_m;
  }

  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE VectorizedArray<value_type>
  calculate_exterior_normal_gradient(VectorizedArray<value_type> const &normal_gradient_m,
                                     unsigned int const                q,
                                     FEEvaluation const                &fe_eval,
                                     OperatorType const                &operator_type,
                                     BoundaryType const                &boundary_type,
                                     types::boundary_id const          boundary_id) const
  {
    VectorizedArray<value_type> normal_gradient_p = make_vectorized_array<value_type>(0.0);

    if(boundary_type == BoundaryType::dirichlet)
    {
      normal_gradient_p = normal_gradient_m;
    }
    else if(boundary_type == BoundaryType::neumann)
    {
      if(operator_type == OperatorType::full ||
         operator_type == OperatorType::inhomogeneous)
      {
        VectorizedArray<value_type> h = make_vectorized_array<value_type>(0.0);
        typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
        it = this->ad.bc->neumann_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        evaluate_scalar_function(h, it->second, q_points, this->eval_time);

        normal_gradient_p = -normal_gradient_m + 2.0 * h;
      }
      else if(operator_type == OperatorType::homogeneous)
      {
        normal_gradient_p = -normal_gradient_m;
      }
      else
      {
        AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
      }
    }
    else
    {
      AssertThrow(false,ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return normal_gradient_p;
  }

  void do_cell_integral(FEEvalCell &fe_eval) const {
    fe_eval.evaluate (false,true,false);

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_gradient (make_vectorized_array<value_type>(diffusivity)*fe_eval.get_gradient(q), q);
    }
    fe_eval.integrate (false,true);
  }

  void do_face_integral(FEEvalFace &fe_eval,FEEvalFace &fe_eval_neighbor) const{
    VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                  fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                           * IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);

    fe_eval.evaluate(true,true);
    fe_eval_neighbor.evaluate(true,true);

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      VectorizedArray<value_type> jump_value = fe_eval.get_value(q) - fe_eval_neighbor.get_value(q);
      VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

      VectorizedArray<value_type> normal_gradient_m = fe_eval.get_normal_gradient(q);
      VectorizedArray<value_type> normal_gradient_p = fe_eval_neighbor.get_normal_gradient(q);
      VectorizedArray<value_type> gradient_flux = calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

      fe_eval.submit_normal_gradient(value_flux,q);
      fe_eval_neighbor.submit_normal_gradient(value_flux,q);

      fe_eval.submit_value(-gradient_flux,q);
      fe_eval_neighbor.submit_value(gradient_flux,q); // + sign since n⁺ = -n⁻
    }

    fe_eval.integrate(true,true);
    fe_eval_neighbor.integrate(true,true);
  }

  void do_face_int_integral(FEEvalFace &fe_eval,FEEvalFace &fe_eval_neighbor) const{
    VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                  fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                           * IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);

    fe_eval.evaluate(true,true);

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      // set exterior value to zero
      VectorizedArray<value_type> jump_value = fe_eval.get_value(q);
      VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

      // set exterior value to zero
      VectorizedArray<value_type> normal_gradient_m = fe_eval.get_normal_gradient(q);
      VectorizedArray<value_type> normal_gradient_p = make_vectorized_array<value_type>(0.0);
      VectorizedArray<value_type> gradient_flux = calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

      fe_eval.submit_normal_gradient(value_flux,q);
      fe_eval.submit_value(-gradient_flux,q);
    }

    fe_eval.integrate(true,true);
  }

  void do_face_ext_integral(FEEvalFace &fe_eval,FEEvalFace &fe_eval_neighbor) const {
    VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                  fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                           * IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);

    fe_eval_neighbor.evaluate(true,true);

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      // set value_m to zero
      VectorizedArray<value_type> jump_value = fe_eval_neighbor.get_value(q);
      VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

      // set gradient_m to zero
      VectorizedArray<value_type> normal_gradient_m = make_vectorized_array<value_type>(0.0);
      // minus sign to get the correct normal vector n⁺ = -n⁻
      VectorizedArray<value_type> normal_gradient_p = - fe_eval_neighbor.get_normal_gradient(q);
      VectorizedArray<value_type> gradient_flux = calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

      fe_eval_neighbor.submit_normal_gradient(-value_flux,q); // minus sign since n⁺ = -n⁻
      fe_eval_neighbor.submit_value(-gradient_flux,q);
    }

    fe_eval_neighbor.integrate(true,true);
  }

  void do_boundary_integral(FEEvalFace &fe_eval, OperatorType const &operator_type, 
          types::boundary_id const &boundary_id) const{
    BoundaryType boundary_type = this->ad.get_boundary_type(boundary_id);

    VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter)
                                          * IP::get_penalty_factor<value_type>(fe_degree, this->ad.IP_factor);

    fe_eval.evaluate(true,true);

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      VectorizedArray<value_type> value_m = calculate_interior_value(q,fe_eval,operator_type);
      VectorizedArray<value_type> value_p = calculate_exterior_value(value_m,q,fe_eval,operator_type,boundary_type,boundary_id);
      VectorizedArray<value_type> jump_value = value_m - value_p;
      VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);

      VectorizedArray<value_type> normal_gradient_m = calculate_interior_normal_gradient(q,fe_eval,operator_type);
      VectorizedArray<value_type> normal_gradient_p = calculate_exterior_normal_gradient(normal_gradient_m,q,fe_eval,operator_type,boundary_type,boundary_id);
      VectorizedArray<value_type> gradient_flux = calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);

      fe_eval.submit_normal_gradient(value_flux,q);
      fe_eval.submit_value(-gradient_flux,q);
    }

    fe_eval.integrate(true,true);
  }

  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter;
  double diffusivity;
};
    
}

#endif