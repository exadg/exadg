#ifndef CONV_DIFF_DIFFUSIVE_OPERATOR
#define CONV_DIFF_DIFFUSIVE_OPERATOR

#include "../../../operators/operation_base.h"

namespace ConvDiff
{
template<int dim>
struct DiffusiveOperatorData
{
  DiffusiveOperatorData ()
    :
    dof_index(0),
    quad_index(0),
    IP_factor(1.0),
    diffusivity(1.0)
  {}

  unsigned int dof_index;
  unsigned int quad_index;

  double IP_factor;

  std::shared_ptr<ConvDiff::BoundaryDescriptor<dim> > bc;

  double diffusivity;
};



template <int dim, int fe_degree, typename value_type>
class DiffusiveOperator
{
public:
  typedef DiffusiveOperator<dim, fe_degree, value_type> This;

  DiffusiveOperator()
    :
    data(nullptr),
    diffusivity(-1.0)
  {}

  void initialize(Mapping<dim> const               &mapping,
                  MatrixFree<dim,value_type> const &mf_data,
                  DiffusiveOperatorData<dim> const &operator_data_in)
  {
    this->data = &mf_data;
    this->operator_data = operator_data_in;

    IP::calculate_penalty_parameter<dim, fe_degree, value_type>(array_penalty_parameter,
                                                                *data,
                                                                mapping,
                                                                operator_data.dof_index);

    diffusivity = operator_data.diffusivity;
  }

  // apply matrix vector multiplication
  void apply (parallel::distributed::Vector<value_type>       &dst,
              parallel::distributed::Vector<value_type> const &src) const
  {
    dst = 0;
    apply_add(dst,src);
  }

  void apply_add (parallel::distributed::Vector<value_type>       &dst,
                  parallel::distributed::Vector<value_type> const &src) const
  {
    AssertThrow(diffusivity > 0.0, ExcMessage("Diffusivity has not been set!"));

    data->loop(&This::cell_loop,
               &This::face_loop,
               &This::boundary_face_loop_hom_operator, this, dst, src);
  }

  // apply "block Jacobi" matrix vector multiplication
  void apply_block_Jacobi (parallel::distributed::Vector<value_type>       &dst,
                           parallel::distributed::Vector<value_type> const &src) const
  {
    dst = 0;
    apply_block_jacobi_add(dst,src);
  }

  void apply_block_jacobi_add (parallel::distributed::Vector<value_type>       &dst,
                               parallel::distributed::Vector<value_type> const &src) const
  {
    AssertThrow(diffusivity > 0.0, ExcMessage("Diffusivity has not been set!"));

    data->loop(&This::cell_loop,
               &This::face_loop_block_jacobi,
               &This::boundary_face_loop_hom_operator, this, dst, src);
  }

  void add_block_jacobi_matrices(std::vector<LAPACKFullMatrix<value_type> > &matrices) const
  {
    parallel::distributed::Vector<value_type> src;

    data->loop(&This::cell_loop_calculate_block_jacobi_matrices,
               &This::face_loop_calculate_block_jacobi_matrices,
               &This::boundary_face_loop_calculate_block_jacobi_matrices, this, matrices, src);
  }

  void rhs (parallel::distributed::Vector<value_type> &dst,
            value_type const                          evaluation_time) const
  {
    dst = 0;
    rhs_add(dst,evaluation_time);
  }

  void rhs_add (parallel::distributed::Vector<value_type> &dst,
                value_type const                          evaluation_time) const
  {
    this->eval_time = evaluation_time;

    parallel::distributed::Vector<value_type> src;
    parallel::distributed::Vector<value_type> tmp(dst);

    data->loop(&This::cell_loop_inhom_operator,
               &This::face_loop_inhom_operator,
               &This::boundary_face_loop_inhom_operator, this, tmp, src);

    // multiply by -1.0 since the boundary face integrals have to be shifted to the right hand side
    dst.add(-1.0,tmp);
  }

  void evaluate (parallel::distributed::Vector<value_type>       &dst,
                 parallel::distributed::Vector<value_type> const &src,
                 value_type const                                evaluation_time) const
  {
    dst = 0;
    evaluate_add(dst,src,evaluation_time);
  }

  void evaluate_add (parallel::distributed::Vector<value_type>       &dst,
                     parallel::distributed::Vector<value_type> const &src,
                     value_type const                                evaluation_time) const
  {
    this->eval_time = evaluation_time;

    data->loop(&This::cell_loop,
               &This::face_loop,
               &This::boundary_face_loop_full_operator, this, dst, src);
  }

  void calculate_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    diagonal = 0;

    add_diagonal(diagonal);
  }

  void add_diagonal(parallel::distributed::Vector<value_type> &diagonal) const
  {
    parallel::distributed::Vector<value_type>  src_dummy(diagonal);

    data->loop(&This::cell_loop_diagonal,
               &This::face_loop_diagonal,
               &This::boundary_face_loop_diagonal, this, diagonal, src_dummy);
  }

  DiffusiveOperatorData<dim> const & get_operator_data() const
  {
    return operator_data;
  }

private:
  /*
   *  Calculate boundary type.
   */
  inline DEAL_II_ALWAYS_INLINE ConvDiff::BoundaryType
  get_boundary_type(types::boundary_id const &boundary_id) const
  {
    BoundaryType boundary_type = BoundaryType::undefined;

    if(operator_data.bc->dirichlet_bc.find(boundary_id) != operator_data.bc->dirichlet_bc.end())
      boundary_type = BoundaryType::dirichlet;
    else if(operator_data.bc->neumann_bc.find(boundary_id) != operator_data.bc->neumann_bc.end())
      boundary_type = BoundaryType::neumann;

    AssertThrow(boundary_type != BoundaryType::undefined,
        ExcMessage("Boundary type of face is invalid or not implemented."));

    return boundary_type;
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
        it = operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        evaluate_scalar_function(g, it->second, q_points, eval_time);

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
        it = operator_data.bc->neumann_bc.find(boundary_id);
        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        evaluate_scalar_function(h, it->second, q_points, eval_time);

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

  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const
  {
    fe_eval.evaluate (false,true,false);

    for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
    {
      fe_eval.submit_gradient (make_vectorized_array<value_type>(diffusivity)*fe_eval.get_gradient(q), q);
    }
    fe_eval.integrate (false,true);
  }

  template<typename FEEvaluation>
  inline void do_interior_face_integral(FEEvaluation &fe_eval,
                                        FEEvaluation &fe_eval_neighbor) const
  {
    VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                  fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                           * IP::get_penalty_factor<value_type>(fe_degree, operator_data.IP_factor);

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

  template<typename FEEvaluation>
  inline void do_interior_face_integral_block_jacobi_interior(FEEvaluation &fe_eval,
                                                              FEEvaluation &fe_eval_neighbor) const
  {
    VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                  fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                           * IP::get_penalty_factor<value_type>(fe_degree, operator_data.IP_factor);

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

  /*
   *  When performing face integrals in a cell-based manner, i.e., looping over all
   *  faces of a cell instead of all interior or boundary faces, this function will
   *  not be necessary anymore.
   */
  template<typename FEEvaluation>
  inline void do_interior_face_integral_block_jacobi_exterior(FEEvaluation &fe_eval,
                                                              FEEvaluation &fe_eval_neighbor) const
  {
    VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
                                                  fe_eval_neighbor.read_cell_data(array_penalty_parameter))
                                           * IP::get_penalty_factor<value_type>(fe_degree, operator_data.IP_factor);

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

  template<typename FEEvaluation>
  inline void do_boundary_face_integral(FEEvaluation             &fe_eval,
                                        OperatorType const       &operator_type,
                                        types::boundary_id const &boundary_id) const
  {
    BoundaryType boundary_type = get_boundary_type(boundary_id);

    VectorizedArray<value_type> tau_IP = fe_eval.read_cell_data(array_penalty_parameter)
                                          * IP::get_penalty_factor<value_type>(fe_degree, operator_data.IP_factor);

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


  /*
   *  Calculate cell integrals.
   */
  void cell_loop (MatrixFree<dim,value_type> const                 &data,
                  parallel::distributed::Vector<value_type>        &dst,
                  parallel::distributed::Vector<value_type> const  &src,
                  std::pair<unsigned int,unsigned int> const       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);
      fe_eval.read_dof_values(src);

      do_cell_integral(fe_eval);

      fe_eval.distribute_local_to_global (dst);
    }
  }

  /*
   *  Calculate interior face integrals.
   */
  void face_loop (MatrixFree<dim,value_type> const                &data,
                  parallel::distributed::Vector<value_type>       &dst,
                  parallel::distributed::Vector<value_type> const &src,
                  std::pair<unsigned int,unsigned int> const      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval_neighbor.read_dof_values(src);

      do_interior_face_integral(fe_eval, fe_eval_neighbor);

      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  /*
   *  Calculate boundary face integrals for homogeneous operator.
   */
  void boundary_face_loop_hom_operator (MatrixFree<dim,value_type> const                &data,
                                        parallel::distributed::Vector<value_type>       &dst,
                                        parallel::distributed::Vector<value_type> const &src,
                                        std::pair<unsigned int,unsigned int> const      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);

      do_boundary_face_integral(fe_eval, OperatorType::homogeneous, boundary_id);

      fe_eval.distribute_local_to_global(dst);
    }
  }

  /*
   *  Calculate boundary face integrals for full operator (homogeneous + inhomogeneous parts).
   */
  void boundary_face_loop_full_operator (MatrixFree<dim,value_type> const                &data,
                                         parallel::distributed::Vector<value_type>       &dst,
                                         parallel::distributed::Vector<value_type> const &src,
                                         std::pair<unsigned int,unsigned int> const      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);

      do_boundary_face_integral(fe_eval, OperatorType::full, boundary_id);

      fe_eval.distribute_local_to_global(dst);
    }
  }


  /*
   *  Evaluate inhomogeneous operator. Note that these integrals are multiplied by
   *  a factor of -1.0 since these integrals apppear on the right-hand side of the equations.
   */
  void cell_loop_inhom_operator (MatrixFree<dim,value_type> const                 &,
                                 parallel::distributed::Vector<value_type>        &,
                                 parallel::distributed::Vector<value_type> const  &,
                                 std::pair<unsigned int,unsigned int> const       &) const
  {}

  void face_loop_inhom_operator (MatrixFree<dim,value_type> const                &,
                                 parallel::distributed::Vector<value_type>       &,
                                 parallel::distributed::Vector<value_type> const &,
                                 std::pair<unsigned int,unsigned int> const      &) const
  {}

  void boundary_face_loop_inhom_operator (MatrixFree<dim,value_type> const                &data,
                                          parallel::distributed::Vector<value_type>       &dst,
                                          parallel::distributed::Vector<value_type> const &/*src*/,
                                          std::pair<unsigned int,unsigned int> const      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      fe_eval.reinit (face);

      do_boundary_face_integral(fe_eval, OperatorType::inhomogeneous, boundary_id);

      fe_eval.distribute_local_to_global(dst);
    }
  }


  /*
   *  Calculation of diagonal: cell integrals
   */
  void cell_loop_diagonal (MatrixFree<dim,value_type> const                 &data,
                           parallel::distributed::Vector<value_type>        &dst,
                           parallel::distributed::Vector<value_type> const  &,
                           std::pair<unsigned int,unsigned int> const       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global (dst);
    }
  }

  void face_loop_diagonal (MatrixFree<dim,value_type> const                &data,
                           parallel::distributed::Vector<value_type>       &dst,
                           parallel::distributed::Vector<value_type> const &,
                           std::pair<unsigned int,unsigned int> const      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);


    // Perform face intergrals for element e⁻.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_interior_face_integral_block_jacobi_interior(fe_eval, fe_eval_neighbor);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }



    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // Perform face intergrals for element e⁺.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_interior_face_integral_block_jacobi_exterior(fe_eval, fe_eval_neighbor);

        local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
        fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  // TODO: This function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element.
  void boundary_face_loop_diagonal (MatrixFree<dim,value_type> const                &data,
                                    parallel::distributed::Vector<value_type>       &dst,
                                    parallel::distributed::Vector<value_type> const &/*src*/,
                                    std::pair<unsigned int,unsigned int> const      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      fe_eval.reinit (face);

      VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_boundary_face_integral(fe_eval, OperatorType::homogeneous, boundary_id);

        local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];

      fe_eval.distribute_local_to_global(dst);
    }
  }

  // TODO
//  void cell_loop_diagonal_new (MatrixFree<dim,value_type> const                &data,
//                               parallel::distributed::Vector<value_type>       &dst,
//                               parallel::distributed::Vector<value_type> const &,
//                               std::pair<unsigned int,unsigned int> const      &cell_range) const
//  {
//    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index); // ??
//
//    // Loop over all cells
//    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
//    {
//      // local diagonal vector (set to zero in initialization!)
//      VectorizedArray<value_type> local_diagonal_vector[fe_eval.static_dofs_per_cell];
//      for(unsigned int i=0; i<fe_eval.static_dofs_per_cell; ++i)
//        local_diagonal_vector[i] = 0.0;
//
//
//      // TODO compute contribution of cell integral
//
//
//
//      // Loop over all faces of current cell
//      for(unsigned int face=0; face<GeometryInfo<dim>::faces_per_cell; ++face)
//      {
//        fe_eval.reinit(cell, face);
//
//        std::array<types::boundary_id,VectorizedArray<value_type>::n_array_elements>
//          boundary_ids = data.get_faces_by_cells_boundary_id(cell, face);
//
//        // zero for interior faces (set value_p to zero for interior faces)!
//        VectorizedArray<value_type> map_int_to_ext_values = make_vectorized_array<value_type>(0.0);
//        // zero for interior faces (set normal_gradient_p to zero for interior faces)!
//        VectorizedArray<value_type> map_int_to_ext_gradients = make_vectorized_array<value_type>(0.0);
//
//        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
//        {
//          if(operator_data.bc->dirichlet_bc.find(boundary_ids[v]) != operator_data.bc->dirichlet_bc.end())
//          {
//            map_int_to_ext_values[v] = -1.0;
//            map_int_to_ext_gradients[v] = 1.0;
//          }
//          else if(operator_data.bc->neumann_bc.find(boundary_ids[v]) != operator_data.bc->neumann_bc.end())
//          {
//            map_int_to_ext_values[v] = 1.0;
//            map_int_to_ext_gradients[v] = -1.0;
//          }
//          else
//          {
//
//          }
//        }
//
//        // TODO
//        VectorizedArray<value_type> tau_IP = std::max(fe_eval.read_cell_data(array_penalty_parameter),
//                                                      fe_eval_neighbor.read_cell_data(array_penalty_parameter))
//                                                * get_penalty_factor();
//
//        for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
//        {
//          // set dof value j of element- to 1 and all other dof values of element- to zero
//          for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
//            fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
//          fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);
//
//          fe_eval.evaluate(true,true);
//
//          for(unsigned int q=0;q<fe_eval.n_q_points;++q)
//          {
//            VectorizedArray<value_type> value_m = fe_eval.get_value(q);
//            VectorizedArray<value_type> value_p = map_int_to_ext_values * value_m;
//
//            VectorizedArray<value_type> jump_value = value_m - value_p;
//            VectorizedArray<value_type> value_flux = calculate_value_flux(jump_value);
//
//            // set exterior value to zero
//            VectorizedArray<value_type> normal_gradient_m = fe_eval.get_normal_gradient(q);
//            VectorizedArray<value_type> normal_gradient_p = map_int_to_ext_gradients * normal_gradient_m;
//            VectorizedArray<value_type> gradient_flux = calculate_gradient_flux(normal_gradient_m, normal_gradient_p, jump_value, tau_IP);
//
//            fe_eval.submit_normal_gradient(value_flux,q);
//            fe_eval.submit_value(-gradient_flux,q);
//          }
//          fe_eval.integrate(true,true);
//
//          local_diagonal_vector[j] += fe_eval.begin_dof_values()[j];
//        }
//      }
//      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
//        fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
//
//      fe_eval.set_dof_values(dst);
//    }
//  }

  /*
   *  Block-Jacobi operator:
   *
   *  face integrals for block Jacobi, use homogeneous operator for cell and boundary face integrals
   *
   *  This function is only needed for testing, i.e., to make sure that the block Jacobi matrices
   *  are calculated correctly.
   */
  void face_loop_block_jacobi (MatrixFree<dim,value_type> const                &data,
                               parallel::distributed::Vector<value_type>       &dst,
                               parallel::distributed::Vector<value_type> const &src,
                               std::pair<unsigned int,unsigned int> const      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    // perform face integral for element e⁻
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);
      fe_eval.read_dof_values(src);

      do_interior_face_integral_block_jacobi_interior(fe_eval,fe_eval_neighbor);

      fe_eval.distribute_local_to_global(dst);
    }

    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // perform face integral for element e⁺
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);
      fe_eval_neighbor.read_dof_values(src);

      do_interior_face_integral_block_jacobi_exterior(fe_eval,fe_eval_neighbor);

      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  void cell_loop_calculate_block_jacobi_matrices (MatrixFree<dim,value_type> const                 &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >       &matrices,
                                                  parallel::distributed::Vector<value_type> const  &,
                                                  std::pair<unsigned int,unsigned int> const       &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,
                                                                 operator_data.dof_index,
                                                                 operator_data.quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      fe_eval.reinit(cell);

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_cell_integral(fe_eval);

        for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
            matrices[cell*VectorizedArray<value_type>::n_array_elements+v](i,j) += fe_eval.begin_dof_values()[i][v];
      }
    }
  }

  /*
   *  Calculation of Block-Jacobi matrices
   */
  void face_loop_calculate_block_jacobi_matrices (MatrixFree<dim,value_type> const                &data,
                                                  std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                  parallel::distributed::Vector<value_type> const &,
                                                  std::pair<unsigned int,unsigned int> const      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,operator_data.dof_index,operator_data.quad_index);

    // Perform face intergrals for element e⁻.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_interior_face_integral_block_jacobi_interior(fe_eval,fe_eval_neighbor);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }



    // TODO: This has to be removed as soon as the new infrastructure is used that
    // allows to perform face integrals over all faces of the current element.
    // Perform face intergrals for element e⁺.
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
      {
        // set dof value j of element+ to 1 and all other dof values of element+ to zero
        for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
          fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_interior_face_integral_block_jacobi_exterior(fe_eval,fe_eval_neighbor);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_exterior[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval_neighbor.begin_dof_values()[i][v];
        }
      }
    }
  }

  // TODO: This function has to be removed as soon as the new infrastructure is used that
  // allows to perform face integrals over all faces of the current element.
  void boundary_face_loop_calculate_block_jacobi_matrices (MatrixFree<dim,value_type> const                &data,
                                                           std::vector<LAPACKFullMatrix<value_type> >      &matrices,
                                                           parallel::distributed::Vector<value_type> const &,
                                                           std::pair<unsigned int,unsigned int> const      &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,operator_data.dof_index,operator_data.quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      types::boundary_id boundary_id = data.get_boundary_id(face);

      fe_eval.reinit (face);

      for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      {
        // set dof value j of element- to 1 and all other dof values of element- to zero
        for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
          fe_eval.begin_dof_values()[i] = make_vectorized_array<value_type>(0.);
        fe_eval.begin_dof_values()[j] = make_vectorized_array<value_type>(1.);

        do_boundary_face_integral(fe_eval, OperatorType::homogeneous, boundary_id);

        for (unsigned int v=0; v<VectorizedArray<value_type>::n_array_elements; ++v)
        {
          const unsigned int cell_number = data.get_face_info(face).cells_interior[v];
          if (cell_number != numbers::invalid_unsigned_int)
            for(unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
              matrices[cell_number](i,j) += fe_eval.begin_dof_values()[i][v];
        }
      }
    }
  }

  MatrixFree<dim,value_type> const * data;
  DiffusiveOperatorData<dim> operator_data;
  AlignedVector<VectorizedArray<value_type> > array_penalty_parameter;
  double diffusivity;
  mutable value_type eval_time;
};
    
}

#endif