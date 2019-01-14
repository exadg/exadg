#ifndef TEST_VECTOR_HELMHOLTZ_OPERATOR
#define TEST_VECTOR_HELMHOLTZ_OPERATOR

#include "../../../../../include/functionalities/evaluate_functions.h"
#include "../../../../../include/incompressible_navier_stokes/user_interface/boundary_descriptor.h"
#include "../../../../../include/incompressible_navier_stokes/user_interface/input_parameters.h"
#include "../../../../../include/operators/interior_penalty_parameter.h"
#include "../../../../../include/operators/operator_base.h"

#include <deal.II/fe/mapping_q.h>

namespace IncNS
{
template<int dim>
struct HelmholtzOperatorDataNew : public OperatorBaseData<dim>
{
  HelmholtzOperatorDataNew()
    // clang-format off
    : OperatorBaseData<dim>(0, 0,
          false, true, false, false, true, false, // cell
          true,  true,        true,  true         // face
    ),
      // clang-format on
      formulation_viscous_term(FormulationViscousTerm::DivergenceFormulation),
      penalty_term_div_formulation(PenaltyTermDivergenceFormulation::Symmetrized),
      IP_formulation(InteriorPenaltyFormulation::SIPG),
      IP_factor(1.0),
      viscosity(1.0)
  {
    this->mapping_update_flags = update_gradients | update_JxW_values | update_quadrature_points;
    this->mapping_update_flags_inner_faces =
      this->mapping_update_flags | update_values | update_normal_vectors;
    this->mapping_update_flags_boundary_faces = this->mapping_update_flags_inner_faces;
  }

  FormulationViscousTerm           formulation_viscous_term;
  PenaltyTermDivergenceFormulation penalty_term_div_formulation;
  InteriorPenaltyFormulation       IP_formulation;
  double                           IP_factor;

  std::shared_ptr<BoundaryDescriptorU<dim>> bc;
  std::shared_ptr<Mapping<dim>>             mapping;

  double viscosity;
};

template<int dim, int degree, typename Number>
class HelmholtzOperatorNew
  : public OperatorBase<dim, degree, Number, HelmholtzOperatorDataNew<dim>, dim>
{
private:
  typedef OperatorBase<dim, degree, Number, HelmholtzOperatorDataNew<dim>, dim> Base;

public:
  static const int                  DIM = dim;
  typedef typename Base::VectorType VectorType;

  typedef VectorizedArray<Number>                 scalar;
  typedef Tensor<1, dim, VectorizedArray<Number>> vector;
  typedef Tensor<2, dim, VectorizedArray<Number>> tensor;

  typedef typename Base::FEEvalCell FEEvalCell;
  typedef typename Base::FEEvalFace FEEvalFace;

  HelmholtzOperatorNew() : const_viscosity(-1.0)
  {
  }

  void
  reinit(MatrixFree<dim, Number> const &       mf_data,
         AffineConstraints<double> const &     constraint_matrix,
         HelmholtzOperatorDataNew<dim> const & operator_data) const
  {
    Base::reinit(mf_data, constraint_matrix, operator_data);

    IP::calculate_penalty_parameter<dim, degree, Number>(array_penalty_parameter,
                                                         *this->data,
                                                         *operator_data.mapping,
                                                         this->operator_data.dof_index);

    const_viscosity = this->operator_data.viscosity;
  }

  void
  reinit(Mapping<dim> const &                  mapping,
         MatrixFree<dim, Number> const &       mf_data,
         AffineConstraints<double> const &     constraint_matrix,
         HelmholtzOperatorDataNew<dim> const & operator_data_in) const
  {
    HelmholtzOperatorDataNew<dim> operator_data = operator_data_in;
    operator_data.mapping                       = std::move(mapping.clone());

    this->reinit(mf_data, constraint_matrix, operator_data);
  }


  void
  reinit_multigrid(
    DoFHandler<dim> const &,
    Mapping<dim> const &,
    void *,
    MGConstrainedDoFs const &,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &,
    unsigned int const)
  {
    AssertThrow(false, ExcMessage("Function should not be accessed!"));
  }

  void
  vmult(VectorType & dst, VectorType const & src) const
  {
    this->apply(dst, src);
  }

  void
  vmult_add(VectorType & dst, VectorType const & src) const
  {
    this->apply_add(dst, src);
  }

  AffineConstraints<double> const &
  get_constraint_matrix() const
  {
    return this->do_get_constraint_matrix();
  }

  MatrixFree<dim, Number> const &
  get_data() const
  {
    return *this->data;
  }

  unsigned int
  get_dof_index() const
  {
    return this->operator_data.dof_index;
  }

  void
  calculate_inverse_diagonal(VectorType & diagonal) const
  {
    this->calculate_diagonal(diagonal);
    invert_diagonal(diagonal);
  }

  void
  apply_inverse_block_diagonal(VectorType & dst, VectorType const & src) const
  {
    AssertThrow(this->operator_data.implement_block_diagonal_preconditioner_matrix_free == false,
                ExcMessage("Not implemented."));

    this->apply_inverse_block_diagonal_matrix_based(dst, src);
  }

  void
  update_block_diagonal_preconditioner() const
  {
    this->do_update_block_diagonal_preconditioner();
  }

  bool
  is_singular() const
  {
    return this->operator_is_singular();
  }

#ifdef DEAL_II_WITH_TRILINOS
  virtual void
  init_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const
  {
    this->do_init_system_matrix(system_matrix);
  }

  virtual void
  calculate_system_matrix(TrilinosWrappers::SparseMatrix & system_matrix) const
  {
    this->do_calculate_system_matrix(system_matrix);
  }
#endif

  PreconditionableOperator<dim, Number> *
  get_new(unsigned int deg) const
  {
    switch(deg)
    {
      case 1:
        return new HelmholtzOperatorNew<dim, 1, Number>();
      case 2:
        return new HelmholtzOperatorNew<dim, 2, Number>();
      case 3:
        return new HelmholtzOperatorNew<dim, 3, Number>();
      default:
        AssertThrow(false, ExcMessage("This degree is not implemented!"));
        return new HelmholtzOperatorNew<dim, degree, Number>();
    }
  }

  bool
  viscosity_is_variable() const
  {
    return viscous_coefficient_cell.n_elements() > 0;
  }

private:
  void
  do_cell_integral(FEEvalCell & fe_eval, unsigned int const cell) const
  {
    AssertThrow(const_viscosity >= 0.0, ExcMessage("Constant viscosity has not been set!"));

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      scalar viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        viscosity = viscous_coefficient_cell[cell][q];

      if(this->operator_data.formulation_viscous_term ==
         FormulationViscousTerm::DivergenceFormulation)
      {
        fe_eval.submit_gradient(viscosity * make_vectorized_array<Number>(2.) *
                                  fe_eval.get_symmetric_gradient(q),
                                q);
      }
      else if(this->operator_data.formulation_viscous_term ==
              FormulationViscousTerm::LaplaceFormulation)
      {
        fe_eval.submit_gradient(viscosity * fe_eval.get_gradient(q), q);
      }
      else
      {
        AssertThrow(this->operator_data.formulation_viscous_term ==
                        FormulationViscousTerm::DivergenceFormulation ||
                      this->operator_data.formulation_viscous_term ==
                        FormulationViscousTerm::LaplaceFormulation,
                    ExcMessage("Specified formulation of viscous term is not implemented."));
      }
    }
  }

  void
  do_face_integral(FEEvalFace & fe_eval_m, FEEvalFace & fe_eval_p, unsigned int const face) const
  {
    scalar penalty_parameter =
      IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor) *
      std::max(fe_eval_m.read_cell_data(array_penalty_parameter),
               fe_eval_p.read_cell_data(array_penalty_parameter));

    for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
    {
      scalar average_viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        average_viscosity = calculate_average_viscosity(face, q);

      vector value_m = fe_eval_m.get_value(q);
      vector value_p = fe_eval_p.get_value(q);
      vector normal  = fe_eval_m.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_m, value_p, normal, average_viscosity);

      vector normal_gradient_m = calculate_normal_gradient(q, fe_eval_m);
      vector normal_gradient_p = calculate_normal_gradient(q, fe_eval_p);

      vector gradient_flux = calculate_gradient_flux(normal_gradient_m,
                                                     normal_gradient_p,
                                                     value_m,
                                                     value_p,
                                                     normal,
                                                     average_viscosity,
                                                     penalty_parameter);

      fe_eval_m.submit_gradient(value_flux, q);
      fe_eval_p.submit_gradient(value_flux, q);

      fe_eval_m.submit_value(-gradient_flux, q);
      fe_eval_p.submit_value(gradient_flux, q); // + sign since n⁺ = -n⁻
    }
  }

  void
  do_face_int_integral(FEEvalFace &       fe_eval_m,
                       FEEvalFace &       fe_eval_p,
                       unsigned int const face) const
  {
    scalar penalty_parameter =
      IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor) *
      std::max(fe_eval_m.read_cell_data(array_penalty_parameter),
               fe_eval_p.read_cell_data(array_penalty_parameter));

    for(unsigned int q = 0; q < fe_eval_m.n_q_points; ++q)
    {
      scalar average_viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        average_viscosity = calculate_average_viscosity(face, q);

      // set exterior values to zero
      vector value_m = fe_eval_m.get_value(q);
      vector value_p;

      vector normal_m = fe_eval_m.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_m, value_p, normal_m, average_viscosity);

      vector normal_gradient_m = calculate_normal_gradient(q, fe_eval_m);
      vector normal_gradient_p; // set exterior gradient to zero

      vector gradient_flux = calculate_gradient_flux(normal_gradient_m,
                                                     normal_gradient_p,
                                                     value_m,
                                                     value_p,
                                                     normal_m,
                                                     average_viscosity,
                                                     penalty_parameter);

      fe_eval_m.submit_gradient(value_flux, q);
      fe_eval_m.submit_value(-gradient_flux, q);
    }
  }


  void
  do_face_ext_integral(FEEvalFace &       fe_eval_m,
                       FEEvalFace &       fe_eval_p,
                       unsigned int const face) const
  {
    scalar penalty_parameter =
      IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor) *
      std::max(fe_eval_m.read_cell_data(array_penalty_parameter),
               fe_eval_p.read_cell_data(array_penalty_parameter));

    for(unsigned int q = 0; q < fe_eval_p.n_q_points; ++q)
    {
      scalar average_viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        average_viscosity = calculate_average_viscosity(face, q);

      // set exterior values to zero
      vector value_m;
      vector value_p = fe_eval_p.get_value(q);
      // multiply by -1.0 to get the correct normal vector !!!
      vector normal_p = -fe_eval_p.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_p, value_m, normal_p, average_viscosity);

      // set exterior gradient to zero
      vector normal_gradient_m;
      // multiply by -1.0 since normal vector n⁺ = -n⁻ !!!
      vector normal_gradient_p = -calculate_normal_gradient(q, fe_eval_p);

      vector gradient_flux = calculate_gradient_flux(normal_gradient_p,
                                                     normal_gradient_m,
                                                     value_p,
                                                     value_m,
                                                     normal_p,
                                                     average_viscosity,
                                                     penalty_parameter);

      fe_eval_p.submit_gradient(value_flux, q);
      fe_eval_p.submit_value(-gradient_flux, q);
    }
  }

  void
  do_boundary_integral(FEEvalFace &               fe_eval,
                       OperatorType const &       operator_type,
                       types::boundary_id const & boundary_id,
                       unsigned int const         face) const
  {
    BoundaryTypeU boundary_type = this->operator_data.bc->get_boundary_type(boundary_id);

    scalar penalty_parameter =
      IP::get_penalty_factor<Number>(degree, this->operator_data.IP_factor) *
      fe_eval.read_cell_data(array_penalty_parameter);

    for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    {
      scalar viscosity = make_vectorized_array<Number>(const_viscosity);
      if(viscosity_is_variable())
        viscosity = viscous_coefficient_face[face][q];

      vector value_m = calculate_interior_value(q, fe_eval, operator_type);
      vector value_p =
        calculate_exterior_value(value_m, q, fe_eval, operator_type, boundary_type, boundary_id);

      vector normal = fe_eval.get_normal_vector(q);

      tensor value_flux = calculate_value_flux(value_m, value_p, normal, viscosity);

      vector normal_gradient_m = calculate_interior_normal_gradient(q, fe_eval, operator_type);
      vector normal_gradient_p = calculate_exterior_normal_gradient(
        normal_gradient_m, q, fe_eval, operator_type, boundary_type, boundary_id);

      vector gradient_flux = calculate_gradient_flux(normal_gradient_m,
                                                     normal_gradient_p,
                                                     value_m,
                                                     value_p,
                                                     normal,
                                                     viscosity,
                                                     penalty_parameter);

      fe_eval.submit_gradient(value_flux, q);
      fe_eval.submit_value(-gradient_flux, q);
    }
  }

  /*
   *  This function calculates the average viscosity for interior faces.
   */
  inline DEAL_II_ALWAYS_INLINE //
    scalar
    calculate_average_viscosity(unsigned int const face, unsigned int const q) const
  {
    scalar average_viscosity = make_vectorized_array<Number>(0.0);

    // harmonic mean (harmonic weighting according to Schott and Rasthofer et al. (2015))
    average_viscosity =
      2.0 * viscous_coefficient_face[face][q] * viscous_coefficient_face_neighbor[face][q] /
      (viscous_coefficient_face[face][q] + viscous_coefficient_face_neighbor[face][q]);

    // arithmetic mean
    //    average_viscosity = 0.5 * (viscous_coefficient_face[face][q] +
    //    viscous_coefficient_face_neighbor[face][q]);

    // maximum value
    //    average_viscosity = std::max(viscous_coefficient_face[face][q],
    //    viscous_coefficient_face_neighbor[face][q]);

    return average_viscosity;
  }


  /*
   *  Calculation of "value_flux".
   */
  inline DEAL_II_ALWAYS_INLINE //
    tensor
    calculate_value_flux(vector const & value_m,
                         vector const & value_p,
                         vector const & normal,
                         scalar const & viscosity) const
  {
    tensor value_flux;

    vector jump_value  = value_m - value_p;
    tensor jump_tensor = outer_product(jump_value, normal);

    if(this->operator_data.formulation_viscous_term == FormulationViscousTerm::LaplaceFormulation)
    {
      if(this->operator_data.IP_formulation == InteriorPenaltyFormulation::NIPG)
      {
        value_flux = 0.5 * viscosity * jump_tensor;
      }
      else if(this->operator_data.IP_formulation == InteriorPenaltyFormulation::SIPG)
      {
        value_flux = -0.5 * viscosity * jump_tensor;
      }
      else
      {
        AssertThrow(false,
                    ExcMessage("Specified interior penalty formulation is not implemented."));
      }
    }
    else if(this->operator_data.formulation_viscous_term ==
            FormulationViscousTerm::DivergenceFormulation)
    {
      if(this->operator_data.IP_formulation == InteriorPenaltyFormulation::NIPG)
      {
        value_flux = 0.5 * viscosity * (jump_tensor + transpose(jump_tensor));
      }
      else if(this->operator_data.IP_formulation == InteriorPenaltyFormulation::SIPG)
      {
        value_flux = -0.5 * viscosity * (jump_tensor + transpose(jump_tensor));
      }
      else
      {
        AssertThrow(false,
                    ExcMessage("Specified interior penalty formulation is not implemented."));
      }
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    return value_flux;
  }

  // clang-format off
  /*
   *  The following two functions calculate the interior/exterior value for boundary faces depending on the
   *  operator type, the type of the boundary face and the given boundary conditions.
   *
   *                            +-------------------------+--------------------+------------------------------+
   *                            | Dirichlet boundaries    | Neumann boundaries | symmetry boundaries          |
   *  +-------------------------+-------------------------+--------------------+------------------------------+
   *  | full operator           | u⁺ = -u⁻ + 2g           | u⁺ = u⁻            | u⁺ = u⁻ - 2 (u⁻*n)n          |
   *  +-------------------------+-------------------------+--------------------+------------------------------+
   *  | homogeneous operator    | u⁺ = -u⁻                | u⁺ = u⁻            | u⁺ = u⁻ - 2 (u⁻*n)n          |
   *  +-------------------------+-------------------------+--------------------+------------------------------+
   *  | inhomogeneous operator  | u⁺ = -u⁻ + 2g , u⁻ = 0  | u⁺ = u⁻ , u⁻ = 0   | u⁺ = u⁻ - 2 (u⁻*n)n , u⁻ = 0 |
   *  +-------------------------+-------------------------+--------------------+------------------------------+
   *
   */
  // clang-format on
  template<typename FEEvaluationVelocity>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_interior_value(unsigned int const           q,
                             FEEvaluationVelocity const & fe_eval_velocity,
                             OperatorType const &         operator_type) const
  {
    // element e⁻
    vector value_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      value_m = fe_eval_velocity.get_value(q);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, value_m is already initialized with zeros
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    return value_m;
  }

  template<typename FEEvaluationVelocity>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_value(vector const &               value_m,
                             unsigned int const           q,
                             FEEvaluationVelocity const & fe_eval_velocity,
                             OperatorType const &         operator_type,
                             BoundaryTypeU const &        boundary_type,
                             types::boundary_id const     boundary_id = types::boundary_id()) const
  {
    // element e⁺
    vector value_p;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
          this->operator_data.bc->dirichlet_bc.find(boundary_id);
        Point<dim, scalar> q_points = fe_eval_velocity.quadrature_point(q);

        vector g = evaluate_vectorial_function(it->second, q_points, this->eval_time);

        value_p = -value_m + make_vectorized_array<Number>(2.0) * g;
      }
      else if(operator_type == OperatorType::homogeneous)
      {
        value_p = -value_m;
      }
      else
      {
        AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
      }
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      value_p = value_m;
    }
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      vector normal_m = fe_eval_velocity.get_normal_vector(q);

      value_p = value_m - 2.0 * (value_m * normal_m) * normal_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return value_p;
  }

  /*
   *  This function calculates the gradient in normal direction on element e
   *  depending on the formulation of the viscous term.
   */
  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_normal_gradient(unsigned int const q, FEEvaluation & fe_eval) const
  {
    tensor gradient;

    if(this->operator_data.formulation_viscous_term ==
       FormulationViscousTerm::DivergenceFormulation)
    {
      /*
       * F = 2 * nu * symmetric_gradient
       *   = 2.0 * nu * 1/2 (grad(u) + grad(u)^T)
       */
      gradient = make_vectorized_array<Number>(2.0) * fe_eval.get_symmetric_gradient(q);
    }
    else if(this->operator_data.formulation_viscous_term ==
            FormulationViscousTerm::LaplaceFormulation)
    {
      /*
       *  F = nu * grad(u)
       */
      gradient = fe_eval.get_gradient(q);
    }
    else
    {
      AssertThrow(this->operator_data.formulation_viscous_term ==
                      FormulationViscousTerm::DivergenceFormulation ||
                    this->operator_data.formulation_viscous_term ==
                      FormulationViscousTerm::LaplaceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    vector normal_gradient = gradient * fe_eval.get_normal_vector(q);

    return normal_gradient;
  }

  /*
   *  Calculation of gradient flux. Strictly speaking, this value is not a numerical flux since
   *  the flux is multiplied by the normal vector, i.e., "gradient_flux" = numerical_flux * normal,
   *  where normal denotes the normal vector of element e⁻.
   */
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_gradient_flux(vector const & normal_gradient_m,
                            vector const & normal_gradient_p,
                            vector const & value_m,
                            vector const & value_p,
                            vector const & normal,
                            scalar const & viscosity,
                            scalar const & penalty_parameter) const
  {
    vector gradient_flux;

    vector jump_value              = value_m - value_p;
    vector average_normal_gradient = 0.5 * (normal_gradient_m + normal_gradient_p);

    if(this->operator_data.formulation_viscous_term ==
       FormulationViscousTerm::DivergenceFormulation)
    {
      if(this->operator_data.penalty_term_div_formulation ==
         PenaltyTermDivergenceFormulation::Symmetrized)
      {
        gradient_flux =
          viscosity * average_normal_gradient -
          viscosity * penalty_parameter * (jump_value + (jump_value * normal) * normal);
      }
      else if(this->operator_data.penalty_term_div_formulation ==
              PenaltyTermDivergenceFormulation::NotSymmetrized)
      {
        gradient_flux =
          viscosity * average_normal_gradient - viscosity * penalty_parameter * jump_value;
      }
      else
      {
        AssertThrow(this->operator_data.penalty_term_div_formulation ==
                        PenaltyTermDivergenceFormulation::Symmetrized ||
                      this->operator_data.penalty_term_div_formulation ==
                        PenaltyTermDivergenceFormulation::NotSymmetrized,
                    ExcMessage("Specified formulation of viscous term is not implemented."));
      }
    }
    else if(this->operator_data.formulation_viscous_term ==
            FormulationViscousTerm::LaplaceFormulation)
    {
      gradient_flux =
        viscosity * average_normal_gradient - viscosity * penalty_parameter * jump_value;
    }
    else
    {
      AssertThrow(this->operator_data.formulation_viscous_term ==
                      FormulationViscousTerm::DivergenceFormulation ||
                    this->operator_data.formulation_viscous_term ==
                      FormulationViscousTerm::LaplaceFormulation,
                  ExcMessage("Specified formulation of viscous term is not implemented."));
    }

    return gradient_flux;
  }

  // clang-format off
  /*
   *  These two functions calculates the velocity gradient in normal
   *  direction depending on the operator type, the type of the boundary face
   *  and the given boundary conditions.
   *
   *  Divergence formulation: F(u) = nu * ( grad(u) + grad(u)^T )
   *  Laplace formulation: F(u) = nu * grad(u)
   *
   *                            +---------------------------------+---------------------------------------+----------------------------------------------------+
   *                            | Dirichlet boundaries            | Neumann boundaries                    | symmetry boundaries                                |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | full operator           | F(u⁺)*n = F(u⁻)*n               | F(u⁺)*n = -F(u⁻)*n + 2h               | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n              |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | homogeneous operator    | F(u⁺)*n = F(u⁻)*n               | F(u⁺)*n = -F(u⁻)*n                    | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n              |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | inhomogeneous operator  | F(u⁺)*n = F(u⁻)*n, F(u⁻)*n = 0  | F(u⁺)*n = -F(u⁻)*n + 2h , F(u⁻)*n = 0 | F(u⁺)*n = -F(u⁻)*n + 2*[(F(u⁻)*n)*n]n, F(u⁻)*n = 0 |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *
   *                            +---------------------------------+---------------------------------------+----------------------------------------------------+
   *                            | Dirichlet boundaries            | Neumann boundaries                    | symmetry boundaries                                |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | full operator           | {{F(u)}}*n = F(u⁻)*n            | {{F(u)}}*n = h                        | {{F(u)}}*n = 2*[(F(u⁻)*n)*n]n                      |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | homogeneous operator    | {{F(u)}}*n = F(u⁻)*n            | {{F(u)}}*n = 0                        | {{F(u)}}*n = 2*[(F(u⁻)*n)*n]n                      |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   *  | inhomogeneous operator  | {{F(u)}}*n = 0                  | {{F(u)}}*n = h                        | {{F(u)}}*n = 0                                     |
   *  +-------------------------+---------------------------------+---------------------------------------+----------------------------------------------------+
   */
  // clang-format on
  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_interior_normal_gradient(unsigned int const   q,
                                       FEEvaluation const & fe_eval,
                                       OperatorType const & operator_type) const
  {
    vector normal_gradient_m;

    if(operator_type == OperatorType::full || operator_type == OperatorType::homogeneous)
    {
      normal_gradient_m = calculate_normal_gradient(q, fe_eval);
    }
    else if(operator_type == OperatorType::inhomogeneous)
    {
      // do nothing, normal_gradient_m is already intialized with 0
    }
    else
    {
      AssertThrow(false, ExcMessage("Specified OperatorType is not implemented!"));
    }

    return normal_gradient_m;
  }

  template<typename FEEvaluation>
  inline DEAL_II_ALWAYS_INLINE //
    vector
    calculate_exterior_normal_gradient(
      vector const &           normal_gradient_m,
      unsigned int const       q,
      FEEvaluation const &     fe_eval,
      OperatorType const &     operator_type,
      BoundaryTypeU const &    boundary_type,
      types::boundary_id const boundary_id = types::boundary_id()) const
  {
    vector normal_gradient_p;

    if(boundary_type == BoundaryTypeU::Dirichlet)
    {
      normal_gradient_p = normal_gradient_m;
    }
    else if(boundary_type == BoundaryTypeU::Neumann)
    {
      if(operator_type == OperatorType::full || operator_type == OperatorType::inhomogeneous)
      {
        typename std::map<types::boundary_id, std::shared_ptr<Function<dim>>>::iterator it =
          this->operator_data.bc->neumann_bc.find(boundary_id);
        Point<dim, scalar> q_points = fe_eval.quadrature_point(q);

        vector h = evaluate_vectorial_function(it->second, q_points, this->eval_time);

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
    else if(boundary_type == BoundaryTypeU::Symmetry)
    {
      vector normal_m   = fe_eval.get_normal_vector(q);
      normal_gradient_p = -normal_gradient_m + 2.0 * (normal_gradient_m * normal_m) * normal_m;
    }
    else
    {
      AssertThrow(false, ExcMessage("Boundary type of face is invalid or not implemented."));
    }

    return normal_gradient_p;
  }

public:
  // penalty parameter
  mutable AlignedVector<scalar> array_penalty_parameter;

  // viscosity
  mutable Number const_viscosity;

  // variable viscosity
  mutable Table<2, scalar> viscous_coefficient_cell;
  mutable Table<2, scalar> viscous_coefficient_face;
  mutable Table<2, scalar> viscous_coefficient_face_neighbor;
};
} // namespace IncNS

#endif
