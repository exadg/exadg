/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_


#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/momentum_operator.h"

namespace IncNS
{
/*
 * Multigrid preconditioner for momentum operator of the incompressible Navier-Stokes equations.
 */
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
private:
  typedef MomentumOperator<dim, Number>                        PDEOperatorNumber;
  typedef MomentumOperator<dim, MultigridNumber>               PDEOperator;
  typedef MultigridOperatorBase<dim, MultigridNumber>          MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperator> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

public:
  virtual ~MultigridPreconditioner(){};

  void
  initialize(MultigridData const &                data,
             parallel::Triangulation<dim> const * tria,
             FiniteElement<dim> const &           fe,
             Mapping<dim> const &                 mapping,
             MomentumOperatorData<dim> const &    operator_data_in,
             Map const *                          dirichlet_bc        = nullptr,
             PeriodicFacePairs *                  periodic_face_pairs = nullptr)
  {
    operator_data                = operator_data_in;
    operator_data.dof_index      = 0;
    operator_data.quad_index_std = 0;
    // check whether we need a second quadrature formula, i.e., it is possible to
    // use the same quadrature formula for all terms
    if(operator_data_in.quad_index_over != operator_data_in.quad_index_std)
      operator_data.quad_index_over = operator_data.quad_index_std + 1;
    else
      operator_data.quad_index_over = operator_data.quad_index_std;

    // set dof index to zero since matrix free object only contains one dof-handler
    operator_data.mass_matrix_operator_data.dof_index  = operator_data.dof_index;
    operator_data.mass_matrix_operator_data.quad_index = operator_data.quad_index_std;

    // set dof index to zero since matrix free object only contains one dof-handler
    operator_data.viscous_operator_data.dof_index  = operator_data.dof_index;
    operator_data.viscous_operator_data.quad_index = operator_data.quad_index_std;

    // do not forget to update kernel_data inside viscous_operator_data!
    operator_data.viscous_operator_data.kernel_data.dof_index = operator_data.dof_index;

    // set dof index to zero since matrix free object only contains one dof-handler
    operator_data.convective_operator_data.dof_index  = operator_data.dof_index;
    operator_data.convective_operator_data.quad_index = operator_data.quad_index_over;

    // do not forget to update kernel_data inside convective_operator_data!
    operator_data.convective_operator_data.kernel_data.dof_index = operator_data.dof_index;
    operator_data.convective_operator_data.kernel_data.quad_index_linearized =
      operator_data.quad_index_over;

    // When solving the reaction-convection-diffusion problem, it might be possible
    // that one wants to apply the multigrid preconditioner only to the reaction-diffusion
    // operator (which is symmetric, Chebyshev smoother, etc.) instead of the non-symmetric
    // reaction-convection-diffusion operator. Accordingly, we have to reset which
    // operators should be "active" for the multigrid preconditioner, independently of
    // the actual equation type that is solved.
    AssertThrow(operator_data.mg_operator_type != MultigridOperatorType::Undefined,
                ExcMessage("Invalid parameter mg_operator_type."));

    if(operator_data.mg_operator_type == MultigridOperatorType::ReactionDiffusion)
    {
      // deactivate convective term for multigrid preconditioner
      operator_data.convective_problem = false;
    }
    else if(operator_data.mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      AssertThrow(operator_data.convective_problem == true, ExcMessage("Invalid parameter."));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }


    Base::initialize(
      data, tria, fe, mapping, false /*operator_is_singular*/, dirichlet_bc, periodic_face_pairs);
  }

  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  initialize_matrix_free(unsigned int const level, Mapping<dim> const & mapping)
  {
    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);

    auto & dof_handler = *this->dof_handlers[level];

    std::vector<DoFHandler<dim> const *> dof_handler_vec;
    dof_handler_vec.resize(1);
    dof_handler_vec[0] = &dof_handler;

    // constraint matrix
    std::vector<AffineConstraints<double> const *> constraint_matrix_vec;
    constraint_matrix_vec.resize(1);
    constraint_matrix_vec[0] = &*this->constraints[level];

    // quadratures
    std::vector<Quadrature<1>> quadrature_vec;
    quadrature_vec.resize(1);
    quadrature_vec[operator_data.quad_index_std] = QGauss<1>(dof_handler.get_fe().degree + 1);

    // only add another quadrature formula if necessary
    if(operator_data.quad_index_over != operator_data.quad_index_std)
    {
      AssertThrow(operator_data.quad_index_over == operator_data.quad_index_std + 1,
                  ExcMessage(
                    "Assumption made regarding quadrature indices is no longer fulfilled."));

      quadrature_vec.resize(2);
      quadrature_vec[operator_data.quad_index_over] =
        QGauss<1>(dof_handler.get_fe().degree + (dof_handler.get_fe().degree + 2) / 2);
    }

    // additional data
    typename MatrixFree<dim, MultigridNumber>::AdditionalData additional_data;

    additional_data.mapping_update_flags =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    additional_data.mapping_update_flags_inner_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    additional_data.mapping_update_flags_boundary_faces =
      (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
       update_values);

    additional_data.level_mg_handler = this->level_info[level].h_level();

    if(operator_data.use_cell_based_loops)
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &dof_handler.get_triangulation());
      Categorization::do_cell_based_loops(*tria,
                                          additional_data,
                                          this->level_info[level].h_level());
    }

    matrix_free->reinit(
      mapping, dof_handler_vec, constraint_matrix_vec, quadrature_vec, additional_data);

    return matrix_free;
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<PDEOperator> pde_operator(new PDEOperator());
    pde_operator->reinit(*this->matrix_free_objects[level],
                         *this->constraints[level],
                         operator_data);

    // initialize MGOperator which is a wrapper around the PDEOperator
    std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

    return mg_operator;
  }

  /*
   * This function updates the multigrid preconditioner.
   */
  virtual void
  update(LinearOperatorBase const * pde_operator_in)
  {
    PDEOperatorNumber const * pde_operator =
      dynamic_cast<PDEOperatorNumber const *>(pde_operator_in);

    AssertThrow(
      pde_operator != nullptr,
      ExcMessage(
        "Operator used to update multigrid preconditioner does not match actual PDE operator!"));

    MultigridOperatorType mg_operator_type = pde_operator->get_operator_data().mg_operator_type;

    if(mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      VectorType const & vector_linearization = pde_operator->get_velocity();

      // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
      VectorTypeMG         vector_multigrid_type_copy;
      VectorTypeMG const * vector_multigrid_type_ptr;
      if(std::is_same<MultigridNumber, Number>::value)
      {
        vector_multigrid_type_ptr = reinterpret_cast<VectorTypeMG const *>(&vector_linearization);
      }
      else
      {
        vector_multigrid_type_copy = vector_linearization;
        vector_multigrid_type_ptr  = &vector_multigrid_type_copy;
      }

      update_operators(pde_operator->get_evaluation_time(),
                       pde_operator->get_scaling_factor_time_derivative_term(),
                       vector_multigrid_type_ptr);
    }
    else if(mg_operator_type == MultigridOperatorType::ReactionDiffusion)
    {
      update_operators(pde_operator->get_evaluation_time(),
                       pde_operator->get_scaling_factor_time_derivative_term());
    }
    else
    {
      AssertThrow(mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion ||
                    mg_operator_type == MultigridOperatorType::ReactionDiffusion,
                  ExcMessage("Multigrid operator type is invalid or not implemented."));
    }

    update_smoothers();

    // singular operators do not occur for this operator
    this->update_coarse_solver(false /* operator_is_singular */);
  }

private:
  /*
   * This function updates operators
   * To do this, three functions are called:
   *  - set_vector_linearization
   *  - set_evaluation_time
   *  - set_scaling_factor_time_derivative_term
   */
  void
  update_operators(double const &       evaluation_time,
                   double const &       scaling_factor_time_derivative_term,
                   VectorTypeMG const * vector_linearization = nullptr)
  {
    set_evaluation_time(evaluation_time);
    set_scaling_factor_time_derivative_term(scaling_factor_time_derivative_term);

    if(vector_linearization != nullptr)
      set_vector_linearization(*vector_linearization);
  }

  /*
   * This function updates vector_linearization.
   * In order to update operators[level] this function has to be called.
   */
  void
  set_vector_linearization(VectorTypeMG const & vector_linearization)
  {
    // copy velocity to finest level
    this->get_operator(this->fine_level)->set_velocity_copy(vector_linearization);

    // interpolate velocity from fine to coarse level
    for(unsigned int level = this->fine_level; level > this->coarse_level; --level)
    {
      auto & vector_fine_level   = this->get_operator(level - 0)->get_velocity();
      auto   vector_coarse_level = this->get_operator(level - 1)->get_velocity();
      this->transfers.interpolate(level, vector_coarse_level, vector_fine_level);
      this->get_operator(level - 1)->set_velocity_copy(vector_coarse_level);
    }
  }

  /*
   * This function updates the evaluation time. In order to update operators[level] this function
   * has to be called. (This is due to the fact that the linearized convective term does not only
   * depend on the linearized velocity field but also on Dirichlet boundary data which itself
   * depends on the current time.)
   */
  void
  set_evaluation_time(double const & evaluation_time)
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
    {
      get_operator(level)->set_evaluation_time(evaluation_time);
    }
  }

  /*
   * This function updates scaling_factor_time_derivative_term. In order to update
   * operators[level] this function has to be called. This is necessary if adaptive time stepping
   * is used where the scaling factor of the derivative term is variable.
   */
  void
  set_scaling_factor_time_derivative_term(double const & scaling_factor_time_derivative_term)
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
    {
      get_operator(level)->set_scaling_factor_time_derivative_term(
        scaling_factor_time_derivative_term);
    }
  }

  /*
   * This function updates the smoother for all levels of the multigrid algorithm.
   * The prerequisite to call this function is that operators[level] have been updated.
   */
  void
  update_smoothers()
  {
    // Skip coarsest level
    for(unsigned int level = this->coarse_level + 1; level <= this->fine_level; ++level)
    {
      this->update_smoother(level);
    }
  }

  std::shared_ptr<PDEOperator>
  get_operator(unsigned int level)
  {
    std::shared_ptr<MGOperator> mg_operator =
      std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

    return mg_operator->get_pde_operator();
  }

  MomentumOperatorData<dim> operator_data;
};

/*
 * Multigrid preconditioner for momentum operator of the incompressible Navier-Stokes equations.
 */
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditionerMerged
  : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
private:
  typedef MomentumOperatorMerged<dim, Number>                  PDEOperatorNumber;
  typedef MomentumOperatorMerged<dim, MultigridNumber>         PDEOperator;
  typedef MultigridOperatorBase<dim, MultigridNumber>          MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperator> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

public:
  virtual ~MultigridPreconditionerMerged(){};

  void
  initialize(MultigridData const &                   data,
             parallel::Triangulation<dim> const *    tria,
             FiniteElement<dim> const &              fe,
             Mapping<dim> const &                    mapping,
             MomentumOperatorMergedData<dim> const & operator_data_in,
             Map const *                             dirichlet_bc        = nullptr,
             PeriodicFacePairs *                     periodic_face_pairs = nullptr)
  {
    operator_data            = operator_data_in;
    operator_data.dof_index  = 0;
    operator_data.quad_index = 0;

    // do not forget to update viscous_kernel_data!
    operator_data.viscous_kernel_data.dof_index = operator_data.dof_index;

    // do not forget to update convective_kernel_data!
    operator_data.convective_kernel_data.dof_index             = operator_data.dof_index;
    operator_data.convective_kernel_data.quad_index_linearized = operator_data.quad_index;

    // When solving the reaction-convection-diffusion problem, it might be possible
    // that one wants to apply the multigrid preconditioner only to the reaction-diffusion
    // operator (which is symmetric, Chebyshev smoother, etc.) instead of the non-symmetric
    // reaction-convection-diffusion operator. Accordingly, we have to reset which
    // operators should be "active" for the multigrid preconditioner, independently of
    // the actual equation type that is solved.
    AssertThrow(operator_data.mg_operator_type != MultigridOperatorType::Undefined,
                ExcMessage("Invalid parameter mg_operator_type."));

    if(operator_data.mg_operator_type == MultigridOperatorType::ReactionDiffusion)
    {
      // deactivate convective term for multigrid preconditioner
      operator_data.convective_problem = false;
    }
    else if(operator_data.mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      AssertThrow(operator_data.convective_problem == true, ExcMessage("Invalid parameter."));
    }
    else
    {
      AssertThrow(false, ExcMessage("Not implemented."));
    }

    Base::initialize(
      data, tria, fe, mapping, false /*operator_is_singular*/, dirichlet_bc, periodic_face_pairs);
  }

  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  initialize_matrix_free(unsigned int const level, Mapping<dim> const & mapping)
  {
    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);

    // additional data
    typename MatrixFree<dim, MultigridNumber>::AdditionalData additional_data;

    additional_data.level_mg_handler      = this->level_info[level].h_level();
    additional_data.tasks_parallel_scheme = MatrixFree<dim, MultigridNumber>::AdditionalData::none;

    MappingFlags flags;
    if(operator_data.unsteady_problem)
      flags = flags || Operators::MassMatrixKernel<dim, Number>::get_mapping_flags();
    if(operator_data.convective_problem)
      flags = flags || Operators::ConvectiveKernel<dim, Number>::get_mapping_flags();
    if(operator_data.viscous_problem)
      flags = flags || Operators::ViscousKernel<dim, Number>::get_mapping_flags();

    additional_data.mapping_update_flags = flags.cells;
    if(this->level_info[level].is_dg())
    {
      additional_data.mapping_update_flags_inner_faces    = flags.inner_faces;
      additional_data.mapping_update_flags_boundary_faces = flags.boundary_faces;
    }

    if(operator_data.use_cell_based_loops && this->level_info[level].is_dg())
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &this->dof_handlers[level]->get_triangulation());
      Categorization::do_cell_based_loops(*tria,
                                          additional_data,
                                          this->level_info[level].h_level());
    }

    QGauss<1> quadrature(this->level_info[level].degree() + 1);
    matrix_free->reinit(
      mapping, *this->dof_handlers[level], *this->constraints[level], quadrature, additional_data);

    return matrix_free;
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<PDEOperator> pde_operator(new PDEOperator());
    pde_operator->reinit(*this->matrix_free_objects[level],
                         *this->constraints[level],
                         operator_data);

    // initialize MGOperator which is a wrapper around the PDEOperator
    std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

    return mg_operator;
  }

  /*
   * This function updates the multigrid preconditioner.
   */
  virtual void
  update(LinearOperatorBase const * pde_operator_in)
  {
    PDEOperatorNumber const * pde_operator =
      dynamic_cast<PDEOperatorNumber const *>(pde_operator_in);

    AssertThrow(
      pde_operator != nullptr,
      ExcMessage(
        "Operator used to update multigrid preconditioner does not match actual PDE operator!"));

    update_operators(pde_operator);

    update_smoothers();

    // singular operators do not occur for this operator
    this->update_coarse_solver(false /* operator_is_singular */);
  }

private:
  /*
   * This function updates the multigrid operators for all levels
   */
  void
  update_operators(PDEOperatorNumber const * pde_operator)
  {
    if(operator_data.unsteady_problem)
    {
      set_evaluation_time(pde_operator->get_evaluation_time());
      set_scaling_factor_time_derivative_term(pde_operator->get_scaling_factor_mass_matrix());
    }

    MultigridOperatorType mg_operator_type = pde_operator->get_operator_data().mg_operator_type;

    if(mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      VectorType const & vector_linearization = pde_operator->get_velocity();

      // convert Number --> MultigridNumber, e.g., double --> float, but only if necessary
      VectorTypeMG         vector_multigrid_type_copy;
      VectorTypeMG const * vector_multigrid_type_ptr;
      if(std::is_same<MultigridNumber, Number>::value)
      {
        vector_multigrid_type_ptr = reinterpret_cast<VectorTypeMG const *>(&vector_linearization);
      }
      else
      {
        vector_multigrid_type_copy = vector_linearization;
        vector_multigrid_type_ptr  = &vector_multigrid_type_copy;
      }

      set_vector_linearization(*vector_multigrid_type_ptr);
    }
  }

  /*
   * This function updates vector_linearization.
   * In order to update operators[level] this function has to be called.
   */
  void
  set_vector_linearization(VectorTypeMG const & vector_linearization)
  {
    // copy velocity to finest level
    this->get_operator(this->fine_level)->set_velocity_copy(vector_linearization);

    // interpolate velocity from fine to coarse level
    for(unsigned int level = this->fine_level; level > this->coarse_level; --level)
    {
      auto & vector_fine_level   = this->get_operator(level - 0)->get_velocity();
      auto   vector_coarse_level = this->get_operator(level - 1)->get_velocity();
      this->transfers.interpolate(level, vector_coarse_level, vector_fine_level);
      this->get_operator(level - 1)->set_velocity_copy(vector_coarse_level);
    }
  }

  /*
   * This function updates the evaluation time. In order to update the operators this function
   * has to be called. (This is due to the fact that the linearized convective term does not only
   * depend on the linearized velocity field but also on Dirichlet boundary data which itself
   * depends on the current time.)
   */
  void
  set_evaluation_time(double const & evaluation_time)
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
    {
      get_operator(level)->set_evaluation_time(evaluation_time);
    }
  }

  /*
   * This function updates scaling_factor_time_derivative_term. In order to update the
   * operators this function has to be called. This is necessary if adaptive time stepping
   * is used where the scaling factor of the derivative term is variable.
   */
  void
  set_scaling_factor_time_derivative_term(double const & scaling_factor_time_derivative_term)
  {
    for(unsigned int level = this->coarse_level; level <= this->fine_level; ++level)
    {
      get_operator(level)->set_scaling_factor_mass_matrix(scaling_factor_time_derivative_term);
    }
  }

  /*
   * This function updates the smoother for all multigrid levels.
   * The prerequisite to call this function is that the multigrid operators have been updated.
   */
  void
  update_smoothers()
  {
    // Skip coarsest level
    for(unsigned int level = this->coarse_level + 1; level <= this->fine_level; ++level)
    {
      this->update_smoother(level);
    }
  }

  std::shared_ptr<PDEOperator>
  get_operator(unsigned int level)
  {
    std::shared_ptr<MGOperator> mg_operator =
      std::dynamic_pointer_cast<MGOperator>(this->operators[level]);

    return mg_operator->get_pde_operator();
  }

  MomentumOperatorMergedData<dim> operator_data;
};

} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_ \
        */
