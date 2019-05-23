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

#include "../../solvers_and_preconditioners/util/restrict_vector_to_coarser_level.h"

namespace IncNS
{
/*
 * Multigrid preconditioner for momentum operator of the incompressible Navier-Stokes equations.
 */
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
private:
  typedef MomentumOperator<dim, MultigridNumber>               PDEOperator;
  typedef MultigridOperatorBase<dim, MultigridNumber>          MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperator> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map          Map;
  typedef typename Base::VectorType   VectorType;
  typedef typename Base::VectorTypeMG VectorTypeMG;

public:
  virtual ~MultigridPreconditioner(){};

  void
  initialize(MultigridData const &                mg_data,
             const parallel::Triangulation<dim> * tria,
             const FiniteElement<dim> &           fe,
             Mapping<dim> const &                 mapping,
             MomentumOperatorData<dim> const &    operator_data_in,
             Map const *                          dirichlet_bc = nullptr,
             std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> *
               periodic_face_pairs = nullptr)
  {
    operator_data                 = operator_data_in;
    operator_data.dof_index       = 0;
    operator_data.quad_index_std  = 0;
    operator_data.quad_index_over = 1;

    // set dof index to zero since matrix free object only contains one dof-handler
    operator_data.mass_matrix_operator_data.dof_index  = operator_data.dof_index;
    operator_data.mass_matrix_operator_data.quad_index = operator_data.quad_index_std;

    // set dof index to zero since matrix free object only contains one dof-handler
    operator_data.viscous_operator_data.dof_index  = operator_data.dof_index;
    operator_data.viscous_operator_data.quad_index = operator_data.quad_index_std;

    // set dof index to zero since matrix free object only contains one dof-handler
    operator_data.convective_operator_data.dof_index = operator_data.dof_index;
    // set quad index to 1 since matrix free object only contains two quadrature formulas
    operator_data.convective_operator_data.quad_index = operator_data.quad_index_over;

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


    Base::initialize(mg_data,
                     tria,
                     fe,
                     mapping,
                     false /*operator_is_singular*/,
                     dirichlet_bc,
                     periodic_face_pairs);
  }

  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  initialize_matrix_free(unsigned int const level, Mapping<dim> const & mapping)
  {
    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);

    auto & dof_handler = *this->mg_dofhandler[level];

    std::vector<DoFHandler<dim> const *> dof_handler_vec;
    dof_handler_vec.resize(1);
    dof_handler_vec[0] = &dof_handler;

    // constraint matrix
    std::vector<AffineConstraints<double> const *> constraint_matrix_vec;
    constraint_matrix_vec.resize(1);
    constraint_matrix_vec[0] = &*this->mg_constraints[level];

    // quadratures
    std::vector<Quadrature<1>> quadrature_vec;
    quadrature_vec.resize(2);
    quadrature_vec[operator_data.quad_index_std] = QGauss<1>(dof_handler.get_fe().degree + 1);
    quadrature_vec[operator_data.quad_index_over] =
      QGauss<1>(dof_handler.get_fe().degree + (dof_handler.get_fe().degree + 2) / 2);

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

    additional_data.level_mg_handler = this->global_levels[level].level;

    if(operator_data.use_cell_based_loops)
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &dof_handler.get_triangulation());
      Categorization::do_cell_based_loops(*tria, additional_data, this->global_levels[level].level);
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
    pde_operator->reinit_multigrid(*this->mg_matrixfree[level],
                                   *this->mg_constraints[level],
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
    MomentumOperator<dim, Number> const * pde_operator =
      dynamic_cast<MomentumOperator<dim, Number> const *>(pde_operator_in);

    AssertThrow(
      pde_operator != nullptr,
      ExcMessage(
        "Operator used to update multigrid preconditioner does not match actual PDE operator!"));

    MultigridOperatorType mg_operator_type = pde_operator->get_operator_data().mg_operator_type;

    if(mg_operator_type == MultigridOperatorType::ReactionConvectionDiffusion)
    {
      VectorType const & vector_linearization = pde_operator->get_solution_linearization();

      // convert Number --> Operator::value_type, e.g., double --> float, but only if necessary
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

      update_mg_matrices(pde_operator->get_evaluation_time(),
                         pde_operator->get_scaling_factor_time_derivative_term(),
                         vector_multigrid_type_ptr);
    }
    else if(mg_operator_type == MultigridOperatorType::ReactionDiffusion)
    {
      update_mg_matrices(pde_operator->get_evaluation_time(),
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
   * This function updates mg_matrices
   * To do this, three functions are called:
   *  - set_vector_linearization
   *  - set_evaluation_time
   *  - set_scaling_factor_time_derivative_term
   */
  void
  update_mg_matrices(double const &       evaluation_time,
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
   * In order to update mg_matrices[level] this function has to be called.
   */
  void
  set_vector_linearization(VectorTypeMG const & vector_linearization)
  {
    // copy velocity to finest level
    this->get_matrix(this->max_level)->set_solution_linearization(vector_linearization);

    // interpolate velocity from fine to coarse level
    for(auto level = this->max_level; level > this->min_level; --level)
    {
      auto & vector_fine_level   = this->get_matrix(level - 0)->get_solution_linearization();
      auto   vector_coarse_level = this->get_matrix(level - 1)->get_solution_linearization();
      this->mg_transfer.interpolate(level, vector_coarse_level, vector_fine_level);
      this->get_matrix(level - 1)->set_solution_linearization(vector_coarse_level);
    }
  }

  /*
   * This function updates the evaluation time. In order to update mg_matrices[level] this function
   * has to be called. (This is due to the fact that the linearized convective term does not only
   * depend on the linearized velocity field but also on Dirichlet boundary data which itself
   * depends on the current time.)
   */
  void
  set_evaluation_time(double const & evaluation_time)
  {
    for(int level = this->n_global_levels - 1; level >= 0; --level)
    {
      get_matrix(level)->set_evaluation_time(evaluation_time);
    }
  }

  /*
   * This function updates scaling_factor_time_derivative_term. In order to update
   * mg_matrices[level] this function has to be called. This is necessary if adaptive time stepping
   * is used where the scaling factor of the derivative term is variable.
   */
  void
  set_scaling_factor_time_derivative_term(double const & scaling_factor_time_derivative_term)
  {
    for(int level = this->n_global_levels - 1; level >= 0; --level)
    {
      get_matrix(level)->set_scaling_factor_time_derivative_term(
        scaling_factor_time_derivative_term);
    }
  }

  /*
   * This function updates the smoother for all levels of the multigrid algorithm.
   * The prerequisite to call this function is that mg_matrices[level] have been updated.
   */
  void
  update_smoothers()
  {
    // Start with level = 1!
    for(unsigned int level = 1; level < this->n_global_levels; ++level)
    {
      this->update_smoother(level);
    }
  }

  std::shared_ptr<PDEOperator>
  get_matrix(unsigned int level)
  {
    std::shared_ptr<MGOperator> mg_operator =
      std::dynamic_pointer_cast<MGOperator>(this->mg_matrices[level]);

    return mg_operator->get_pde_operator();
  }

  MomentumOperatorData<dim> operator_data;
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_ \
        */
