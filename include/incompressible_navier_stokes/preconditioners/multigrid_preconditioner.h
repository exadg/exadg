/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_


#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/momentum_operator.h"

#include "../../solvers_and_preconditioners/util/restrict_vector_to_coarser_level.h"

namespace IncNS
{
/*
 * Multigrid preconditioner for velocity (reaction-)convection-diffusion operator of the
 * incompressible Navier-Stokes equations.
 */
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
private:
  typedef MomentumOperator<dim, Number>          PDEOperator;
  typedef MomentumOperator<dim, MultigridNumber> MultigridOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> BASE;
  typedef typename BASE::Map                                        Map;

  typedef typename BASE::VectorType   VectorType;
  typedef typename BASE::VectorTypeMG VectorTypeMG;

public:
  MultigridPreconditioner()
    : MultigridPreconditionerBase<dim, Number, MultigridNumber>(
        std::shared_ptr<PreconditionableOperator<dim, MultigridNumber>>(new MultigridOperator()))
  {
  }

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
    auto operator_data            = operator_data_in;
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

    BASE::initialize(mg_data, tria, fe, mapping, operator_data, dirichlet_bc, periodic_face_pairs);
  }

  virtual ~MultigridPreconditioner(){};


  void
  initialize_matrixfree(std::vector<MGLevelInfo> &                global_levels,
                        Mapping<dim> const &                      mapping,
                        PreconditionableOperatorData<dim> const & operator_data_in)
  {
    const auto & operator_data = static_cast<MomentumOperatorData<dim> const &>(operator_data_in);

    this->mg_matrixfree.resize(this->min_level, this->max_level);

    for(auto level = this->min_level; level <= this->max_level; ++level)
    {
      auto data = new MatrixFree<dim, MultigridNumber>;

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

      additional_data.level_mg_handler = global_levels[level].level;

      if(operator_data.use_cell_based_loops)
      {
        auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
          &dof_handler.get_triangulation());
        Categorization::do_cell_based_loops(*tria, additional_data, global_levels[level].level);
      }

      // reinit
      data->reinit(
        mapping, dof_handler_vec, constraint_matrix_vec, quadrature_vec, additional_data);

      this->mg_matrixfree[level].reset(data);
    }
  }

  /*
   * This function updates the multigrid preconditioner.
   */
  virtual void
  update(LinearOperatorBase const * update_operator)
  {
    PDEOperator const * pde_operator = dynamic_cast<PDEOperator const *>(update_operator);

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
    this->update_coarse_solver();
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

  MomentumOperatorAbstract<dim, MultigridNumber> *
  get_matrix(unsigned int level)
  {
    return dynamic_cast<MomentumOperatorAbstract<dim, MultigridNumber> *>(
      &*this->mg_matrices[level]);
  }
};


} // namespace IncNS

#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_ \
        */
