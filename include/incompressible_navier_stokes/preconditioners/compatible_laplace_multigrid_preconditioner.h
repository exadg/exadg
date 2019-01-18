/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_

#include <vector>

#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "compatible_laplace_operator.h"

namespace IncNS
{
/*
 *  Multigrid preconditioner for compatible Laplace operator.
 */
template<int dim, int degree_u, int degree_p, typename Number, typename MultigridNumber>
class CompatibleLaplaceMultigridPreconditioner
  : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  // TODO: remove unnecessary typedefs
  typedef PreconditionableOperator<dim, MultigridNumber> MG_OPERATOR_BASE;

  typedef CompatibleLaplaceOperator<dim, degree_u, degree_p, Number>          PDEOperator;
  typedef CompatibleLaplaceOperator<dim, degree_u, degree_p, MultigridNumber> MultigridOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> BASE;
  typedef typename BASE::Map                                        Map;

  typedef typename BASE::VectorType   VectorType;
  typedef typename BASE::VectorTypeMG VectorTypeMG;

  CompatibleLaplaceMultigridPreconditioner()
    : MultigridPreconditionerBase<dim, Number, MultigridNumber>(
        std::shared_ptr<MG_OPERATOR_BASE>(new MultigridOperator()))
  {
  }

  void
  initialize(MultigridData const &                      mg_data,
             const parallel::Triangulation<dim> *       tria,
             const FiniteElement<dim> &                 fe,
             Mapping<dim> const &                       mapping,
             CompatibleLaplaceOperatorData<dim> const & operator_data_in,
             Map const *                                dirichlet_bc = nullptr,
             std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> *
               periodic_face_pairs = nullptr)
  {
    auto operator_data               = operator_data_in;
    operator_data.dof_index_velocity = 1;
    operator_data.dof_index_pressure = 0;

    operator_data.gradient_operator_data.dof_index_velocity = operator_data.dof_index_velocity;
    operator_data.gradient_operator_data.dof_index_pressure = operator_data.dof_index_pressure;
    operator_data.gradient_operator_data.quad_index         = 0;

    operator_data.divergence_operator_data.dof_index_velocity = operator_data.dof_index_velocity;
    operator_data.divergence_operator_data.dof_index_pressure = operator_data.dof_index_pressure;
    operator_data.divergence_operator_data.quad_index         = 0;

    BASE::initialize(mg_data, tria, fe, mapping, operator_data, dirichlet_bc, periodic_face_pairs);
  }

  virtual void
  initialize_mg_dof_handler_and_constraints_all(
    bool is_singular,
    std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator>> &
                                                                         periodic_face_pairs,
    FiniteElement<dim> const &                                           fe,
    parallel::Triangulation<dim> const *                                 tria,
    std::vector<MGLevelInfo> &                                           global_levels,
    std::vector<MGDofHandlerIdentifier> &                                p_levels,
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const & dirichlet_bc,
    PreconditionableOperatorData<dim> const &                            operator_data_in)
  {
    BASE::initialize_mg_dof_handler_and_constraints_all(is_singular,
                                                        periodic_face_pairs,
                                                        fe,
                                                        tria,
                                                        global_levels,
                                                        p_levels,
                                                        dirichlet_bc,
                                                        operator_data_in);

    std::vector<MGLevelInfo>            global_levels_vel;
    std::vector<MGDofHandlerIdentifier> p_levels_vel;

    // setup global velocity levels
    for(auto & i : global_levels)
      global_levels_vel.push_back({i.level, i.degree + degree_u - degree_p, i.is_dg});

    // setup p velocity levels
    for(auto i : global_levels_vel)
      p_levels_vel.push_back(i.id);

    sort(p_levels_vel.begin(), p_levels_vel.end());
    p_levels_vel.erase(unique(p_levels_vel.begin(), p_levels_vel.end()), p_levels_vel.end());
    std::reverse(std::begin(p_levels_vel), std::end(p_levels_vel));

    // setup dofhandler and constraint matrices
    FE_DGQ<dim>                                                  temp(degree_u);
    FESystem<dim>                                                fe_vel(temp, dim);
    std::map<types::boundary_id, std::shared_ptr<Function<dim>>> dirichlet_bc_vel;
    BASE::initialize_mg_dof_handler_and_constraints(false,
                                                    periodic_face_pairs,
                                                    fe_vel,
                                                    tria,
                                                    global_levels_vel,
                                                    p_levels_vel,
                                                    dirichlet_bc_vel,
                                                    this->mg_dofhandler_vel,
                                                    this->mg_constrained_dofs_vel,
                                                    this->mg_constraints_vel);
  }


  void
  initialize_matrixfree(std::vector<MGLevelInfo> &                global_levels,
                        Mapping<dim> const &                      mapping,
                        PreconditionableOperatorData<dim> const & operator_data_in)
  {
    const auto & operator_data =
      static_cast<CompatibleLaplaceOperatorData<dim> const &>(operator_data_in);

    this->mg_matrixfree.resize(this->min_level, this->max_level);

    for(auto level = this->min_level; level <= this->max_level; ++level)
    {
      auto data = new MatrixFree<dim, MultigridNumber>;

      auto & dof_handler_p = *this->mg_dofhandler[level];
      auto & dof_handler_u = *this->mg_dofhandler_vel[level];

      // dof_handler
      std::vector<const DoFHandler<dim> *> dof_handler_vec;
      // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
      dof_handler_vec.resize(2);
      dof_handler_vec[operator_data.dof_index_velocity] = &dof_handler_u;
      dof_handler_vec[operator_data.dof_index_pressure] = &dof_handler_p;

      // constraint matrix
      std::vector<AffineConstraints<double> const *> constraint_matrix_vec;
      // TODO: instead of 2 use something more general like DofHandlerSelector::n_variants
      constraint_matrix_vec.resize(2);
      constraint_matrix_vec[operator_data.dof_index_velocity] = &*this->mg_constraints_vel[level];
      constraint_matrix_vec[operator_data.dof_index_pressure] = &*this->mg_constraints[level];

      // quadratures
      // quadrature formula with (fe_degree_velocity+1) quadrature points: this is the quadrature
      // formula that is used for the gradient operator and the divergence operator (and the inverse
      // velocity mass matrix operator
      std::vector<Quadrature<1>> quadrature_vec;
      quadrature_vec.resize(2);
      quadrature_vec[0] = QGauss<1>(global_levels[level].degree + 1 + (degree_u - degree_p));
      // quadrature formula with (fe_degree_velocity+1) quadrature points: this is the quadrature is
      // needed for p-transfer
      quadrature_vec[1] = QGauss<1>(global_levels[level].degree + 1);

      // additional data
      typename MatrixFree<dim, MultigridNumber>::AdditionalData addit_data;

      addit_data.mapping_update_flags =
        (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
         update_values);

      if(global_levels[level].is_dg)
      {
        addit_data.mapping_update_flags_inner_faces =
          (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
           update_values);

        addit_data.mapping_update_flags_boundary_faces =
          (update_gradients | update_JxW_values | update_quadrature_points | update_normal_vectors |
           update_values);
      }

      addit_data.level_mg_handler = global_levels[level].level;

      // if(operator_data.use_cell_based_loops)
      //{
      //  auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
      //    &dof_handler_p.get_triangulation());
      //  Categorization::do_cell_based_loops(*tria, additional_data, global_levels[level].level);
      //}

      // reinit
      data->reinit(mapping, dof_handler_vec, constraint_matrix_vec, quadrature_vec, addit_data);

      this->mg_matrixfree[level].reset(data);
    }
  }


  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>>     mg_dofhandler_vel;
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>>         mg_constrained_dofs_vel;
  MGLevelObject<std::shared_ptr<AffineConstraints<double>>> mg_constraints_vel;
};

} // namespace IncNS


#endif /* INCLUDE_INCOMPRESSIBLE_NAVIER_STOKES_PRECONDITIONERS_COMPATIBLE_LAPLACE_MULTIGRID_PRECONDITIONER_H_ \
        */
