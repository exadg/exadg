/*
 * multigrid_preconditioner.h
 *
 *  Created on: Nov 23, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_

#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"
#include "../spatial_discretization/laplace_operator.h"

namespace Poisson
{
/*
 *  Multigrid preconditioner for scalar Laplace operator.
 */
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef LaplaceOperator<dim, MultigridNumber> Laplace;

  typedef MultigridOperatorBase<dim, MultigridNumber>      MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, Laplace> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> BASE;

  typedef typename BASE::Map               Map;
  typedef typename BASE::PeriodicFacePairs PeriodicFacePairs;

  void
  initialize(MultigridData const &                mg_data,
             const parallel::TriangulationBase<dim> * tria,
             const FiniteElement<dim> &           fe,
             Mapping<dim> const &                 mapping,
             LaplaceOperatorData<dim> const &     data_in,
             Map const *                          dirichlet_bc        = nullptr,
             PeriodicFacePairs *                  periodic_face_pairs = nullptr)
  {
    data            = data_in;
    data.dof_index  = 0;
    data.quad_index = 0;

    BASE::initialize(
      mg_data, tria, fe, mapping, data.operator_is_singular, dirichlet_bc, periodic_face_pairs);
  }

  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  initialize_matrix_free(unsigned int const level, Mapping<dim> const & mapping)
  {
    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);

    // setup MatrixFree::AdditionalData
    typename MatrixFree<dim, MultigridNumber>::AdditionalData additional_data;

    additional_data.level_mg_handler = this->level_info[level].h_level();

    MappingFlags flags = Operators::LaplaceKernel<dim, Number>::get_mapping_flags();

    additional_data.mapping_update_flags = flags.cells;

    if(this->level_info[level].is_dg())
    {
      additional_data.mapping_update_flags_inner_faces    = flags.inner_faces;
      additional_data.mapping_update_flags_boundary_faces = flags.boundary_faces;
    }

    if(data.use_cell_based_loops && this->level_info[level].is_dg())
    {
      auto tria = dynamic_cast<parallel::distributed::Triangulation<dim> const *>(
        &this->dof_handlers[level]->get_triangulation());
      Categorization::do_cell_based_loops(*tria,
                                          additional_data,
                                          this->level_info[level].h_level());
    }

    QGauss<1> const quad(this->level_info[level].degree() + 1);
    matrix_free->reinit(
      mapping, *this->dof_handlers[level], *this->constraints[level], quad, additional_data);

    return matrix_free;
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<Laplace> pde_operator(new Laplace());

    // The polynomial degree changes in case of p-multigrid, so we have to adapt kernel data.
    LaplaceOperatorData<dim> data_level = data;
    data_level.kernel_data.degree       = this->level_info[level].degree();

    pde_operator->reinit(*this->matrix_free_objects[level], *this->constraints[level], data_level);

    // initialize MGOperator which is a wrapper around the PDEOperator
    std::shared_ptr<MGOperator> mg_operator(new MGOperator(pde_operator));

    return mg_operator;
  }

  LaplaceOperatorData<dim> data;
};

} // namespace Poisson


#endif /* INCLUDE_POISSON_MULTIGRID_PRECONDITIONER_H_ */
