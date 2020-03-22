/*
 * multigrid_preconditioner.h
 *
 *  Created on: 21.03.2020
 *      Author: fehn
 */

#ifndef INCLUDE_STRUCTURE_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_
#define INCLUDE_STRUCTURE_PRECONDITIONERS_MULTIGRID_PRECONDITIONER_H_

#include "../../operators/multigrid_operator.h"
#include "../../solvers_and_preconditioners/multigrid/multigrid_preconditioner_base.h"

#include "../../structure/spatial_discretization/operators/linear_operator.h"

namespace Structure
{
template<int dim, typename Number, typename MultigridNumber>
class MultigridPreconditioner : public MultigridPreconditionerBase<dim, Number, MultigridNumber>
{
public:
  typedef LinearOperator<dim, Number>          PDEOperator;
  typedef LinearOperator<dim, MultigridNumber> PDEOperatorMG;

  typedef MultigridOperatorBase<dim, MultigridNumber>            MGOperatorBase;
  typedef MultigridOperator<dim, MultigridNumber, PDEOperatorMG> MGOperator;

  typedef MultigridPreconditionerBase<dim, Number, MultigridNumber> Base;

  typedef typename Base::Map               Map;
  typedef typename Base::PeriodicFacePairs PeriodicFacePairs;
  typedef typename Base::VectorType        VectorType;
  typedef typename Base::VectorTypeMG      VectorTypeMG;

  MultigridPreconditioner(MPI_Comm const & mpi_comm) : Base(mpi_comm), pde_operator(nullptr)
  {
  }

  void
  initialize(MultigridData const &                    mg_data,
             parallel::TriangulationBase<dim> const * tria,
             FiniteElement<dim> const &               fe,
             Mapping<dim> const &                     mapping,
             PDEOperator const &                      pde_operator,
             Map const *                              dirichlet_bc        = nullptr,
             PeriodicFacePairs *                      periodic_face_pairs = nullptr)
  {
    this->pde_operator = &pde_operator;

    data            = this->pde_operator->get_data();
    data.dof_index  = 0;
    data.quad_index = 0;

    Base::initialize(mg_data,
                     tria,
                     fe,
                     mapping,
                     false /*operator_is_singular*/,
                     dirichlet_bc,
                     periodic_face_pairs);
  }


private:
  std::shared_ptr<MatrixFree<dim, MultigridNumber>>
  do_initialize_matrix_free(unsigned int const level) override
  {
    typename MatrixFree<dim, MultigridNumber>::AdditionalData additional_data;

    additional_data.mg_level              = this->level_info[level].h_level();
    additional_data.tasks_parallel_scheme = MatrixFree<dim, MultigridNumber>::AdditionalData::none;

    MappingFlags flags;
    flags = flags || PDEOperator::get_mapping_flags();

    additional_data.mapping_update_flags                = flags.cells;
    additional_data.mapping_update_flags_inner_faces    = flags.inner_faces;
    additional_data.mapping_update_flags_boundary_faces = flags.boundary_faces;

    Quadrature<1> quadrature = QGauss<1>(this->level_info[level].degree() + 1);

    std::shared_ptr<MatrixFree<dim, MultigridNumber>> matrix_free;
    matrix_free.reset(new MatrixFree<dim, MultigridNumber>);
    matrix_free->reinit(*this->mapping,
                        *this->dof_handlers[level],
                        *this->constraints[level],
                        quadrature,
                        additional_data);

    return matrix_free;
  }

  std::shared_ptr<MGOperatorBase>
  initialize_operator(unsigned int const level)
  {
    // initialize pde_operator in a first step
    std::shared_ptr<PDEOperatorMG> pde_operator_level(new PDEOperatorMG());

    pde_operator_level->reinit(*this->matrix_free_objects[level], *this->constraints[level], data);

    // initialize MGOperator which is a wrapper around the PDEOperatorMG
    std::shared_ptr<MGOperator> mg_operator_level(new MGOperator(pde_operator_level));

    return mg_operator_level;
  }

private:
  OperatorData<dim> data;

  PDEOperator const * pde_operator;
};

} // namespace Structure

#endif
