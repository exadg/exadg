//
// Created by max on 02.02.21.
//

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer_global_coarsening.h>


namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number, typename VectorType>
void
MGTransferGlobalCoarsening<dim, Number, VectorType>::restrict_and_add(const unsigned int level,
                                                                      VectorType &       dst,
                                                                      const VectorType & src) const
{
  mg_transfer_global_coarsening->restrict_and_add(level, dst, src);
}

template<int dim, typename Number, typename VectorType>
void
MGTransferGlobalCoarsening<dim, Number, VectorType>::prolongate(const unsigned int level,
                                                                VectorType &       dst,
                                                                const VectorType & src) const
{
  mg_transfer_global_coarsening->prolongate(level, dst, src);
}

template<int dim, typename Number, typename VectorType>
void
MGTransferGlobalCoarsening<dim, Number, VectorType>::reinit(
  const Mapping<dim> &                                        mapping,
  MGLevelObject<std::shared_ptr<MatrixFree<dim, Number>>> &   mg_matrixfree,
  MGLevelObject<std::shared_ptr<AffineConstraints<Number>>> & mg_constraints,
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &         mg_constrained_dofs,
  const unsigned int                                          dof_handler_index)
{
  (void)mapping;
  (void)mg_constrained_dofs;
  {
    std::vector<MGLevelInfo>            global_levels;
    std::vector<MGDoFHandlerIdentifier> p_levels;

    const unsigned int min_level = mg_matrixfree.min_level();
    AssertThrow(min_level == 0, ExcMessage("Currently, we expect min_level==0!"));

    const unsigned int max_level = mg_matrixfree.max_level();

    // construct global_levels
    for(unsigned int global_level = min_level; global_level <= max_level; global_level++)
    {
      const auto &       matrixfree = mg_matrixfree[global_level];
      const auto &       fe         = matrixfree->get_dof_handler(dof_handler_index).get_fe();
      const bool         is_dg      = fe.dofs_per_vertex == 0;
      const unsigned int level =
        matrixfree->get_dof_handler().get_triangulation().n_global_levels();
      const unsigned int degree =
        (int)round(std::pow(fe.n_dofs_per_cell() / fe.n_components(), 1.0 / dim)) - 1;

      global_levels.push_back(MGLevelInfo(level, degree, is_dg));
    }

    // construct and p_levels
    for(auto i : global_levels)
      p_levels.push_back(i.dof_handler_id());

    sort(p_levels.begin(), p_levels.end());
    p_levels.erase(unique(p_levels.begin(), p_levels.end()), p_levels.end());
    std::reverse(std::begin(p_levels), std::end(p_levels));

    // create transfer-operator instances
    transfers.resize(0, global_levels.size() - 1);

    // fill mg_transfer with the correct transfers
    for(unsigned int i = 1; i < global_levels.size(); i++)
    {
      auto coarse_level = global_levels[i - 1];
      auto fine_level   = global_levels[i];

      if(coarse_level.h_level() != fine_level.h_level()) // h-transfer
      {
        transfers[i].reinit_geometric_transfer(mg_matrixfree[i]->get_dof_handler(dof_handler_index),
                                               mg_matrixfree[i - 1]->get_dof_handler(
                                                 dof_handler_index),
                                               *mg_constraints[i],
                                               *mg_constraints[i - 1]);
      }
      else if(coarse_level.degree() != fine_level.degree() || // p-transfer
              coarse_level.is_dg() != fine_level.is_dg())     // c-transfer
      {
        transfers[i].reinit_polynomial_transfer(
          mg_matrixfree[i]->get_dof_handler(dof_handler_index),
          mg_matrixfree[i - 1]->get_dof_handler(dof_handler_index),
          *mg_constraints[i],
          *mg_constraints[i - 1]);
      }
      else
      {
        AssertThrow(false, ExcMessage("Cannot create MGTransfer!"));
      }
    }
  }
  mg_transfer_global_coarsening =
    std::make_unique<dealii::MGTransferGlobalCoarsening<dim, VectorType>>(transfers);
}

template<int dim, typename Number, typename VectorType>
void
MGTransferGlobalCoarsening<dim, Number, VectorType>::interpolate(const unsigned int level,
                                                                 VectorType &       dst,
                                                                 const VectorType & src) const
{
  (void)level;
  (void)dst;
  (void)src;
  AssertThrow(false, ExcNotImplemented());
}

typedef dealii::LinearAlgebra::distributed::Vector<float>  VectorTypeFloat;
typedef dealii::LinearAlgebra::distributed::Vector<double> VectorTypeDouble;

template class MGTransferGlobalCoarsening<2, float, VectorTypeFloat>;

template class MGTransferGlobalCoarsening<3, float, VectorTypeFloat>;

template class MGTransferGlobalCoarsening<2, double, VectorTypeDouble>;

template class MGTransferGlobalCoarsening<3, double, VectorTypeDouble>;

} // namespace ExaDG