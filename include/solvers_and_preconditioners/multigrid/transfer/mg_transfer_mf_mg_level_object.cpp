#include "mg_transfer_mf_mg_level_object.h"

#include "mg_transfer_mf_c.h"
#include "mg_transfer_mf_h.h"
#include "mg_transfer_mf_p.h"


template<int dim, typename VectorType>
template<typename MultigridNumber, typename MatrixFree, typename Constraints>
void
MGTransferMF_MGLevelObject<dim, VectorType>::reinit(
  MGLevelObject<std::shared_ptr<MatrixFree>> &        mg_data,
  MGLevelObject<std::shared_ptr<Constraints>> &       mg_Constraints,
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> & mg_constrained_dofs,
  const unsigned int                                  dof_handler_index)
{
  std::vector<MGLevelInfo>            global_levels;
  std::vector<MGDoFHandlerIdentifier> p_levels;

  const unsigned int min_level = mg_data.min_level();
  AssertThrow(min_level == 0, ExcMessage("Currently, we expect min_level==0!"));

  const unsigned int max_level = mg_data.max_level();
  const int          n_components =
    mg_data[max_level]->get_dof_handler(dof_handler_index).get_fe().n_components();

  // extract relevant information and construct global_levels...
  for(unsigned int global_level = min_level; global_level <= max_level; global_level++)
  {
    const auto &       data  = mg_data[global_level];
    const auto &       fe    = data->get_dof_handler(dof_handler_index).get_fe();
    const bool         is_dg = fe.dofs_per_vertex == 0;
    const unsigned int level = data->get_level_mg_handler();
    const unsigned int degree =
      (int)round(std::pow(fe.n_dofs_per_cell() / fe.n_components(), 1.0 / dim)) - 1;
    global_levels.push_back(MGLevelInfo(level, degree, is_dg));
  }

  // .. and p_levels
  for(auto i : global_levels)
    p_levels.push_back(i.dof_handler_id());

  sort(p_levels.begin(), p_levels.end());
  p_levels.erase(unique(p_levels.begin(), p_levels.end()), p_levels.end());
  std::reverse(std::begin(p_levels), std::end(p_levels));

  // create transfer-operator instances
  mg_level_object.resize(0, global_levels.size() - 1);

  std::map<MGDoFHandlerIdentifier, std::shared_ptr<MGTransferMFH<dim, MultigridNumber>>>
    mg_tranfers_temp;
  std::map<MGDoFHandlerIdentifier, std::map<unsigned int, unsigned int>>
    map_global_level_to_h_levels;

  // initialize maps so that we do not have to check existence later on
  for(auto deg : p_levels)
    map_global_level_to_h_levels[deg] = {};

  // fill the maps
  for(unsigned int i = 0; i < global_levels.size(); i++)
  {
    auto level = global_levels[i];

    map_global_level_to_h_levels[level.dof_handler_id()][i] = level.h_level();
  }

  // create h-transfer operators between levels
  for(auto deg : p_levels)
  {
    if(map_global_level_to_h_levels[deg].size() > 1)
    {
      // create actual h-transfer-operator
      unsigned int global_level = map_global_level_to_h_levels[deg].begin()->first;
      std::shared_ptr<MGTransferMFH<dim, MultigridNumber>> transfer(
        new MGTransferMFH<dim, MultigridNumber>(map_global_level_to_h_levels[deg],
                                                mg_data[global_level]->get_dof_handler(
                                                  dof_handler_index)));

      // dof-handlers and constrains are saved for global levels
      // so we have to convert degree to any global level which has this degree
      // (these share the same dof-handlers and constraints)
      transfer->initialize_constraints(*mg_constrained_dofs[global_level]);
      transfer->build(mg_data[global_level]->get_dof_handler(dof_handler_index));
      mg_tranfers_temp[deg] = transfer;
    } // else: there is only one global level (and one h-level) on this p-level
  }

  // fill mg_transfer with the correct transfers
  for(unsigned int i = 1; i < global_levels.size(); i++)
  {
    auto coarse_level = global_levels[i - 1];
    auto fine_level   = global_levels[i];

    std::shared_ptr<MGTransferMF<VectorType>> temp;

    if(coarse_level.h_level() != fine_level.h_level()) // h-transfer
    {
#ifdef DEBUG
      printf("  h-MG (l = %d, k = %d, dg = %d) -> (l = %d, k = %d, dg = %d)\n",
             coarse_level.h_level(),
             coarse_level.degree(),
             coarse_level.is_dg(),
             fine_level.h_level(),
             fine_level.degree(),
             fine_level.is_dg());
#endif

      temp =
        mg_tranfers_temp[coarse_level.dof_handler_id()]; // get the previously h-transfer operator
    }
    else if(coarse_level.degree() != fine_level.degree()) // p-transfer
    {
#ifdef DEBUG
      printf("  p-MG (l = %d, k = %d, dg = %d) -> (l = %d, k = %d, dg = %d)\n",
             coarse_level.h_level(),
             coarse_level.degree(),
             coarse_level.is_dg(),
             fine_level.h_level(),
             fine_level.degree(),
             fine_level.is_dg());
#endif

      if(n_components == 1)
        temp.reset(new MGTransferMFP<dim, MultigridNumber, VectorType, 1>(&*mg_data[i],
                                                                          &*mg_data[i - 1],
                                                                          fine_level.degree(),
                                                                          coarse_level.degree(),
                                                                          dof_handler_index));
      else if(n_components == dim)
        temp.reset(new MGTransferMFP<dim, MultigridNumber, VectorType, dim>(&*mg_data[i],
                                                                            &*mg_data[i - 1],
                                                                            fine_level.degree(),
                                                                            coarse_level.degree(),
                                                                            dof_handler_index));
      else
        AssertThrow(false, ExcMessage("Cannot create MGTransferMFP!"));
    }
    else if(coarse_level.is_dg() != fine_level.is_dg()) // c-transfer
    {
#ifdef DEBUG
      printf("  c-MG (l = %d, k = %d, dg = %d) -> (l = %d, k = %d, dg = %d)\n",
             coarse_level.h_level(),
             coarse_level.degree(),
             coarse_level.is_dg(),
             fine_level.h_level(),
             fine_level.degree(),
             fine_level.is_dg());
#endif

      if(n_components == 1)
        temp.reset(new MGTransferMFC<dim, typename MatrixFree::value_type, VectorType, 1>(
          *mg_data[i],
          *mg_data[i - 1],
          *mg_Constraints[i],
          *mg_Constraints[i - 1],
          fine_level.h_level(),
          coarse_level.degree(),
          dof_handler_index));
      else if(n_components == dim)
        temp.reset(new MGTransferMFC<dim, typename MatrixFree::value_type, VectorType, dim>(
          *mg_data[i],
          *mg_data[i - 1],
          *mg_Constraints[i],
          *mg_Constraints[i - 1],
          fine_level.h_level(),
          coarse_level.degree(),
          dof_handler_index));
      else
        AssertThrow(false, ExcMessage("Cannot create MGTransferMFP!"));
    }
    mg_level_object[i] = temp;
  }
}

template<int dim, typename VectorType>
void
MGTransferMF_MGLevelObject<dim, VectorType>::interpolate(const unsigned int level,
                                                         VectorType &       dst,
                                                         const VectorType & src) const
{
  this->mg_level_object[level]->interpolate(level, dst, src);
}

template<int dim, typename VectorType>
void
MGTransferMF_MGLevelObject<dim, VectorType>::restrict_and_add(const unsigned int level,
                                                              VectorType &       dst,
                                                              const VectorType & src) const
{
  this->mg_level_object[level]->restrict_and_add(level, dst, src);
}

template<int dim, typename VectorType>
void
MGTransferMF_MGLevelObject<dim, VectorType>::prolongate(const unsigned int level,
                                                        VectorType &       dst,
                                                        const VectorType & src) const
{
  this->mg_level_object[level]->prolongate(level, dst, src);
}
