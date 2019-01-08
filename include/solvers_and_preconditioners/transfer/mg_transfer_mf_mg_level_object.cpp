#include "mg_transfer_mf_c.h"
#include "mg_transfer_mf_h.h"
#include "mg_transfer_mf_p.h"


template<int dim, typename VectorType>
template<typename MultigridNumber, typename MatrixFree, typename Constraints>
void
MGTransferMF_MGLevelObject<dim, VectorType>::reinit(
  const int                                               n_components,
  const int                                               rank,
  std::vector<MGLevelIdentifier> &                        global_levels,
  std::vector<MGDofHandlerIdentifier> &                   p_levels,
  MGLevelObject<std::shared_ptr<MatrixFree>> &            mg_data,
  MGLevelObject<std::shared_ptr<Constraints>> &           mg_Constraints,
  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> & mg_dofhandler,
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &     mg_constrained_dofs)
{
  mg_level_object.resize(0, global_levels.size() - 1);

#ifndef DEBUG
  (void)rank; // avoid compiler warning
#endif

  std::map<MGDofHandlerIdentifier, std::shared_ptr<MGTransferMFH<dim, MultigridNumber>>>
    mg_tranfers_temp;
  std::map<MGDofHandlerIdentifier, std::map<unsigned int, unsigned int>>
    map_global_level_to_h_levels;

  // initialize maps so that we do not have to check existence later on
  for(auto deg : p_levels)
    map_global_level_to_h_levels[deg] = {};

  // fill the maps
  for(unsigned int i = 0; i < global_levels.size(); i++)
  {
    auto level = global_levels[i];

    map_global_level_to_h_levels[level.id][i] = level.level;
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
                                                *mg_dofhandler[global_level]));

      // dof-handlers and constrains are saved for global levels
      // so we have to convert degree to any global level which has this degree
      // (these share the same dof-handlers and constraints)
      transfer->initialize_constraints(*mg_constrained_dofs[global_level]);
      transfer->build(*mg_dofhandler[global_level]);
      mg_tranfers_temp[deg] = transfer;
    } // else: there is only one global level (and one h-level) on this p-level
  }

  // fill mg_transfer with the correct transfers
  for(unsigned int i = 1; i < global_levels.size(); i++)
  {
    auto coarse_level = global_levels[i - 1];
    auto fine_level   = global_levels[i];

    std::shared_ptr<MGTransferMF<VectorType>> temp;

    if(coarse_level.level != fine_level.level) // h-transfer
    {
#ifdef DEBUG
      if(rank == 0)
        printf("  h-MG (l=%2d,k=%2d) -> (l=%2d,k=%2d)\n",
               coarse_level.level,
               coarse_level.degree,
               fine_level.level,
               fine_level.degree);
#endif

      temp = mg_tranfers_temp[coarse_level.id]; // get the previously h-transfer operator
    }
    else if(coarse_level.degree != fine_level.degree) // p-transfer
    {
#ifdef DEBUG
      if(rank == 0)
        printf("  p-MG (l=%2d,k=%2d) -> (l=%2d,k=%2d)\n",
               coarse_level.level,
               coarse_level.degree,
               fine_level.level,
               fine_level.degree);
#endif

      if(n_components == 1)
        temp.reset(new MGTransferMFP<dim, MultigridNumber, VectorType, 1>(
          &*mg_data[i], &*mg_data[i - 1], fine_level.degree, coarse_level.degree));
      else if(n_components == dim)
        temp.reset(new MGTransferMFP<dim, MultigridNumber, VectorType, dim>(
          &*mg_data[i], &*mg_data[i - 1], fine_level.degree, coarse_level.degree));
      else
        AssertThrow(false, ExcMessage("Cannot create MGTransferMFP!"));
    }
    else if(coarse_level.is_dg != fine_level.is_dg) // c-transfer
    {
#ifdef DEBUG
      if(rank == 0)
        printf("  c-MG (l=%2d,k=%2d) -> (l=%2d,k=%2d)\n",
               coarse_level.level,
               coarse_level.degree,
               fine_level.level,
               fine_level.degree);
#endif

      if(n_components == 1)
        temp.reset(new MGTransferMFC<dim, typename MatrixFree::value_type, VectorType, 1>(
          *mg_data[i],
          *mg_data[i - 1],
          *mg_Constraints[i],
          *mg_Constraints[i - 1],
          fine_level.level,
          coarse_level.degree));
      else if(n_components == dim)
        temp.reset(new MGTransferMFC<dim, typename MatrixFree::value_type, VectorType, dim>(
          *mg_data[i],
          *mg_data[i - 1],
          *mg_Constraints[i],
          *mg_Constraints[i - 1],
          fine_level.level,
          coarse_level.degree));
      else
        AssertThrow(false, ExcMessage("Cannot create MGTransferMFP!"));
    }
    mg_level_object[i] = temp;
  }
}


template<int dim, typename VectorType>
template<typename MultigridNumber, typename Operator>
void
MGTransferMF_MGLevelObject<dim, VectorType>::reinit(
  const int                                               n_components,
  const int                                               rank,
  std::vector<MGLevelIdentifier> &                        global_levels,
  std::vector<MGDofHandlerIdentifier> &                   p_levels,
  MGLevelObject<std::shared_ptr<Operator>> &              mg_matrices,
  MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> & mg_dofhandler,
  MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &     mg_constrained_dofs)
{
  mg_level_object.resize(0, global_levels.size() - 1);

#ifndef DEBUG
  (void)rank; // avoid compiler warning
#endif

  std::map<MGDofHandlerIdentifier, std::shared_ptr<MGTransferMFH<dim, MultigridNumber>>>
    mg_tranfers_temp;
  std::map<MGDofHandlerIdentifier, std::map<unsigned int, unsigned int>>
    map_global_level_to_h_levels;

  // initialize maps so that we do not have to check existence later on
  for(auto deg : p_levels)
    map_global_level_to_h_levels[deg] = {};

  // fill the maps
  for(unsigned int i = 0; i < global_levels.size(); i++)
  {
    auto level = global_levels[i];

    map_global_level_to_h_levels[level.id][i] = level.level;
  }

  // create h-transfer operators between levels
  for(auto deg : p_levels)
  {
    if(map_global_level_to_h_levels[deg].size() > 1)
    {
      unsigned int global_level = map_global_level_to_h_levels[deg].begin()->first;
      // create actual h-transfer-operator
      std::shared_ptr<MGTransferMFH<dim, MultigridNumber>> transfer(
        new MGTransferMFH<dim, MultigridNumber>(map_global_level_to_h_levels[deg],
                                                *mg_dofhandler[global_level]));

      // dof-handlers and constrains are saved for global levels
      // so we have to convert degree to any global level which has this degree
      // (these share the same dof-handlers and constraints)
      transfer->initialize_constraints(*mg_constrained_dofs[global_level]);
      transfer->build(*mg_dofhandler[global_level]);
      mg_tranfers_temp[deg] = transfer;
    } // else: there is only one global level (and one h-level) on this p-level
  }

  // fill mg_transfer with the correct transfers
  for(unsigned int i = 1; i < global_levels.size(); i++)
  {
    auto coarse_level = global_levels[i - 1];
    auto fine_level   = global_levels[i];

    std::shared_ptr<MGTransferMF<VectorType>> temp;

    if(coarse_level.level != fine_level.level) // h-transfer
    {
#ifdef DEBUG
      if(rank == 0)
        printf("  h-MG (l=%2d,k=%2d) -> (l=%2d,k=%2d)\n",
               coarse_level.level,
               coarse_level.degree,
               fine_level.level,
               fine_level.degree);
#endif

      temp = mg_tranfers_temp[coarse_level.id]; // get the previously h-transfer operator
    }
    else if(coarse_level.degree != fine_level.degree) // p-transfer
    {
#ifdef DEBUG
      if(rank == 0)
        printf("  p-MG (l=%2d,k=%2d) -> (l=%2d,k=%2d)\n",
               coarse_level.level,
               coarse_level.degree,
               fine_level.level,
               fine_level.degree);
#endif

      if(n_components == 1)
        temp.reset(
          new MGTransferMFP<dim, MultigridNumber, VectorType, 1>(&mg_matrices[i]->get_data(),
                                                                 &mg_matrices[i - 1]->get_data(),
                                                                 fine_level.degree,
                                                                 coarse_level.degree));
      else if(n_components == dim)
        temp.reset(
          new MGTransferMFP<dim, MultigridNumber, VectorType, dim>(&mg_matrices[i]->get_data(),
                                                                   &mg_matrices[i - 1]->get_data(),
                                                                   fine_level.degree,
                                                                   coarse_level.degree));
      else
        AssertThrow(false, ExcMessage("Cannot create MGTransferMFP!"));
    }
    else if(coarse_level.is_dg != fine_level.is_dg) // c-transfer
    {
#ifdef DEBUG
      if(rank == 0)
        printf("  c-MG (l=%2d,k=%2d) -> (l=%2d,k=%2d)\n",
               coarse_level.level,
               coarse_level.degree,
               fine_level.level,
               fine_level.degree);
#endif

      if(n_components == 1)
        temp.reset(new MGTransferMFC<dim, typename Operator::value_type, VectorType, 1>(
          mg_matrices[i]->get_data(),
          mg_matrices[i - 1]->get_data(),
          mg_matrices[i]->get_constraint_matrix(),
          mg_matrices[i - 1]->get_constraint_matrix(),
          fine_level.level,
          coarse_level.degree));
      else if(n_components == dim)
        temp.reset(new MGTransferMFC<dim, typename Operator::value_type, VectorType, dim>(
          mg_matrices[i]->get_data(),
          mg_matrices[i - 1]->get_data(),
          mg_matrices[i]->get_constraint_matrix(),
          mg_matrices[i - 1]->get_constraint_matrix(),
          fine_level.level,
          coarse_level.degree));
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