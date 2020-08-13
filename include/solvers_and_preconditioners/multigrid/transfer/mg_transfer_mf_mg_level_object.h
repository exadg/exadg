#ifndef MG_TRANSFER_MF_MG_LEVEL_OBJECT
#define MG_TRANSFER_MF_MG_LEVEL_OBJECT

#include "mg_transfer_mf.h"

#include "../levels_hybrid_multigrid.h"

namespace ExaDG
{
using namespace dealii;

template<int dim, typename VectorType>
class MGTransferMF_MGLevelObject : virtual public MGTransferMF<VectorType>
{
public:
  virtual ~MGTransferMF_MGLevelObject()
  {
  }

  template<typename MultigridNumber, typename MatrixFree, typename Constraints>
  void
  reinit(MGLevelObject<std::shared_ptr<MatrixFree>> &        mg_data,
         MGLevelObject<std::shared_ptr<Constraints>> &       mg_Constraints,
         MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> & mg_constrained_dofs,
         const unsigned int                                  dof_handler_index = 0);

  virtual void
  interpolate(const unsigned int level, VectorType & dst, const VectorType & src) const;

  virtual void
  restrict_and_add(const unsigned int level, VectorType & dst, const VectorType & src) const;

  virtual void
  prolongate(const unsigned int level, VectorType & dst, const VectorType & src) const;

private:
  MGLevelObject<std::shared_ptr<MGTransferMF<VectorType>>> mg_level_object;
};

} // namespace ExaDG

#include "mg_transfer_mf_mg_level_object.cpp"

#endif
