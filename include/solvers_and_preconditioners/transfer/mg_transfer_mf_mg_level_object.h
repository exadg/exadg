#ifndef MG_TRANSFER_MF_MG_LEVEL_OBJECT
#define MG_TRANSFER_MF_MG_LEVEL_OBJECT

#include "mg_transfer_mf.h"

struct MGDofHandlerIdentifier
{
  MGDofHandlerIdentifier(unsigned int degree, bool is_dg) : degree(degree), is_dg(is_dg)
  {
  }
  unsigned int degree;
  bool         is_dg;

  bool
  operator<(const MGDofHandlerIdentifier & rhs) const
  {
    return !((degree >= rhs.degree) && (is_dg >= rhs.is_dg));
  }

  bool
  operator==(const MGDofHandlerIdentifier & rhs) const
  {
    return (degree == rhs.degree) && (is_dg == rhs.is_dg);
  }
};

struct MGLevelInfo
{
  MGLevelInfo(unsigned int level, unsigned int degree, bool is_dg)
    : level(level), degree(degree), is_dg(is_dg), id(degree, is_dg)
  {
  }
  MGLevelInfo(unsigned int level, MGDofHandlerIdentifier p)
    : level(level), degree(p.degree), is_dg(p.is_dg), id(p)
  {
  }

  unsigned int           level;
  unsigned int           degree;
  bool                   is_dg;
  MGDofHandlerIdentifier id;
};

template<int dim, typename VectorType>
class MGTransferMF_MGLevelObject : virtual public MGTransferMF<VectorType>
{
public:
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

#include "mg_transfer_mf_mg_level_object.cpp"

#endif