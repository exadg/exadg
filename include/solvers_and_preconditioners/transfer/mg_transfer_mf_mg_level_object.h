#ifndef MG_TRANSFER_MF_MG_LEVEL_OBJECT
#define MG_TRANSFER_MF_MG_LEVEL_OBJECT

#include "mg_transfer_mf.h"

struct MGDoFHandlerIdentifier
{
  MGDoFHandlerIdentifier(unsigned int degree, bool is_dg) : degree(degree), is_dg(is_dg)
  {
  }

  bool
  operator<(const MGDoFHandlerIdentifier & other) const
  {
    return !((degree >= other.degree) && (is_dg >= other.is_dg));
  }

  bool
  operator==(const MGDoFHandlerIdentifier & other) const
  {
    return (degree == other.degree) && (is_dg == other.is_dg);
  }

  unsigned int degree;
  bool         is_dg;
};

struct MGLevelInfo
{
  MGLevelInfo(unsigned int h_level, unsigned int degree, bool is_dg)
    : _h_level(h_level), _dof_handler_id(degree, is_dg)
  {
  }
  MGLevelInfo(unsigned int h_level, MGDoFHandlerIdentifier dof_handler_id)
    : _h_level(h_level), _dof_handler_id(dof_handler_id)
  {
  }

  unsigned int
  h_level() const
  {
    return _h_level;
  }

  unsigned int
  degree() const
  {
    return _dof_handler_id.degree;
  }

  bool
  is_dg() const
  {
    return _dof_handler_id.is_dg;
  }

  MGDoFHandlerIdentifier
  dof_handler_id() const
  {
    return _dof_handler_id;
  }

private:
  unsigned int           _h_level;
  MGDoFHandlerIdentifier _dof_handler_id;
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
