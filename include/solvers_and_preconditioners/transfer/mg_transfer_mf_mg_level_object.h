#ifndef MG_TRANSFER_MF_MG_LEVEL_OBJECT
#define MG_TRANSFER_MF_MG_LEVEL_OBJECT

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
};

struct MGLevelIdentifier
{
  MGLevelIdentifier(unsigned int level, unsigned int degree, bool is_dg)
    : level(level), degree(degree), is_dg(is_dg), id(degree, is_dg)
  {
  }
  MGLevelIdentifier(unsigned int level, MGDofHandlerIdentifier p)
    : level(level), degree(p.degree), is_dg(p.is_dg), id(p)
  {
  }

  unsigned int           level;
  unsigned int           degree;
  bool                   is_dg;
  MGDofHandlerIdentifier id;
};

template<typename VectorType>
class MGTransferMF_MGLevelObject : virtual public MGTransferBase<VectorType>
{
public:
  template<int dim, typename MultigridNumber, typename Operator>
  void
  reinit(const int                                               n_components,
         const int                                               rank,
         std::vector<MGLevelIdentifier> &                        global_levels,
         std::vector<MGDofHandlerIdentifier> &                   p_levels,
         MGLevelObject<std::shared_ptr<Operator>> &              mg_matrices,
         MGLevelObject<std::shared_ptr<const DoFHandler<dim>>> & mg_dofhandler,
         MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &     mg_constrained_dofs);

  virtual void
  restrict_and_add(const unsigned int level, VectorType & dst, const VectorType & src) const;

  virtual void
  prolongate(const unsigned int level, VectorType & dst, const VectorType & src) const;

private:
  MGLevelObject<std::shared_ptr<MGTransferBase<VectorType>>> mg_level_object;
};

#include "mg_transfer_mf_mg_level_object.cpp"

#endif