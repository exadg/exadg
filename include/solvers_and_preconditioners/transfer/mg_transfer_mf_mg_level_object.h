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

template<typename VectorTypeMG>
class MGTransferMF_MGLevelObject
{
public:
  MGLevelObject<std::shared_ptr<MGTransferBase<VectorTypeMG>>> mg_level_object;
};

#endif