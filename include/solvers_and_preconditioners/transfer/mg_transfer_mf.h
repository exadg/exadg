#ifndef MG_TRANSFER_MF
#define MG_TRANSFER_MF

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_base.h>

using namespace dealii;

template<typename VectorType>
class MGTransferMF //: virtual public MGTransferBase<VectorType>
{
public:
  virtual void
  interpolate(const unsigned int level, VectorType & dst, const VectorType & src) const = 0;

  virtual void
  restrict_and_add(const unsigned int level, VectorType & dst, const VectorType & src) const = 0;

  virtual void
  prolongate(const unsigned int level, VectorType & dst, const VectorType & src) const = 0;
};

#endif