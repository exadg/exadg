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
  interpolate(const unsigned int level, VectorType & dst, const VectorType & src) const
  {
    (void)level;
    (void)dst;
    (void)src;

    AssertThrow(false, ExcMessage("MGTransferMF::interpolate(): should not be called!"));
  }

  virtual void
  restrict_and_add(const unsigned int level, VectorType & dst, const VectorType & src) const
  {
    (void)level;
    (void)dst;
    (void)src;

    AssertThrow(false, ExcMessage("MGTransferMF::interpolate(): should not be called!"));
  }

  virtual void
  prolongate(const unsigned int level, VectorType & dst, const VectorType & src) const
  {
    (void)level;
    (void)dst;
    (void)src;

    AssertThrow(false, ExcMessage("MGTransferMF::interpolate(): should not be called!"));
  }
};

#endif