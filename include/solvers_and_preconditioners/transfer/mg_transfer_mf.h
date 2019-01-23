#ifndef MG_TRANSFER_MF
#define MG_TRANSFER_MF

template<typename VectorType>
class MGTransferMF
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