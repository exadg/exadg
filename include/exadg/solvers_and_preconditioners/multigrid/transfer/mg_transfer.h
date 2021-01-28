#ifndef MG_TRANSFER_MF
#define MG_TRANSFER_MF

namespace ExaDG
{
template<typename VectorType>
class MGTransfer
{
public:
  virtual ~MGTransfer()
  {
  }

  virtual void
  interpolate(const unsigned int level, VectorType & dst, const VectorType & src) const = 0;

  virtual void
  restrict_and_add(const unsigned int level, VectorType & dst, const VectorType & src) const = 0;

  virtual void
  prolongate(const unsigned int level, VectorType & dst, const VectorType & src) const = 0;
};
} // namespace ExaDG

#endif
