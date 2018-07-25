#ifndef DG_CG_TRANSFER
#define DG_CG_TRANSFER

#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

template <int dim, typename Number> class CGToDGTransfer {
public:
  typedef LinearAlgebra::distributed::Vector<Number> VNumber;
  typedef MatrixFree<dim, Number> MF;

  CGToDGTransfer(const MF &data_dg, const MF &data_cg, const unsigned int level,
                 const unsigned int fe_degree);

  virtual ~CGToDGTransfer();

  void toCG(VNumber &dst, const VNumber &src) const;

  void toDG(VNumber &dst, const VNumber &src) const;

private:
  void transfer(VNumber &dst, const VNumber &src, const MF &data_dst,
                const MF &data_src) const;

  const MF &data_dg;
  const MF &data_cg;
  const unsigned int level;
  mutable Vector<Number> temp_src;
  mutable Vector<Number> temp_dst;
};

#endif