#ifndef DG_CG_TRANSFER
#define DG_CG_TRANSFER

#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

template<int dim, typename Number>
class CGToDGTransfer
{
public:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;
  typedef MatrixFree<dim, Number>                    MF;

  CGToDGTransfer(const MF &         data_dg,
                 const MF &         data_cg,
                 const unsigned int level,
                 const unsigned int fe_degree);

  virtual ~CGToDGTransfer();

  void
  toCG(VectorType & dst, const VectorType & src) const;

  void
  toDG(VectorType & dst, const VectorType & src) const;

private:
  void
  transfer(VectorType & dst, const VectorType & src, const MF & data_dst, const MF & data_src) const;

  const MF &         data_dg;
  const MF &         data_cg;
  const unsigned int level;

  std::vector<types::global_dof_index> dof_indices_cg;
  std::vector<types::global_dof_index> dof_indices_dg;
};

#endif