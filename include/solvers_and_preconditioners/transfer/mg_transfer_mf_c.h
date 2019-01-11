#ifndef DG_CG_TRANSFER
#define DG_CG_TRANSFER

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_base.h>

#include "mg_transfer_mf.h"

using namespace dealii;

template<int dim,
         typename Number,
         typename VectorType = LinearAlgebra::distributed::Vector<Number>,
         int components      = 1>
class MGTransferMFC : virtual public MGTransferMF<VectorType>
{
public:
  typedef MatrixFree<dim, Number> MF;

  MGTransferMFC(const MF &                        data_dg,
                const MF &                        data_cg,
                const AffineConstraints<double> & cm_dg,
                const AffineConstraints<double> & cm_cg,
                const unsigned int                level,
                const unsigned int                fe_degree);

  virtual ~MGTransferMFC();

  void
  interpolate(const unsigned int level, VectorType & dst, const VectorType & src) const;

  void
  restrict_and_add(const unsigned int /*level*/, VectorType & dst, const VectorType & src) const;

  void
  prolongate(const unsigned int /*level*/, VectorType & dst, const VectorType & src) const;

private:
  template<int degree>
  void
  do_interpolate(VectorType & dst, const VectorType & src) const;

  template<int degree>
  void
  do_restrict_and_add(VectorType & dst, const VectorType & src) const;

  template<int degree>
  void
  do_prolongate(VectorType & dst, const VectorType & src) const;

  const unsigned int      fe_degree;
  MatrixFree<dim, Number> data_composite;
};

#endif