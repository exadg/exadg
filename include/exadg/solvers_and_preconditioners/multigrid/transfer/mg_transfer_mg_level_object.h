#ifndef MG_TRANSFER_MF_MG_LEVEL_OBJECT
#define MG_TRANSFER_MF_MG_LEVEL_OBJECT

// deal.II
#include <deal.II/base/mg_level_object.h>
#include <deal.II/fe/mapping.h>
#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/multigrid/mg_constrained_dofs.h>

// ExaDG
#include <exadg/solvers_and_preconditioners/multigrid/transfer/mg_transfer.h>

namespace ExaDG
{
using namespace dealii;

template<int dim, typename Number, typename VectorType = LinearAlgebra::distributed::Vector<Number>>
class MGTransfer_MGLevelObject : virtual public MGTransfer<VectorType>
{
public:
  virtual ~MGTransfer_MGLevelObject()
  {
  }

  void
  reinit(const Mapping<dim> &                                        mapping,
         MGLevelObject<std::shared_ptr<MatrixFree<dim, Number>>> &   mg_matrixfree,
         MGLevelObject<std::shared_ptr<AffineConstraints<double>>> & mg_constraints,
         MGLevelObject<std::shared_ptr<MGConstrainedDoFs>> &         mg_constrained_dofs,
         const unsigned int                                          dof_handler_index = 0);

  virtual void
  interpolate(const unsigned int level, VectorType & dst, const VectorType & src) const;

  virtual void
  restrict_and_add(const unsigned int level, VectorType & dst, const VectorType & src) const;

  virtual void
  prolongate(const unsigned int level, VectorType & dst, const VectorType & src) const;

private:
  MGLevelObject<std::shared_ptr<MGTransfer<VectorType>>> mg_level_object;
};

} // namespace ExaDG

#endif
