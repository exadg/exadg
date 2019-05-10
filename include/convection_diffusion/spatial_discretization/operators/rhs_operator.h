#ifndef INCLUDE_CONVECTION_DIFFUSION_RHS
#define INCLUDE_CONVECTION_DIFFUSION_RHS

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/matrix_free/matrix_free.h>

using namespace dealii;

namespace ConvDiff
{
template<int dim>
struct RHSOperatorData
{
  RHSOperatorData() : dof_index(0), quad_index(0)
  {
  }

  unsigned int dof_index;
  unsigned int quad_index;

  std::shared_ptr<Function<dim>> rhs;
};

template<int dim, typename Number>
class RHSOperator
{
private:
  typedef LinearAlgebra::distributed::Vector<Number> VectorType;

  typedef RHSOperator<dim, Number> This;

  typedef VectorizedArray<Number> scalar;

  typedef std::pair<unsigned int, unsigned int> Range;

public:
  /*
   * Constructor.
   */
  RHSOperator();

  /*
   * Initialization.
   */
  void
  reinit(MatrixFree<dim, Number> const & matrix_free_in,
         RHSOperatorData<dim> const &    operator_data_in);

  /*
   * Evaluate operator and overwrite dst-vector.
   */
  void
  evaluate(VectorType & dst, double const evaluation_time) const;

  /*
   * Evaluate operator and add to dst-vector.
   */
  void
  evaluate_add(VectorType & dst, double const evaluation_time) const;

private:
  template<typename Integrator>
  void
  do_cell_integral(Integrator & integrator) const;

  /*
   * The right-hand side operator involves only cell integrals so we only need a function looping
   * over all cells and computing the cell integrals.
   */
  void
  cell_loop(MatrixFree<dim, Number> const & matrix_free,
            VectorType &                    dst,
            VectorType const & /*src*/,
            Range const & cell_range) const;

  MatrixFree<dim, Number> const * matrix_free;

  RHSOperatorData<dim> operator_data;

  double mutable eval_time;
};

} // namespace ConvDiff

#endif
