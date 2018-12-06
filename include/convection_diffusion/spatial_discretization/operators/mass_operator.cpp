#include "mass_operator.h"

namespace ConvDiff
{
template<int dim, int degree, typename Number>
void
MassMatrixOperator<dim, degree, Number>::do_cell_integral(FEEvalCell & fe_eval,
                                                          unsigned int const /*cell*/) const
{
  for(unsigned int q = 0; q < fe_eval.n_q_points; ++q)
    fe_eval.submit_value(fe_eval.get_value(q), q);
}

} // namespace ConvDiff

#include "mass_operator.hpp"
