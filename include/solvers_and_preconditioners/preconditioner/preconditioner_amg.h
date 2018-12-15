#ifndef PRECONDITIONER_AMG
#define PRECONDITIONER_AMG

#include <deal.II/lac/la_parallel_vector.h>
#include <deal.II/lac/trilinos_precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include "../../functionalities/print_functions.h"
#include "../multigrid/multigrid_input_parameters.h"
#include "../preconditioner/preconditioner_base.h"


template<typename Operator, typename TrilinosNumber>
class PreconditionerAMG : public PreconditionerBase<TrilinosNumber>
{
private:
  typedef LinearAlgebra::distributed::Vector<TrilinosNumber> VectorTypeTrilinos;

  // reference to matrix-free operator
  Operator const & pde_operator;

  AMGData amg_data;

#ifdef DEAL_II_WITH_TRILINOS
public:
  // distributed sparse system matrix
  TrilinosWrappers::SparseMatrix system_matrix;
  
private:
  TrilinosWrappers::PreconditionAMG amg;
#endif

public:
  PreconditionerAMG(Operator const & op, AMGData data = AMGData()) : pde_operator(op)
  {
    amg_data = data;

    // initialize system matrix
    pde_operator.init_system_matrix(system_matrix);

    // calculate_matrix
    pde_operator.calculate_system_matrix(system_matrix);

#ifdef DEAL_II_WITH_TRILINOS
    // initialize Trilinos' AMG
    amg.initialize(system_matrix, amg_data.data);
#else
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
  }

  void
  update(LinearOperatorBase const * /*linear_operator*/)
  {
#ifdef DEAL_II_WITH_TRILINOS
    // clear content of matrix since the next calculate_system_matrix-commands add their result
    system_matrix *= 0.0;

    // re-calculate matrix
    pde_operator.calculate_system_matrix(system_matrix);

    // initialize Trilinos' AMG
    amg.initialize(system_matrix, amg_data.data);
#else
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
  }

  void
  vmult(VectorTypeTrilinos & dst, VectorTypeTrilinos const & src) const
  {
#ifdef DEAL_II_WITH_TRILINOS
    amg.vmult(dst, src);
#else
    void(dst);
    void(src);
    AssertThrow(false, ExcMessage("deal.II is not compiled with Trilinos!"));
#endif
  }
};

#endif
