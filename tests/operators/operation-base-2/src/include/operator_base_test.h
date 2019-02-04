#ifndef OPERATION_BASE_TEST_H
#define OPERATION_BASE_TEST_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>

#include "../../../operation-base-util/interpolate.h"

template<int dim>
class TestSolution : public Function<dim>
{
public:
  TestSolution(const double time = 0.) : Function<dim>(1, time), wave_number(1.)
  {
  }

  virtual double
  value(const Point<dim> & p, const unsigned int = 0) const
  {
    double result = std::sin(wave_number * p[0] * numbers::PI);
    for(unsigned int d = 1; d < dim; ++d)
      result *= std::sin(wave_number * p[d] * numbers::PI);
    return result;
  }

private:
  const double wave_number;
};

class OperatorBaseTest
{
public:
  template<typename Operator>
  static void
  test(Operator & operotor)
  {
    // create table
    ConvergenceTable convergence_table;

    // run test
    OperatorBaseTest::test<Operator>(operotor, convergence_table);

    // print table
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      convergence_table.write_text(std::cout);
  }

  /**
   * Following comparisons are performed:
   *
   *  -----------------+---+---+---+---+
   *  matrix-free      |   |   |   |   |
   *  diag             | 2 |   |   |   |
   *  block diagonal   | 4 |   |   |   |
   *  sparse matrix    | 3 | 1 |   |   |
   *  -----------------+---+---+---+---+
   *                     MF  D   BD  SM
   *
   * The numbers indicate the sequence of the tests.
   *
   * @param op
   * @param convergence_table
   */
  template<typename Operator>
  static void
  test(Operator &         op,
       ConvergenceTable & convergence_table,
       bool               do_sm_vs_d  = true,
       bool               do_sm_vs_mf = true,
       bool               do_mf_vs_d  = true,
       bool               do_mf_vs_b  = true)
  {
    typedef typename Operator::VectorType VectorType;
    const int                             dim = Operator::DIM;

    const auto & data        = op.do_get_data();
    auto &       dof_handler = data.get_dof_handler(/*TODO*/);
    const int    fe_degree   = dof_handler.get_fe(/*TODO*/).degree;

#ifdef DEAL_II_WITH_TRILINOS
    // compute system matrix
    TrilinosWrappers::SparseMatrix system_matrix;
    op.do_init_system_matrix(system_matrix);
    op.do_calculate_system_matrix(system_matrix);
#endif

    // compute diagonal
    VectorType vec_diag;
    op.calculate_diagonal(vec_diag);

    convergence_table.add_value("dim", dim);
    convergence_table.add_value("degree", fe_degree);
    convergence_table.add_value("dofs", vec_diag.size());

#ifdef DEAL_II_WITH_TRILINOS
    if(do_sm_vs_d)
    {
      // create temporal vector for diagonal of sparse matrix
      VectorType vec_diag_sm;
      op.do_initialize_dof_vector(vec_diag_sm);

      // extract diagonal from sparse matrix
      auto p = system_matrix.local_range();
      for(unsigned int i = p.first; i < p.second; i++)
        vec_diag_sm[i] = system_matrix(i, i);

      // print l2-norms
      print_l2(convergence_table, vec_diag, vec_diag_sm, "(D)_L2", "(D-D(S))_L2");
    }
#else
    (void)do_sm_vs_d;
#endif

    if(do_mf_vs_d)
    {
      // initialize vectors
      VectorType vec_src, vec_diag_mf;
      op.do_initialize_dof_vector(vec_src);
      op.do_initialize_dof_vector(vec_diag_mf);

      // fill source vector
      vec_src = 1;

      // apply block-diagonal matrix of size: 1 x 1
      apply_block(op, 1, vec_diag_mf, vec_src);

      // auto local_range = vec_diag_mf.local_range();
      // auto & conatraint_matrix = op.get_constraint_matrix();
      //// set diagonal to 1.0 if necessary
      // for(int i = local_range.first; i < local_range.second; i++)
      //  if(vec_diag_mf[i] == 0.0/* || conatraint_matrix.is_constrained(i)*/)
      //    vec_diag_mf[i] == 1.0;
      for(auto i : op.do_get_data().get_constrained_dofs())
        vec_diag_mf.local_element(i) = 1.0;

      // print l2-norms
      print_l2(convergence_table, vec_diag, vec_diag_mf, "", "(D-D(F))_L2");
    }

#ifdef DEAL_II_WITH_TRILINOS
    if(do_sm_vs_mf)
    {
      // initialize vectors
      VectorType vec_src, vec_dst_sm, vec_dst_mf;
      op.do_initialize_dof_vector(vec_src);
      op.do_initialize_dof_vector(vec_dst_sm);
      op.do_initialize_dof_vector(vec_dst_mf);

      // fill source vector
      MGTools::interpolate(dof_handler, TestSolution<dim>(0), vec_src, op.get_level());

      // perform vmult with system matrix
      system_matrix.vmult(vec_dst_sm, vec_src);
      // perform matirx-free vmult
      op.apply(vec_dst_mf, vec_src);

      // print l2-norms
      print_l2(convergence_table, vec_dst_sm, vec_dst_mf, "(S*v)_L2", "(S*v-F*v)_L2");
    }
#else
    (void)do_sm_vs_mf;
#endif

    // TODO: Block-Jacobi currently not working
    if(do_mf_vs_b)
    {
      // initialize vectors
      VectorType vec_src, vec_dst_mf, vec_dst_op;
      op.do_initialize_dof_vector(vec_src);
      op.do_initialize_dof_vector(vec_dst_mf);
      op.do_initialize_dof_vector(vec_dst_op);

      // fill source vector
      MGTools::interpolate(dof_handler, TestSolution<dim>(0), vec_src, op.get_level());

      // perform block-jacobi with operator
      op.calculate_block_diagonal_matrices();
      op.apply_block_diagonal_matrix_based(vec_dst_op, vec_src);

      // apply block-diagonal matrix of size: dofs_per_cell x dofs_per_cell
      const unsigned int dofs_per_cell = std::pow(fe_degree + 1, dim);
      apply_block(op, dofs_per_cell, vec_dst_mf, vec_src);

      // print l2-norms
      print_l2(convergence_table, vec_dst_op, vec_dst_mf, "(B*v)_L2", "(B*v-B(S)*v)_L2");
    }
  }

private:
  template<typename Operator, typename VectorType>
  static void
  apply_block(Operator &         op,
              const unsigned int dofs_per_block,
              VectorType &       vec_dst,
              VectorType &       vec_src)
  {
    // initialize temporal vectors
    VectorType vec_src_temp, vec_dst_temp;
    vec_src_temp.reinit(vec_src);
    vec_dst_temp.reinit(vec_src);

    const unsigned int n_blocks = vec_src.size() / dofs_per_block;
    // local range
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    auto local_range = vec_src.local_range();
#pragma GCC diagnostic pop

    const unsigned int loc_start = local_range.first;
    const unsigned int loc_end   = local_range.second;
    // iterate over all block diagonals
    for(unsigned int block = 0; block < n_blocks; block++)
    {
      // compute intersection of ranges:
      //    b_s                   b_e
      //     +---------------------+                  -> block
      //           +---------------------------+      -> local range
      //          l_s                         l_e
      //
      //           +---------------+                  -> intersection
      //      max(l_s,b_l)    min(l_e,b_e)
      const unsigned int block_start = block * dofs_per_block;
      const unsigned int block_end   = (block + 1) * dofs_per_block;
      const unsigned int start       = std::max(loc_start, block_start);
      const unsigned int end         = std::min(loc_end, block_end);

      // clear entries in source vector and ...
      vec_src_temp = 0;
      // only set the entries corresponding to the current block
      for(unsigned int i = start; i < end; i++)
        vec_src_temp[i] = vec_src[i];
      // perform vmult
      op.apply(vec_dst_temp, vec_src_temp);
      // extract result
      for(unsigned int i = start; i < end; i++)
        vec_dst[i] = vec_dst_temp[i];
    }
  }

  template<typename vector_type>
  static void
  print_l2(ConvergenceTable & convergence_table,
           vector_type &      vec_1,
           vector_type &      vec_2,
           std::string        label_1,
           std::string        label_2)
  {
    auto vec_temp = vec_1;
    // compute L2-norm of vector
    if(label_1 != "")
    {
      convergence_table.add_value(label_1, vec_temp.l2_norm());
      convergence_table.set_scientific(label_1, true);
    }

    // compute error and ...
    vec_temp -= vec_2;
    // vec_temp.print(std::cout);exit(0);
    // ... its norm
    if(label_2 != "")
    {
      convergence_table.add_value(label_2, vec_temp.l2_norm());
      convergence_table.set_scientific(label_2, true);
    }
  }
};

#endif