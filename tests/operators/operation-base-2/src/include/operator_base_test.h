#ifndef OPERATION_BASE_TEST_H
#define OPERATION_BASE_TEST_H

#include <deal.II/base/conditional_ostream.h>

class OperatorBaseTest {
public:
  template <typename Operator> static void test(Operator &operotor) {

    // create table
    ConvergenceTable convergence_table;

    // run test
    OperatorBaseTest::test<Operator>(operotor, convergence_table);

    // print table
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
        convergence_table.write_text(std::cout);
  }

private:
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
  template <typename Operator>
  static void test(Operator &op, ConvergenceTable & convergence_table) {
    typedef typename Operator::VNumber VNumber;
    
    // compute system matrix
    TrilinosWrappers::SparseMatrix system_matrix;
    op.init_system_matrix(system_matrix);
    op.calculate_system_matrix(system_matrix);
    
    // compute diagonal
    VNumber vec_diag;
    op.calculate_diagonal(vec_diag);
    
    if(true){
        // create temporal vector for diagonal of sparse matrix
        VNumber vec_diag_sm;
        op.initialize_dof_vector(vec_diag_sm);
        
        // extract diagonal from sparse matrix
        for (auto i : system_matrix)
            if(i.row() == i.column())
                vec_diag_sm[i.row()] = i.value();
        
        convergence_table.add_value("D_L2", vec_diag.l2_norm());
        convergence_table.set_scientific("D_L2", true);
        vec_diag_sm -= vec_diag;
        convergence_table.add_value("(D-V)_L2", vec_diag_sm.l2_norm());
        convergence_table.set_scientific("(D-V)_L2", true);
    }
  }
};

#endif