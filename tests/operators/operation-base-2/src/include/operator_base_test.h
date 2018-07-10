#ifndef OPERATION_BASE_TEST_H
#define OPERATION_BASE_TEST_H

#include <deal.II/base/conditional_ostream.h>
#include <deal.II/base/function.h>


template<int dim>
class TestSolution : public Function<dim>{
  public:
  TestSolution(const double time = 0.)
      : Function<dim>(1, time), wave_number(1.) {}

  virtual double value(const Point<dim> &p, const unsigned int = 0) const {
    double result = std::sin(wave_number * p[0] * numbers::PI);
    for (unsigned int d = 1; d < dim; ++d)
      result *= std::sin((d + 1) * wave_number * p[d] * numbers::PI);
    return result;
  }

private:
  const double wave_number;
};

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
    const int dim = Operator::DIM;
    
    const auto& data = op.get_data();
    auto& dof_handler = data.get_dof_handler(/*TODO*/);
    
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
    
    if(true){
        // initialize vectors 
        VNumber vec_src, vec_dst_sm, vec_dst_mf;
        op.initialize_dof_vector(vec_src);
        op.initialize_dof_vector(vec_dst_sm);
        op.initialize_dof_vector(vec_dst_mf);
        
        // fill source vector
        VectorTools::interpolate(dof_handler, TestSolution<dim>(0), vec_src);
        
        // perform vmult with system matrix
        system_matrix.vmult(vec_dst_sm, vec_src);
        // perform matirx-free vmult
        op.vmult(vec_dst_mf, vec_src);
        
        // compute L2-norm of vector
        convergence_table.add_value("(S*v)_L2", vec_dst_sm.l2_norm());
        convergence_table.set_scientific("(S*v)_L2", true);
        
        // compute error and ...
        vec_dst_sm -= vec_dst_mf;
        // ... its norm
        convergence_table.add_value("(S*v-MF*v)_L2", vec_dst_sm.l2_norm());
        convergence_table.set_scientific("(S*v-MF*v)_L2", true);
    }
  }
  
};

#endif