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
  static void test(Operator &op, ConvergenceTable & convergence_table, 
          bool do_sm_vs_d=true, bool do_sm_vs_mf=true, 
          bool /*do_mf_vs_d*/=true, bool /*do_mf_vs_b*/=true) {
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
    
    if(do_sm_vs_d){
        // create temporal vector for diagonal of sparse matrix
        VNumber vec_diag_sm;
        op.initialize_dof_vector(vec_diag_sm);
        
        // extract diagonal from sparse matrix
        for (auto i : system_matrix)
            if(i.row() == i.column())
                vec_diag_sm[i.row()] = i.value();
        
        // print l2-norms
        print_l2(convergence_table, vec_diag, vec_diag_sm, 
                "(D)_L2", "(D-D(S))_L2");
    }
    
    if(do_sm_vs_mf){
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
        
        // print l2-norms
        print_l2(convergence_table, vec_dst_sm, vec_dst_mf, 
                "(S*v)_L2", "(S*v-MF*v)_L2");
    }
  }
  
  
  template<typename vector_type>
  static void print_l2(ConvergenceTable & convergence_table, 
    vector_type& vec_1, vector_type& vec_2,
    std::string label_1, std::string label_2) {
        auto vec_temp = vec_1;
        // compute L2-norm of vector
        if(label_1!=""){
          convergence_table.add_value(label_1, vec_temp.l2_norm());
          convergence_table.set_scientific(label_1, true);
        }
        
        // compute error and ...
        vec_temp -= vec_2;
        // ... its norm
        if(label_2!=""){
          convergence_table.add_value(label_2, vec_temp.l2_norm());
          convergence_table.set_scientific(label_2, true);
        }
  }
  
};

#endif