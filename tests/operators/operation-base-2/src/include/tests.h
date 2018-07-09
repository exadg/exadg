#ifndef TESTS
#define TESTS

template <typename value_type> struct Tests {
  typedef LinearAlgebra::distributed::Vector<value_type> Vector;
  typedef TrilinosWrappers::SparseMatrix SMatrix;
  typedef std::vector<LAPACKFullMatrix<value_type>> BMatrix;

  static bool test_sm_vs_mf(Vector &vec_mf, Vector &vec_sm) {
    std::cout << "Norm 1: " << vec_mf.l2_norm() << std::endl;
    std::cout << "Norm 2: " << vec_sm.l2_norm() << std::endl;
    vec_sm -= vec_mf;
    std::cout << "Error:  " << vec_sm.l2_norm() << std::endl;

    return true;
  }

  static bool test_sm_vs_diag(SMatrix &sm, Vector &vec_diag) {

    // create new vector and set it zero
    Vector sm_diag;
    sm_diag.reinit(vec_diag);

    // extract diagonal from sparse matrix
    for (unsigned int i = 0; i < sm.m() && i < sm.n(); i++)
      sm_diag[i] = sm(i, i);

    std::cout << "Norm 1: " << vec_diag.l2_norm() << std::endl;
    std::cout << "Norm 2: " << sm_diag.l2_norm() << std::endl;
    sm_diag -= vec_diag;
    std::cout << "Error:  " << sm_diag.l2_norm() << std::endl;

    return true;
  }

  template<typename OPERATOR>
  static bool test_block_diag(Vector &  vec_src, Vector &vec_dst_3,
    OPERATOR& laplace, const int dim, const int fe_degree) {

        LinearAlgebra::distributed::Vector<value_type> vec_src_temp;
        LinearAlgebra::distributed::Vector<value_type> vec_dst_temp;
        LinearAlgebra::distributed::Vector<value_type> vec_dst_5;
        vec_src_temp.reinit(vec_src);
        vec_dst_temp.reinit(vec_src);
        vec_dst_5.reinit(vec_src);
        
        unsigned const int delta = std::pow(fe_degree+1,dim);
        for(unsigned int start = 0; start < vec_src.size(); start += delta){
            vec_src_temp = 0;
            for(unsigned int i = 0; i < delta; i++)
                vec_src_temp[start + i] = vec_src[start + i];
            laplace.vmult(vec_dst_temp, vec_src_temp);
            for(unsigned int i = 0; i < delta; i++)
                vec_dst_5[start + i] = vec_dst_temp[start + i];
        }
        test_sm_vs_mf(vec_dst_5, vec_dst_3);
      
    return true;
  }
  
  
};

#endif