#include "multigrid_preconditioner_adapter_base.h"

#include <navier-constants.h>

template <int dim, typename value_type, typename Operator>
MyMultigridPreconditionerBase<dim, value_type, Operator>::
    MyMultigridPreconditionerBase(std::shared_ptr<Operator> underlying_operator)
    : underlying_operator(underlying_operator) {}

template <int dim, typename value_type, typename Operator>
MyMultigridPreconditionerBase<dim, value_type,
                              Operator>::~MyMultigridPreconditionerBase() {}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::initialize(
    const MultigridData &mg_data_in, const DoFHandler<dim> &dof_handler,
    const Mapping<dim> &mapping,
          std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const &dirichlet_bc, void* operator_data_in) {

  // save mg-setup
  this->mg_data = mg_data_in;

  // get triangulation
  const parallel::Triangulation<dim> *tria =
      dynamic_cast<const parallel::Triangulation<dim> *>(
          &dof_handler.get_triangulation());

  // which mg-level should be processed
  const bool pgmg = this->mg_data.type == MultigridType::PGMG;
  const bool both = this->mg_data.two_levels;
  unsigned int global = tria->n_global_levels();
  unsigned int degree = dof_handler.get_fe().degree;
  
  // determine number of components
  const unsigned int n_components =  dof_handler.n_dofs() / tria->n_active_cells() / std::pow(1+degree,dim);

  std::vector<unsigned int> seq_geo, seq_deg;

  for (unsigned int i = 0; i < global; i++)
    seq_geo.push_back(i);

  unsigned int temp = degree;
  do {
    seq_deg.push_back(temp);
    temp = get_next(temp);
  } while (temp != seq_deg.back());
  std::reverse(std::begin(seq_deg), std::end(seq_deg));

  std::vector<std::pair<unsigned int, unsigned int>> seq;

  if (pgmg) {
    // top level: p-gmg
    if (both) // low level: h-gmg
      for (unsigned int i = 0; i < seq_geo.size() - 1; i++)
        seq.push_back(std::pair<int, int>(seq_geo[i], seq_deg.front()));
    for (auto deg : seq_deg)
      seq.push_back(std::pair<int, int>(seq_geo.back(), deg));
  } else {
    // top level: h-gmg
    if (both) // low level: p-gmg
      for (unsigned int i = 0; i < seq_deg.size() - 1; i++)
        seq.push_back(std::pair<int, int>(seq_geo.front(), seq_deg[i]));
    for (auto geo : seq_geo)
      seq.push_back(std::pair<int, int>(geo, seq_deg.back()));
  }

  int min_level = 0;
  int max_level = seq.size() - 1;
  this->n_global_levels = seq.size();

  this->mg_constrained_dofs.resize(min_level, max_level);
  this->mg_matrices.resize(min_level, max_level);
  this->mg_smoother.resize(min_level, max_level);
  this->mg_dofhandler.resize(min_level, max_level);
  this->mg_transfer.resize(min_level, max_level);

  // setup dof-handler and constrained dofs for each level
  for (unsigned int deg : seq_deg) {
    // setup dof_handler: create dof_handler...
    auto dof_handler = new DoFHandler<dim>(*tria);
    // ... create FE and distribute on all mg-levels
    dof_handler->distribute_dofs(FESystem<dim>(FE_DGQ<dim>(deg),n_components));
    dof_handler->distribute_mg_dofs();
    // setup constrained dofs:
    auto constrained_dofs = new MGConstrainedDoFs();
    constrained_dofs->clear();
    this->initialize_mg_constrained_dofs(*dof_handler, *constrained_dofs, dirichlet_bc);

    // populate dofhandler and constrained dofs all levels with the same degree
    std::shared_ptr<const DoFHandler<dim>> temp_dofh(dof_handler);
    std::shared_ptr<MGConstrainedDoFs> temp_constraint(constrained_dofs);
    for (unsigned int i = 0; i < seq.size(); i++)
      if (seq[i].second == deg) {
        mg_dofhandler[i] = temp_dofh;
        mg_constrained_dofs[i] = temp_constraint;
      }
  }

  // create and setup operator on each level
  for (int i = min_level; i <= max_level; i++) {
    // TODO: remove static cast
    auto matrix =
        static_cast<Operator *>(underlying_operator->get_new(seq[i].second));
    matrix->reinit(*mg_dofhandler[i], mapping, operator_data_in, 
                   *this->mg_constrained_dofs[i], seq[i].first);
    mg_matrices[i].reset(matrix);

    if (i == min_level) {
      if(mg_data_in.coarse_solver == MultigridCoarseGridSolver::AMG_ML){
        // create coarse matrix with fe_q
        auto dof_handler_q = new DoFHandler<dim>(*tria);
        dof_handler_q->distribute_dofs(FE_Q<dim>(seq[i].second));
        dof_handler_q->distribute_mg_dofs();
        this->cg_dofhandler.reset(dof_handler_q);
  
        auto constrained_dofs_q = new MGConstrainedDoFs();
        constrained_dofs_q->clear();
        this->initialize_mg_constrained_dofs(*dof_handler_q, *constrained_dofs_q, dirichlet_bc);
        this->cg_constrained_dofs.reset(constrained_dofs_q);
  
        // TODO: remove static cast
        auto matrix_q =
            static_cast<Operator *>(underlying_operator->get_new(seq[i].second));
        matrix_q->reinit(*dof_handler_q, mapping, operator_data_in, 
                         *this->cg_constrained_dofs, seq[i].first);
        this->cg_matrices.reset(matrix_q);
      }

      // create coarse solver with coarse matrix fe_q and fe_dgq
      this->initialize_coarse_solver(*matrix, *this->cg_matrices, seq[i].first);
    } else{
      this->initialize_smoother(*matrix, i);
    }
  }

  // setup transfer for h-gmg
  for (unsigned int deg : seq_deg) {
    std::map<unsigned int, unsigned int> m;

    for (unsigned int i = 1; i < seq.size(); i++) {
      auto prev = seq[i - 1];
      auto curr = seq[i];
      if (prev.first != curr.first && deg == prev.second &&
          deg == curr.second) {
        printf("  h-gmg (%d,%d) -> (%d,%d)\n", prev.first, prev.second,
               curr.first, curr.second);
        m[i] = curr.first;
      }
    }

    if (m.empty())
      continue;

    std::shared_ptr<MGTransferMF<dim, value_type_operator>> transfer(
        new MGTransferMF<dim, value_type_operator>(m));
    transfer->initialize_constraints(
        *mg_constrained_dofs[m.begin()->first]);
    transfer->build(*mg_dofhandler[m.begin()->first]);

    for (auto i : m) 
      mg_transfer[i.first] = transfer;
  }

  // setup transfer for p-gmg
  for (unsigned int i = 1; i < seq.size(); i++) {
    auto prev = seq[i - 1];
    auto curr = seq[i];
    if (prev.second != curr.second) {
      printf("  p-gmg (%d,%d) -> (%d,%d)\n", prev.first, prev.second, curr.first,
             curr.second);
      MGTransferBase<VECTOR_TYPE> *temp;

      const unsigned int from = curr.second, to = prev.second;
      
#if DEGREE_9 && DEGREE_4
      if (from == 9 && to == 4) {
        temp = new MGTransferMatrixFreeP<dim, 9, 4, value_type_operator, VECTOR_TYPE>( *mg_dofhandler[i], *mg_dofhandler[i - 1], curr.first);
      } else 
#endif
#if DEGREE_8 && DEGREE_4
      if (from == 8 && to == 4) {
        temp = new MGTransferMatrixFreeP<dim, 8, 4, value_type_operator, VECTOR_TYPE>( *mg_dofhandler[i], *mg_dofhandler[i - 1], curr.first);
      } else 
#endif
#if DEGREE_7 && DEGREE_3
      if (from == 7 && to == 3) {
        temp = new MGTransferMatrixFreeP<dim, 7, 3, value_type_operator, VECTOR_TYPE>( *mg_dofhandler[i], *mg_dofhandler[i - 1], curr.first);
      } else 
#endif
#if DEGREE_6 && DEGREE_3
      if (from == 7 && to == 3) {
        temp = new MGTransferMatrixFreeP<dim, 6, 3, value_type_operator, VECTOR_TYPE>( *mg_dofhandler[i], *mg_dofhandler[i - 1], curr.first);
      } else 
#endif
#if DEGREE_5 && DEGREE_2
      if (from == 5 && to == 2) {
        temp = new MGTransferMatrixFreeP<dim, 5, 2, value_type_operator, VECTOR_TYPE>( *mg_dofhandler[i], *mg_dofhandler[i - 1], curr.first);
      } else 
#endif
#if DEGREE_4 && DEGREE_2
      if (from == 4 && to == 2) {
        temp = new MGTransferMatrixFreeP<dim, 4, 2, value_type_operator, VECTOR_TYPE>( *mg_dofhandler[i], *mg_dofhandler[i - 1], curr.first);
      } else 
#endif
#if DEGREE_3 && DEGREE_1
      if (from == 3 && to == 1) {
        temp = new MGTransferMatrixFreeP<dim, 3, 1, value_type_operator, VECTOR_TYPE>( *mg_dofhandler[i], *mg_dofhandler[i - 1], curr.first);
      } else 
#endif
#if DEGREE_2 && DEGREE_1
      if (from == 2 && to == 1) {
        temp = new MGTransferMatrixFreeP<dim, 2, 1, value_type_operator, VECTOR_TYPE>( *mg_dofhandler[i], *mg_dofhandler[i - 1], curr.first);
      } else 
#endif
      {
        AssertThrow(false,
                    ExcMessage("This type of p-transfer is not implemented"));
      }

      mg_transfer[i].reset(temp);
    }
  }

  // finalize setup of preconditioner
  this->initialize_multigrid_preconditioner(dof_handler);
}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::
    initialize_mg_constrained_dofs(const DoFHandler<dim> &dof_handler,
                                   MGConstrainedDoFs &constrained_dofs,
          std::map<types::boundary_id, std::shared_ptr<Function<dim>>> const &dirichlet_bc) {
    std::set<types::boundary_id> dirichlet_boundary;
    for (auto& it : dirichlet_bc)
      dirichlet_boundary.insert(it.first);
    constrained_dofs.initialize(dof_handler);
    constrained_dofs.make_zero_boundary_constraints(dof_handler,
                                                    dirichlet_boundary);

}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::update(
    MatrixOperatorBase const * /*matrix_operator*/) {}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::vmult(
    parallel::distributed::Vector<value_type> &dst,
    const parallel::distributed::Vector<value_type> &src) const {
  multigrid_preconditioner->vmult(dst, src);
}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::operator()(
    const unsigned int /*level*/,
    parallel::distributed::Vector<value_type_operator> &dst,
    const parallel::distributed::Vector<value_type_operator> &src) const {
  multigrid_preconditioner->vmult(dst, src);
}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::
    apply_smoother_on_fine_level(
        parallel::distributed::Vector<value_type_operator> &dst,
        const parallel::distributed::Vector<value_type_operator> &src) const {
  this->mg_smoother[this->mg_smoother.max_level()]->vmult(dst, src);
}


template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::update_smoother(unsigned int level){

    AssertThrow(level > 0, ExcMessage("Multigrid level is invalid when initializing multigrid smoother!"));

    switch (mg_data.smoother)
    {
      case MultigridSmoother::Chebyshev:
      {
        initialize_chebyshev_smoother(*mg_matrices[level], level);
        break;
      }
      case MultigridSmoother::ChebyshevNonsymmetricOperator:
      {
        initialize_chebyshev_smoother_nonsymmetric_operator(*mg_matrices[level], level);
        break;
      }
      case MultigridSmoother::GMRES:
      {
        typedef GMRESSmoother<Operator,VECTOR_TYPE> GMRES_SMOOTHER;
        std::shared_ptr<GMRES_SMOOTHER> smoother = std::dynamic_pointer_cast<GMRES_SMOOTHER>(mg_smoother[level]);
        smoother->update();
        break;
      }
      case MultigridSmoother::CG:
      {
        typedef CGSmoother<Operator,VECTOR_TYPE> CG_SMOOTHER;
        std::shared_ptr<CG_SMOOTHER> smoother = std::dynamic_pointer_cast<CG_SMOOTHER>(mg_smoother[level]);
        smoother->update();
        break;
      }
      case MultigridSmoother::Jacobi:
      {
        typedef JacobiSmoother<Operator,VECTOR_TYPE> JACOBI_SMOOTHER;
        std::shared_ptr<JACOBI_SMOOTHER> smoother = std::dynamic_pointer_cast<JACOBI_SMOOTHER>(mg_smoother[level]);
        smoother->update();
        break;
      }
      default:
      {
        AssertThrow(false, ExcMessage("Specified MultigridSmoother not implemented!"));
      }
    }
  }

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::update_coarse_solver(){
      
    switch (mg_data.coarse_solver)
    {
      case MultigridCoarseGridSolver::Chebyshev:
      {
        initialize_chebyshev_smoother_coarse_grid(*mg_matrices[0]);
        break;
      }
      case MultigridCoarseGridSolver::ChebyshevNonsymmetricOperator:
      {
        initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(*mg_matrices[0]);
        break;
      }
      case MultigridCoarseGridSolver::PCG_NoPreconditioner:
      {
        // do nothing
        break;
      }
      case MultigridCoarseGridSolver::PCG_PointJacobi:
      {
        std::shared_ptr<MGCoarsePCG<Operator> >
          coarse_solver = std::dynamic_pointer_cast<MGCoarsePCG<Operator> >(mg_coarse);
        coarse_solver->update_preconditioner(*this->mg_matrices[0]);

        break;
      }
      case MultigridCoarseGridSolver::PCG_BlockJacobi:
      {
        std::shared_ptr<MGCoarsePCG<Operator> >
          coarse_solver = std::dynamic_pointer_cast<MGCoarsePCG<Operator> >(mg_coarse);
        coarse_solver->update_preconditioner(*this->mg_matrices[0]);

        break;
      }
      case MultigridCoarseGridSolver::GMRES_NoPreconditioner:
      {
        // do nothing
        break;
      }
      case MultigridCoarseGridSolver::GMRES_PointJacobi:
      {
        std::shared_ptr<MGCoarseGMRES<Operator> >
          coarse_solver = std::dynamic_pointer_cast<MGCoarseGMRES<Operator> >(mg_coarse);
        coarse_solver->update_preconditioner(*this->mg_matrices[0]);
        break;
      }
      case MultigridCoarseGridSolver::GMRES_BlockJacobi:
      {
        std::shared_ptr<MGCoarseGMRES<Operator> >
          coarse_solver = std::dynamic_pointer_cast<MGCoarseGMRES<Operator> >(mg_coarse);
        coarse_solver->update_preconditioner(*this->mg_matrices[0]);
        break;
      }
      default:
      {
        AssertThrow(false, ExcMessage("Unknown coarse-grid solver given"));
      }
    }
  }

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<
    dim, value_type, Operator>::initialize_smoother(Operator &matrix,
                                                    unsigned int level) {
  AssertThrow(
      level > 0,
      ExcMessage(
          "Multigrid level is invalid when initializing multigrid smoother!"));

  switch (mg_data.smoother) {
  case MultigridSmoother::Chebyshev: {
    mg_smoother[level].reset(new ChebyshevSmoother<Operator, VECTOR_TYPE>());
    initialize_chebyshev_smoother(matrix, level);
    break;
  }
  case MultigridSmoother::ChebyshevNonsymmetricOperator: {
    mg_smoother[level].reset(new ChebyshevSmoother<Operator, VECTOR_TYPE>());
    initialize_chebyshev_smoother_nonsymmetric_operator(matrix, level);
    break;
  }
  case MultigridSmoother::GMRES: {
    typedef GMRESSmoother<Operator, VECTOR_TYPE> GMRES_SMOOTHER;
    mg_smoother[level].reset(new GMRES_SMOOTHER());

    typename GMRES_SMOOTHER::AdditionalData smoother_data;
    smoother_data.preconditioner = mg_data.gmres_smoother_data.preconditioner;
    smoother_data.number_of_iterations =
        mg_data.gmres_smoother_data.number_of_iterations;

    std::shared_ptr<GMRES_SMOOTHER> smoother =
        std::dynamic_pointer_cast<GMRES_SMOOTHER>(mg_smoother[level]);
    smoother->initialize(matrix, smoother_data);
    break;
  }
  case MultigridSmoother::CG: {
    typedef CGSmoother<Operator, VECTOR_TYPE> CG_SMOOTHER;
    mg_smoother[level].reset(new CG_SMOOTHER());

    typename CG_SMOOTHER::AdditionalData smoother_data;
    smoother_data.preconditioner = mg_data.cg_smoother_data.preconditioner;
    smoother_data.number_of_iterations =
        mg_data.cg_smoother_data.number_of_iterations;

    std::shared_ptr<CG_SMOOTHER> smoother =
        std::dynamic_pointer_cast<CG_SMOOTHER>(mg_smoother[level]);
    smoother->initialize(matrix, smoother_data);
    break;
  }
  case MultigridSmoother::Jacobi: {
    typedef JacobiSmoother<Operator, VECTOR_TYPE> JACOBI_SMOOTHER;
    mg_smoother[level].reset(new JACOBI_SMOOTHER());

    typename JACOBI_SMOOTHER::AdditionalData smoother_data;
    smoother_data.preconditioner = mg_data.jacobi_smoother_data.preconditioner;
    smoother_data.number_of_smoothing_steps =
        mg_data.jacobi_smoother_data.number_of_smoothing_steps;
    smoother_data.damping_factor = mg_data.jacobi_smoother_data.damping_factor;

    std::shared_ptr<JACOBI_SMOOTHER> smoother =
        std::dynamic_pointer_cast<JACOBI_SMOOTHER>(mg_smoother[level]);
    smoother->initialize(matrix, smoother_data);
    break;
  }
  default: {
    AssertThrow(false,
                ExcMessage("Specified MultigridSmoother not implemented!"));
  }
  }
}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::
    initialize_coarse_solver(Operator &matrix, Operator &matrix_q,
                             const unsigned int coarse_level) {

  switch (mg_data.coarse_solver) {
  case MultigridCoarseGridSolver::Chebyshev: {
    mg_smoother[0].reset(new ChebyshevSmoother<Operator, VECTOR_TYPE>());
    initialize_chebyshev_smoother_coarse_grid(matrix);

    mg_coarse.reset(
        new MGCoarseInverseOperator<
            parallel::distributed::Vector<value_type_operator>, SMOOTHER>(
            mg_smoother[0]));
    break;
  }
  case MultigridCoarseGridSolver::ChebyshevNonsymmetricOperator: {
    mg_smoother[0].reset(new ChebyshevSmoother<Operator, VECTOR_TYPE>());
    initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(matrix);

    mg_coarse.reset(
        new MGCoarseInverseOperator<
            parallel::distributed::Vector<value_type_operator>, SMOOTHER>(
            mg_smoother[0]));
    break;
  }
  case MultigridCoarseGridSolver::PCG_NoPreconditioner: {
    typename MGCoarsePCG<Operator>::AdditionalData additional_data;
    additional_data.preconditioner = PreconditionerCoarseGridSolver::None;

    mg_coarse.reset(new MGCoarsePCG<Operator>(matrix, additional_data));
    break;
  }
  case MultigridCoarseGridSolver::PCG_PointJacobi: {
    typename MGCoarsePCG<Operator>::AdditionalData additional_data;
    additional_data.preconditioner =
        PreconditionerCoarseGridSolver::PointJacobi;

    mg_coarse.reset(new MGCoarsePCG<Operator>(matrix, additional_data));
    break;
  }
  case MultigridCoarseGridSolver::PCG_BlockJacobi: {
    typename MGCoarsePCG<Operator>::AdditionalData additional_data;
    additional_data.preconditioner =
        PreconditionerCoarseGridSolver::BlockJacobi;

    mg_coarse.reset(new MGCoarsePCG<Operator>(matrix, additional_data));
    break;
  }
  case MultigridCoarseGridSolver::GMRES_NoPreconditioner: {
    typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
    additional_data.preconditioner = PreconditionerCoarseGridSolver::None;

    mg_coarse.reset(new MGCoarseGMRES<Operator>(matrix, additional_data));
    break;
  }
  case MultigridCoarseGridSolver::GMRES_PointJacobi: {
    typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
    additional_data.preconditioner =
        PreconditionerCoarseGridSolver::PointJacobi;

    mg_coarse.reset(new MGCoarseGMRES<Operator>(matrix, additional_data));
    break;
  }
  case MultigridCoarseGridSolver::GMRES_BlockJacobi: {
    typename MGCoarseGMRES<Operator>::AdditionalData additional_data;
    additional_data.preconditioner =
        PreconditionerCoarseGridSolver::BlockJacobi;

    mg_coarse.reset(new MGCoarseGMRES<Operator>(matrix, additional_data));
    break;
  }
  case MultigridCoarseGridSolver::AMG_ML: {
    // TODO modify template arguments
    mg_coarse.reset(
        new MGCoarseML<Operator>(matrix, matrix_q, true, coarse_level));
    return;
  }
  default: {
    AssertThrow(false, ExcMessage("Unknown coarse-grid solver specified."));
  }
  }
}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::
    initialize_multigrid_preconditioner(
        DoFHandler<dim> const & /*dof_handler*/) {
  this->multigrid_preconditioner.reset(
      new MultigridPreconditioner<VECTOR_TYPE, Operator, MG_TRANSFER, SMOOTHER>(
          this->mg_matrices, *this->mg_coarse, this->mg_transfer,
          this->mg_smoother));
}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::
    initialize_chebyshev_smoother(Operator &matrix, unsigned int level) {
  typedef ChebyshevSmoother<Operator, VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  /*
  std::pair<double,double> eigenvalues = compute_eigenvalues(mg_matrices[level],
  smoother_data.matrix_diagonal_inverse);
  if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << "Eigenvalues on level l = " << level << std::endl;
    std::cout << std::scientific << std::setprecision(3)
              <<"Max EV = " << eigenvalues.second << " : Min EV = " <<
  eigenvalues.first << std::endl;
  }
  */

  smoother_data.smoothing_range =
      mg_data.chebyshev_smoother_data.smoother_smoothing_range;
  smoother_data.degree = mg_data.chebyshev_smoother_data.smoother_poly_degree;
  smoother_data.eig_cg_n_iterations =
      mg_data.chebyshev_smoother_data.eig_cg_n_iterations;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
      std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[level]);
  smoother->initialize(matrix, smoother_data);
}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::
    initialize_chebyshev_smoother_coarse_grid(Operator &matrix) {
  // use Chebyshev smoother of high degree to solve the coarse grid problem
  // approximately
  typedef ChebyshevSmoother<Operator, VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<double, double> eigenvalues =
      compute_eigenvalues(matrix, smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop
  double const factor = 1.1;
  smoother_data.max_eigenvalue = factor * eigenvalues.second;
  smoother_data.smoothing_range =
      eigenvalues.second / eigenvalues.first * factor;
  double sigma = (1. - std::sqrt(1. / smoother_data.smoothing_range)) /
                 (1. + std::sqrt(1. / smoother_data.smoothing_range));
  const double eps = 1.e-3;
  smoother_data.degree = std::log(1. / eps + std::sqrt(1. / eps / eps - 1.)) /
                         std::log(1. / sigma);
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
      std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[0]);
  smoother->initialize(matrix, smoother_data);
}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::
    initialize_chebyshev_smoother_nonsymmetric_operator(Operator &matrix,
                                                        unsigned int level) {
  typedef ChebyshevSmoother<Operator, VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  /*
  std::pair<double,double> eigenvalues =
  compute_eigenvalues_gmres(mg_matrices[level],
  smoother_data.matrix_diagonal_inverse);
  std::cout<<"Max EW = "<< eigenvalues.second <<" : Min EW =
  "<<eigenvalues.first<<std::endl;
  */

  // use gmres to calculate eigenvalues for nonsymmetric problem
  const unsigned int eig_n_iter = 20;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<std::complex<double>, std::complex<double>> eigenvalues =
      compute_eigenvalues_gmres(matrix, smoother_data.matrix_diagonal_inverse,
                                eig_n_iter);
#pragma GCC diagnostic pop
  const double factor = 1.1;
  smoother_data.max_eigenvalue = factor * std::abs(eigenvalues.second);
  smoother_data.smoothing_range =
      mg_data.chebyshev_smoother_data.smoother_smoothing_range;
  smoother_data.degree = mg_data.chebyshev_smoother_data.smoother_poly_degree;
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
      std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[level]);
  smoother->initialize(matrix, smoother_data);
}

template <int dim, typename value_type, typename Operator>
void MyMultigridPreconditionerBase<dim, value_type, Operator>::
    initialize_chebyshev_smoother_nonsymmetric_operator_coarse_grid(
        Operator &matrix) {
  // use Chebyshev smoother of high degree to solve the coarse grid problem
  // approximately
  typedef ChebyshevSmoother<Operator, VECTOR_TYPE> CHEBYSHEV_SMOOTHER;
  typename CHEBYSHEV_SMOOTHER::AdditionalData smoother_data;

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  matrix.initialize_dof_vector(smoother_data.matrix_diagonal_inverse);
  matrix.calculate_inverse_diagonal(smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop

  const double factor = 1.1;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  std::pair<std::complex<double>, std::complex<double>> eigenvalues =
      compute_eigenvalues_gmres(matrix, smoother_data.matrix_diagonal_inverse);
#pragma GCC diagnostic pop
  smoother_data.max_eigenvalue = factor * std::abs(eigenvalues.second);
  smoother_data.smoothing_range =
      factor * std::abs(eigenvalues.second) / std::abs(eigenvalues.first);
  double sigma = (1. - std::sqrt(1. / smoother_data.smoothing_range)) /
                 (1. + std::sqrt(1. / smoother_data.smoothing_range));
  const double eps = 1e-3;
  smoother_data.degree =
      std::log(1. / eps + std::sqrt(1. / eps / eps - 1)) / std::log(1. / sigma);
  smoother_data.eig_cg_n_iterations = 0;

  std::shared_ptr<CHEBYSHEV_SMOOTHER> smoother =
      std::dynamic_pointer_cast<CHEBYSHEV_SMOOTHER>(mg_smoother[0]);
  smoother->initialize(matrix, smoother_data);
}

#include "multigrid_preconditioner_adapter_base.hpp"

//template class MyMultigridPreconditionerBase<2, float,
//                                             MatrixOperatorBaseNew<2, float>>;
//template class MyMultigridPreconditionerBase<2, double,
//                                             MatrixOperatorBaseNew<2, float>>;