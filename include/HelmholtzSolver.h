/*
 * HelmholtzSolver.h
 *
 *  Created on: May 11, 2016
 *      Author: fehn
 */

#ifndef INCLUDE_HELMHOLTZSOLVER_H_
#define INCLUDE_HELMHOLTZSOLVER_H_

#include "NavierStokesOperators.h"
#include "Preconditioner.h"

template<int dim>
struct HelmholtzOperatorData
{
  HelmholtzOperatorData ()
    :
    dof_index(0),
    formulation_viscous_term(FormulationViscousTerm::DivergenceFormulation),
    IP_formulation_viscous(InteriorPenaltyFormulationViscous::SIPG),
    IP_factor_viscous(1.0),
    viscosity(1.0),
    mass_matrix_coefficient(1.0)
  {}

  unsigned int dof_index;
  FormulationViscousTerm formulation_viscous_term;
  InteriorPenaltyFormulationViscous IP_formulation_viscous;
  double IP_factor_viscous;
  std::set<types::boundary_id> dirichlet_boundaries;
  std::set<types::boundary_id> neumann_boundaries;
  double viscosity;
  double mass_matrix_coefficient;
  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs_level0;
};

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d_xwall,typename Number = double>
class HelmholtzOperator : public Subscriptor
{
public:
  typedef Number value_type;

  HelmholtzOperator()
    :
    data(nullptr),
    mass_matrix_coefficient(1.0),
    needs_mean_value_constraint (false),
    apply_mean_value_constraint_in_matvec (false)
  {}

  void reinit(MatrixFree<dim,Number> const     &mf_data,
              Mapping<dim> const               &mapping,
              HelmholtzOperatorData<dim> const &operator_data,
              FEParameters const               &fe_param)
  {
    this->data = &mf_data;
    this->helmholtz_operator_data = operator_data;
    // set mass matrix coefficient !
    this->mass_matrix_coefficient = helmholtz_operator_data.mass_matrix_coefficient;

    // MassMatrixOperator
    MassMatrixOperatorData mass_matrix_operator_data;
    mass_matrix_operator_data.dof_index = helmholtz_operator_data.dof_index;
    mass_matrix_operator.initialize(*data, fe_param, mass_matrix_operator_data);

    // ViscousOperator
    ViscousOperatorData viscous_operator_data;
    viscous_operator_data.dof_index = helmholtz_operator_data.dof_index;
    viscous_operator_data.formulation_viscous_term = helmholtz_operator_data.formulation_viscous_term;
    viscous_operator_data.IP_formulation_viscous = helmholtz_operator_data.IP_formulation_viscous;
    viscous_operator_data.IP_factor_viscous = helmholtz_operator_data.IP_factor_viscous;
    viscous_operator_data.dirichlet_boundaries = helmholtz_operator_data.dirichlet_boundaries;
    viscous_operator_data.neumann_boundaries = helmholtz_operator_data.neumann_boundaries;
    viscous_operator.initialize(mapping,*data,fe_param,viscous_operator_data);
    viscous_operator.set_constant_viscosity(helmholtz_operator_data.viscosity);

    // initialize temp vector
    initialize_dof_vector(temp);

    // Check whether the matrix is singular when applied to a vector
    // consisting of only ones (except for constrained entries)
    parallel::distributed::Vector<Number> in_vec, out_vec;
    initialize_dof_vector(in_vec);
    initialize_dof_vector(out_vec);
    in_vec = 1;
    const std::vector<unsigned int> &constrained_entries =
      mf_data.get_constrained_dofs(operator_data.dof_index);
    for (unsigned int i=0; i<constrained_entries.size(); ++i)
      in_vec.local_element(constrained_entries[i]) = 0;
    vmult_add(out_vec, in_vec);
    const double linfty_norm = out_vec.linfty_norm();

    // since we cannot know the magnitude of the entries at this point (the
    // diagonal entries would be a guideline but they are not available here),
    // we instead multiply by a random vector
    for (unsigned int i=0; i<in_vec.local_size(); ++i)
      in_vec.local_element(i) = (double)rand()/RAND_MAX;
    vmult(out_vec, in_vec);
    const double linfty_norm_compare = out_vec.linfty_norm();

    // use mean value constraint if the infty norm with the one vector is very small
    needs_mean_value_constraint =
      linfty_norm / linfty_norm_compare < std::pow(std::numeric_limits<Number>::epsilon(), 2./3.);
    apply_mean_value_constraint_in_matvec = needs_mean_value_constraint;
  }


  void reinit (const DoFHandler<dim>            &dof_handler,
               const Mapping<dim>               &mapping,
               const HelmholtzOperatorData<dim> &operator_data,
               const MGConstrainedDoFs          &/*mg_constrained_dofs*/,
               const unsigned int               level = numbers::invalid_unsigned_int,
               FEParameters const               &fe_param = FEParameters())
  {
    this->helmholtz_operator_data = operator_data;

    const QGauss<1> quad(dof_handler.get_fe().degree+1);
    typename MatrixFree<dim,Number>::AdditionalData addit_data;
    addit_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::none;
    if (dof_handler.get_fe().dofs_per_vertex == 0)
      addit_data.build_face_info = true;
    addit_data.level_mg_handler = level;
    addit_data.mpi_communicator =
      dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()) ?
      (dynamic_cast<const parallel::Triangulation<dim> *>(&dof_handler.get_triangulation()))->get_communicator() : MPI_COMM_SELF;
    addit_data.periodic_face_pairs_level_0 = operator_data.periodic_face_pairs_level0;

    HelmholtzOperatorData<dim> my_operator_data = operator_data;
    my_operator_data.dof_index = 0;

    ConstraintMatrix constraints;
    own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

    reinit(own_matrix_free_storage, mapping, my_operator_data,fe_param);
  }

  void set_mass_matrix_coefficient(Number const coefficient_in)
  {
    mass_matrix_coefficient = coefficient_in;
  }

  void apply_nullspace_projection(parallel::distributed::Vector<Number> &vec) const
  {
    if (needs_mean_value_constraint)
      {
        const Number mean_val = vec.mean_value();
        vec.add(-mean_val);
      }
  }

  // apply matrix vector multiplication
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    mass_matrix_operator.apply(dst,src);
    dst *= mass_matrix_coefficient;

    viscous_operator.apply_add(dst,src);
  }

  void Tvmult(parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    vmult(dst,src);
  }

  void Tvmult_add(parallel::distributed::Vector<Number>       &dst,
                  const parallel::distributed::Vector<Number> &src) const
  {
    vmult_add(dst,src);
  }

  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    mass_matrix_operator.apply(temp,src);
    temp *= mass_matrix_coefficient;
    dst += temp;

    viscous_operator.apply_add(dst,src);
  }

  void vmult_interface_down(parallel::distributed::Vector<Number>       &dst,
                            const parallel::distributed::Vector<Number> &src) const
  {
    vmult(dst,src);
  }

  void vmult_add_interface_up(parallel::distributed::Vector<Number>       &dst,
                              const parallel::distributed::Vector<Number> &src) const
  {
    vmult_add(dst,src);
  }

  types::global_dof_index m() const
  {
    return data->get_vector_partitioner(helmholtz_operator_data.dof_index)->size();
  }

  types::global_dof_index n() const
  {
    return data->get_vector_partitioner(helmholtz_operator_data.dof_index)->size();
  }

  Number el (const unsigned int,  const unsigned int) const
  {
    AssertThrow(false, ExcMessage("Matrix-free does not allow for entry access"));
    return Number();
  }

  MatrixFree<dim,Number> const  & get_data() const
  {
    return *data;
  }

  HelmholtzOperatorData<dim> const & get_operator_data() const
  {
    return helmholtz_operator_data;
  }

  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    mass_matrix_operator.calculate_diagonal(diagonal);
    diagonal *= mass_matrix_coefficient;

    viscous_operator.add_diagonal(diagonal);

    // verify_calculation_of_diagonal(diagonal);
  }

  void verify_calculation_of_diagonal(parallel::distributed::Vector<Number> const &diagonal) const
  {
    parallel::distributed::Vector<Number>  diagonal2(diagonal);
    diagonal2 = 0.0;
    parallel::distributed::Vector<Number>  src(diagonal2);
    parallel::distributed::Vector<Number>  dst(diagonal2);
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      src.local_element(i) = 1.0;
      vmult(dst,src);
      diagonal2.local_element(i) = dst.local_element(i);
      src.local_element(i) = 0.0;
    }

    std::cout<<"L2 norm diagonal - Variant 1: "<<diagonal.l2_norm()<<std::endl;
    std::cout<<"L2 norm diagonal - Variant 2: "<<diagonal2.l2_norm()<<std::endl;
    diagonal2.add(-1.0,diagonal);
    std::cout<<"L2 error diagonal: "<<diagonal2.l2_norm()<<std::endl;
  }

  void invert_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    for (unsigned int i=0;i<diagonal.local_size();++i)
    {
      if( std::abs(diagonal.local_element(i)) > 1.0e-10 )
        diagonal.local_element(i) = 1.0/diagonal.local_element(i);
      else
        diagonal.local_element(i) = 1.0;
    }
  }

  void calculate_inverse_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    calculate_diagonal(diagonal);

    invert_diagonal(diagonal);
  }

  void initialize_dof_vector(parallel::distributed::Vector<Number> &vector) const
  {
    data->initialize_dof_vector(vector,helmholtz_operator_data.dof_index);
  }

private:
  MatrixFree<dim,Number> const * data;
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> viscous_operator;
  HelmholtzOperatorData<dim> helmholtz_operator_data;
  parallel::distributed::Vector<Number> mutable temp;
  Number mass_matrix_coefficient;
  bool needs_mean_value_constraint;
  bool apply_mean_value_constraint_in_matvec;
};

struct HelmholtzSolverData
{
  HelmholtzSolverData()
    :
    max_iter(1e4),
    solver_tolerance_abs(1.e-12),
    solver_tolerance_rel(1.e-6),
    solver_viscous(SolverViscous::PCG),
    preconditioner_viscous(PreconditionerViscous::None),
    smoother_poly_degree(5),
    smoother_smoothing_range(20),
    multigrid_smoother(MultigridSmoother::Chebyshev),
    coarse_solver(MultigridCoarseGridSolver::coarse_chebyshev_smoother)
    {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  SolverViscous solver_viscous;
  PreconditionerViscous preconditioner_viscous;

  // Sets the polynomial degree of the Chebyshev smoother (Chebyshev
  // accelerated Jacobi smoother)
  double smoother_poly_degree;
  // Sets the smoothing range of the Chebyshev smoother
  double smoother_smoothing_range;

  // multigrid smoother
  MultigridSmoother multigrid_smoother;
  // Sets the coarse grid solver
  MultigridCoarseGridSolver coarse_solver;
};


template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type=double>
class HelmholtzSolver
{
public:
  typedef float Number;

  HelmholtzSolver()
    :
    global_matrix(nullptr),
    preconditioner(nullptr)
  {}

  ~HelmholtzSolver()
  {}

  void initialize(const HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> &helmholtz_operator,
                  const Mapping<dim>                                                         &mapping,
                  const MatrixFree<dim,value_type>                                           &matrix_free,
                  const HelmholtzSolverData                                                  &solver_data,
                  const unsigned int                                                         dof_index,
                  const unsigned int                                                         quad_index,
                  FEParameters const                                                         &fe_param)
  {
    this->global_matrix = &helmholtz_operator;
    this->solver_data = solver_data;

    const DoFHandler<dim> &dof_handler = matrix_free.get_dof_handler(global_matrix->get_operator_data().dof_index);

    if(solver_data.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
      preconditioner.reset(new InverseMassMatrixPreconditioner<dim,fe_degree,value_type>(matrix_free,dof_index,quad_index));
    else if(solver_data.preconditioner_viscous == PreconditionerViscous::Jacobi)
      preconditioner.reset(new JacobiPreconditioner<value_type, HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall, value_type> >(*global_matrix));
    else if(solver_data.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
    {
      MultigridData mg_data;
      mg_data.multigrid_smoother = solver_data.multigrid_smoother;
      mg_data.coarse_solver = solver_data.coarse_solver;
      mg_data.smoother_poly_degree = solver_data.smoother_poly_degree;
      mg_data.smoother_smoothing_range = solver_data.smoother_smoothing_range;

      preconditioner.reset(new MyMultigridPreconditioner<dim,value_type,HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall,Number>, HelmholtzOperatorData<dim> >
                           (mg_data, dof_handler, mapping, global_matrix->get_operator_data(),fe_param));
    }
  }

  unsigned int solve(parallel::distributed::Vector<value_type>       &dst,
                     const parallel::distributed::Vector<value_type> &src) const;

private:
  HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> const *global_matrix;
  HelmholtzSolverData solver_data;
  std_cxx11::shared_ptr<PreconditionerBase<value_type> > preconditioner;
};

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type>
unsigned int HelmholtzSolver<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type> ::
solve(parallel::distributed::Vector<value_type>       &dst,
      const parallel::distributed::Vector<value_type> &src) const
{
  ReductionControl solver_control (solver_data.max_iter,
                                   solver_data.solver_tolerance_abs,
                                   solver_data.solver_tolerance_rel);
  try
  {
    if(solver_data.solver_viscous == SolverViscous::PCG)
    {
      SolverCG<parallel::distributed::Vector<value_type> > solver (solver_control);
      if(solver_data.preconditioner_viscous == PreconditionerViscous::None)
        solver.solve (*global_matrix, dst, src, PreconditionIdentity());
      else if(solver_data.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
        solver.solve (*global_matrix, dst, src, *preconditioner);
      else if(solver_data.preconditioner_viscous == PreconditionerViscous::Jacobi)
      {
        // TODO: recalculate diagonal (say every 10, 100 time steps) in case of varying parameters
        // of mass matrix term or viscous term, e.g. strongly varying time step sizes (adaptive time step control)
        // or strongly varying viscosity (turbulence)
//        std_cxx11::shared_ptr<JacobiPreconditioner<value_type,HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> > >
//          jacobi_preconditioner = std::dynamic_pointer_cast<JacobiPreconditioner<value_type,HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> > >(preconditioner);
//        jacobi_preconditioner->recalculate_diagonal(*global_matrix);

        solver.solve (*global_matrix, dst, src, *preconditioner);
      }
      else if(solver_data.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
      {
//        parallel::distributed::Vector<value_type> check1;
//        global_matrix->initialize_dof_vector(check1);
//        parallel::distributed::Vector<value_type> check2(check1), tmp(check1);
//        parallel::distributed::Vector<Number> check3;
//        check3 = check1;
//        for (unsigned int i=0; i<check1.size(); ++i)
//          check1(i) = (double)rand()/RAND_MAX;
//        global_matrix->vmult(tmp, check1);
//        tmp *= -1.0;
//        preconditioner->vmult(check2, tmp);
//        check2 += check1;
//
//        parallel::distributed::Vector<Number> tmp_float, check1_float;
//        tmp_float = tmp;
//        check1_float = check1;
//        std_cxx11::shared_ptr<MyMultigridPreconditioner<dim,value_type,HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall,Number>, HelmholtzOperatorData<dim> > >
//          my_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditioner<dim,value_type,HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall,Number>, HelmholtzOperatorData<dim> > >(preconditioner);
//        my_preconditioner->mg_smoother[my_preconditioner->mg_smoother.max_level()].vmult(check3,tmp_float);
//        check3 += check1_float;
//        /*
//        my_preconditioner->mg_matrices[my_preconditioner->mg_matrices.max_level()].vmult(tmp_float,check1_float);
//        check1_float = tmp;
//        tmp_float *= -1.0;
//        std::cout<<"L2 norm tmp = "<<tmp_float.l2_norm()<<std::endl;
//        std::cout<<"L2 norm check = "<<check1_float.l2_norm()<<std::endl;
//        */
//        DataOut<dim> data_out;
//        data_out.attach_dof_handler (global_matrix->get_data().get_dof_handler(global_matrix->get_operator_data().dof_index));
//
//        std::vector<std::string> initial (dim, "initial");
//        std::vector<DataComponentInterpretation::DataComponentInterpretation>
//          initial_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
//        data_out.add_data_vector (global_matrix->get_data().get_dof_handler(global_matrix->get_operator_data().dof_index),check1, initial, initial_component_interpretation);
//
//        std::vector<std::string> mg_cycle (dim, "mg_cycle");
//        std::vector<DataComponentInterpretation::DataComponentInterpretation>
//          mg_cylce_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
//        data_out.add_data_vector (global_matrix->get_data().get_dof_handler(global_matrix->get_operator_data().dof_index),check2, mg_cycle, mg_cylce_component_interpretation);
//
//        std::vector<std::string> smoother (dim, "smoother");
//        std::vector<DataComponentInterpretation::DataComponentInterpretation>
//          smoother_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
//        data_out.add_data_vector (global_matrix->get_data().get_dof_handler(global_matrix->get_operator_data().dof_index),check3, smoother, smoother_component_interpretation);
//
//        data_out.build_patches (global_matrix->get_data().get_dof_handler(global_matrix->get_operator_data().dof_index).get_fe().degree*3);
//        std::ostringstream filename;
//        filename << "smoothing.vtk";
//
//        std::ofstream output (filename.str().c_str());
//        data_out.write_vtk(output);
//        std::abort();

        // TODO: update multigrid preconditioner (diagonals) in case of varying parameters
        solver.solve (*global_matrix, dst, src, *preconditioner);
      }
    }
    else if(solver_data.solver_viscous == SolverViscous::GMRES)
    {
      SolverGMRES<parallel::distributed::Vector<value_type> > solver (solver_control);
      if(solver_data.preconditioner_viscous == PreconditionerViscous::None)
        solver.solve (*global_matrix, dst, src, PreconditionIdentity());
      else if(solver_data.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
        solver.solve (*global_matrix, dst, src, *preconditioner);
      else if(solver_data.preconditioner_viscous == PreconditionerViscous::Jacobi)
      {
        // TODO: recalculate diagonal (say every 10, 100 time steps) in case of varying parameters
        // of mass matrix term or viscous term, e.g. strongly varying time step sizes (adaptive time step control)
        // or strongly varying viscosity (turbulence)
//        std_cxx11::shared_ptr<JacobiPreconditioner<value_type,HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> > >
//          jacobi_preconditioner = std::dynamic_pointer_cast<JacobiPreconditioner<value_type,HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> > >(preconditioner);
//        jacobi_preconditioner->recalculate_diagonal(*global_matrix);

        solver.solve (*global_matrix, dst, src, *preconditioner);
      }
      else if(solver_data.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
      {
        // TODO: update multigrid preconditioner (diagonals) in case of varying parameters
        solver.solve (*global_matrix, dst, src, *preconditioner);
      }
    }
    else
      AssertThrow(false,ExcMessage("Specified Viscous Solver not implemented - possibilities are PCG and GMRES"));
  }
  catch (SolverControl::NoConvergence &)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      std::cout << std::endl << "Viscous solver failed to solve to given tolerance." << std::endl;
  }

  return solver_control.last_step();
}

#endif /* INCLUDE_HELMHOLTZSOLVER_H_ */
