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
    mass_matrix_coefficient(1.0)
  {}

  unsigned int dof_index;

  MassMatrixOperatorData mass_matrix_operator_data;
  ViscousOperatorData<dim> viscous_operator_data;

  /*
   * This variable 'mass_matrix_coefficient' is only used when initializing the HelmholtzOperator.
   * In order to change/update this coefficient during the simulation (e.g., varying time step sizes)
   * use the element variable 'mass_matrix_coefficient' of HelmholtzOperator and the corresponding setter
   * set_mass_matrix_coefficient().
   */
  double mass_matrix_coefficient;

  std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs_level0;

  std::set<types::boundary_id> const & get_dirichlet_boundaries() const
  {
    return viscous_operator_data.get_dirichlet_boundaries();
  }

  void set_dof_index(unsigned int dof_index_in)
  {
    this->dof_index = dof_index_in;

    // don't forget to set the dof_indices of the mass_matrix_operator_data and viscous_operator_data
    mass_matrix_operator_data.dof_index = dof_index_in;
    viscous_operator_data.dof_index = dof_index_in;
  }
};

template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d_xwall,typename Number = double>
class HelmholtzOperator : public Subscriptor
{
public:
  typedef Number value_type;

  HelmholtzOperator()
    :
    data(nullptr),
    mass_matrix_operator(nullptr),
    viscous_operator(nullptr),
    mass_matrix_coefficient(1.0)
  {}

  void initialize(MatrixFree<dim,Number> const                                                            &mf_data_in,
                  HelmholtzOperatorData<dim> const                                                        &helmholtz_operator_data_in,
                  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>  const &mass_matrix_operator_in,
                  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> const     &viscous_operator_in)
  {
    // copy parameters into element variables
    this->data = &mf_data_in;
    this->helmholtz_operator_data = helmholtz_operator_data_in;
    this->mass_matrix_operator = &mass_matrix_operator_in;
    this->viscous_operator = &viscous_operator_in;

    // set mass matrix coefficient!
    this->mass_matrix_coefficient = helmholtz_operator_data.mass_matrix_coefficient;

    // initialize temp vector
    initialize_dof_vector(temp);
  }

  void reinit (const DoFHandler<dim>            &dof_handler,
               const Mapping<dim>               &mapping,
               const HelmholtzOperatorData<dim> &operator_data,
               const MGConstrainedDoFs          &/*mg_constrained_dofs*/,
               const unsigned int               level = numbers::invalid_unsigned_int,
               FEParameters<dim> const          &fe_param = FEParameters<dim>())
  {
    // set the dof index to zero (for the HelmholtzOperator and also
    // for the basic Operators (MassMatrixOperator and ViscousOperator))
    HelmholtzOperatorData<dim> my_operator_data = operator_data;
    my_operator_data.set_dof_index(0);

    // setup own matrix free object
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

    ConstraintMatrix constraints;
    own_matrix_free_storage.reinit(mapping, dof_handler, constraints, quad, addit_data);

    // setup own mass matrix operator
    MassMatrixOperatorData mass_matrix_operator_data = my_operator_data.mass_matrix_operator_data;
    own_mass_matrix_operator_storage.initialize(own_matrix_free_storage,fe_param,mass_matrix_operator_data);

    // setup own viscous operator
    ViscousOperatorData<dim> viscous_operator_data = my_operator_data.viscous_operator_data;
    own_viscous_operator_storage.initialize(mapping,own_matrix_free_storage,fe_param,viscous_operator_data);

    // setup Helmholtz operator
    initialize(own_matrix_free_storage, my_operator_data, own_mass_matrix_operator_storage, own_viscous_operator_storage);
  }

  void set_mass_matrix_coefficient(Number const coefficient_in)
  {
    mass_matrix_coefficient = coefficient_in;
  }

  void apply_nullspace_projection(parallel::distributed::Vector<Number> &/*vec*/) const
  {
    // does nothing in case of the Helmholtz equation
    // this function is only necessary due to the interface of the multigrid preconditioner
    // and especially the coarse grid solver that calls this function
  }

  // apply matrix vector multiplication
  void vmult (parallel::distributed::Vector<Number>       &dst,
              const parallel::distributed::Vector<Number> &src) const
  {
    // helmholtz operator = mass_matrix_operator + viscous_operator
    mass_matrix_operator->apply(dst,src);
    dst *= mass_matrix_coefficient;

    viscous_operator->apply_add(dst,src);
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
    mass_matrix_operator->apply(temp,src);
    temp *= mass_matrix_coefficient;
    dst += temp;

    viscous_operator->apply_add(dst,src);
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

  unsigned int get_dof_index() const
  {
    return helmholtz_operator_data.dof_index;
  }

  void calculate_diagonal(parallel::distributed::Vector<Number> &diagonal) const
  {
    mass_matrix_operator->calculate_diagonal(diagonal);
    diagonal *= mass_matrix_coefficient;

    viscous_operator->add_diagonal(diagonal);

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
    data->initialize_dof_vector(vector,get_dof_index());
  }

private:
  MatrixFree<dim,Number> const * data;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>  const *mass_matrix_operator;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number>  const *viscous_operator;
  HelmholtzOperatorData<dim> helmholtz_operator_data;
  parallel::distributed::Vector<Number> mutable temp;
  Number mass_matrix_coefficient;

  /*
   * The following variables are necessary when applying the multigrid preconditioner to the Helmholtz equation:
   * In that case, the HelmholtzOperator has to be generated for each level of the multigrid algorithm.
   * Accordingly, in a first step one has to setup own objects of MatrixFree, MassMatrixOperator, ViscousOperator,
   *  e.g., own_matrix_free_storage.reinit(...);
   * and later initialize the HelmholtzOperator with these ojects by setting the above pointers to the own_objects_storage,
   *  e.g., data = &own_matrix_free_storage;
   */
  MatrixFree<dim,Number> own_matrix_free_storage;
  MassMatrixOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> own_mass_matrix_operator_storage;
  ViscousOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall, Number> own_viscous_operator_storage;

};

struct HelmholtzSolverData
{
  HelmholtzSolverData()
    :
    max_iter(1e4),
    solver_tolerance_abs(1.e-12),
    solver_tolerance_rel(1.e-6),
    solver_viscous(SolverViscous::PCG),
    preconditioner_viscous(PreconditionerViscous::None)
    {}

  unsigned int max_iter;
  double solver_tolerance_abs;
  double solver_tolerance_rel;
  SolverViscous solver_viscous;
  PreconditionerViscous preconditioner_viscous;
};


template <int dim, int fe_degree, int fe_degree_xwall, int n_q_points_1d_xwall, typename value_type=double>
class HelmholtzSolver
{
public:
  HelmholtzSolver()
    :
    global_matrix(nullptr),
    preconditioner(nullptr)
  {}

  ~HelmholtzSolver()
  {}

  void initialize(const HelmholtzOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall> &helmholtz_operator,
                  std_cxx11::shared_ptr<PreconditionerBase<value_type> >                     helmholtz_preconditioner,
                  const HelmholtzSolverData                                                  &solver_data)
  {
    this->global_matrix = &helmholtz_operator;
    this->solver_data = solver_data;
    this->preconditioner = helmholtz_preconditioner;
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
      {
        solver.solve (*global_matrix, dst, src, PreconditionIdentity());
      }
      /*
      else if(solver_data.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
      {

        parallel::distributed::Vector<value_type> check1;
        global_matrix->initialize_dof_vector(check1);
        parallel::distributed::Vector<value_type> check2(check1), tmp(check1);
        parallel::distributed::Vector<Number> check3;
        check3 = check1;
        for (unsigned int i=0; i<check1.size(); ++i)
          check1(i) = (double)rand()/RAND_MAX;
        global_matrix->vmult(tmp, check1);
        tmp *= -1.0;
        preconditioner->vmult(check2, tmp);
        check2 += check1;

        parallel::distributed::Vector<Number> tmp_float, check1_float;
        tmp_float = tmp;
        check1_float = check1;
        std_cxx11::shared_ptr<MyMultigridPreconditioner<dim,value_type,HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall,Number>, HelmholtzOperatorData<dim> > >
          my_preconditioner = std::dynamic_pointer_cast<MyMultigridPreconditioner<dim,value_type,HelmholtzOperator<dim, fe_degree, fe_degree_xwall, n_q_points_1d_xwall,Number>, HelmholtzOperatorData<dim> > >(preconditioner);
        my_preconditioner->mg_smoother[my_preconditioner->mg_smoother.max_level()].vmult(check3,tmp_float);
        check3 += check1_float;

        //my_preconditioner->mg_matrices[my_preconditioner->mg_matrices.max_level()].vmult(tmp_float,check1_float);
        //check1_float = tmp;
        //tmp_float *= -1.0;
        //std::cout<<"L2 norm tmp = "<<tmp_float.l2_norm()<<std::endl;
        //std::cout<<"L2 norm check = "<<check1_float.l2_norm()<<std::endl;

        DataOut<dim> data_out;
        data_out.attach_dof_handler (global_matrix->get_data().get_dof_handler(global_matrix->get_operator_data().dof_index));

        std::vector<std::string> initial (dim, "initial");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          initial_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
        data_out.add_data_vector (global_matrix->get_data().get_dof_handler(global_matrix->get_operator_data().dof_index),check1, initial, initial_component_interpretation);

        std::vector<std::string> mg_cycle (dim, "mg_cycle");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          mg_cylce_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
        data_out.add_data_vector (global_matrix->get_data().get_dof_handler(global_matrix->get_operator_data().dof_index),check2, mg_cycle, mg_cylce_component_interpretation);

        std::vector<std::string> smoother (dim, "smoother");
        std::vector<DataComponentInterpretation::DataComponentInterpretation>
          smoother_component_interpretation(dim, DataComponentInterpretation::component_is_part_of_vector);
        data_out.add_data_vector (global_matrix->get_data().get_dof_handler(global_matrix->get_operator_data().dof_index),check3, smoother, smoother_component_interpretation);

        data_out.build_patches (global_matrix->get_data().get_dof_handler(global_matrix->get_operator_data().dof_index).get_fe().degree*3);
        std::ostringstream filename;
        filename << "smoothing.vtk";

        std::ofstream output (filename.str().c_str());
        data_out.write_vtk(output);
        std::abort();

        // TODO: update multigrid preconditioner (diagonals) in case of varying parameters
        solver.solve (*global_matrix, dst, src, *preconditioner);
      }
      */
      else
      {
        solver.solve (*global_matrix, dst, src, *preconditioner);
      }
    }
    else if(solver_data.solver_viscous == SolverViscous::GMRES)
    {
      SolverGMRES<parallel::distributed::Vector<value_type> > solver (solver_control);
      if(solver_data.preconditioner_viscous == PreconditionerViscous::None)
      {
        solver.solve (*global_matrix, dst, src, PreconditionIdentity());
      }
      else
      {
        solver.solve (*global_matrix, dst, src, *preconditioner);
      }
    }
    else
    {
      AssertThrow(false,ExcMessage("Specified Viscous Solver not implemented - possibilities are PCG and GMRES"));
    }
  }
  catch (SolverControl::NoConvergence &)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0)
      std::cout << std::endl << "Viscous solver failed to solve to given tolerance." << std::endl;
  }

  return solver_control.last_step();
}

#endif /* INCLUDE_HELMHOLTZSOLVER_H_ */
