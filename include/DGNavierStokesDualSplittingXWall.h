/*
 * DGNavierStokesDualSplittingXWall.h
 *
 *  Created on: Jul 7, 2016
 *      Author: krank
 */

#ifndef INCLUDE_DGNAVIERSTOKESDUALSPLITTINGXWALL_H_
#define INCLUDE_DGNAVIERSTOKESDUALSPLITTINGXWALL_H_

#include "DGNavierStokesDualSplitting.h"
#include "InverseMassMatrixXWall.h"
#include <deal.II/base/utilities.h>

template<int dim> class Enrichment;

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
class DGNavierStokesDualSplittingXWall : public DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>
{
public:
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::value_type value_type;
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::FEFaceEval_Velocity_Velocity_linear FEFaceEval_Velocity_Velocity_linear;
  typedef typename DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::FEEval_Velocity_Velocity_linear FEEval_Velocity_Velocity_linear;


  enum class DofHandlerSelector{
    velocity = static_cast<int>(DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::velocity),
    pressure = static_cast<int>(DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::pressure),
    wdist_tauw = static_cast<int>(DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::DofHandlerSelector::n_variants),
    n_variants = static_cast<typename std::underlying_type<DofHandlerSelector>::type >(wdist_tauw)+1
  };

  enum class QuadratureSelector{
    velocity = static_cast<int>(DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::velocity),
    pressure = static_cast<int>(DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::pressure),
    velocity_nonlinear = static_cast<int>(DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::velocity_nonlinear),
    enriched = static_cast<int>(DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::QuadratureSelector::n_variants),
    n_variants = static_cast<typename std::underlying_type<QuadratureSelector>::type >(enriched)+1
  };

  DGNavierStokesDualSplittingXWall(parallel::distributed::Triangulation<dim> const &triangulation,
                                    InputParametersNavierStokes<dim> const         &parameter)
    :
      DGNavierStokesDualSplitting<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>(triangulation,parameter),
      fe_wdist(QGaussLobatto<1>(1+1)),
      dof_handler_wdist(triangulation),
      fe_param_n(this->param),
      inverse_mass_matrix_operator_xwall(nullptr)
  {
    this->fe_u.reset(new FESystem<dim>(FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(fe_degree+1)),dim,FE_DGQArbitraryNodes<dim>(QGaussLobatto<1>(fe_degree_xwall+1)),dim));
  }

  virtual ~DGNavierStokesDualSplittingXWall(){}

  void setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
              std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_velocity,
              std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> > boundary_descriptor_pressure,
              std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> >     field_functions);

  void update_tauw(parallel::distributed::Vector<value_type> &velocity);

  void precompute_inverse_mass_matrix();

  void rhs_for_inverse_mass_matrix(parallel::distributed::Vector<value_type> &velocity);

  void xwall_projection(parallel::distributed::Vector<value_type> & velocity);

  FE_Q<dim> const & get_fe_wdist() const
  {
    return fe_wdist;
  }

  DoFHandler<dim> const & get_dof_handler_wdist() const
  {
    return  dof_handler_wdist;
  }


private:
  virtual void create_dofs();

  virtual void data_reinit(typename MatrixFree<dim,value_type>::AdditionalData & additional_data);

  void init_wdist();

  void calculate_wall_shear_stress(const parallel::distributed::Vector<value_type> &src);

  void local_rhs_dummy (const MatrixFree<dim,value_type>                &,
                        parallel::distributed::Vector<value_type>      &,
                        const parallel::distributed::Vector<value_type>  &,
                        const std::pair<unsigned int,unsigned int>           &) const;

  void local_rhs_dummy_face (const MatrixFree<dim,value_type>                 &,
                             parallel::distributed::Vector<value_type>      &,
                             const parallel::distributed::Vector<value_type>  &,
                             const std::pair<unsigned int,unsigned int>          &) const;

  void local_rhs_wss_boundary_face (const MatrixFree<dim,value_type>             &data,
                                    parallel::distributed::Vector<value_type>    &dst,
                                    const parallel::distributed::Vector<value_type>  &src,
                                    const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_normalization_boundary_face (const MatrixFree<dim,value_type>             &data,
                                              parallel::distributed::Vector<value_type>    &dst,
                                              const parallel::distributed::Vector<value_type>  &,
                                              const std::pair<unsigned int,unsigned int>   &face_range) const;

  void precompute_spaldings_law();

  void local_precompute_spaldings_law (const MatrixFree<dim,value_type>        &data,
                                       parallel::distributed::Vector<value_type>    &,
                                       const parallel::distributed::Vector<value_type>  &,
                                       const std::pair<unsigned int,unsigned int>   &cell_range);

  // inverse mass matrix velocity
  void local_project_xwall(const MatrixFree<dim,value_type>                &data,
                      parallel::distributed::Vector<value_type>    &dst,
                      const parallel::distributed::Vector<value_type>  &src,
                      const std::pair<unsigned int,unsigned int>          &cell_range);

  // inverse mass matrix velocity
  void local_rhs_for_inverse_mass_matrix(const MatrixFree<dim,value_type>                &data,
                      parallel::distributed::Vector<value_type>    &dst,
                      const parallel::distributed::Vector<value_type>  &src,
                      const std::pair<unsigned int,unsigned int>          &cell_range);

  void setup_helmholtz_preconditioner(HelmholtzOperatorData<dim> &helmholtz_operator_data);

  void setup_projection_solver();

protected:
  void initialize_constraints(const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > &periodic_face_pairs);

  FE_Q<dim>        fe_wdist;
  DoFHandler<dim>  dof_handler_wdist;
  ConstraintMatrix constraint_periodic;
private:
  parallel::distributed::Vector<double> wdist;
  parallel::distributed::Vector<double> tauw;
  parallel::distributed::Vector<double> tauw_n;
  parallel::distributed::Vector<double> tauw_boundary;
  parallel::distributed::Vector<value_type> normalization;
  parallel::distributed::Vector<value_type> force;
  std::vector<unsigned int> vector_to_tauw_boundary;

  FEParameters<dim> fe_param_n;
  std_cxx11::shared_ptr< InverseMassMatrixXWallOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type> > inverse_mass_matrix_operator_xwall;
  AlignedVector<AlignedVector<VectorizedArray<value_type> > > enrichment;
  AlignedVector<AlignedVector<Tensor<1,dim,VectorizedArray<value_type> > > > enrichment_gradient;
};

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup (const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > periodic_face_pairs,
        std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> >                                boundary_descriptor_velocity,
        std_cxx11::shared_ptr<BoundaryDescriptorNavierStokes<dim> >                                boundary_descriptor_pressure,
        std_cxx11::shared_ptr<FieldFunctionsNavierStokes<dim> >                                    field_functions)
{
  DGNavierStokesBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::setup(periodic_face_pairs,boundary_descriptor_velocity,boundary_descriptor_pressure,field_functions);

  //set fe_param in all operators
  this->mass_matrix_operator.set_fe_param(&this->fe_param);
  this->body_force_operator.set_fe_param(&this->fe_param);
  this->gradient_operator.set_fe_param(&this->fe_param);
  this->divergence_operator.set_fe_param(&this->fe_param);
  this->convective_operator.set_fe_param(&this->fe_param);
  this->viscous_operator.set_fe_param(&this->fe_param);
  this->viscous_operator.initialize_viscous_coefficients();

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "\nXWall Initialization:" << std::endl;

  //initialize wall distance and closest wall-node connectivity
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "Initialize wall distance:...";
  init_wdist();
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << " done!" << std::endl;

  //initialize some vectors
  this->data.initialize_dof_vector(tauw, 2);
  tauw = 1.0;
  tauw_n =tauw;
  normalization = tauw;
  force = tauw;
  tauw.update_ghost_values();
  tauw_n.update_ghost_values();

  this->fe_param.setup(&wdist,&tauw,&enrichment,&enrichment_gradient);
  std_cxx11::shared_ptr<Function<dim> > test;
  test.reset(new Enrichment<dim>(this->param.max_wdist_xwall));
  this->fe_param.enrichment_is_within = test;
  fe_param_n.setup(&wdist,&tauw_n);
  fe_param_n.enrichment_is_within = test;

  enrichment.resize(this->data.n_macro_cells());
  enrichment_gradient.resize(this->data.n_macro_cells());
  this->precompute_spaldings_law();

  this->inverse_mass_matrix_operator.reset(new InverseMassMatrixXWallOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type>());
  inverse_mass_matrix_operator_xwall = std::dynamic_pointer_cast<InverseMassMatrixXWallOperator<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,value_type> > (this->inverse_mass_matrix_operator);
  inverse_mass_matrix_operator_xwall->initialize(this->data,this->fe_param,
          static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
          static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity));

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
create_dofs()
{

  // enumerate degrees of freedom
  // multigrid solvers for enrichment not supported
  this->dof_handler_u.distribute_dofs(*this->fe_u);
  this->dof_handler_p.distribute_dofs(this->fe_p);
  this->dof_handler_p.distribute_mg_dofs(this->fe_p);
  dof_handler_wdist.distribute_dofs(fe_wdist);

  unsigned int ndofs_per_cell_velocity    = Utilities::fixed_int_power<fe_degree+1,dim>::value*dim;
  unsigned int ndofs_per_cell_xwall    = Utilities::fixed_int_power<fe_degree_xwall+1,dim>::value*dim;
  unsigned int ndofs_per_cell_pressure    = Utilities::fixed_int_power<fe_degree_p+1,dim>::value;

  ConditionalOStream pcout(std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0);
  pcout << std::endl << "Discontinuous finite element discretization:" << std::endl << std::endl
    << "Velocity:" << std::endl
    << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree
    << " (polynomial) and " << std::setw(10) << std::right << fe_degree_xwall << " (enrichment) " << std::endl
    << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << ndofs_per_cell_velocity
    << " (polynomial) and " << std::setw(10) << std::right << ndofs_per_cell_xwall << " (enrichment) " << std::endl
    << "  number of dofs (velocity):\t" << std::fixed << std::setw(10) << std::right << this->dof_handler_u.n_dofs() << std::endl
    << "Pressure:" << std::endl
    << "  degree of 1D polynomials:\t"  << std::fixed << std::setw(10) << std::right << fe_degree_p << std::endl
    << "  number of dofs per cell:\t"   << std::fixed << std::setw(10) << std::right << ndofs_per_cell_pressure << std::endl
    << "  number of dofs (pressure):\t" << std::fixed << std::setw(10) << std::right << this->dof_handler_p.n_dofs() << std::endl;
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
data_reinit(typename MatrixFree<dim,value_type>::AdditionalData & additional_data)
{
  std::vector<const DoFHandler<dim> * >  dof_handler_vec;

  dof_handler_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::n_variants));
  dof_handler_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity)] = &this->dof_handler_u;
  dof_handler_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure)] = &this->dof_handler_p;
  dof_handler_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::wdist_tauw)] = &dof_handler_wdist;

  std::vector<const ConstraintMatrix *> constraint_matrix_vec;
  constraint_matrix_vec.resize(static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::n_variants));
  ConstraintMatrix constraint_u, constraint_p;
  constraint_u.close();
  constraint_p.close();
  initialize_constraints(additional_data.periodic_face_pairs_level_0);
  constraint_matrix_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity)] = &constraint_u;
  constraint_matrix_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::pressure)] = &constraint_p;
  constraint_matrix_vec[static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::wdist_tauw)] = &constraint_periodic;

  std::vector<Quadrature<1> > quadratures;

  // resize quadratures
  quadratures.resize(static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::n_variants));
  // velocity
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity)]
              = QGauss<1>(fe_degree+1);
  // pressure
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::pressure)]
              = QGauss<1>(fe_degree_p+1);
  // exact integration of nonlinear convective term
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity_nonlinear)]
              = QGauss<1>(fe_degree + (fe_degree+2)/2);
  // high-order integration of enrichment
  quadratures[static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::enriched)]
              = QGauss<1>(n_q_points_1d_xwall);

  this->data.reinit (this->mapping, dof_handler_vec, constraint_matrix_vec, quadratures, additional_data);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
init_wdist()
{
  // layout of aux_vector: 0-dim: normal, dim: distance, dim+1: nearest dof
  // index, dim+2: touch count (for computing weighted normals); normals not
  // currently used
  std::vector<parallel::distributed::Vector<double> > aux_vectors(dim+3);

  // store integer indices in a double. In order not to get overflow, we
  // need to make sure the global index fits into a double -> this limits
  // the maximum size in the dof indices to 2^53 (approx 10^15)
#ifdef DEAL_II_WITH_64BIT_INTEGERS
  AssertThrow(dof_handler_wdist.n_dofs() <
              (types::global_dof_index(1ull) << 53),
              ExcMessage("Sizes larger than 2^53 currently not supported"));
#endif

  IndexSet locally_relevant_set;
  DoFTools::extract_locally_relevant_dofs(this->dof_handler_wdist,
                                          locally_relevant_set);
  aux_vectors[0].reinit(this->dof_handler_wdist.locally_owned_dofs(),
                        locally_relevant_set, MPI_COMM_WORLD);
  for (unsigned int d=1; d<aux_vectors.size(); ++d)
    aux_vectors[d].reinit(aux_vectors[0]);

  // assign distance to close to infinity (we would like to use inf here but
  // there are checks in deal.II whether numbers are finite so we must use a
  // finite number here)
  const double unreached = 1e305;
  aux_vectors[dim] = unreached;

  // TODO: get the actual set of wall (Dirichlet) boundaries as input
  // arguments. Currently, this is matched with what is set in the outer
  // problem type.
  std::set<types::boundary_id> wall_boundaries;
  wall_boundaries.insert(0);

  // set the initial distance for the wall to zero and initialize the normal
  // directions
  {
    QGauss<dim-1> face_quadrature(1);
    FEFaceValues<dim> fe_face_values(this->fe_wdist, face_quadrature,
                                     update_normal_vectors);
    std::vector<types::global_dof_index> dof_indices(this->fe_wdist.dofs_per_face);
    int found = 0;
    for (typename DoFHandler<dim>::active_cell_iterator cell = this->dof_handler_wdist.begin_active(); cell != this->dof_handler_wdist.end(); ++cell)
      if (cell->is_locally_owned())
        for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          if (cell->at_boundary(f) &&
              wall_boundaries.find(cell->face(f)->boundary_id()) !=
              wall_boundaries.end())
            {
              found = 1;
              cell->face(f)->get_dof_indices(dof_indices);
              // get normal vector on face
              fe_face_values.reinit(cell, f);
              const Tensor<1,dim> normal = fe_face_values.normal_vector(0);
              for (unsigned int i=0; i<dof_indices.size(); ++i)
                {
                  for (unsigned int d=0; d<dim; ++d)
                    aux_vectors[d](dof_indices[i]) += normal[d];
                  aux_vectors[dim](dof_indices[i]) = 0.;
                  if(constraint_periodic.is_constrained(dof_indices[i]))
                    aux_vectors[dim+1](dof_indices[i]) = (*constraint_periodic.get_constraint_entries(dof_indices[i]))[0].first;
                  else
                    aux_vectors[dim+1](dof_indices[i]) = dof_indices[i];
                  aux_vectors[dim+2](dof_indices[i]) += 1.;
                }
            }
    int found_global = Utilities::MPI::sum(found,MPI_COMM_WORLD);
    //at least one processor has to have walls
    AssertThrow(found_global>0, ExcMessage("Could not find any wall. Aborting."));
    for (unsigned int i=0; i<aux_vectors[0].local_size(); ++i)
      if (aux_vectors[dim+2].local_element(i) != 0)
        for (unsigned int d=0; d<dim; ++d)
          aux_vectors[d].local_element(i) /= aux_vectors[dim+2].local_element(i);
  }

  // this algorithm finds the closest point on the interface by simply
  // searching locally on each element. This algorithm is only correct for
  // simple meshes (as it searches purely locally and can result in zig-zag
  // paths that are nowhere near optimal on general meshes) but it works in
  // parallel when the mesh can be arbitrarily decomposed among
  // processors. A generic class of algorithms to find the closest point on
  // the wall (not necessarily on a node of the mesh) is by some interface
  // evolution method similar to finding signed distance functions to a
  // given interface (see e.g. Sethian, Level Set Methods and Fast Marching
  // Methods, 2000, Chapter 6). But I do not know how to keep track of the
  // point of origin in those algorithms which is essential here, so skip
  // that for the moment. -- MK, Dec 2015

  // loop as long as we have untracked degrees of freedom. this loop should
  // terminate after a number of steps that is approximately half the width
  // of the mesh in elements
  while (aux_vectors[dim].linfty_norm() == unreached)
    {
      aux_vectors[dim+2] = 0.;
      for (unsigned int d=0; d<dim+2; ++d)
        aux_vectors[d].update_ghost_values();

      // get a pristine vector with the content of the distances at the
      // beginning of the step to distinguish which degrees of freedom were
      // already touched before the current loop and which are in the
      // process of being updated
      parallel::distributed::Vector<double> distances_step(aux_vectors[dim]);
      distances_step.update_ghost_values();

      AssertThrow(this->fe_wdist.dofs_per_cell ==
                  GeometryInfo<dim>::vertices_per_cell, ExcNotImplemented());
      Quadrature<dim> quadrature(this->fe_wdist.get_unit_support_points());
      FEValues<dim> fe_values(this->fe_wdist, quadrature, update_quadrature_points);
      std::vector<types::global_dof_index> dof_indices(this->fe_wdist.dofs_per_cell);

      // go through all locally owned and ghosted cells and compute the
      // nearest point from within the element. Since we have both ghosted
      // and owned cells, we can be sure that the locally owned vector
      // elements get the closest point from the neighborhood
      for (typename DoFHandler<dim>::active_cell_iterator cell =
             this->dof_handler_wdist.begin_active();
           cell != this->dof_handler_wdist.end(); ++cell)
        if (!cell->is_artificial())
          {
            bool cell_is_initialized = false;
            cell->get_dof_indices(dof_indices);

            for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
              // point is unreached -> find the closest point within cell
              // that is already reached
              if (distances_step(dof_indices[v]) == unreached)
                {
                  for (unsigned int w=0; w<GeometryInfo<dim>::vertices_per_cell; ++w)
                    if (distances_step(dof_indices[w]) < unreached)
                      {
                        if (! cell_is_initialized)
                          {
                            fe_values.reinit(cell);
                            cell_is_initialized = true;
                          }

                        // here are the normal vectors in case they should
                        // be necessary in a refined version of the
                        // algorithm
                        /*
                        Tensor<1,dim> normal;
                        for (unsigned int d=0; d<dim; ++d)
                          normal[d] = aux_vectors[d](dof_indices[w]);
                        */
                        const Tensor<1,dim> distance_vec =
                          fe_values.quadrature_point(v) - fe_values.quadrature_point(w);
                        if (distances_step(dof_indices[w]) + distance_vec.norm() <
                            aux_vectors[dim](dof_indices[v]))
                          {
                            aux_vectors[dim](dof_indices[v]) =
                              distances_step(dof_indices[w]) + distance_vec.norm();
                            aux_vectors[dim+1](dof_indices[v]) =
                              aux_vectors[dim+1](dof_indices[w]);
                            for (unsigned int d=0; d<dim; ++d)
                              aux_vectors[d](dof_indices[v]) +=
                                aux_vectors[d](dof_indices[w]);
                            aux_vectors[dim+2](dof_indices[v]) += 1;
                          }
                      }
                }
          }
      for (unsigned int i=0; i<aux_vectors[0].local_size(); ++i)
        if (aux_vectors[dim+2].local_element(i) != 0)
          for (unsigned int d=0; d<dim; ++d)
            aux_vectors[d].local_element(i) /= aux_vectors[dim+2].local_element(i);
    }
  aux_vectors[dim+1].update_ghost_values();

  // at this point we could do a search for closer points in the
  // neighborhood of the points identified before (but it is probably quite
  // difficult to do and one needs to search in layers around a given point
  // to have all data available locally; I currently do not have a good idea
  // to sort out this mess and I am not sure whether we really need
  // something better than the local search above). -- MK, Dec 2015

  // copy the aux vector with extended ghosting into a vector that fits the
  // matrix-free partitioner
  this->data.initialize_dof_vector(wdist, 2);
  AssertThrow(wdist.local_size() == aux_vectors[dim].local_size(),
              ExcMessage("Vector sizes do not match, cannot import wall distances"));
  wdist = aux_vectors[dim];
  wdist.update_ghost_values();

  IndexSet accessed_indices(aux_vectors[dim+1].size());
  {
    // copy the accumulated indices into an index vector
    std::vector<types::global_dof_index> my_indices;
    my_indices.reserve(aux_vectors[dim+1].local_size());
    for (unsigned int i=0; i<aux_vectors[dim+1].local_size(); ++i)
      my_indices.push_back(static_cast<types::global_dof_index>(aux_vectors[dim+1].local_element(i)));
    // sort and compress out duplicates
    std::sort(my_indices.begin(), my_indices.end());
    my_indices.erase(std::unique(my_indices.begin(), my_indices.end()),
                     my_indices.end());
    accessed_indices.add_indices(my_indices.begin(),
                                 my_indices.end());
  }

  // create partitioner for exchange of ghost data (after having computed
  // the vector of wall shear stresses)
  std_cxx11::shared_ptr<const Utilities::MPI::Partitioner> vector_partitioner
    (new Utilities::MPI::Partitioner(this->dof_handler_wdist.locally_owned_dofs(),
                                     accessed_indices, MPI_COMM_WORLD));
  tauw_boundary.reinit(vector_partitioner);

  vector_to_tauw_boundary.resize(wdist.local_size());
  for (unsigned int i=0; i<wdist.local_size(); ++i)
    vector_to_tauw_boundary[i] = vector_partitioner->global_to_local
      (static_cast<types::global_dof_index>(aux_vectors[dim+1].local_element(i)));
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
precompute_inverse_mass_matrix()
{
  inverse_mass_matrix_operator_xwall->reinit();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
rhs_for_inverse_mass_matrix(parallel::distributed::Vector<value_type> & velocity)
{
  this->data.cell_loop(&DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_for_inverse_mass_matrix,
                  this, velocity, velocity);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_for_inverse_mass_matrix (const MatrixFree<dim,value_type>        &data,
                     parallel::distributed::Vector<value_type>    &dst,
                     const parallel::distributed::Vector<value_type>  &src,
                     const std::pair<unsigned int,unsigned int>   &cell_range)
{
  FEEval_Velocity_Velocity_linear fe_eval_velocity(data,this->fe_param,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_velocity.reinit (cell);
    fe_eval_velocity.reinit (cell);
  //  if(fe_eval_velocity.enriched)
    {
      Vector<value_type> vector_result(fe_eval_velocity.dofs_per_cell);
      fe_eval_velocity.read_dof_values(src);
      fe_eval_velocity.evaluate(true,false);
      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; q++)
        fe_eval_velocity.submit_value(fe_eval_velocity.get_value(q),q);
      fe_eval_velocity.integrate(true,false);
      fe_eval_velocity.set_dof_values (dst);
    }
  }
}


template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
xwall_projection(parallel::distributed::Vector<value_type> & velocity)
{

  this->data.cell_loop(&DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_project_xwall,
                   this, velocity, velocity);
  this->apply_inverse_mass_matrix(velocity,velocity);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_project_xwall (const MatrixFree<dim,value_type>        &data,
                     parallel::distributed::Vector<value_type>    &dst,
                     const parallel::distributed::Vector<value_type>  &src,
                     const std::pair<unsigned int,unsigned int>   &cell_range)
{

  FEEval_Velocity_Velocity_linear fe_eval_velocity_n(data,&this->fe_param_n,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));
  //FEEvaluationXWall<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,1,value_type> fe_eval_xwall_n (data,xwallstatevec[0],*xwall.ReturnTauWN(),0,3);
  FEEval_Velocity_Velocity_linear fe_eval_velocity(data,&this->fe_param,
      static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    //first, check if we have an enriched element
    //if so, perform the routine for the enriched elements
    fe_eval_velocity_n.reinit (cell);
    fe_eval_velocity.reinit (cell);
  //  if(fe_eval_velocity.enriched)
    {
      Vector<value_type> vector_result(fe_eval_velocity.dofs_per_cell);
      fe_eval_velocity_n.read_dof_values(src);
      fe_eval_velocity_n.evaluate(true,false);
      for (unsigned int q=0; q<fe_eval_velocity.n_q_points; q++)
        fe_eval_velocity.submit_value(fe_eval_velocity_n.get_value(q),q);
      fe_eval_velocity.integrate(true,false);
      fe_eval_velocity.set_dof_values (dst);
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
update_tauw(parallel::distributed::Vector<value_type> &velocity)
{

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "\nCompute new tauw: ";
  calculate_wall_shear_stress(velocity);
  //mean does not work currently because of all off-wall nodes in the vector
  double tauwmean = tauw.mean_value();
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "mean applied = " << tauwmean << " ";

  value_type tauwmax = tauw.linfty_norm();
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "max = " << tauwmax << " ";

  value_type minloc = 1e9;
  for(unsigned int i = 0; i < tauw.local_size(); ++i)
  {
    if(tauw.local_element(i)>0.0)
    {
      if(minloc > tauw.local_element(i))
        minloc = tauw.local_element(i);
    }
  }
  const value_type minglob = Utilities::MPI::min(minloc, MPI_COMM_WORLD);

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "min = " << minglob << " ";
  if(not this->param.variabletauw)
  {
    if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
      std::cout << "(manually set to 1.0) ";
    tauw = 1.0;
  }

  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << std::endl;

  tauw.update_ghost_values();

  if(this->param.variabletauw)
    this->precompute_spaldings_law();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
calculate_wall_shear_stress (const parallel::distributed::Vector<value_type>      &src)
{
  // initialize
  force = 0.0;
  normalization = 0.0;

  // run loop to compute the local integrals
  // here, we still need tauw, so it should first be swapped with tauw_n afterwards
  this->data.loop (&DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_dummy,
      &DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_dummy_face,
      &DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_wss_boundary_face,
            this, force, src);

  this->data.loop (&DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_dummy,
      &DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_dummy_face,
      &DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_rhs_normalization_boundary_face,
            this, normalization, src);

  // run normalization
  value_type mean = 0.0;
  unsigned int count = 0;
  const double EPSILON = 1e-12;


  for(unsigned int i = 0; i < force.local_size(); ++i)
  {
    if(normalization.local_element(i)>EPSILON)
    {
      tauw_boundary.local_element(i) = force.local_element(i) / normalization.local_element(i);
      mean += tauw_boundary.local_element(i);
      count++;
    }
  }

  mean = Utilities::MPI::sum(mean,MPI_COMM_WORLD);
  count = Utilities::MPI::sum(count,MPI_COMM_WORLD);
  mean /= (value_type)count;
  //prescribe 1% of mean as minimum value
  const double fac = mean * 0.02;
  for(unsigned int i = 0; i < force.local_size(); ++i)
  {
    if(normalization.local_element(i)>EPSILON && tauw_boundary.local_element(i) < fac)
    {
      tauw_boundary.local_element(i) = fac;
    }
  }
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    std::cout << "mean = " << mean << " ";

  // communicate the boundary values for the shear stress to the calling
  // processor and access the data according to the vector_to_tauw_boundary
  // field
  tauw_boundary.update_ghost_values();

  //store old wall shear stress
  tauw_n.swap(tauw);

  for (unsigned int i=0; i<tauw.local_size(); ++i)
    tauw.local_element(i) = (1.-this->param.dtauw)*tauw_n.local_element(i)+this->param.dtauw*tauw_boundary.local_element(vector_to_tauw_boundary[i]);

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
precompute_spaldings_law()
{
  parallel::distributed::Vector<value_type> dummy;
  this->data.cell_loop (&DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::local_precompute_spaldings_law,
            this, dummy, dummy);
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_precompute_spaldings_law (const MatrixFree<dim,value_type>        &data,
                                     parallel::distributed::Vector<value_type>    &,
                                     const parallel::distributed::Vector<value_type>  &,
                                     const std::pair<unsigned int,unsigned int>   &cell_range)
{
  FEEvaluationWrapper<dim,fe_degree,fe_degree_xwall,n_q_points_1d_xwall,dim,value_type,true> fe_eval(data,&this->fe_param,
                              static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity));
  AlignedVector<VectorizedArray<value_type> > enrichment_cell;
  AlignedVector<Tensor<1,dim,VectorizedArray<value_type> > > enrichment_gradient_cell;
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    fe_eval.reinit(cell,enrichment_cell,enrichment_gradient_cell);
    enrichment[cell].resize(enrichment_cell.size());
    enrichment_gradient[cell].resize(enrichment_cell.size());
    for(unsigned int i = 0; i< enrichment_cell.size(); i++)
    {
      enrichment[cell][i] = enrichment_cell[i];
      enrichment_gradient[cell][i] = enrichment_gradient_cell[i];
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_dummy (const MatrixFree<dim,value_type>                &,
            parallel::distributed::Vector<value_type>      &,
            const parallel::distributed::Vector<value_type>  &,
            const std::pair<unsigned int,unsigned int>           &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_dummy_face (const MatrixFree<dim,value_type>                 &,
              parallel::distributed::Vector<value_type>      &,
              const parallel::distributed::Vector<value_type>  &,
              const std::pair<unsigned int,unsigned int>          &) const
{

}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_wss_boundary_face (const MatrixFree<dim,value_type>             &data,
                       parallel::distributed::Vector<value_type>    &dst,
                       const parallel::distributed::Vector<value_type>  &src,
                       const std::pair<unsigned int,unsigned int>          &face_range) const
{
  const double EPSILON = 1e-10;
  //this case is difficult to handle in a general way and would require another template parameter
  //for the case that we are running through here but actually do not have an enriched element
  if(this->fe_param.max_wdist_xwall > EPSILON)
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_face(data,&this->fe_param,true,0);
    //these faces should always be enriched, therefore quadrature rule enriched (3)
    FEFaceEvaluation<dim,1,n_q_points_1d_xwall,1,value_type> fe_eval_tauw(data,true,2,3);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
      {
        fe_eval_velocity_face.reinit (face);
        fe_eval_tauw.reinit (face);
        fe_eval_velocity_face.read_dof_values(src);
        fe_eval_velocity_face.evaluate(false,true);
        AssertThrow(fe_eval_velocity_face.n_q_points == fe_eval_tauw.n_q_points,ExcMessage("\nwrong number of quadrature points"));
        for(unsigned int q=0;q<fe_eval_velocity_face.n_q_points;++q)
        {
          Tensor<1, dim, VectorizedArray<value_type> > average_gradient = fe_eval_velocity_face.get_normal_gradient(q);

          VectorizedArray<value_type> tauwsc = make_vectorized_array<value_type>(0.0);
          tauwsc = average_gradient.norm();
          tauwsc = tauwsc * this->get_viscosity();
          fe_eval_tauw.submit_value(tauwsc,q);
        }
        fe_eval_tauw.integrate(true,false);
        fe_eval_tauw.distribute_local_to_global(dst);
      }
    }
  }
  else
  {
    FEFaceEval_Velocity_Velocity_linear fe_eval_velocity_face(data,&this->fe_param,true,0);
    //these faces should always be enriched, therefore quadrature rule enriched (3)
    FEFaceEvaluation<dim,1,fe_degree+(fe_degree+2)/2,1,value_type> fe_eval_tauw(data,true,2,2);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
      {
        fe_eval_velocity_face.reinit (face);
        fe_eval_tauw.reinit (face);
        fe_eval_velocity_face.read_dof_values(src);
        fe_eval_velocity_face.evaluate(false,true);
        AssertThrow(fe_eval_velocity_face.n_q_points == fe_eval_tauw.n_q_points,ExcMessage("\nwrong number of quadrature points"));
        for(unsigned int q=0;q<fe_eval_velocity_face.n_q_points;++q)
        {
          Tensor<1, dim, VectorizedArray<value_type> > average_gradient = fe_eval_velocity_face.get_normal_gradient(q);

          VectorizedArray<value_type> tauwsc = make_vectorized_array<value_type>(0.0);
          tauwsc = average_gradient.norm();
          tauwsc = tauwsc * this->get_viscosity();
          fe_eval_tauw.submit_value(tauwsc,q);
        }
        fe_eval_tauw.integrate(true,false);
        fe_eval_tauw.distribute_local_to_global(dst);
      }
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
local_rhs_normalization_boundary_face (const MatrixFree<dim,value_type>             &data,
                       parallel::distributed::Vector<value_type>    &dst,
                       const parallel::distributed::Vector<value_type>  &,
                       const std::pair<unsigned int,unsigned int>          &face_range) const
{
  FEFaceEvaluation<dim,1,fe_degree+1,1,value_type> fe_eval_tauw(data,true,2,0);
  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
    {
      fe_eval_tauw.reinit (face);

      for(unsigned int q=0;q<fe_eval_tauw.n_q_points;++q)
        fe_eval_tauw.submit_value(make_vectorized_array<value_type>(1.0),q);

      fe_eval_tauw.integrate(true,false);
      fe_eval_tauw.distribute_local_to_global(dst);
    }
  }
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
initialize_constraints(const std::vector<GridTools::PeriodicFacePair<typename Triangulation<dim>::cell_iterator> > &periodic_face_pairs)
{
  IndexSet xwall_relevant_set;
  DoFTools::extract_locally_relevant_dofs(this->dof_handler_wdist,
                                          xwall_relevant_set);
  constraint_periodic.clear();
  constraint_periodic.reinit(xwall_relevant_set);
  std::vector<GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> >
    periodic_face_pairs_dh(periodic_face_pairs.size());
  for (unsigned int i=0; i<periodic_face_pairs.size(); ++i)
    {
      GridTools::PeriodicFacePair<typename DoFHandler<dim>::cell_iterator> pair;
      pair.cell[0] = typename DoFHandler<dim>::cell_iterator
        (&periodic_face_pairs[i].cell[0]->get_triangulation(),
         periodic_face_pairs[i].cell[0]->level(),
         periodic_face_pairs[i].cell[0]->index(),
         &this->dof_handler_wdist);
      pair.cell[1] = typename DoFHandler<dim>::cell_iterator
        (&periodic_face_pairs[i].cell[1]->get_triangulation(),
         periodic_face_pairs[i].cell[1]->level(),
         periodic_face_pairs[i].cell[1]->index(),
         &this->dof_handler_wdist);
      pair.face_idx[0] = periodic_face_pairs[i].face_idx[0];
      pair.face_idx[1] = periodic_face_pairs[i].face_idx[1];
      pair.orientation = periodic_face_pairs[i].orientation;
      pair.matrix = periodic_face_pairs[i].matrix;
      periodic_face_pairs_dh[i] = pair;
    }
  DoFTools::make_periodicity_constraints<DoFHandler<dim> >(periodic_face_pairs_dh, constraint_periodic);
  DoFTools::make_hanging_node_constraints(this->dof_handler_wdist,
                                          constraint_periodic);

  constraint_periodic.close();
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_helmholtz_preconditioner(HelmholtzOperatorData<dim> &)
{
  this->helmholtz_operator.initialize_strong_homogeneous_dirichlet_boundary_conditions();
  if(this->param.preconditioner_viscous == PreconditionerViscous::InverseMassMatrix)
  {
    this->helmholtz_preconditioner.reset(new InverseMassMatrixPreconditionerPtr<dim,fe_degree,value_type>(
    this->inverse_mass_matrix_operator));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::Jacobi)
  {
    AssertThrow(false,ExcMessage("Jacobi is currently not supported as Helmholtz preconditioner"));
  }
  else if(this->param.preconditioner_viscous == PreconditionerViscous::GeometricMultigrid)
  {
    AssertThrow(false,ExcMessage("GeometricMultigrid is currently not supported as Helmholtz preconditioner"));
  }

  //some further safety checks
  AssertThrow(this->param.solver_viscous == SolverViscous::GMRES,ExcMessage("only gmres allowed"));
  AssertThrow(this->param.IP_formulation_viscous == InteriorPenaltyFormulation::NIPG,ExcMessage("need non-symmetric formulation of viscous part for stability"));
}

template<int dim, int fe_degree, int fe_degree_p, int fe_degree_xwall, int n_q_points_1d_xwall>
void DGNavierStokesDualSplittingXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall>::
setup_projection_solver ()
{
  // initialize projection solver
  ProjectionOperatorData projection_operator_data;
  projection_operator_data.penalty_parameter_divergence = this->param.penalty_factor_divergence;
  projection_operator_data.penalty_parameter_continuity = this->param.penalty_factor_continuity;
  projection_operator_data.solve_stokes_equations = (this->param.equation_type == EquationType::Stokes);

  if(this->param.projection_type == ProjectionType::NoPenalty)
  {
    AssertThrow(false,ExcMessage("This is not implemented, use the local penalty-projection method with zero penalty parameter"));
  }
  else if(this->param.projection_type == ProjectionType::DivergencePenalty &&
          this->param.solver_projection == SolverProjection::LU)
  {
    if(this->projection_operator != nullptr)
    {
      delete this->projection_operator;
      this->projection_operator = nullptr;
    }

    this->projection_operator = new ProjectionOperatorBase<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
        this->data,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity),
        projection_operator_data);
    this->projection_operator->set_fe_param(&this->fe_param);

    this->projection_solver.reset(new DirectProjectionSolverDivergencePenalty
        <dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(this->projection_operator));
  }
  else if(this->param.projection_type == ProjectionType::DivergencePenalty &&
          this->param.solver_projection == SolverProjection::PCG)
  {
    if(this->projection_operator != nullptr)
    {
      delete this->projection_operator;
      this->projection_operator = nullptr;
    }

    this->projection_operator = new ProjectionOperatorDivergencePenaltyXWall<dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
        this->data,
        &this->fe_param,
        static_cast<typename std::underlying_type<DofHandlerSelector>::type >(DofHandlerSelector::velocity),
        static_cast<typename std::underlying_type<QuadratureSelector>::type >(QuadratureSelector::velocity),
        projection_operator_data,
        inverse_mass_matrix_operator_xwall);
    this->projection_operator->set_fe_param(&this->fe_param);

    ProjectionSolverData projection_solver_data;
    projection_solver_data.solver_tolerance_abs = this->param.abs_tol_projection;
    projection_solver_data.solver_tolerance_rel = this->param.rel_tol_projection;
    projection_solver_data.solver_projection = this->param.solver_projection;
    projection_solver_data.preconditioner_projection = this->param.preconditioner_projection;

    this->projection_solver.reset(new IterativeProjectionSolverDivergencePenaltyXWall
        <dim, fe_degree, fe_degree_p, fe_degree_xwall, n_q_points_1d_xwall, value_type>(
                            this->projection_operator,
                              projection_solver_data));
  }
}

#endif /* INCLUDE_DGNAVIERSTOKESDUALSPLITTINGXWALL_H_ */
