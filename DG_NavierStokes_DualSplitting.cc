//---------------------------------------------------------------------------
//    $Id: program.cc 56 2015-02-06 13:05:10Z kronbichler $
//    Version: $Name$
//
//    Copyright (C) 2013 - 2015 by Katharina Kormann and Martin Kronbichler
//
//---------------------------------------------------------------------------

// program based on step-37 but implementing interior penalty DG (currently
// without multigrid)

#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/base/convergence_table.h>
#include <deal.II/base/timer.h>

#include <deal.II/lac/parallel_vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/solver_gmres.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/lac/trilinos_sparse_matrix.h>

#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/fe_system.h>
#include <deal.II/grid/tria.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/tria_boundary_lib.h>
#include <deal.II/grid/grid_generator.h>

#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>

#include <deal.II/numerics/data_out.h>
#include <deal.II/numerics/vector_tools.h>

#include <deal.II/matrix_free/matrix_free.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>

#include <deal.II/integrators/laplace.h>

#include <fstream>
#include <sstream>

// Need to provide a realization of MGTransferPrebuilt with
// parallel::distributed::Vector.
namespace dealii
{
  template <int dim, int spacedim>
  void
  reinit_vector (const DoFHandler<dim,spacedim> &mg_dof,
                 const std::vector<unsigned int> &,
                 MGLevelObject<parallel::distributed::Vector<double> > &v)
  {
    const parallel::distributed::Triangulation<dim,spacedim> *tria =
      (dynamic_cast<const parallel::distributed::Triangulation<dim,spacedim>*>
       (&mg_dof.get_tria()));

    if (tria != 0)
      {
        for (unsigned int level=v.min_level(); level<=v.max_level(); ++level)
          {
            v[level].reinit(mg_dof.locally_owned_mg_dofs(level), tria->get_communicator());
          }
      }
    else
      {
        for (unsigned int level=v.min_level(); level<=v.max_level(); ++level)
          {
            unsigned int n = mg_dof.n_dofs (level);
            v[level].reinit(n);
          }
      }
  }
}

#include <deal.II/multigrid/mg_transfer.templates.h>

namespace dealii
{
  template<class VECTOR>
    MGTransferPrebuilt<VECTOR>::MGTransferPrebuilt ()
  {}


  template<class VECTOR>
    MGTransferPrebuilt<VECTOR>::MGTransferPrebuilt (const ConstraintMatrix &c, const MGConstrainedDoFs &mg_c)
    :
    constraints(&c),
    mg_constrained_dofs(&mg_c)
      {}


  template <class VECTOR>
    MGTransferPrebuilt<VECTOR>::~MGTransferPrebuilt ()
  {}


  template <class VECTOR>
    void MGTransferPrebuilt<VECTOR>::initialize_constraints (
                                                             const ConstraintMatrix &c, const MGConstrainedDoFs &mg_c)
  {
    constraints = &c;
    mg_constrained_dofs = &mg_c;
  }


  template <class VECTOR>
    void MGTransferPrebuilt<VECTOR>::clear ()
  {
    sizes.resize(0);
    prolongation_matrices.resize(0);
    prolongation_sparsities.resize(0);
    copy_indices.resize(0);
    copy_indices_to_me.resize(0);
    copy_indices_from_me.resize(0);
    component_to_block_map.resize(0);
    interface_dofs.resize(0);
    constraints = 0;
    mg_constrained_dofs = 0;
  }


  template <class VECTOR>
    void MGTransferPrebuilt<VECTOR>::prolongate (
                                                 const unsigned int to_level,
                                                 VECTOR            &dst,
                                                 const VECTOR      &src) const
  {
    Assert ((to_level >= 1) && (to_level<=prolongation_matrices.size()),
            ExcIndexRange (to_level, 1, prolongation_matrices.size()+1));

    prolongation_matrices[to_level-1]->vmult (dst, src);
  }


  template <class VECTOR>
    void MGTransferPrebuilt<VECTOR>::restrict_and_add (
                                                       const unsigned int   from_level,
                                                       VECTOR       &dst,
                                                       const VECTOR &src) const
  {
    Assert ((from_level >= 1) && (from_level<=prolongation_matrices.size()),
            ExcIndexRange (from_level, 1, prolongation_matrices.size()+1));
    (void)from_level;

    prolongation_matrices[from_level-1]->Tvmult_add (dst, src);
  }


  template <typename VECTOR>
    template <int dim, int spacedim>
    void MGTransferPrebuilt<VECTOR>::build_matrices (
                                                     const DoFHandler<dim,spacedim>  &mg_dof)
  {
    const unsigned int n_levels      = mg_dof.get_tria().n_global_levels();
    const unsigned int dofs_per_cell = mg_dof.get_fe().dofs_per_cell;

    sizes.resize(n_levels);
    for (unsigned int l=0; l<n_levels; ++l)
      sizes[l] = mg_dof.n_dofs(l);

    // reset the size of the array of
    // matrices. call resize(0) first,
    // in order to delete all elements
    // and clear their memory. then
    // repopulate these arrays
    //
    // note that on resize(0), the
    // shared_ptr class takes care of
    // deleting the object it points to
    // by itself
    prolongation_matrices.resize (0);
    prolongation_sparsities.resize (0);

    for (unsigned int i=0; i<n_levels-1; ++i)
      {
        prolongation_sparsities.push_back
          (std_cxx11::shared_ptr<typename internal::MatrixSelector<VECTOR>::Sparsity> (new typename internal::MatrixSelector<VECTOR>::Sparsity));
        prolongation_matrices.push_back
          (std_cxx11::shared_ptr<typename internal::MatrixSelector<VECTOR>::Matrix> (new typename internal::MatrixSelector<VECTOR>::Matrix));
      }

    // two fields which will store the
    // indices of the multigrid dofs
    // for a cell and one of its children
    std::vector<types::global_dof_index> dof_indices_parent (dofs_per_cell);
    std::vector<types::global_dof_index> dof_indices_child (dofs_per_cell);

    // for each level: first build the sparsity
    // pattern of the matrices and then build the
    // matrices themselves. note that we only
    // need to take care of cells on the coarser
    // level which have children
    for (unsigned int level=0; level<n_levels-1; ++level)
      {

        // reset the dimension of the structure.
        // note that for the number of entries
        // per row, the number of parent dofs
        // coupling to a child dof is
        // necessary. this, of course, is the
        // number of degrees of freedom per
        // cell
        // increment dofs_per_cell
        // since a useless diagonal
        // element will be stored
        DynamicSparsityPattern dsp (sizes[level+1],
                                    sizes[level]);
        std::vector<types::global_dof_index> entries (dofs_per_cell);
        for (typename DoFHandler<dim,spacedim>::cell_iterator cell=mg_dof.begin(level);
             cell != mg_dof.end(level); ++cell)
          if (cell->has_children() &&
              ( mg_dof.get_tria().locally_owned_subdomain()==numbers::invalid_subdomain_id
                || cell->level_subdomain_id()==mg_dof.get_tria().locally_owned_subdomain()
                ))
            {
              cell->get_mg_dof_indices (dof_indices_parent);

              Assert(cell->n_children()==GeometryInfo<dim>::max_children_per_cell,
                     ExcNotImplemented());
              for (unsigned int child=0; child<cell->n_children(); ++child)
                {
                  // set an alias to the prolongation matrix for this child
                  const FullMatrix<double> &prolongation
                    = mg_dof.get_fe().get_prolongation_matrix (child,
                                                               cell->refinement_case());

                  Assert (prolongation.n() != 0, ExcNoProlongation());

                  cell->child(child)->get_mg_dof_indices (dof_indices_child);

                  // now tag the entries in the
                  // matrix which will be used
                  // for this pair of parent/child
                  for (unsigned int i=0; i<dofs_per_cell; ++i)
                    {
                      entries.resize(0);
                      for (unsigned int j=0; j<dofs_per_cell; ++j)
                        if (prolongation(i,j) != 0)
                          entries.push_back (dof_indices_parent[j]);
                      dsp.add_entries (dof_indices_child[i],
                                       entries.begin(), entries.end());
                    }
                }
            }

        internal::MatrixSelector<VECTOR>::reinit(*prolongation_matrices[level],
                                                 *prolongation_sparsities[level],
                                                 level,
                                                 dsp,
                                                 mg_dof);
        dsp.reinit(0,0);

        FullMatrix<double> prolongation;

        // now actually build the matrices
        for (typename DoFHandler<dim,spacedim>::cell_iterator cell=mg_dof.begin(level);
             cell != mg_dof.end(level); ++cell)
          if (cell->has_children() &&
              (mg_dof.get_tria().locally_owned_subdomain()==numbers::invalid_subdomain_id
               || cell->level_subdomain_id()==mg_dof.get_tria().locally_owned_subdomain())
              )
            {
              cell->get_mg_dof_indices (dof_indices_parent);

              Assert(cell->n_children()==GeometryInfo<dim>::max_children_per_cell,
                     ExcNotImplemented());
              for (unsigned int child=0; child<cell->n_children(); ++child)
                {
                  // set an alias to the prolongation matrix for this child
                  prolongation
                    = mg_dof.get_fe().get_prolongation_matrix (child,
                                                               cell->refinement_case());

                  if (mg_constrained_dofs != 0 && mg_constrained_dofs->set_boundary_values())
                    for (unsigned int j=0; j<dofs_per_cell; ++j)
                      if (mg_constrained_dofs->is_boundary_index(level, dof_indices_parent[j]))
                        for (unsigned int i=0; i<dofs_per_cell; ++i)
                          prolongation(i,j) = 0.;

                  cell->child(child)->get_mg_dof_indices (dof_indices_child);

                  // now set the entries in the matrix
                  for (unsigned int i=0; i<dofs_per_cell; ++i)
                    prolongation_matrices[level]->set (dof_indices_child[i],
                                                       dofs_per_cell,
                                                       &dof_indices_parent[0],
                                                       &prolongation(i,0),
                                                       true);
                }
            }
        prolongation_matrices[level]->compress(VectorOperation::insert);
      }

    // Now we are filling the variables copy_indices*, which are essentially
    // maps from global to mgdof for each level stored as a std::vector of
    // pairs. We need to split this map on each level depending on the ownership
    // of the global and mgdof, so that we later not access non-local elements
    // in copy_to/from_mg.
    // We keep track in the bitfield dof_touched which global dof has
    // been processed already (on the current level). This is the same as
    // the multigrid running in serial.
    // Only entering on the finest level gives wrong results (why?)

    copy_indices.resize(n_levels);
    copy_indices_from_me.resize(n_levels);
    copy_indices_to_me.resize(n_levels);
    IndexSet globally_relevant;
    DoFTools::extract_locally_relevant_dofs(mg_dof, globally_relevant);

    std::vector<types::global_dof_index> global_dof_indices (dofs_per_cell);
    std::vector<types::global_dof_index> level_dof_indices  (dofs_per_cell);
    //  for (int level=mg_dof.get_tria().n_levels()-1; level>=0; --level)
    for (unsigned int level=0; level<mg_dof.get_tria().n_levels(); ++level)
      {
        std::vector<bool> dof_touched(globally_relevant.n_elements(), false);
        copy_indices[level].clear();
        copy_indices_from_me[level].clear();
        copy_indices_to_me[level].clear();

        typename DoFHandler<dim,spacedim>::active_cell_iterator
          level_cell = mg_dof.begin_active(level);
        const typename DoFHandler<dim,spacedim>::active_cell_iterator
          level_end  = mg_dof.end_active(level);

        for (; level_cell!=level_end; ++level_cell)
          {
            if (mg_dof.get_tria().locally_owned_subdomain()!=numbers::invalid_subdomain_id
                &&  (level_cell->level_subdomain_id()==numbers::artificial_subdomain_id
                     ||  level_cell->subdomain_id()==numbers::artificial_subdomain_id)
                )
              continue;

            // get the dof numbers of this cell for the global and the level-wise
            // numbering
            level_cell->get_dof_indices (global_dof_indices);
            level_cell->get_mg_dof_indices (level_dof_indices);

            for (unsigned int i=0; i<dofs_per_cell; ++i)
              {
                // we need to ignore if the DoF is on a refinement edge (hanging node)
                if (mg_constrained_dofs != 0
                    && mg_constrained_dofs->at_refinement_edge(level, level_dof_indices[i]))
                  continue;
                unsigned int global_idx = globally_relevant.index_within_set(global_dof_indices[i]);
                //skip if we did this global dof already (on this or a coarser level)
                if (dof_touched[global_idx])
                  continue;
                bool global_mine = mg_dof.locally_owned_dofs().is_element(global_dof_indices[i]);
                bool level_mine = mg_dof.locally_owned_mg_dofs(level).is_element(level_dof_indices[i]);

                if (global_mine && level_mine)
                  copy_indices[level].push_back(
                                                std::pair<unsigned int, unsigned int> (global_dof_indices[i], level_dof_indices[i]));
                else if (level_mine)
                  copy_indices_from_me[level].push_back(
                                                        std::pair<unsigned int, unsigned int> (global_dof_indices[i], level_dof_indices[i]));
                else if (global_mine)
                  copy_indices_to_me[level].push_back(
                                                      std::pair<unsigned int, unsigned int> (global_dof_indices[i], level_dof_indices[i]));
                else
                  continue;

                dof_touched[global_idx] = true;
              }
          }
      }

    // If we are in debugging mode, we order the copy indices, so we get
    // more reliable output for regression texts
#ifdef DEBUG
    std::less<std::pair<types::global_dof_index, unsigned int> > compare;
    for (unsigned int level=0; level<copy_indices.size(); ++level)
      std::sort(copy_indices[level].begin(), copy_indices[level].end(), compare);
    for (unsigned int level=0; level<copy_indices_from_me.size(); ++level)
      std::sort(copy_indices_from_me[level].begin(), copy_indices_from_me[level].end(), compare);
    for (unsigned int level=0; level<copy_indices_to_me.size(); ++level)
      std::sort(copy_indices_to_me[level].begin(), copy_indices_to_me[level].end(), compare);
#endif
  }


  template <class VECTOR>
    void
    MGTransferPrebuilt<VECTOR>::print_matrices (std::ostream &os) const
  {
    for (unsigned int level = 0; level<prolongation_matrices.size(); ++level)
      {
        os << "Level " << level << std::endl;
        prolongation_matrices[level]->print(os);
        os << std::endl;
      }
  }

  template <class VECTOR>
    void
    MGTransferPrebuilt<VECTOR>::print_indices (std::ostream &os) const
  {
    for (unsigned int level = 0; level<copy_indices.size(); ++level)
      {
        for (unsigned int i=0; i<copy_indices[level].size(); ++i)
          os << "copy_indices[" << level
             << "]\t" << copy_indices[level][i].first << '\t' << copy_indices[level][i].second << std::endl;
      }

    for (unsigned int level = 0; level<copy_indices_from_me.size(); ++level)
      {
        for (unsigned int i=0; i<copy_indices_from_me[level].size(); ++i)
          os << "copy_ifrom  [" << level
             << "]\t" << copy_indices_from_me[level][i].first << '\t' << copy_indices_from_me[level][i].second << std::endl;
      }
    for (unsigned int level = 0; level<copy_indices_to_me.size(); ++level)
      {
        for (unsigned int i=0; i<copy_indices_to_me[level].size(); ++i)
          os << "copy_ito    [" << level
             << "]\t" << copy_indices_to_me[level][i].first << '\t' << copy_indices_to_me[level][i].second << std::endl;
      }
  }
}

namespace DG_NavierStokes
{
  using namespace dealii;

  const unsigned int fe_degree = 5;
  const unsigned int fe_degree_p = fe_degree;//fe_degree-1;
  const unsigned int dimension = 2; // dimension >= 2
  const unsigned int refine_steps_min = 3;
  const unsigned int refine_steps_max = 3;

  const double START_TIME = 0.0;
  const double END_TIME = 5.0; // Poisseuille 5.0;  Kovasznay 1.0

  const double VISCOSITY = 0.005; // Taylor vortex: 0.01; vortex problem (Hesthaven): 0.025; Poisseuille 0.005; Kovasznay 0.025; Stokes 1.0
  const double MAX_VELOCITY = 1.0; // Taylor vortex: 1; vortex problem (Hesthaven): 1.5; Poisseuille 1.0; Kovasznay 4.0
  const double stab_factor = 4.0;

  bool pure_dirichlet_bc = false;

  const double lambda = 0.5/VISCOSITY - std::pow(0.25/std::pow(VISCOSITY,2.0)+4.0*std::pow(numbers::PI,2.0),0.5);

  template<int dim>
  class AnalyticalSolution : public Function<dim>
  {
  public:
  AnalyticalSolution (const unsigned int   component,
            const double     time = 0.) : Function<dim>(1, time),component(component) {}

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double AnalyticalSolution<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
      double t = this->get_time();
    double result = 0.0;

    /*********************** cavitiy flow *******************************/
  /*  const double T = 0.1;
    if(component == 0 && (std::abs(p[1]-1.0)<1.0e-15))
      result = t<T? (t/T) : 1.0; */
    /********************************************************************/

    /*********************** Cuette flow problem ************************/
    // stationary
  /*  if(component == 0)
          result = ((p[1]+1.0)*0.5); */

    // instationary
   /* const double T = 1.0;
    if(component == 0)
      result = ((p[1]+1.0)*0.5)*(t<T? (t/T) : 1.0); */
    /********************************************************************/

    /****************** Poisseuille flow problem ************************/
    // constant velocity profile at inflow
   /* const double pressure_gradient = -0.01;
    double T = 0.5;
    if(component == 0 && (std::abs(p[0]+1.0)<1.0e-12))
    result = (t<T? (t/T) : 1.0); */

    // parabolic velocity profile at inflow - stationary
    /*const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
    if(component == 0)
    result = 1.0/VISCOSITY*pressure_gradient*(pow(p[1],2.0)-1.0)/2.0;
    if(component == dim)
    result = (p[0]-1.0)*pressure_gradient;*/

    // parabolic velocity profile at inflow - instationary
    const double pressure_gradient = -2.*VISCOSITY*MAX_VELOCITY;
    double T = 0.5;
    if(component == 0)
      result = 1.0/VISCOSITY*pressure_gradient*(pow(p[1],2.0)-1.0)/2.0*(t<T? (t/T) : 1.0);
    if(component == dim)
    result = (p[0]-1.0)*pressure_gradient*(t<T? (t/T) : 1.0);
    /********************************************************************/

    /************************* vortex problem ***************************/
    //Taylor vortex problem (Shahbazi et al.,2007)
//    const double pi = numbers::PI;
//    if(component == 0)
//      result = (-std::cos(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//    else if(component == 1)
//      result = (+std::sin(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//    else if(component == 2)
//      result = -0.25*(std::cos(2*pi*p[0])+std::cos(2*pi*p[1]))*std::exp(-4.0*pi*pi*t*VISCOSITY);

    // vortex problem (Hesthaven)
//    const double pi = numbers::PI;
//    if(component == 0)
//      result = -std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//    else if(component == 1)
//      result = std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//    else if(component == dim)
//      result = -std::cos(2*pi*p[0])*std::cos(2*pi*p[1])*std::exp(-8.0*pi*pi*VISCOSITY*t);
    /********************************************************************/

    /************************* Kovasznay flow ***************************/
//    const double pi = numbers::PI;
//    if (component == 0)
//      result = 1.0 - std::exp(lambda*p[0])*std::cos(2*pi*p[1]);
//    else if (component == 1)
//      result = lambda/2.0/pi*std::exp(lambda*p[0])*std::sin(2*pi*p[1]);
//    else if (component == dim)
//      result = 0.5*(1.0-std::exp(2.0*lambda*p[0]));
    /********************************************************************/

    /************************* Beltrami flow ****************************/
    /*const double pi = numbers::PI;
    const double a = 0.25*pi;
    const double d = 2*a;
    if (component == 0)
      result = -a*(std::exp(a*p[0])*std::sin(a*p[1]+d*p[2]) + std::exp(a*p[2])*std::cos(a*p[0]+d*p[1]))*std::exp(-VISCOSITY*d*d*t);
    else if (component == 1)
      result = -a*(std::exp(a*p[1])*std::sin(a*p[2]+d*p[0]) + std::exp(a*p[0])*std::cos(a*p[1]+d*p[2]))*std::exp(-VISCOSITY*d*d*t);
    else if (component == 2)
      result = -a*(std::exp(a*p[2])*std::sin(a*p[0]+d*p[1]) + std::exp(a*p[1])*std::cos(a*p[2]+d*p[0]))*std::exp(-VISCOSITY*d*d*t);
    else if (component == dim)
        result = -a*a*0.5*(std::exp(2*a*p[0]) + std::exp(2*a*p[1]) + std::exp(2*a*p[2]) +
                           2*std::sin(a*p[0]+d*p[1])*std::cos(a*p[2]+d*p[0])*std::exp(a*(p[1]+p[2])) +
                           2*std::sin(a*p[1]+d*p[2])*std::cos(a*p[0]+d*p[1])*std::exp(a*(p[2]+p[0])) +
                           2*std::sin(a*p[2]+d*p[0])*std::cos(a*p[1]+d*p[2])*std::exp(a*(p[0]+p[1]))) * std::exp(-2*VISCOSITY*d*d*t);*/
    /********************************************************************/

    /************* Stokes problem (Guermond,2003 & 2006) ****************/
//    const double pi = numbers::PI;
//    double sint = std::sin(t);
//    double sinx = std::sin(pi*p[0]);
//    double siny = std::sin(pi*p[1]);
//    double cosx = std::cos(pi*p[0]);
//    double sin2x = std::sin(2.*pi*p[0]);
//    double sin2y = std::sin(2.*pi*p[1]);
//    if (component == 0)
//      result = pi*sint*sin2y*pow(sinx,2.);
//    else if (component == 1)
//      result = -pi*sint*sin2x*pow(siny,2.);
//    else if (component == dim)
//      result = cosx*siny*sint;
    /********************************************************************/

  return result;
  }

  template<int dim>
  class NeumannBoundaryVelocity : public Function<dim>
  {
  public:
    NeumannBoundaryVelocity (const unsigned int   component,
            const double     time = 0.) : Function<dim>(1, time),component(component) {}

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double NeumannBoundaryVelocity<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
//    double t = this->get_time();
    double result = 0.0;

    // Kovasznay flow
//    const double pi = numbers::PI;
//    if (component == 0)
//      result = -lambda*std::exp(lambda)*std::cos(2*pi*p[1]);
//    else if (component == 1)
//      result = std::pow(lambda,2.0)/2/pi*std::exp(lambda)*std::sin(2*pi*p[1]);

    //Taylor vortex (Shahbazi et al.,2007)
//    const double pi = numbers::PI;
//    if(component == 0)
//      result = (pi*std::sin(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//    else if(component == 1)
//      result = (+pi*std::cos(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);

    // vortex problem (Hesthaven)
//    const double pi = numbers::PI;
//    if(component==0)
//    {
//      if( (std::abs(p[1]+0.5)< 1e-12) && (p[0]<0) )
//        result = 2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//      else if( (std::abs(p[1]-0.5)< 1e-12) && (p[0]>0) )
//        result = -2.0*pi*std::cos(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//    }
//    else if(component==1)
//    {
//      if( (std::abs(p[0]+0.5)< 1e-12) && (p[1]>0) )
//        result = -2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//      else if((std::abs(p[0]-0.5)< 1e-12) && (p[1]<0) )
//        result = 2.0*pi*std::cos(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//    }
    return result;
  }

  template<int dim>
  class NeumannBoundaryPressure : public Function<dim>
  {
  public:
  NeumannBoundaryPressure (const unsigned int   n_components = 1,
                 const double       time = 0.) : Function<dim>(n_components, time) {}

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;
  };

  template<int dim>
  double NeumannBoundaryPressure<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
    double result = 0.0;
    // Kovasznay flow
//    if(std::abs(p[0]+1.0)<1.0e-12)
//      result = lambda*std::exp(2.0*lambda*p[0]);

    // Poiseuille
//    const double pressure_gradient = -0.01;
//    if(std::abs(p[0]+1.0)<1.0e-12)
//      result = -pressure_gradient;

    return result;
  }

  template <int dim, typename Number>
  Tensor<1,dim,Number> get_rhs(Point<dim,Number> &point, double time)
  {
    Tensor<1,dim,Number> rhs;
    for(unsigned int d=0;d<dim;++d)
      rhs[d] = 0.0;

    return rhs;
  }

  template<int dim>
  class RHS : public Function<dim>
  {
  public:
    RHS (const unsigned int   component,
      const double     time = 0.) : Function<dim>(1, time),component(component) {}

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double RHS<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
//  double t = this->get_time();
  double result = 0.0;

  // Stokes problem (Guermond,2003 & 2006)
//  const double pi = numbers::PI;
//  double sint = std::sin(t);
//  double cost = std::cos(t);
//  double sinx = std::sin(pi*p[0]);
//  double siny = std::sin(pi*p[1]);
//  double cosx = std::cos(pi*p[0]);
//  double cosy = std::cos(pi*p[1]);
//  double sin2x = std::sin(2.*pi*p[0]);
//  double sin2y = std::sin(2.*pi*p[1]);
//  if (component == 0)
//    result = pi*cost*sin2y*pow(sinx,2.)
//        - 2.*pow(pi,3.)*sint*sin2y*(1.-4.*pow(sinx,2.))
//        - pi*sint*sinx*siny;
//  else if (component == 1)
//    result = -pi*cost*sin2x*pow(siny,2.)
//        + 2.*pow(pi,3.)*sint*sin2x*(1.-4.*pow(siny,2.))
//        + pi*sint*cosx*cosy;

  return result;
  }

  template<int dim>
  class PressureBC_dudt : public Function<dim>
  {
  public:
    PressureBC_dudt (const unsigned int   component,
            const double     time = 0.) : Function<dim>(1, time),component(component) {}

    virtual double value (const Point<dim> &p,const unsigned int component = 0) const;

  private:
    const unsigned int component;
  };

  template<int dim>
  double PressureBC_dudt<dim>::value(const Point<dim> &p,const unsigned int /* component */) const
  {
//  double t = this->get_time();
  double result = 0.0;

  //Taylor vortex (Shahbazi et al.,2007)
//  const double pi = numbers::PI;
//  if(component == 0)
//    result = (2.0*pi*pi*VISCOSITY*std::cos(pi*p[0])*std::sin(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);
//  else if(component == 1)
//    result = (-2.0*pi*pi*VISCOSITY*std::sin(pi*p[0])*std::cos(pi*p[1]))*std::exp(-2.0*pi*pi*t*VISCOSITY);

  // vortex problem (Hesthaven)
//  const double pi = numbers::PI;
//  if(component == 0)
//    result = 4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[1])*std::exp(-4.0*pi*pi*VISCOSITY*t);
//  else if(component == 1)
//    result = -4.0*pi*pi*VISCOSITY*std::sin(2.0*pi*p[0])*std::exp(-4.0*pi*pi*VISCOSITY*t);

  // Beltrami flow
//  const double pi = numbers::PI;
//  const double a = 0.25*pi;
//  const double d = 2*a;
//  if (component == 0)
//    result = a*VISCOSITY*d*d*(std::exp(a*p[0])*std::sin(a*p[1]+d*p[2]) + std::exp(a*p[2])*std::cos(a*p[0]+d*p[1]))*std::exp(-VISCOSITY*d*d*t);
//  else if (component == 1)
//    result = a*VISCOSITY*d*d*(std::exp(a*p[1])*std::sin(a*p[2]+d*p[0]) + std::exp(a*p[0])*std::cos(a*p[1]+d*p[2]))*std::exp(-VISCOSITY*d*d*t);
//  else if (component == 2)
//    result = a*VISCOSITY*d*d*(std::exp(a*p[2])*std::sin(a*p[0]+d*p[1]) + std::exp(a*p[1])*std::cos(a*p[2]+d*p[0]))*std::exp(-VISCOSITY*d*d*t);

  // Stokes problem (Guermond,2003 & 2006)
//  const double pi = numbers::PI;
//  double cost = std::cos(t);
//  double sinx = std::sin(pi*p[0]);
//  double siny = std::sin(pi*p[1]);
//  double cosx = std::cos(pi*p[0]);
//  double cosy = std::cos(pi*p[1]);
//  double sin2x = std::sin(2.*pi*p[0]);
//  double sin2y = std::sin(2.*pi*p[1]);
//  if (component == 0)
//    result = pi*cost*sin2y*pow(sinx,2.);
//  else if (component == 1)
//    result = -pi*cost*sin2x*pow(siny,2.);

  return result;
  }

  template<int dim, int fe_degree, int fe_degree_p> struct NavierStokesPressureMatrix;
  template<int dim, int fe_degree, int fe_degree_p> struct NavierStokesViscousMatrix;

  template<int dim, int fe_degree, int fe_degree_p>
  class MGCoarsePressure : public MGCoarseGridBase<parallel::distributed::Vector<double> >
  {
  public:
    MGCoarsePressure() {}

    void initialize(const NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p> &pressure)
    {
      ns_pressure_coarse = &pressure;
    }

    virtual void operator() (const unsigned int   level,
                             parallel::distributed::Vector<double> &dst,
                             const parallel::distributed::Vector<double> &src) const
    {
      SolverControl solver_control (1e3, 1e-6);
      SolverCG<parallel::distributed::Vector<double> > solver_coarse (solver_control);
      solver_coarse.solve (*ns_pressure_coarse, dst, src, PreconditionIdentity());
    }

    const  NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p> *ns_pressure_coarse;
  };

  template<int dim, int fe_degree, int fe_degree_p>
  class MGCoarseViscous : public MGCoarseGridBase<parallel::distributed::Vector<double> >
  {
  public:
     MGCoarseViscous() {}

     void initialize(const NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p> &viscous)
     {
       ns_viscous_coarse = &viscous;
     }

     virtual void operator() (const unsigned int   level,
                              parallel::distributed::Vector<double> &dst,
                              const parallel::distributed::Vector<double> &src) const
     {
       SolverControl solver_control (1e3, 1e-6);
       SolverCG<parallel::distributed::Vector<double> > solver_coarse (solver_control);
       solver_coarse.solve (*ns_viscous_coarse, dst, src, PreconditionIdentity());
     }

     const  NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p> *ns_viscous_coarse;
  };

  template<int dim, int fe_degree>
  class XWall
  {
  //time-integration-level routines for xwall
  public:
    XWall(const DoFHandler<dim> &dof_handler,
        std::vector<MatrixFree<dim,double> >* data);

    //initialize everything, e.g.
    //setup of wall distance
    //setup of communication of tauw to off-wall nodes
    //setup quadrature rules
    //possibly setup new matrixfree data object only including the xwall elements
    void initialize()
    {
      std::cout << "\nXWall Initialization:" << std::endl;

      //initialize wall distance and closest wall-node connectivity
      std::cout << "Initialize wall distance:...";
      InitWDist();
      std::cout << " done!" << std::endl;

      //initialize some vectors
      (*mydata).back().initialize_dof_vector(tauw, 2);
    }

    //Update wall shear stress at the beginning of every time step
    void UpdateTauW();

    DoFHandler<dim>* ReturnDofHandlerWallDistance(){return &dof_handler_wall_distance;}
    parallel::distributed::Vector<double>* ReturnWDist(){return &wall_distance;}
  private:

    void InitWDist();

    //calculate wall shear stress based on current solution
    void CalculateWallShearStress(){};

    void L2Projection(){};

    //continuous vectors with linear interpolation
    FE_Q<dim> fe_wall_distance;
    DoFHandler<dim> dof_handler_wall_distance;
    parallel::distributed::Vector<double> wall_distance;
    parallel::distributed::Vector<double> tauw;
    //MatrixFree<dim,value_type> data;
    std::vector<MatrixFree<dim,double> >* mydata;

  public:

  };

  template<int dim, int fe_degree>
  XWall<dim,fe_degree>::XWall(const DoFHandler<dim> &dof_handler,
      std::vector<MatrixFree<dim,double> >* data)
  :fe_wall_distance(1),
   dof_handler_wall_distance(dof_handler.get_tria()),
   mydata(data)
  {
    dof_handler_wall_distance.distribute_dofs(fe_wall_distance);
    dof_handler_wall_distance.distribute_mg_dofs(fe_wall_distance);
  }

  template<int dim, int fe_degree>
  void XWall<dim,fe_degree>::InitWDist()
  {
    // compute wall distance
    (*mydata).back().initialize_dof_vector(wall_distance, 2);
    std::vector<types::global_dof_index> element_dof_indices(fe_wall_distance.dofs_per_cell);
//    for (typename DoFHandler<dim>::active_face_iterator face=dof_handler_wall_distance.begin_active();
//        face != dof_handler_wall_distance.end(); ++face)
//    {
//      //is Dirichlet boundary
//      std::cout << "checking a face" << std::endl;
//      if((((*mydata).back())).get_boundary_indicator(face->index()) == 0)
//        std::cout << "found a face" << std::endl;
//    }

    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler_wall_distance.begin_active();
        cell != dof_handler_wall_distance.end(); ++cell)
      if (cell->is_locally_owned())
      {
        std::vector<double> tmpwdist(GeometryInfo<dim>::vertices_per_cell,1e9);
        //TODO Benjamin
        //this won't work in parallel
        for (typename DoFHandler<dim>::active_cell_iterator cellw=dof_handler_wall_distance.begin_active();
            cellw != dof_handler_wall_distance.end(); ++cellw)
        {
          for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
          {
            typename DoFHandler<dim>::face_iterator face=cellw->face(f);
            //this is a face with dirichlet boundary
            if((*mydata).at(0).get_boundary_indicator(face->boundary_id()) == 0)
            {
              for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
              {
                Point<dim> p = cell->vertex(v);
                for (unsigned int vw=0; vw<GeometryInfo<dim>::vertices_per_face; ++vw)
                {
                  Point<dim> pw =face->vertex(vw);
                  double wdist=1e9;
                  if(dim==2)
                    wdist=std::sqrt((p[0]-pw[0])*(p[0]-pw[0])+(p[1]-pw[1])*(p[1]-pw[1]));
                  else if(dim==3)
                    wdist=std::sqrt((p[0]-pw[0])*(p[0]-pw[0])+(p[1]-pw[1])*(p[1]-pw[1])+(p[2]-pw[2])*(p[2]-pw[2]));
                  else std::cerr << "\nstop" << std::endl;
                  if(wdist < tmpwdist.at(v))
                    tmpwdist.at(v)=wdist;
                }
              }
            }
          }
        }
        //TODO also store the connectivity of the enrichment nodes to these Dirichlet nodes
        //to efficiently communicate the wall shear stress later on
        cell->get_dof_indices(element_dof_indices);
        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
        {
          wall_distance(element_dof_indices[v]) = tmpwdist.at(v);
        }
      }
//    wall_distance.print(std::cout);
//    Vector<double> local_distance_values(fe_wall_distance.dofs_per_cell);
//    for (typename DoFHandler<dim>::active_cell_iterator cell=dof_handler_wall_distance.begin_active();
//        cell != dof_handler_wall_distance.end(); ++cell)
//      if (cell->is_locally_owned())
//      {
//        cell->get_dof_values(wall_distance, local_distance_values);
//        std::cout << "Element with center: " << cell->center() << ": ";
//        for (unsigned int v=0; v<GeometryInfo<dim>::vertices_per_cell; ++v)
//        {
//          std::cout << local_distance_values[v] << " ";
//        }
//        std::cout << std::endl;
//      }
  }

  template<int dim, int fe_degree>
  void XWall<dim,fe_degree>::UpdateTauW()
  {
    std::cout << "\nCompute new tauw: ";
    CalculateWallShearStress();
    std::cout << "L2-project... ";
    L2Projection();
    std::cout << "done!" << std::endl;
  }

  template<int dim, int fe_degree, int fe_degree_p>
  class NavierStokesOperation
  {
  public:
  typedef double value_type;
  static const unsigned int number_vorticity_components = (dim==2) ? 1 : dim;

  NavierStokesOperation(const DoFHandler<dim> &dof_handler,const DoFHandler<dim> &dof_handler_p, const double time_step_size);

  void do_timestep (const double  &cur_time,const double  &delta_t, const unsigned int &time_step_number);

  void  rhs_convection (const std::vector<parallel::distributed::Vector<value_type> > &src,
                std::vector<parallel::distributed::Vector<value_type> >    &dst);

  void  compute_rhs (std::vector<parallel::distributed::Vector<value_type> >  &dst);

  void  apply_viscous (const parallel::distributed::Vector<value_type>     &src,
                   parallel::distributed::Vector<value_type>      &dst) const;

  void  apply_viscous (const parallel::distributed::Vector<value_type> &src,
                parallel::distributed::Vector<value_type>     &dst,
                const unsigned int                 &level) const;

  void  rhs_viscous (const std::vector<parallel::distributed::Vector<value_type> >   &src,
                   std::vector<parallel::distributed::Vector<value_type> >  &dst);

  void  apply_pressure (const parallel::distributed::Vector<value_type>     &src,
                   parallel::distributed::Vector<value_type>      &dst) const;

  void  apply_pressure (const parallel::distributed::Vector<value_type>   &src,
                   parallel::distributed::Vector<value_type>    &dst,
                   const unsigned int                &level) const;

  void  apply_P (parallel::distributed::Vector<value_type> &dst) const;

  void  shift_pressure (parallel::distributed::Vector<value_type>  &pressure);

  void apply_inverse_mass_matrix(const parallel::distributed::Vector<value_type>  &src,
                  parallel::distributed::Vector<value_type>    &dst) const;

  void  rhs_pressure (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                std::vector<parallel::distributed::Vector<value_type> >      &dst);

  void  apply_projection (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                   std::vector<parallel::distributed::Vector<value_type> >      &dst);

  void compute_vorticity (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                      std::vector<parallel::distributed::Vector<value_type> >      &dst);

  void analyse_computing_times();

  std::vector<parallel::distributed::Vector<value_type> > solution_nm, solution_n, velocity_temp, velocity_temp2, solution_np;
  std::vector<parallel::distributed::Vector<value_type> > vorticity_nm, vorticity_n;
  std::vector<parallel::distributed::Vector<value_type> > rhs_convection_nm, rhs_convection_n;
  std::vector<parallel::distributed::Vector<value_type> > f;

  const MatrixFree<dim,value_type> & get_data() const
  {
    return data.back();
  }

  const MatrixFree<dim,value_type> & get_data(const unsigned int level) const
  {
    return data[level];
  }

  void calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal) const;

  void calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal, unsigned int level) const;

  void calculate_diagonal_viscous(parallel::distributed::Vector<value_type> &diagonal, unsigned int level) const;

  XWall<dim,fe_degree>* ReturnXWall(){return &xwall;}

  private:
  //MatrixFree<dim,value_type> data;
  std::vector<MatrixFree<dim,value_type> > data;

  double time, time_step;
  const double viscosity;
  double gamma0;
  double alpha[2], beta[2];
  std::vector<double> computing_times;
  std::vector<double> times_cg_velo;
  std::vector<unsigned int> iterations_cg_velo;
  std::vector<double> times_cg_pressure;
  std::vector<unsigned int> iterations_cg_pressure;

  //NavierStokesPressureMatrix<dim,fe_degree, fe_degree_p> ns_pressure_matrix;
  MGLevelObject<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p> > mg_matrices_pressure;
  MGTransferPrebuilt<parallel::distributed::Vector<double> > mg_transfer_pressure;

  typedef PreconditionChebyshev<NavierStokesPressureMatrix<dim,fe_degree, fe_degree_p>,
                  parallel::distributed::Vector<double> > SMOOTHER_PRESSURE;
  typename SMOOTHER_PRESSURE::AdditionalData smoother_data_pressure;
  MGSmootherPrecondition<NavierStokesPressureMatrix<dim,fe_degree,fe_degree_p>,
    SMOOTHER_PRESSURE, parallel::distributed::Vector<double> > mg_smoother_pressure;
    MGCoarsePressure<dim,fe_degree,fe_degree_p> mg_coarse_pressure;

  MGLevelObject<NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p> > mg_matrices_viscous;
  MGTransferPrebuilt<parallel::distributed::Vector<double> > mg_transfer_viscous;

  typedef PreconditionChebyshev<NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p>,
                  parallel::distributed::Vector<double> > SMOOTHER_VISCOUS;
  typename SMOOTHER_VISCOUS::AdditionalData smoother_data_viscous;
  MGSmootherPrecondition<NavierStokesViscousMatrix<dim,fe_degree,fe_degree_p>,
    SMOOTHER_VISCOUS, parallel::distributed::Vector<double> > mg_smoother_viscous;
    MGCoarseViscous<dim,fe_degree,fe_degree_p> mg_coarse_viscous;

    Point<dim> first_point;
    types::global_dof_index dof_index_first_point;

    XWall<dim,fe_degree> xwall;

  void update_time_integrator();
  void check_time_integrator();

  // impulse equation
  void local_rhs_convection (const MatrixFree<dim,value_type>                &data,
                        std::vector<parallel::distributed::Vector<double> >      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &src,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_rhs_convection_face (const MatrixFree<dim,value_type>              &data,
                  std::vector<parallel::distributed::Vector<double> >      &dst,
                  const std::vector<parallel::distributed::Vector<double> >  &src,
                  const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_convection_boundary_face(const MatrixFree<dim,value_type>              &data,
                      std::vector<parallel::distributed::Vector<double> >      &dst,
                      const std::vector<parallel::distributed::Vector<double> >  &src,
                      const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_compute_rhs (const MatrixFree<dim,value_type>                &data,
                        std::vector<parallel::distributed::Vector<double> >      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_apply_viscous (const MatrixFree<dim,value_type>        &data,
                        parallel::distributed::Vector<double>      &dst,
                        const parallel::distributed::Vector<double>  &src,
                        const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_apply_viscous_face (const MatrixFree<dim,value_type>      &data,
                  parallel::distributed::Vector<double>    &dst,
                  const parallel::distributed::Vector<double>  &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_apply_viscous_boundary_face(const MatrixFree<dim,value_type>      &data,
                      parallel::distributed::Vector<double>    &dst,
                      const parallel::distributed::Vector<double>  &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_rhs_viscous (const MatrixFree<dim,value_type>                &data,
                        std::vector<parallel::distributed::Vector<double> >      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &src,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_rhs_viscous_face (const MatrixFree<dim,value_type>              &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_viscous_boundary_face(const MatrixFree<dim,value_type>              &data,
                    std::vector<parallel::distributed::Vector<double> >      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &src,
                    const std::pair<unsigned int,unsigned int>          &face_range) const;

  // poisson equation
  void local_apply_pressure (const MatrixFree<dim,value_type>          &data,
                            parallel::distributed::Vector<double>      &dst,
                            const parallel::distributed::Vector<double>  &src,
                            const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_apply_pressure_face (const MatrixFree<dim,value_type>      &data,
                  parallel::distributed::Vector<double>    &dst,
                  const parallel::distributed::Vector<double>  &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_apply_pressure_boundary_face(const MatrixFree<dim,value_type>      &data,
                      parallel::distributed::Vector<double>    &dst,
                      const parallel::distributed::Vector<double>  &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_laplace_diagonal(const MatrixFree<dim,value_type>        &data,
                            parallel::distributed::Vector<double>      &dst,
                            const parallel::distributed::Vector<double>  &src,
                            const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_laplace_diagonal_face (const MatrixFree<dim,value_type>      &data,
                  parallel::distributed::Vector<double>    &dst,
                  const parallel::distributed::Vector<double>  &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_laplace_diagonal_boundary_face(const MatrixFree<dim,value_type>      &data,
                      parallel::distributed::Vector<double>    &dst,
                      const parallel::distributed::Vector<double>  &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_diagonal_viscous(const MatrixFree<dim,value_type>        &data,
                            parallel::distributed::Vector<double>      &dst,
                            const parallel::distributed::Vector<double>  &src,
                            const std::pair<unsigned int,unsigned int>  &cell_range) const;

  void local_diagonal_viscous_face (const MatrixFree<dim,value_type>      &data,
                  parallel::distributed::Vector<double>    &dst,
                  const parallel::distributed::Vector<double>  &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_diagonal_viscous_boundary_face(const MatrixFree<dim,value_type>      &data,
                      parallel::distributed::Vector<double>    &dst,
                      const parallel::distributed::Vector<double>  &src,
                      const std::pair<unsigned int,unsigned int>  &face_range) const;

  void local_rhs_pressure (const MatrixFree<dim,value_type>                &data,
                        std::vector<parallel::distributed::Vector<double> >      &dst,
                        const std::vector<parallel::distributed::Vector<double> >    &src,
                        const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_rhs_pressure_face (const MatrixFree<dim,value_type>              &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const;

  void local_rhs_pressure_boundary_face(const MatrixFree<dim,value_type>              &data,
                    std::vector<parallel::distributed::Vector<double> >      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &src,
                    const std::pair<unsigned int,unsigned int>          &face_range) const;

  // inverse mass matrix velocity
  void local_apply_mass_matrix(const MatrixFree<dim,value_type>                &data,
                      std::vector<parallel::distributed::Vector<value_type> >    &dst,
                      const std::vector<parallel::distributed::Vector<value_type> >  &src,
                      const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_apply_mass_matrix(const MatrixFree<dim,value_type>          &data,
                      parallel::distributed::Vector<value_type>      &dst,
                      const parallel::distributed::Vector<value_type>   &src,
                      const std::pair<unsigned int,unsigned int>    &cell_range) const;

  // projection step
  void local_projection (const MatrixFree<dim,value_type>                &data,
                    std::vector<parallel::distributed::Vector<double> >      &dst,
                    const std::vector<parallel::distributed::Vector<double> >    &src,
                    const std::pair<unsigned int,unsigned int>          &cell_range) const;

  void local_compute_vorticity (const MatrixFree<dim,value_type>                &data,
                            std::vector<parallel::distributed::Vector<double> >      &dst,
                            const std::vector<parallel::distributed::Vector<double> >    &src,
                            const std::pair<unsigned int,unsigned int>          &cell_range) const;

  //penalty parameter
  void calculate_penalty_parameter(double &factor) const;
  };

  template<int dim, int fe_degree, int fe_degree_p>
  NavierStokesOperation<dim,fe_degree, fe_degree_p>::NavierStokesOperation(const DoFHandler<dim> &dof_handler,
                                                                       const DoFHandler<dim> &dof_handler_p,
                                                                       const double time_step_size):
  time(0.0),
  time_step(time_step_size),
  viscosity(VISCOSITY),
  gamma0(1.0),
  computing_times(4),
  times_cg_velo(3),
  iterations_cg_velo(3),
  times_cg_pressure(2),
  iterations_cg_pressure(2),
  xwall(dof_handler,&data)
  {
    alpha[0] = 1.0;
    alpha[1] = 0.0;
    beta[0] = 1.0;
    beta[1] = 0.0;

  data.resize(dof_handler_p.get_tria().n_levels());
  //mg_matrices_pressure.resize(dof_handler_p.get_tria().n_levels()-2, dof_handler_p.get_tria().n_levels()-1);
  mg_matrices_pressure.resize(0, dof_handler_p.get_tria().n_levels()-1);
  mg_matrices_viscous.resize(0, dof_handler.get_tria().n_levels()-1);
  gamma0 = 3.0/2.0;
  for (unsigned int level=mg_matrices_pressure.min_level();level<=mg_matrices_pressure.max_level(); ++level)
  {
    // initialize matrix_free_data
    typename MatrixFree<dim,value_type>::AdditionalData additional_data;
    additional_data.mpi_communicator = MPI_COMM_WORLD;
    additional_data.tasks_parallel_scheme =
    MatrixFree<dim,value_type>::AdditionalData::partition_partition;
    additional_data.build_face_info = true;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                          update_quadrature_points | update_normal_vectors |
                          update_values);
    additional_data.level_mg_handler = level;

    std::vector<const DoFHandler<dim> * >  dof_handler_vec;
    dof_handler_vec.push_back(&dof_handler);
    dof_handler_vec.push_back(&dof_handler_p);
    dof_handler_vec.push_back((xwall.ReturnDofHandlerWallDistance()));

    ConstraintMatrix constraint, constraint_p;
    constraint.close();
    constraint_p.close();
    std::vector<const ConstraintMatrix *> constraint_matrix_vec;
    constraint_matrix_vec.push_back(&constraint);
    constraint_matrix_vec.push_back(&constraint_p);
    constraint_matrix_vec.push_back(&constraint);

    std::vector<Quadrature<1> > quadratures;
    quadratures.push_back(QGauss<1>(fe_degree+1));
    quadratures.push_back(QGauss<1>(fe_degree_p+1));
    // quadrature formula 2: exact integration of convective term
    quadratures.push_back(QGauss<1>(fe_degree + (fe_degree+2)/2));

    data[level].reinit (dof_handler_vec, constraint_matrix_vec,
                  quadratures, additional_data);

    mg_matrices_pressure[level].initialize(*this, level);
    mg_matrices_viscous[level].initialize(*this, level);
  }

  mg_transfer_pressure.build_matrices(dof_handler_p);
  mg_coarse_pressure.initialize(mg_matrices_pressure[mg_matrices_pressure.min_level()]);

  mg_transfer_viscous.build_matrices(dof_handler);
  mg_coarse_viscous.initialize(mg_matrices_viscous[mg_matrices_viscous.min_level()]);

  smoother_data_pressure.smoothing_range = 30;
  smoother_data_pressure.degree = 5; //empirically: use degree = 3 - 6
  smoother_data_pressure.eig_cg_n_iterations = 20;
  mg_smoother_pressure.initialize(mg_matrices_pressure, smoother_data_pressure);

  smoother_data_viscous.smoothing_range = 30;
  smoother_data_viscous.degree = 5; //empirically: use degree = 3 - 6
  smoother_data_viscous.eig_cg_n_iterations = 30;
  mg_smoother_viscous.initialize(mg_matrices_viscous, smoother_data_viscous);
  gamma0 = 1.0;

  // initialize solution vectors
  solution_n.resize(dim+1);
  data.back().initialize_dof_vector(solution_n[0], 0);
  for (unsigned int d=1;d<dim;++d)
  {
    solution_n[d] = solution_n[0];
  }
  data.back().initialize_dof_vector(solution_n[dim], 1);
  solution_nm = solution_n;
  solution_np = solution_n;

  velocity_temp.resize(dim);
  data.back().initialize_dof_vector(velocity_temp[0]);
  for (unsigned int d=1;d<dim;++d)
  {
    velocity_temp[d] = velocity_temp[0];
  }
  velocity_temp2 = velocity_temp;

  vorticity_n.resize(number_vorticity_components);
  data.back().initialize_dof_vector(vorticity_n[0]);
  for (unsigned int d=1;d<number_vorticity_components;++d)
  {
    vorticity_n[d] = vorticity_n[0];
  }
  vorticity_nm = vorticity_n;

  rhs_convection_n.resize(dim);
  data.back().initialize_dof_vector(rhs_convection_n[0]);
  for (unsigned int d=1;d<dim;++d)
  {
    rhs_convection_n[d] = rhs_convection_n[0];
  }
  rhs_convection_nm = rhs_convection_n;
  f = rhs_convection_n;

  typename DoFHandler<dim>::active_cell_iterator first_cell = dof_handler_p.begin_active();
  FEValues<dim> fe_values(dof_handler_p.get_fe(),
              Quadrature<dim>(dof_handler_p.get_fe().get_unit_support_points()),
              update_quadrature_points);
  fe_values.reinit(first_cell);
  first_point = fe_values.quadrature_point(0);
  std::vector<types::global_dof_index>
  dof_indices(dof_handler_p.get_fe().dofs_per_cell);
  first_cell->get_dof_indices(dof_indices);
  dof_index_first_point = dof_indices[0];

  xwall.initialize();
  }

  template <int dim, int fe_degree, int fe_degree_p>
  struct NavierStokesPressureMatrix : public Subscriptor
  {
    void initialize(const NavierStokesOperation<dim, fe_degree, fe_degree_p> &ns_op, unsigned int lvl)
    {
      ns_operation = &ns_op;
      level = lvl;
      ns_operation->get_data(level).initialize_dof_vector(diagonal,1);
      ns_operation->calculate_laplace_diagonal(diagonal,level);
    }

    unsigned int m() const
    {
      return ns_operation->get_data(level).get_vector_partitioner(1)->size();
    }

    double el(const unsigned int row,const unsigned int /*col*/) const
    {
      return diagonal(row);
    }

    void vmult (parallel::distributed::Vector<double> &dst,
        const parallel::distributed::Vector<double> &src) const
    {
      dst = 0;
      vmult_add(dst,src);
    }

    void Tvmult (parallel::distributed::Vector<double> &dst,
        const parallel::distributed::Vector<double> &src) const
    {
      dst = 0;
      vmult_add(dst,src);
    }

    void Tvmult_add (parallel::distributed::Vector<double> &dst,
        const parallel::distributed::Vector<double> &src) const
    {
      vmult_add(dst,src);
    }

    void vmult_add (parallel::distributed::Vector<double> &dst,
        const parallel::distributed::Vector<double> &src) const
    {
      if(pure_dirichlet_bc)
      {
        parallel::distributed::Vector<double> temp1(src);
        ns_operation->apply_P(temp1);
        ns_operation->apply_pressure(temp1,dst,level);
        ns_operation->apply_P(dst);
      }
      else
      {
        ns_operation->apply_pressure(src,dst,level);
      }
    }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p> *ns_operation;
    unsigned int level;
    parallel::distributed::Vector<double> diagonal;
  };

/*  template <int dim, int fe_degree, int fe_degree_p>
  struct NavierStokesViscousMatrix
  {
    NavierStokesViscousMatrix(const NavierStokesOperation<dim, fe_degree, fe_degree_p> &ns_op)
    :
      ns_op(ns_op)
    {}

    void vmult (parallel::distributed::Vector<double> &dst,
        const parallel::distributed::Vector<double> &src) const
    {
      ns_op.apply_viscous(src,dst);
    }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p> ns_op;
  }; */

  template <int dim, int fe_degree, int fe_degree_p>
  struct NavierStokesViscousMatrix : public Subscriptor
  {
      void initialize(const NavierStokesOperation<dim, fe_degree, fe_degree_p> &ns_op, unsigned int lvl)
      {
        ns_operation = &ns_op;
        level = lvl;
        ns_operation->get_data(level).initialize_dof_vector(diagonal,0);
        ns_operation->calculate_diagonal_viscous(diagonal,level);
      }

      unsigned int m() const
      {
        return ns_operation->get_data(level).get_vector_partitioner(0)->size();
      }

      double el(const unsigned int row,const unsigned int /*col*/) const
      {
        return diagonal(row);
      }

      void vmult (parallel::distributed::Vector<double> &dst,
          const parallel::distributed::Vector<double> &src) const
      {
        dst = 0;
        vmult_add(dst,src);
      }

      void Tvmult (parallel::distributed::Vector<double> &dst,
          const parallel::distributed::Vector<double> &src) const
      {
        dst = 0;
        vmult_add(dst,src);
      }

      void Tvmult_add (parallel::distributed::Vector<double> &dst,
          const parallel::distributed::Vector<double> &src) const
      {
        vmult_add(dst,src);
      }

      void vmult_add (parallel::distributed::Vector<double> &dst,
          const parallel::distributed::Vector<double> &src) const
      {
        ns_operation->apply_viscous(src,dst,level);
      }

      const NavierStokesOperation<dim,fe_degree, fe_degree_p> *ns_operation;
      unsigned int level;
      parallel::distributed::Vector<double> diagonal;
  };

  template <int dim, int fe_degree, int fe_degree_p>
  struct PreconditionerInverseMassMatrix
  {
    PreconditionerInverseMassMatrix(const NavierStokesOperation<dim, fe_degree, fe_degree_p> &ns_op)
    :
      ns_op(ns_op)
    {}

    void vmult (parallel::distributed::Vector<double> &dst,
        const parallel::distributed::Vector<double> &src) const
    {
      ns_op.apply_inverse_mass_matrix(src,dst);
    }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p> &ns_op;
  };

  template<int dim, int fe_degree, int fe_degree_p>
  struct PreconditionerJacobi
  {
    PreconditionerJacobi(const NavierStokesOperation<dim, fe_degree, fe_degree_p> &ns_op)
    :
        ns_operation(ns_op)
    {
      ns_operation.get_data().initialize_dof_vector(diagonal,1);
      ns_operation.calculate_laplace_diagonal(diagonal);
    }

    void vmult (parallel::distributed::Vector<double> &dst,
        const parallel::distributed::Vector<double> &src) const
    {
      for (unsigned int i=0;i<src.local_size();++i)
      {
          dst.local_element(i) = src.local_element(i)/diagonal.local_element(i);
      }
    }

    const NavierStokesOperation<dim,fe_degree, fe_degree_p> ns_operation;
    parallel::distributed::Vector<double> diagonal;
  };

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  do_timestep (const double  &cur_time,const double  &delta_t, const unsigned int &time_step_number)
  {
  if(time_step_number == 1)
    check_time_integrator();

    const unsigned int output_solver_info_every_timesteps = 1e5;

    time = cur_time;
    time_step = delta_t;

    Timer timer;
    timer.restart();
  /***************** STEP 0: xwall update **********************************/
    xwall.UpdateTauW();
  /*************************************************************************/

  /***************** STEP 1: convective (nonlinear) term ********************/
    rhs_convection(solution_n,rhs_convection_n);
    compute_rhs(f);
    for (unsigned int d=0; d<dim; ++d)
    {
      velocity_temp[d].equ(beta[0],rhs_convection_n[d],beta[1],rhs_convection_nm[d],1.,f[d]); // Stokes problem: velocity_temp[d] = f[d];
      velocity_temp[d].sadd(time_step,alpha[0],solution_n[d],alpha[1],solution_nm[d]);
    }
//    DataOut<dim> data_out;
//    data_out.add_data_vector (data.back().get_dof_handler(0),velocity_temp[0], "velocity1");
//    data_out.add_data_vector (data.back().get_dof_handler(0),velocity_temp[1], "velocity2");
//    data_out.add_data_vector (data.back().get_dof_handler(0),velocity_temp[2], "velocity3");
//    data_out.build_patches ();
//    std::ostringstream filename;
//    filename << "velocity"
//         << ".vtk";
//    std::ofstream output (filename.str().c_str());
//    data_out.write_vtk(output);

    rhs_convection_nm = rhs_convection_n;

    computing_times[0] += timer.wall_time();
  /*************************************************************************/

  /************ STEP 2: solve poisson equation for pressure ****************/
    timer.restart();

    rhs_pressure(velocity_temp,solution_np);
    solution_np[dim] *= -1.0/time_step;

  // set maximum number of iterations, tolerance
  SolverControl solver_control (1e3, 1.e-15);
  SolverCG<parallel::distributed::Vector<double> > solver (solver_control);

//  Timer cg_timer;
//  cg_timer.restart();

  // start CG-iterations with pressure solution at time t_n
  parallel::distributed::Vector<value_type> solution(solution_n[dim]);

  // CG-Solver without preconditioning
//    solver.solve (mg_matrices_pressure[mg_matrices_pressure.max_level()], solution, solution_np[dim], PreconditionIdentity());

//    times_cg_pressure[0] += cg_timer.wall_time();
//    iterations_cg_pressure[0] += solver_control.last_step();
//    cg_timer.restart();
//    solution = solution_n[dim];

    // PCG-Solver with GMG + Chebyshev smoother as a preconditioner
  mg::Matrix<parallel::distributed::Vector<double> > mgmatrix_pressure(mg_matrices_pressure);
  Multigrid<parallel::distributed::Vector<double> > mg_pressure(data.back().get_dof_handler(1),
                             mgmatrix_pressure,
                               mg_coarse_pressure,
                               mg_transfer_pressure,
                               mg_smoother_pressure,
                               mg_smoother_pressure);
  PreconditionMG<dim, parallel::distributed::Vector<double>, MGTransferPrebuilt<parallel::distributed::Vector<double> > >
  preconditioner_pressure(data.back().get_dof_handler(1), mg_pressure, mg_transfer_pressure);
  try
  {
    solver.solve (mg_matrices_pressure[mg_matrices_pressure.max_level()], solution, solution_np[dim], preconditioner_pressure);
  }
  catch (SolverControl::NoConvergence)
  {
    std::cout<<"Multigrid failed. Try CG ..." << std::endl;
    solution=solution_n[dim];
    SolverControl solver_control (1e3, 1.e-15);
    SolverCG<parallel::distributed::Vector<double> > solver (solver_control);
    solver.solve (mg_matrices_pressure[mg_matrices_pressure.max_level()], solution, solution_np[dim], PreconditionIdentity());
  }

//    times_cg_pressure[1] += cg_timer.wall_time();
//    iterations_cg_pressure[1] += solver_control.last_step();

//    if(time_step_number%10 == 0)
//    std::cout << std::endl << "Solve pressure Poisson equation: Number of timesteps: " << time_step_number << std::endl
//          << "CG (no preconditioning):  wall time: " << times_cg_pressure[0]/time_step_number << " Iterations: " << (double)iterations_cg_pressure[0]/time_step_number << std::endl
//          << "PCG (GMG with Chebyshev): wall time: " << times_cg_pressure[1]/time_step_number << " Iterations: " << (double)iterations_cg_pressure[1]/time_step_number << std::endl
//          << std::endl;

//  if (false)
//  {
//    parallel::distributed::Vector<double> check1(mg_matrices_pressure[mg_matrices_pressure.max_level()].m()),
//        check2(check1), tmp(check1),
//        check3(check1);
//    for (unsigned int i=0; i<check1.size(); ++i)
//      check1(i) = (double)rand()/RAND_MAX;
//    mg_matrices_pressure[mg_matrices_pressure.max_level()].vmult(tmp, check1);
//    tmp *= -1.0;
//    preconditioner_pressure.vmult(check2, tmp);
//    check2 += check1;
//
//    mg_smoother_pressure.smooth(mg_matrices_pressure.max_level(), check3, tmp);
//    //check3 += check1;
//
//    DataOut<dim> data_out;
//    data_out.attach_dof_handler (data.back().get_dof_handler(1));
//    data_out.add_data_vector (check1, "initial");
//    data_out.add_data_vector (check2, "mg_cycle");
//    data_out.add_data_vector (check3, "smoother");
//    data_out.build_patches (data.back().get_dof_handler(1).get_fe().degree*3);
//    std::ostringstream filename;
//    filename << "smoothing-"
//         << Utilities::int_to_string(data.back().get_dof_handler(1).get_tria().n_active_cells(), 6)
//         << ".vtk";
//    std::ofstream output (filename.str().c_str());
//    data_out.write_vtk(output);
//    std::abort();
//  }

  if(pure_dirichlet_bc)
  {
    shift_pressure(solution);
  }
  solution_np[dim] = solution;

    if(time_step_number%output_solver_info_every_timesteps == 0)
  {
      std::cout << std::endl << "Number of timesteps: " << time_step_number << std::endl;
      std::cout << "Solve Poisson equation for p: PCG iterations: " << std::setw(3) << solver_control.last_step() << "  Wall time: " << timer.wall_time() << std::endl;
  }

  computing_times[1] += timer.wall_time();
  /*************************************************************************/

  /********************** STEP 3: projection *******************************/
    timer.restart();

  apply_projection(solution_np,velocity_temp2);
  for (unsigned int d=0; d<dim; ++d)
  {
    velocity_temp2[d].sadd(time_step,1.0,velocity_temp[d]);
  }
  computing_times[2] += timer.wall_time();
  /*************************************************************************/

  /************************ STEP 4: viscous term ***************************/
    timer.restart();

  rhs_viscous(velocity_temp2,solution_np);

  // set maximum number of iterations, tolerance
  SolverControl solver_control_velocity (1e3, 1.e-15);
  SolverCG<parallel::distributed::Vector<double> > solver_velocity (solver_control_velocity);
  //NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p> ns_viscous_matrix(*this);
  for (unsigned int d=0;d<dim;++d)
  {
    double wall_time_temp = timer.wall_time();

//    Timer cg_timer_viscous;
//    cg_timer_viscous.restart();

    // start CG-iterations with solution_n
    parallel::distributed::Vector<value_type> solution(solution_n[d]);

    // CG-Solver without preconditioning
    //solver_velocity.solve (ns_viscous_matrix, solution, solution_np[d], PreconditionIdentity());
//    solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], PreconditionIdentity());

//    times_cg_velo[0] += cg_timer_viscous.wall_time();
//    iterations_cg_velo[0] += solver_control_velocity.last_step();
//    cg_timer_viscous.restart();
//    solution = solution_n[d];

    // PCG-Solver with inverse mass matrix as a preconditioner
    // solver_velocity.solve (ns_viscous_matrix, solution, solution_np[d], preconditioner);
    PreconditionerInverseMassMatrix<dim,fe_degree, fe_degree_p> preconditioner(*this);
    solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], preconditioner);

//    times_cg_velo[1] += cg_timer_viscous.wall_time();
//    iterations_cg_velo[1] += solver_control_velocity.last_step();
//    cg_timer_viscous.restart();
//    solution = solution_n[d];

    // PCG-Solver with GMG + Chebyshev smoother as a preconditioner
//    mg::Matrix<parallel::distributed::Vector<double> > mgmatrix_viscous(mg_matrices_viscous);
//    Multigrid<parallel::distributed::Vector<double> > mg_viscous(data.back().get_dof_handler(0),
//                               mgmatrix_viscous,
//                               mg_coarse_viscous,
//                               mg_transfer_viscous,
//                               mg_smoother_viscous,
//                               mg_smoother_viscous);
//    PreconditionMG<dim, parallel::distributed::Vector<double>, MGTransferPrebuilt<parallel::distributed::Vector<double> > >
//    preconditioner_viscous(data.back().get_dof_handler(0), mg_viscous, mg_transfer_viscous);
//    solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], preconditioner_viscous);

    // PCG-Solver with Chebyshev preconditioner
//    PreconditionChebyshev<NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p>,parallel::distributed::Vector<value_type> > precondition_chebyshev;
//    typename PreconditionChebyshev<NavierStokesViscousMatrix<dim,fe_degree, fe_degree_p>,parallel::distributed::Vector<value_type> >::AdditionalData smoother_data;
//    smoother_data.smoothing_range = 30;
//    smoother_data.degree = 5;
//    smoother_data.eig_cg_n_iterations = 30;
//    precondition_chebyshev.initialize(mg_matrices_viscous[mg_matrices_viscous.max_level()], smoother_data);
//    solver_velocity.solve (mg_matrices_viscous[mg_matrices_viscous.max_level()], solution, solution_np[d], precondition_chebyshev);

//    times_cg_velo[2] += cg_timer_viscous.wall_time();
//    iterations_cg_velo[2] += solver_control_velocity.last_step();

    solution_np[d] = solution;

    if(time_step_number%output_solver_info_every_timesteps == 0)
    {
    std::cout << "Solve viscous step for u" << d+1 <<":    PCG iterations: " << std::setw(3) << solver_control_velocity.last_step() << "  Wall time: " << timer.wall_time()-wall_time_temp << std::endl;
    }
  }
//  if(time_step_number%10 == 0)
//    std::cout << "Solve viscous step for u: Number of timesteps: " << time_step_number << std::endl
//          << "CG (no preconditioning):  wall time: " << times_cg_velo[0]/time_step_number << " Iterations: " << (double)iterations_cg_velo[0]/time_step_number/dim << std::endl
//          << "PCG (inv mass precond.):  wall time: " << times_cg_velo[1]/time_step_number << " Iterations: " << (double)iterations_cg_velo[1]/time_step_number/dim << std::endl
//          << "PCG (GMG with Chebyshev): wall time: " << times_cg_velo[2]/time_step_number << " Iterations: " << (double)iterations_cg_velo[2]/time_step_number/dim << std::endl
//          << std::endl;

  computing_times[3] += timer.wall_time();
  /*************************************************************************/

  // solution at t_n -> solution at t_n-1    and    solution at t_n+1 -> solution at t_n
  solution_nm.swap(solution_n);
  solution_n.swap(solution_np);

  vorticity_nm = vorticity_n;
  compute_vorticity(solution_n,vorticity_n);

  if(time_step_number == 1)
    update_time_integrator();
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  update_time_integrator ()
  {
    gamma0 = 3.0/2.0;
    alpha[0] = 2.0;
    alpha[1] = -0.5;
    beta[0] = 2.0;
    beta[1] = -1.0;
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  check_time_integrator()
  {
    if (std::abs(gamma0-1.0)>1.e-12 || std::abs(alpha[0]-1.0)>1.e-12 || std::abs(alpha[1]-0.0)>1.e-12 || std::abs(beta[0]-1.0)>1.e-12 || std::abs(beta[1]-0.0)>1.e-12)
    {
      std::cout<< "Time integrator constants invalid!" << std::endl;
      std::abort();
    }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  analyse_computing_times()
  {
  double time=0.0;
  for(unsigned int i=0;i<4;++i)
    time+=computing_times[i];
  std::cout<<std::endl<<"Computing times:"
       <<std::endl<<"Step 1: Convection:\t"<<computing_times[0]/time
       <<std::endl<<"Step 2: Pressure:\t"<<computing_times[1]/time
       <<std::endl<<"Step 3: Projection:\t"<<computing_times[2]/time
       <<std::endl<<"Step 4: Viscous:\t"<<computing_times[3]/time
       <<std::endl<<"Time (Step 1-4):\t"<<time<<std::endl;
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  calculate_penalty_parameter(double &factor) const
  {
//TODO Benjamin: why is h missing here?
  // penalty parameter = stab_factor*(p+1)(p+d)*2/h
  factor = stab_factor * (fe_degree +1.0) * (fe_degree + dim) * 2.0;
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal) const
  {
    parallel::distributed::Vector<value_type> src(laplace_diagonal);
    data.back().loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal_face,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal_boundary_face,
              this, laplace_diagonal, src);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  calculate_laplace_diagonal(parallel::distributed::Vector<value_type> &laplace_diagonal, unsigned int level) const
  {
    parallel::distributed::Vector<value_type> src(laplace_diagonal);
    data[level].loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal_face,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_laplace_diagonal_boundary_face,
              this, laplace_diagonal, src);

    if(pure_dirichlet_bc)
    {
      parallel::distributed::Vector<value_type> vec1(laplace_diagonal);
      for(unsigned int i=0;i<vec1.local_size();++i)
        vec1.local_element(i) = 1.;
      parallel::distributed::Vector<value_type> d;
      d.reinit(laplace_diagonal);
      apply_pressure(vec1,d,level);
      double length = vec1*vec1;
      double factor = vec1*d;
      laplace_diagonal.add(-2./length,d,factor/pow(length,2.),vec1);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_laplace_diagonal (const MatrixFree<dim,value_type>        &data,
            parallel::distributed::Vector<double>      &dst,
            const parallel::distributed::Vector<double>    &,
            const std::pair<unsigned int,unsigned int>     &cell_range) const
  {
  FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    pressure.reinit (cell);

    VectorizedArray<value_type> local_diagonal_vector[pressure.tensor_dofs_per_cell];
    for (unsigned int j=0; j<pressure.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<pressure.dofs_per_cell; ++i)
        pressure.begin_dof_values()[i] = make_vectorized_array(0.);
      pressure.begin_dof_values()[j] = make_vectorized_array(1.);
      pressure.evaluate (false,true,false);
      for (unsigned int q=0; q<pressure.n_q_points; ++q)
      {
      pressure.submit_gradient (pressure.get_gradient(q), q);
      }
      pressure.integrate (false,true);
      local_diagonal_vector[j] = pressure.begin_dof_values()[j];
    }
    for (unsigned int j=0; j<pressure.dofs_per_cell; ++j)
      pressure.begin_dof_values()[j] = local_diagonal_vector[j];
    pressure.distribute_local_to_global (dst);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_laplace_diagonal_face (const MatrixFree<dim,value_type>       &data,
                  parallel::distributed::Vector<double>    &dst,
                  const parallel::distributed::Vector<double>  &,
                  const std::pair<unsigned int,unsigned int>  &face_range) const
  {
  FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);
  FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval_neighbor(data,false,1,1);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval.reinit (face);
    fe_eval_neighbor.reinit (face);

    /*VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction()) +
             std::abs(fe_eval_neighbor.get_normal_volume_fraction())) *
      (value_type)(fe_degree * (fe_degree + 1.0)) * 0.5   *stab_factor; */

      double factor = 1.;
      calculate_penalty_parameter(factor);
      VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;

    // element-
    VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
        fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
      for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
        fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array(0.);

      fe_eval.begin_dof_values()[j] = make_vectorized_array(1.);

      fe_eval.evaluate(true,true);
      fe_eval_neighbor.evaluate(true,true);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> valueM = fe_eval.get_value(q);
        VectorizedArray<value_type> valueP = fe_eval_neighbor.get_value(q);

        VectorizedArray<value_type> jump_value = valueM - valueP;
        VectorizedArray<value_type> average_gradient =
            ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
        fe_eval.submit_value(-average_gradient,q);
      }
      fe_eval.integrate(true,true);
      local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
    }
    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
    fe_eval.distribute_local_to_global(dst);

    // neighbor (element+)
    VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
        fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
      for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
        fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array(0.);

      fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array(1.);

      fe_eval.evaluate(true,true);
      fe_eval_neighbor.evaluate(true,true);

      for(unsigned int q=0;q<fe_eval_neighbor.n_q_points;++q)
      {
        VectorizedArray<value_type> valueM = fe_eval.get_value(q);
        VectorizedArray<value_type> valueP = fe_eval_neighbor.get_value(q);

        VectorizedArray<value_type> jump_value = valueM - valueP;
        VectorizedArray<value_type> average_gradient =
            ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval_neighbor.submit_normal_gradient(-0.5*jump_value,q);
        fe_eval_neighbor.submit_value(average_gradient,q);
      }
      fe_eval_neighbor.integrate(true,true);
      local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
    }
    for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
      fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];
    fe_eval_neighbor.distribute_local_to_global(dst);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_laplace_diagonal_boundary_face (const MatrixFree<dim,value_type>         &data,
                        parallel::distributed::Vector<double>      &dst,
                        const parallel::distributed::Vector<double>    &,
                        const std::pair<unsigned int,unsigned int>    &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      //VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
      //  (value_type)(fe_degree * (fe_degree + 1.0))  *stab_factor;

      double factor = 1.;
      calculate_penalty_parameter(factor);
      VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;

    VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
      {
        fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
      }
      fe_eval.begin_dof_values()[j] = make_vectorized_array(1.);
      fe_eval.evaluate(true,true);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
        {
          //set pressure gradient in normal direction to zero, i.e. pressure+ = pressure-, grad+ = -grad-
          VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
          VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval.submit_normal_gradient(-0.5*jump_value,q);
          fe_eval.submit_value(-average_gradient,q);
        }
        else if (data.get_boundary_indicator(face) == 1) // outflow boundaries
        {
          //set pressure to zero, i.e. pressure+ = - pressure- , grad+ = grad-
          VectorizedArray<value_type> valueM = fe_eval.get_value(q);

          VectorizedArray<value_type> jump_value = 2.0*valueM;
          VectorizedArray<value_type> average_gradient = fe_eval.get_normal_gradient(q);
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval.submit_normal_gradient(-0.5*jump_value,q);
          fe_eval.submit_value(-average_gradient,q);
        }
      }
      fe_eval.integrate(true,true);
      local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
      }
    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
      fe_eval.distribute_local_to_global(dst);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  calculate_diagonal_viscous(parallel::distributed::Vector<value_type> &diagonal, unsigned int level) const
  {
    parallel::distributed::Vector<value_type> src(diagonal);
    data[level].loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_diagonal_viscous,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_diagonal_viscous_face,
              &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_diagonal_viscous_boundary_face,
              this, diagonal, src);
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_diagonal_viscous (const MatrixFree<dim,value_type>        &data,
               parallel::distributed::Vector<double>    &dst,
               const parallel::distributed::Vector<double>  &src,
               const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
   FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> velocity (data,0,0);

   for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
   {
     velocity.reinit (cell);

    VectorizedArray<value_type> local_diagonal_vector[velocity.tensor_dofs_per_cell];
    for (unsigned int j=0; j<velocity.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<velocity.dofs_per_cell; ++i)
        velocity.begin_dof_values()[i] = make_vectorized_array(0.);
      velocity.begin_dof_values()[j] = make_vectorized_array(1.);
      velocity.evaluate (true,true,false);
      for (unsigned int q=0; q<velocity.n_q_points; ++q)
      {
      velocity.submit_value (gamma0/time_step*velocity.get_value(q), q);
      velocity.submit_gradient (make_vectorized_array<value_type>(viscosity)*velocity.get_gradient(q), q);
      }
      velocity.integrate (true,true);
      local_diagonal_vector[j] = velocity.begin_dof_values()[j];
    }
    for (unsigned int j=0; j<velocity.dofs_per_cell; ++j)
      velocity.begin_dof_values()[j] = local_diagonal_vector[j];
     velocity.distribute_local_to_global (dst);
   }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_diagonal_viscous_face (const MatrixFree<dim,value_type>       &data,
                   parallel::distributed::Vector<double>    &dst,
                   const parallel::distributed::Vector<double>  &src,
                   const std::pair<unsigned int,unsigned int>  &face_range) const
  {
     FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,0,0);
     FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,0,0);

     for(unsigned int face=face_range.first; face<face_range.second; face++)
     {
       fe_eval.reinit (face);
       fe_eval_neighbor.reinit (face);

       double factor = 1.;
       calculate_penalty_parameter(factor);
       VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;

       // element-
       VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
        fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
      for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
        fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array(0.);

      fe_eval.begin_dof_values()[j] = make_vectorized_array(1.);

      fe_eval.evaluate(true,true);
      fe_eval_neighbor.evaluate(true,true);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> uM = fe_eval.get_value(q);
        VectorizedArray<value_type> uP = fe_eval_neighbor.get_value(q);

        VectorizedArray<value_type> jump_value = uM - uP;
        VectorizedArray<value_type> average_gradient =
            ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * make_vectorized_array<value_type>(0.5);
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
        fe_eval.submit_value(-viscosity*average_gradient,q);
      }
      fe_eval.integrate(true,true);
      local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
       }
    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
       fe_eval.distribute_local_to_global(dst);

       // neighbor (element+)
    VectorizedArray<value_type> local_diagonal_vector_neighbor[fe_eval_neighbor.tensor_dofs_per_cell];
    for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
    {
      for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
        fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
      for (unsigned int i=0; i<fe_eval_neighbor.dofs_per_cell; ++i)
        fe_eval_neighbor.begin_dof_values()[i] = make_vectorized_array(0.);

      fe_eval_neighbor.begin_dof_values()[j] = make_vectorized_array(1.);

      fe_eval.evaluate(true,true);
      fe_eval_neighbor.evaluate(true,true);

        for(unsigned int q=0;q<fe_eval.n_q_points;++q)
        {
          VectorizedArray<value_type> uM = fe_eval.get_value(q);
          VectorizedArray<value_type> uP = fe_eval_neighbor.get_value(q);

          VectorizedArray<value_type> jump_value = uM - uP;
          VectorizedArray<value_type> average_gradient =
              ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * make_vectorized_array<value_type>(0.5);
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval_neighbor.submit_normal_gradient(-0.5*viscosity*jump_value,q);
          fe_eval_neighbor.submit_value(viscosity*average_gradient,q);
        }
      fe_eval_neighbor.integrate(true,true);
      local_diagonal_vector_neighbor[j] = fe_eval_neighbor.begin_dof_values()[j];
    }
    for (unsigned int j=0; j<fe_eval_neighbor.dofs_per_cell; ++j)
      fe_eval_neighbor.begin_dof_values()[j] = local_diagonal_vector_neighbor[j];
    fe_eval_neighbor.distribute_local_to_global(dst);
     }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_diagonal_viscous_boundary_face (const MatrixFree<dim,value_type>       &data,
                       parallel::distributed::Vector<double>    &dst,
                       const parallel::distributed::Vector<double>  &src,
                       const std::pair<unsigned int,unsigned int>  &face_range) const
  {
     FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,0,0);

     for(unsigned int face=face_range.first; face<face_range.second; face++)
     {
       fe_eval.reinit (face);

       double factor = 1.;
       calculate_penalty_parameter(factor);
       VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;

       VectorizedArray<value_type> local_diagonal_vector[fe_eval.tensor_dofs_per_cell];
       for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
       {
         for (unsigned int i=0; i<fe_eval.dofs_per_cell; ++i)
      {
        fe_eval.begin_dof_values()[i] = make_vectorized_array(0.);
      }
      fe_eval.begin_dof_values()[j] = make_vectorized_array(1.);
      fe_eval.evaluate(true,true);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          VectorizedArray<value_type> uM = fe_eval.get_value(q);
          VectorizedArray<value_type> uP = -uM;

          VectorizedArray<value_type> jump_value = uM - uP;
          VectorizedArray<value_type> average_gradient = fe_eval.get_normal_gradient(q);
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
          fe_eval.submit_value(-viscosity*average_gradient,q);
        }
        else if (data.get_boundary_indicator(face) == 1) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
          VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
          VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
          fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
          fe_eval.submit_value(-viscosity*average_gradient,q);
        }
      }
      fe_eval.integrate(true,true);
      local_diagonal_vector[j] = fe_eval.begin_dof_values()[j];
       }
    for (unsigned int j=0; j<fe_eval.dofs_per_cell; ++j)
      fe_eval.begin_dof_values()[j] = local_diagonal_vector[j];
       fe_eval.distribute_local_to_global(dst);
     }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  rhs_convection (const std::vector<parallel::distributed::Vector<value_type> >   &src,
            std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<dim;++d)
    dst[d] = 0;

  // data.loop
  data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_convection,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_convection_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_convection_boundary_face,
            this, dst, src);
  // data.cell_loop
  data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_mass_matrix,
                             this, dst, dst);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  compute_rhs (std::vector<parallel::distributed::Vector<value_type> >  &dst)
  {
  for(unsigned int d=0;d<dim;++d)
    dst[d] = 0;

  // data.loop
  data.back().cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_compute_rhs,this, dst, dst);
  // data.cell_loop
  data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_mass_matrix,
                             this, dst, dst);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_viscous (const parallel::distributed::Vector<value_type>   &src,
              parallel::distributed::Vector<value_type>      &dst) const
  {
  dst = 0;

  data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous_boundary_face,
            this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_viscous (const parallel::distributed::Vector<value_type>   &src,
              parallel::distributed::Vector<value_type>      &dst,
            const unsigned int                &level) const
  {
  //dst = 0;
  data[level].loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_viscous_boundary_face,
            this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  rhs_viscous (const std::vector<parallel::distributed::Vector<value_type> >   &src,
            std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<dim;++d)
    dst[d] = 0;

  data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_viscous,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_viscous_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_viscous_boundary_face,
            this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_convection (const MatrixFree<dim,value_type>              &data,
            std::vector<parallel::distributed::Vector<double> >      &dst,
            const std::vector<parallel::distributed::Vector<double> >  &src,
            const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
  // inexact integration  (data,0,0) : second argument: which dof-handler, third argument: which quadrature
//  FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity (data,0,0);

  // exact integration of convective term
  FEEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> velocity (data,0,2);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit (cell);
    velocity.read_dof_values(src,0);
    velocity.evaluate (true,false,false);

    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      // nonlinear convective flux F(u) = uu
      Tensor<1,dim,VectorizedArray<value_type> > u = velocity.get_value(q);
      Tensor<2,dim,VectorizedArray<value_type> > F;
      outer_product(F,u,u);
    velocity.submit_gradient (F, q);
    }
    velocity.integrate (false,true);
    velocity.distribute_local_to_global (dst,0);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_convection_face (const MatrixFree<dim,value_type>               &data,
              std::vector<parallel::distributed::Vector<double> >      &dst,
              const std::vector<parallel::distributed::Vector<double> >  &src,
              const std::pair<unsigned int,unsigned int>          &face_range) const
  {
  // inexact integration
//  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,true,0,0);
//  FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval_neighbor(data,false,0,0);

  // exact integration
  FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval(data,true,0,2);
  FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval_neighbor(data,false,0,2);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval.reinit (face);
    fe_eval_neighbor.reinit (face);

    fe_eval.read_dof_values(src,0);
    fe_eval.evaluate(true,false);
    fe_eval_neighbor.read_dof_values(src,0);
    fe_eval_neighbor.evaluate(true,false);

  /*  VectorizedArray<value_type> lambda = make_vectorized_array<value_type>(0.0);
    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
      VectorizedArray<value_type> uM_n = uM*normal;
      VectorizedArray<value_type> uP_n = uP*normal;
      VectorizedArray<value_type> lambda_qpoint = std::max(std::abs(uM_n), std::abs(uP_n));
      lambda = std::max(lambda_qpoint,lambda);
    } */

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > uP = fe_eval_neighbor.get_value(q);
      Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
      VectorizedArray<value_type> uM_n = uM*normal;
      VectorizedArray<value_type> uP_n = uP*normal;
      VectorizedArray<value_type> lambda; //lambda = std::max(std::abs(uM_n), std::abs(uP_n));
      for(unsigned int k=0;k<lambda.n_array_elements;++k)
        lambda[k] = std::abs(uM_n[k]) > std::abs(uP_n[k]) ? std::abs(uM_n[k]) : std::abs(uP_n[k]);

      Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
      Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = ( uM*uM_n + uP*uP_n) * make_vectorized_array<value_type>(0.5);
      Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

      fe_eval.submit_value(-lf_flux,q);
      fe_eval_neighbor.submit_value(lf_flux,q);
    }
    fe_eval.integrate(true,false);
    fe_eval.distribute_local_to_global(dst,0);
    fe_eval_neighbor.integrate(true,false);
    fe_eval_neighbor.distribute_local_to_global(dst,0);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_convection_boundary_face (const MatrixFree<dim,value_type>             &data,
                       std::vector<parallel::distributed::Vector<double> >    &dst,
                       const std::vector<parallel::distributed::Vector<double> >  &src,
                       const std::pair<unsigned int,unsigned int>          &face_range) const
  {
  // inexact integration
//    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,true,0,0);

  // exact integration
    FEFaceEvaluation<dim,fe_degree,fe_degree+(fe_degree+2)/2,dim,value_type> fe_eval(data,true,0,2);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval.reinit (face);
    fe_eval.read_dof_values(src,0);
    fe_eval.evaluate(true,false);

  /*  VectorizedArray<value_type> lambda = make_vectorized_array<value_type>(0.0);
    if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
    {
      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);

        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        Tensor<1,dim,VectorizedArray<value_type> > g_n;
        for(unsigned int d=0;d<dim;++d)
        {
          AnalyticalSolution<dim> dirichlet_boundary(d,time);
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
            array[n] = dirichlet_boundary.value(q_point);
          }
          g_n[d].load(&array[0]);
        }
        Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g_n;
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        VectorizedArray<value_type> uM_n = uM*normal;
        VectorizedArray<value_type> uP_n = uP*normal;
        VectorizedArray<value_type> lambda_qpoint = std::max(std::abs(uM_n), std::abs(uP_n));
        lambda = std::max(lambda_qpoint,lambda);
      }
    } */

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
      {
        // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);

        Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
        Tensor<1,dim,VectorizedArray<value_type> > g_n;
        for(unsigned int d=0;d<dim;++d)
        {
          AnalyticalSolution<dim> dirichlet_boundary(d,time);
          value_type array [VectorizedArray<value_type>::n_array_elements];
          for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
          {
            Point<dim> q_point;
            for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
            array[n] = dirichlet_boundary.value(q_point);
          }
          g_n[d].load(&array[0]);
        }

        Tensor<1,dim,VectorizedArray<value_type> > uP = -uM + make_vectorized_array<value_type>(2.0)*g_n;
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        VectorizedArray<value_type> uM_n = uM*normal;
        VectorizedArray<value_type> uP_n = uP*normal;
        VectorizedArray<value_type> lambda;
        for(unsigned int k=0;k<lambda.n_array_elements;++k)
          lambda[k] = std::abs(uM_n[k]) > std::abs(uP_n[k]) ? std::abs(uM_n[k]) : std::abs(uP_n[k]);

        Tensor<1,dim,VectorizedArray<value_type> > jump_value = uM - uP;
        Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = ( uM*uM_n + uP*uP_n) * make_vectorized_array<value_type>(0.5);
        Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

        fe_eval.submit_value(-lf_flux,q);
      }
      else if (data.get_boundary_indicator(face) == 1) // Outflow boundary
      {
        // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
        Tensor<1,dim,VectorizedArray<value_type> > uM = fe_eval.get_value(q);
        Tensor<1,dim,VectorizedArray<value_type> > normal = fe_eval.get_normal_vector(q);
        VectorizedArray<value_type> uM_n = uM*normal;
        VectorizedArray<value_type> lambda;
        for(unsigned int k=0;k<lambda.n_array_elements;++k)
          lambda[k] = std::abs(uM_n[k]);

        Tensor<1,dim,VectorizedArray<value_type> > jump_value;
        for(unsigned d=0;d<dim;++d)
          jump_value[d] = 0.0;
        Tensor<1,dim,VectorizedArray<value_type> > average_normal_flux = uM*uM_n;
        Tensor<1,dim,VectorizedArray<value_type> > lf_flux = average_normal_flux + 0.5 * lambda * jump_value;

        fe_eval.submit_value(-lf_flux,q);
      }
    }

    fe_eval.integrate(true,false);
    fe_eval.distribute_local_to_global(dst,0);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_compute_rhs (const MatrixFree<dim,value_type>              &data,
          std::vector<parallel::distributed::Vector<double> >      &dst,
          const std::vector<parallel::distributed::Vector<double> >  &,
          const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
    // (data,0,0) : second argument: which dof-handler, third argument: which quadrature
  FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity (data,0,0);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit (cell);

    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      Point<dim,VectorizedArray<value_type> > q_points = velocity.quadrature_point(q);
    Tensor<1,dim,VectorizedArray<value_type> > rhs;
    for(unsigned int d=0;d<dim;++d)
    {
      RHS<dim> f(d,time+time_step);
      value_type array [VectorizedArray<value_type>::n_array_elements];
      for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
      {
        Point<dim> q_point;
        for (unsigned int d=0; d<dim; ++d)
        q_point[d] = q_points[d][n];
        array[n] = f.value(q_point);
      }
      rhs[d].load(&array[0]);
    }
      velocity.submit_value (rhs, q);
    }
    velocity.integrate (true,false);
    velocity.distribute_local_to_global (dst,0);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_viscous (const MatrixFree<dim,value_type>        &data,
            parallel::distributed::Vector<double>    &dst,
            const parallel::distributed::Vector<double>  &src,
            const std::pair<unsigned int,unsigned int>   &cell_range) const
  {
  FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> velocity (data,0,0);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit (cell);
    velocity.read_dof_values(src);
    velocity.evaluate (true,true,false);

    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
    velocity.submit_value (gamma0/time_step*velocity.get_value(q), q);
    velocity.submit_gradient (make_vectorized_array<value_type>(viscosity)*velocity.get_gradient(q), q);
    }
    velocity.integrate (true,true);
    velocity.distribute_local_to_global (dst);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_viscous_face (const MatrixFree<dim,value_type>       &data,
                parallel::distributed::Vector<double>    &dst,
                const parallel::distributed::Vector<double>  &src,
                const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,0,0);
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval_neighbor(data,false,0,0);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,true);

//      VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction()) +
//               std::abs(fe_eval_neighbor.get_normal_volume_fraction())) *
//        (value_type)(fe_degree * (fe_degree + 1.0)) * 0.5    *stab_factor;

      double factor = 1.;
      calculate_penalty_parameter(factor);
      VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> uM = fe_eval.get_value(q);
        VectorizedArray<value_type> uP = fe_eval_neighbor.get_value(q);

        VectorizedArray<value_type> jump_value = uM - uP;
        VectorizedArray<value_type> average_gradient =
            ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * make_vectorized_array<value_type>(0.5);
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
        fe_eval_neighbor.submit_normal_gradient(-0.5*viscosity*jump_value,q);
        fe_eval.submit_value(-viscosity*average_gradient,q);
        fe_eval_neighbor.submit_value(viscosity*average_gradient,q);
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.integrate(true,true);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_viscous_boundary_face (const MatrixFree<dim,value_type>       &data,
                    parallel::distributed::Vector<double>    &dst,
                    const parallel::distributed::Vector<double>  &src,
                    const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,1,value_type> fe_eval(data,true,0,0);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);

//    VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
//      (value_type)(fe_degree * (fe_degree + 1.0))   *stab_factor;

      double factor = 1.;
      calculate_penalty_parameter(factor);
      VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
      {
        // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
        VectorizedArray<value_type> uM = fe_eval.get_value(q);
        VectorizedArray<value_type> uP = -uM;

        VectorizedArray<value_type> jump_value = uM - uP;
        VectorizedArray<value_type> average_gradient = fe_eval.get_normal_gradient(q);
        average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
          fe_eval.submit_value(-viscosity*average_gradient,q);
      }
      else if (data.get_boundary_indicator(face) == 1) // Outflow boundary
      {
        // applying inhomogeneous Neumann BC (value+ = value- , grad+ =  - grad- +2h)
        VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
        VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
          fe_eval.submit_normal_gradient(-0.5*viscosity*jump_value,q);
          fe_eval.submit_value(-viscosity*average_gradient,q);
      }
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_viscous (const MatrixFree<dim,value_type>                &data,
              std::vector<parallel::distributed::Vector<double> >      &dst,
              const std::vector<parallel::distributed::Vector<double> >  &src,
              const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
      // (data,0,0) : second argument: which dof-handler, third argument: which quadrature
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity (data,0,0);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      velocity.reinit (cell);
      velocity.read_dof_values(src,0);
      velocity.evaluate (true,false,false);

      for (unsigned int q=0; q<velocity.n_q_points; ++q)
      {
      Tensor<1,dim,VectorizedArray<value_type> > u = velocity.get_value(q);
        velocity.submit_value (make_vectorized_array<value_type>(1.0/time_step)*u, q);
      }
      velocity.integrate (true,false);
      velocity.distribute_local_to_global (dst,0);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_viscous_face (const MatrixFree<dim,value_type>                 &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_viscous_boundary_face (const MatrixFree<dim,value_type>             &data,
                         std::vector<parallel::distributed::Vector<double> >    &dst,
                         const std::vector<parallel::distributed::Vector<double> >  &src,
                         const std::pair<unsigned int,unsigned int>          &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> fe_eval(data,true,0,0);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);

      /* VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
        (value_type)(fe_degree * (fe_degree + 1.0))   *stab_factor; */

      double factor = 1.;
      calculate_penalty_parameter(factor);
      VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
        {
          // applying inhomogeneous Dirichlet BC (value+ = - value- + 2g , grad+ = grad-)
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<value_type> > g_np;
          for(unsigned int d=0;d<dim;++d)
          {
            AnalyticalSolution<dim> dirichlet_boundary(d,time+time_step);
            value_type array [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
              array[n] = dirichlet_boundary.value(q_point);
            }
            g_np[d].load(&array[0]);
          }

          VectorizedArray<value_type> nu = make_vectorized_array<value_type>(viscosity);
          fe_eval.submit_normal_gradient(-nu*g_np,q);
          fe_eval.submit_value(2.0*nu*sigmaF*g_np,q);
        }
        else if (data.get_boundary_indicator(face) == 1) // Outflow boundary
        {
          // applying inhomogeneous Neumann BC (value+ = value- , grad+ = - grad- +2h)
          Point<dim,VectorizedArray<value_type> > q_points = fe_eval.quadrature_point(q);
          Tensor<1,dim,VectorizedArray<value_type> > h;
          for(unsigned int d=0;d<dim;++d)
          {
            NeumannBoundaryVelocity<dim> neumann_boundary(d,time+time_step);
            value_type array [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
              array[n] = neumann_boundary.value(q_point);
            }
            h[d].load(&array[0]);
          }
          Tensor<1,dim,VectorizedArray<value_type> > jump_value;
          for(unsigned d=0;d<dim;++d)
            jump_value[d] = 0.0;

          VectorizedArray<value_type> nu = make_vectorized_array<value_type>(viscosity);
          fe_eval.submit_normal_gradient(jump_value,q);
          fe_eval.submit_value(nu*h,q);
        }
      }

      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst,0);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_inverse_mass_matrix (const parallel::distributed::Vector<value_type>  &src,
                          parallel::distributed::Vector<value_type>      &dst) const
  {
  dst = 0;

  data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_mass_matrix,
                   this, dst, src);

  dst *= time_step/gamma0;
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_mass_matrix(const MatrixFree<dim,value_type>                &data,
                std::vector<parallel::distributed::Vector<value_type> >    &dst,
                const std::vector<parallel::distributed::Vector<value_type> >  &src,
                const std::pair<unsigned int,unsigned int>          &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> phi(data,0,0);

    const unsigned int dofs_per_cell = phi.dofs_per_cell;

    AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, dim, value_type> inverse(phi);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
    phi.reinit(cell);
    phi.read_dof_values(src,0);

    inverse.fill_inverse_JxW_values(coefficients);
    inverse.apply(coefficients,dim,phi.begin_dof_values(),phi.begin_dof_values());

    phi.set_dof_values(dst,0);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_mass_matrix(const MatrixFree<dim,value_type>          &data,
                parallel::distributed::Vector<value_type>      &dst,
                const parallel::distributed::Vector<value_type>  &src,
                const std::pair<unsigned int,unsigned int>    &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,1,value_type> phi(data,0,0);

    const unsigned int dofs_per_cell = phi.dofs_per_cell;

    AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, 1, value_type> inverse(phi);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
    phi.reinit(cell);
    phi.read_dof_values(src);

    inverse.fill_inverse_JxW_values(coefficients);
    inverse.apply(coefficients,1,phi.begin_dof_values(),phi.begin_dof_values());

    phi.set_dof_values(dst);
    }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  compute_vorticity (const std::vector<parallel::distributed::Vector<value_type> >   &src,
              std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<number_vorticity_components;++d)
    dst[d] = 0;
  // data.loop
  data.back().cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_compute_vorticity,this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_compute_vorticity(const MatrixFree<dim,value_type>                  &data,
                std::vector<parallel::distributed::Vector<value_type> >      &dst,
                const std::vector<parallel::distributed::Vector<value_type> >  &src,
                const std::pair<unsigned int,unsigned int>            &cell_range) const
  {
    FEEvaluation<dim,fe_degree,fe_degree+1,dim,value_type> velocity(data,0,0);
    FEEvaluation<dim,fe_degree,fe_degree+1,number_vorticity_components,value_type> phi(data,0,0);

    const unsigned int dofs_per_cell = phi.dofs_per_cell;

    AlignedVector<VectorizedArray<value_type> > coefficients(dofs_per_cell);
    MatrixFreeOperators::CellwiseInverseMassMatrix<dim, fe_degree, number_vorticity_components, value_type> inverse(phi);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
    velocity.reinit(cell);
    velocity.read_dof_values(src,0);
      velocity.evaluate (false,true,false);

      phi.reinit(cell);

      for (unsigned int q=0; q<phi.n_q_points; ++q)
      {
      Tensor<1,number_vorticity_components,VectorizedArray<value_type> > omega = velocity.get_curl(q);
        phi.submit_value (omega, q);
      }
      phi.integrate (true,false);

      inverse.fill_inverse_JxW_values(coefficients);
    inverse.apply(coefficients,number_vorticity_components,phi.begin_dof_values(),phi.begin_dof_values());

    phi.set_dof_values(dst,0);
    }
  }

  template <int dim, typename FEEval>
  struct CurlCompute
  {
    static
    Tensor<1,dim,VectorizedArray<typename FEEval::number_type> >
    compute(const FEEval     &fe_eval,
        const unsigned int   q_point)
    {
    return fe_eval.get_curl(q_point);
    }
  };

  template <typename FEEval>
  struct CurlCompute<2,FEEval>
  {
  static
    Tensor<1,2,VectorizedArray<typename FEEval::number_type> >
    compute(const FEEval     &fe_eval,
        const unsigned int   q_point)
    {
    Tensor<1,2,VectorizedArray<typename FEEval::number_type> > rot;
    Tensor<1,2,VectorizedArray<typename FEEval::number_type> > temp = fe_eval.get_gradient(q_point);
    rot[0] = temp[1];
    rot[1] = - temp[0];
    return rot;
    }
  };

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_P (parallel::distributed::Vector<value_type> &vector) const
  {
    parallel::distributed::Vector<value_type> vec1(vector);
    for(unsigned int i=0;i<vec1.local_size();++i)
      vec1.local_element(i) = 1.;
    double scalar = vec1*vector;
    double length = vec1*vec1;
    vector.add(-scalar/length,vec1);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  shift_pressure (parallel::distributed::Vector<value_type>  &pressure)
  {
    parallel::distributed::Vector<value_type> vec1(pressure);
    for(unsigned int i=0;i<vec1.local_size();++i)
      vec1.local_element(i) = 1.;
    AnalyticalSolution<dim> analytical_solution(dim,time+time_step);
    double exact = analytical_solution.value(first_point);
    double current = pressure(dof_index_first_point);
    pressure.add(exact-current,vec1);
  }


  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_pressure (const parallel::distributed::Vector<value_type>    &src,
                  parallel::distributed::Vector<value_type>      &dst) const
  {
  dst = 0;

  data.loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure_face,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure_boundary_face,
        this, dst, src);
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_pressure (const parallel::distributed::Vector<value_type>    &src,
                  parallel::distributed::Vector<value_type>      &dst,
                  const unsigned int                 &level) const
  {
  //dst = 0;
  data[level].loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure_face,
            &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_pressure_boundary_face,
            this, dst, src);
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_pressure (const MatrixFree<dim,value_type>        &data,
            parallel::distributed::Vector<double>      &dst,
            const parallel::distributed::Vector<double>    &src,
            const std::pair<unsigned int,unsigned int>     &cell_range) const
  {
  FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);
  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    pressure.reinit (cell);
    pressure.read_dof_values(src);
    pressure.evaluate (false,true,false);
    for (unsigned int q=0; q<pressure.n_q_points; ++q)
    {
    pressure.submit_gradient (pressure.get_gradient(q), q);
    }
    pressure.integrate (false,true);
    pressure.distribute_local_to_global (dst);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_pressure_face (const MatrixFree<dim,value_type>       &data,
                parallel::distributed::Vector<double>    &dst,
                const parallel::distributed::Vector<double>  &src,
                const std::pair<unsigned int,unsigned int>  &face_range) const
  {
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);
    FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval_neighbor(data,false,1,1);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,true);
//      VectorizedArray<value_type> sigmaF = (std::abs(fe_eval.get_normal_volume_fraction()) +
//               std::abs(fe_eval_neighbor.get_normal_volume_fraction())) *
//        (value_type)(fe_degree * (fe_degree + 1.0)) * 0.5;//   *stab_factor;

      double factor = 1.;
      calculate_penalty_parameter(factor);
      VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<value_type> valueM = fe_eval.get_value(q);
        VectorizedArray<value_type> valueP = fe_eval_neighbor.get_value(q);

        VectorizedArray<value_type> jump_value = valueM - valueP;
        VectorizedArray<value_type> average_gradient =
            ( fe_eval.get_normal_gradient(q) + fe_eval_neighbor.get_normal_gradient(q) ) * 0.5;
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
        fe_eval_neighbor.submit_normal_gradient(-0.5*jump_value,q);
        fe_eval.submit_value(-average_gradient,q);
        fe_eval_neighbor.submit_value(average_gradient,q);
      }
      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
      fe_eval_neighbor.integrate(true,true);
      fe_eval_neighbor.distribute_local_to_global(dst);
    }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_apply_pressure_boundary_face (const MatrixFree<dim,value_type>           &data,
                      parallel::distributed::Vector<double>      &dst,
                      const parallel::distributed::Vector<double>    &src,
                      const std::pair<unsigned int,unsigned int>    &face_range) const
  {
  FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> fe_eval(data,true,1,1);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    fe_eval.reinit (face);

    fe_eval.read_dof_values(src);
    fe_eval.evaluate(true,true);

//    VectorizedArray<value_type> sigmaF = (std::abs( fe_eval.get_normal_volume_fraction()) ) *
//      (value_type)(fe_degree * (fe_degree + 1.0));//  *stab_factor;

      double factor = 1.;
      calculate_penalty_parameter(factor);
      VectorizedArray<value_type> sigmaF = std::abs(fe_eval.get_normal_volume_fraction()) * (value_type)factor;

    for(unsigned int q=0;q<fe_eval.n_q_points;++q)
    {
      if (data.get_boundary_indicator(face) == 0) // Infow and wall boundaries
      {
        //set pressure gradient in normal direction to zero, i.e. pressure+ = pressure-, grad+ = -grad-
        VectorizedArray<value_type> jump_value = make_vectorized_array<value_type>(0.0);
        VectorizedArray<value_type> average_gradient = make_vectorized_array<value_type>(0.0);
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
        fe_eval.submit_value(-average_gradient,q);
      }
      else if (data.get_boundary_indicator(face) == 1) // outflow boundaries
      {
        //set pressure to zero, i.e. pressure+ = - pressure- , grad+ = grad-
        VectorizedArray<value_type> valueM = fe_eval.get_value(q);

        VectorizedArray<value_type> jump_value = 2.0*valueM;
        VectorizedArray<value_type> average_gradient = fe_eval.get_normal_gradient(q);
        average_gradient = average_gradient - jump_value * sigmaF;

        fe_eval.submit_normal_gradient(-0.5*jump_value,q);
        fe_eval.submit_value(-average_gradient,q);
      }
    }
    fe_eval.integrate(true,true);
    fe_eval.distribute_local_to_global(dst);
  }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  rhs_pressure (const std::vector<parallel::distributed::Vector<value_type> >     &src,
             std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  dst[dim] = 0;
  // data.loop
  data.back().loop (  &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_pressure,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_pressure_face,
        &NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_rhs_pressure_boundary_face,
        this, dst, src);

  if(pure_dirichlet_bc)
  {  apply_P(dst[dim]);  }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_pressure (const MatrixFree<dim,value_type>                &data,
            std::vector<parallel::distributed::Vector<double> >      &dst,
            const std::vector<parallel::distributed::Vector<double> >  &src,
            const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
  FEEvaluation<dim,fe_degree,fe_degree_p+1,dim,value_type> velocity (data,0,1);
  FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit (cell);
    pressure.reinit (cell);
    velocity.read_dof_values(src,0);
    velocity.evaluate (false,true,false);
    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
    VectorizedArray<value_type> divergence = velocity.get_divergence(q);
    pressure.submit_value (divergence, q);
    }
    pressure.integrate (true,false);
    pressure.distribute_local_to_global (dst,dim);
  }
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_pressure_face (const MatrixFree<dim,value_type>               &data,
                std::vector<parallel::distributed::Vector<double> >      &dst,
                const std::vector<parallel::distributed::Vector<double> >  &src,
                const std::pair<unsigned int,unsigned int>          &face_range) const
  {

  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_rhs_pressure_boundary_face (const MatrixFree<dim,value_type>               &data,
                    std::vector<parallel::distributed::Vector<double> >      &dst,
                    const std::vector<parallel::distributed::Vector<double> >  &src,
                    const std::pair<unsigned int,unsigned int>          &face_range) const
  {
    // inhomogene Dirichlet-RB für Druck p
  FEFaceEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure(data,true,1,1);

  FEFaceEvaluation<dim,fe_degree,fe_degree_p+1,dim,value_type> velocity_n(data,true,0,1);
  FEFaceEvaluation<dim,fe_degree,fe_degree_p+1,dim,value_type> velocity_nm(data,true,0,1);
  FEFaceEvaluation<dim,fe_degree,fe_degree_p+1,number_vorticity_components,value_type> omega_n(data,true,0,1);
  FEFaceEvaluation<dim,fe_degree,fe_degree_p+1,number_vorticity_components,value_type> omega_nm(data,true,0,1);

  for(unsigned int face=face_range.first; face<face_range.second; face++)
  {
    pressure.reinit (face);
    velocity_n.reinit (face);
    velocity_n.read_dof_values(solution_n,0);
    velocity_n.evaluate (true,true);
    velocity_nm.reinit (face);
    velocity_nm.read_dof_values(solution_nm,0);
    velocity_nm.evaluate (true,true);

    omega_n.reinit (face);
    omega_n.read_dof_values(vorticity_n,0);
    omega_n.evaluate (false,true);
    omega_nm.reinit (face);
    omega_nm.read_dof_values(vorticity_nm,0);
    omega_nm.evaluate (false,true);

    //VectorizedArray<value_type> sigmaF = (std::abs( pressure.get_normal_volume_fraction()) ) *
    //  (value_type)(fe_degree * (fe_degree + 1.0)) *stab_factor;

      double factor = 1.;
      calculate_penalty_parameter(factor);
      VectorizedArray<value_type> sigmaF = std::abs(pressure.get_normal_volume_fraction()) * (value_type)factor;

    for(unsigned int q=0;q<pressure.n_q_points;++q)
    {
      if (data.get_boundary_indicator(face) == 0) // Inflow and wall boundaries
      {
        // p+ =  p-
        Point<dim,VectorizedArray<value_type> > q_points = pressure.quadrature_point(q);
        VectorizedArray<value_type> h;

//        NeumannBoundaryPressure<dim> neumann_boundary(1,time+time_step);
//        value_type array [VectorizedArray<value_type>::n_array_elements];
//        for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
//        {
//          Point<dim> q_point;
//          for (unsigned int d=0; d<dim; ++d)
//          q_point[d] = q_points[d][n];
//          array[n] = neumann_boundary.value(q_point);
//        }
//        h.load(&array[0]);

//          Tensor<1,dim,VectorizedArray<value_type> > dudt_n, rhs_n;
//          for(unsigned int d=0;d<dim;++d)
//          {
//            PressureBC_dudt<dim> neumann_boundary_pressure(d,time);
//            value_type array_dudt [VectorizedArray<value_type>::n_array_elements];
//            value_type array_f [VectorizedArray<value_type>::n_array_elements];
//            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
//            {
//              Point<dim> q_point;
//              for (unsigned int d=0; d<dim; ++d)
//              q_point[d] = q_points[d][n];
//              array_dudt[n] = neumann_boundary_pressure.value(q_point);
//              array_f[n] = f.value(q_point);
//            }
//            dudt_n[d].load(&array_dudt[0]);
//            rhs_n[d].load(&array_f[0]);
//          }
//          Tensor<1,dim,VectorizedArray<value_type> > dudt_nm, rhs_nm;
//          for(unsigned int d=0;d<dim;++d)
//          {
//            PressureBC_dudt<dim> neumann_boundary_pressure(d,time-time_step);
//            RHS<dim> f(d,time-time_step);
//            value_type array_dudt [VectorizedArray<value_type>::n_array_elements];
//            value_type array_f [VectorizedArray<value_type>::n_array_elements];
//            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
//            {
//              Point<dim> q_point;
//              for (unsigned int d=0; d<dim; ++d)
//              q_point[d] = q_points[d][n];
//              array_dudt[n] = neumann_boundary_pressure.value(q_point);
//              array_f[n] = f.value(q_point);
//            }
//            dudt_nm[d].load(&array_dudt[0]);
//            rhs_nm[d].load(&array_f[0]);
//          }

          Tensor<1,dim,VectorizedArray<value_type> > dudt_np, rhs_np;
          for(unsigned int d=0;d<dim;++d)
          {
            PressureBC_dudt<dim> neumann_boundary_pressure(d,time+time_step);
            RHS<dim> f(d,time+time_step);
            value_type array_dudt [VectorizedArray<value_type>::n_array_elements];
            value_type array_f [VectorizedArray<value_type>::n_array_elements];
            for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
            {
              Point<dim> q_point;
              for (unsigned int d=0; d<dim; ++d)
              q_point[d] = q_points[d][n];
              array_dudt[n] = neumann_boundary_pressure.value(q_point);
              array_f[n] = f.value(q_point);
            }
            dudt_np[d].load(&array_dudt[0]);
            rhs_np[d].load(&array_f[0]);
          }

        Tensor<1,dim,VectorizedArray<value_type> > normal = pressure.get_normal_vector(q);
          Tensor<1,dim,VectorizedArray<value_type> > u_n = velocity_n.get_value(q);
          Tensor<2,dim,VectorizedArray<value_type> > grad_u_n = velocity_n.get_gradient(q);
          Tensor<1,dim,VectorizedArray<value_type> > conv_n = grad_u_n * u_n;
          Tensor<1,dim,VectorizedArray<value_type> > u_nm = velocity_nm.get_value(q);
          Tensor<2,dim,VectorizedArray<value_type> > grad_u_nm = velocity_nm.get_gradient(q);
          Tensor<1,dim,VectorizedArray<value_type> > conv_nm = grad_u_nm * u_nm;
          Tensor<1,dim,VectorizedArray<value_type> > rot_n = CurlCompute<dim,decltype(omega_n)>::compute(omega_n,q);
          Tensor<1,dim,VectorizedArray<value_type> > rot_nm = CurlCompute<dim,decltype(omega_nm)>::compute(omega_nm,q);

          // 2nd order extrapolation
//        h = - normal * (make_vectorized_array<value_type>(beta[0])*(dudt_n + conv_n + make_vectorized_array<value_type>(viscosity)*rot_n - rhs_n)
//                + make_vectorized_array<value_type>(beta[1])*(dudt_nm + conv_nm + make_vectorized_array<value_type>(viscosity)*rot_nm - rhs_nm));

        h = - normal * (dudt_np - rhs_np + make_vectorized_array<value_type>(beta[0])*(conv_n + make_vectorized_array<value_type>(viscosity)*rot_n)
                + make_vectorized_array<value_type>(beta[1])*(conv_nm + make_vectorized_array<value_type>(viscosity)*rot_nm));
        // Stokes
//        h = - normal * (dudt_np - rhs_np + make_vectorized_array<value_type>(beta[0])*(make_vectorized_array<value_type>(viscosity)*rot_n)
//                        + make_vectorized_array<value_type>(beta[1])*(make_vectorized_array<value_type>(viscosity)*rot_nm));
        // 1st order extrapolation
//        h = - normal * (dudt_np - rhs_np + conv_n + make_vectorized_array<value_type>(viscosity)*rot_n);

        pressure.submit_normal_gradient(make_vectorized_array<value_type>(0.0),q);
        pressure.submit_value(-time_step*h,q);
      }
      else if (data.get_boundary_indicator(face) == 1) // Outflow boundary
      {
        // p+ = - p- + 2g
        Point<dim,VectorizedArray<value_type> > q_points = pressure.quadrature_point(q);
        VectorizedArray<value_type> g;

        AnalyticalSolution<dim> dirichlet_boundary(dim,time+time_step);
        value_type array [VectorizedArray<value_type>::n_array_elements];
        for (unsigned int n=0; n<VectorizedArray<value_type>::n_array_elements; ++n)
        {
          Point<dim> q_point;
          for (unsigned int d=0; d<dim; ++d)
            q_point[d] = q_points[d][n];
          array[n] = dirichlet_boundary.value(q_point);
        }
        g.load(&array[0]);

        pressure.submit_normal_gradient(time_step*g,q);
        pressure.submit_value(-time_step * 2.0 *sigmaF * g,q);
      }
    }
    pressure.integrate(true,true);
    pressure.distribute_local_to_global(dst,dim);
  }
  }

  template<int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim, fe_degree, fe_degree_p>::
  apply_projection (const std::vector<parallel::distributed::Vector<value_type> >     &src,
                  std::vector<parallel::distributed::Vector<value_type> >      &dst)
  {
  for(unsigned int d=0;d<dim;++d)
    dst[d] = 0;
  // data.cell_loop
  data.back().cell_loop (&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_projection,this, dst, src);
  // data.cell_loop
  data.back().cell_loop(&NavierStokesOperation<dim, fe_degree, fe_degree_p>::local_apply_mass_matrix,
                   this, dst, dst);
  }

  template <int dim, int fe_degree, int fe_degree_p>
  void NavierStokesOperation<dim,fe_degree, fe_degree_p>::
  local_projection (const MatrixFree<dim,value_type>              &data,
          std::vector<parallel::distributed::Vector<double> >      &dst,
          const std::vector<parallel::distributed::Vector<double> >  &src,
          const std::pair<unsigned int,unsigned int>           &cell_range) const
  {
  FEEvaluation<dim,fe_degree,fe_degree_p+1,dim,value_type> velocity (data,0,1);
  FEEvaluation<dim,fe_degree_p,fe_degree_p+1,1,value_type> pressure (data,1,1);

  for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
  {
    velocity.reinit (cell);
    pressure.reinit (cell);
    pressure.read_dof_values(src,dim);
    pressure.evaluate (false,true,false);
    for (unsigned int q=0; q<velocity.n_q_points; ++q)
    {
      Tensor<1,dim,VectorizedArray<value_type> > pressure_gradient = pressure.get_gradient(q);
      velocity.submit_value (-pressure_gradient, q);
    }
    velocity.integrate (true,false);
    velocity.distribute_local_to_global (dst,0);
  }
  }

  template<int dim>
  class NavierStokesProblem
  {
  public:
  typedef typename NavierStokesOperation<dim, fe_degree, fe_degree_p>::value_type value_type;
  NavierStokesProblem(const unsigned int n_refinements);
  void run();

  private:
  void make_grid_and_dofs ();
  void write_output(std::vector<parallel::distributed::Vector<value_type>>   &solution_n,
             std::vector<parallel::distributed::Vector<value_type>>   &vorticity,
             XWall<dim,fe_degree>* xwall,
             const unsigned int                     timestep_number);
  void calculate_error(std::vector<parallel::distributed::Vector<value_type>> &solution_n, const double delta_t=0.0);
  void calculate_time_step();

  ConditionalOStream pcout;

  double time, time_step;

  Triangulation<dim> triangulation;
    FE_DGQArbitraryNodes<dim>  fe;
    FE_DGQArbitraryNodes<dim>  fe_p;
    DoFHandler<dim>  dof_handler;
    DoFHandler<dim>  dof_handler_p;

  const double cfl;
  const unsigned int n_refinements;
  const double output_interval_time;
  };

  template<int dim>
  NavierStokesProblem<dim>::NavierStokesProblem(const unsigned int refine_steps):
  pcout (std::cout,
         Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  time(START_TIME),
  fe(QGaussLobatto<1>(fe_degree+1)),
  fe_p(QGaussLobatto<1>(fe_degree_p+1)),
  dof_handler(triangulation),
  dof_handler_p(triangulation),
  cfl(0.5/pow(fe_degree,2.0)),
  n_refinements(refine_steps),
  output_interval_time(0.01)
  {
  pcout << std::endl << std::endl << std::endl
  << "/******************************************************************/" << std::endl
  << "/*                                                                */" << std::endl
  << "/*     Solver for the incompressible Navier-Stokes equations      */" << std::endl
  << "/*                                                                */" << std::endl
  << "/******************************************************************/" << std::endl
  << std::endl;
  }

  template<int dim>
  void NavierStokesProblem<dim>::make_grid_and_dofs ()
  {
    /* --------------- Generate grid ------------------- */

    // hypercube: line in 1D, square in 2D, etc., hypercube volume is [left,right]^dim
    const double left = -1.0, right = 1.0;
    GridGenerator::hyper_cube(triangulation,left,right);

    // set boundary indicator
    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
    for(;cell!=endc;++cell)
    {
    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
    {
    //  if ((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12)||
    //      (std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
     if ((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12))
        cell->face(face_number)->set_boundary_indicator (1);
    }
    }
    triangulation.refine_global(n_refinements);

    // vortex problem
//    const double left = -0.5, right = 0.5;
//    GridGenerator::subdivided_hyper_cube(triangulation,2,left,right);
//
//    triangulation.refine_global(n_refinements);
//
//    typename Triangulation<dim>::cell_iterator cell = triangulation.begin(), endc = triangulation.end();
//    for(;cell!=endc;++cell)
//    {
//    for(unsigned int face_number=0;face_number < GeometryInfo<dim>::faces_per_cell;++face_number)
//    {
//     if (((std::fabs(cell->face(face_number)->center()(0) - right)< 1e-12) && (cell->face(face_number)->center()(1)<0))||
//         ((std::fabs(cell->face(face_number)->center()(0) - left)< 1e-12) && (cell->face(face_number)->center()(1)>0))||
//         ((std::fabs(cell->face(face_number)->center()(1) - left)< 1e-12) && (cell->face(face_number)->center()(0)<0))||
//         ((std::fabs(cell->face(face_number)->center()(1) - right)< 1e-12) && (cell->face(face_number)->center()(0)>0)))
//        cell->face(face_number)->set_boundary_indicator (1);
//    }
//    }
    // vortex problem

    pcout << std::endl << "Generating grid for " << dim << "-dimensional problem" << std::endl << std::endl
      << "  number of refinements:" << std::setw(10) << n_refinements << std::endl
      << "  number of cells:      " << std::setw(10) << triangulation.n_global_active_cells() << std::endl
      << "  number of faces:      " << std::setw(10) << triangulation.n_active_faces() << std::endl
      << "  number of vertices:   " << std::setw(10) << triangulation.n_vertices() << std::endl;

    // enumerate degrees of freedom
    dof_handler.distribute_dofs(fe);
    dof_handler_p.distribute_dofs(fe_p);
    dof_handler.distribute_mg_dofs(fe);
    dof_handler_p.distribute_mg_dofs(fe_p);

    float ndofs_per_cell_velocity = pow(float(fe_degree+1),dim)*dim;
    float ndofs_per_cell_pressure = pow(float(fe_degree_p+1),dim);
    pcout << std::endl << "Discontinuous finite element discretization:" << std::endl << std::endl
      << "Velocity:" << std::endl
      << "  degree of 1D polynomials:\t" << std::setw(10) << fe_degree << std::endl
      << "  number of dofs per cell:\t" << std::setw(10) << ndofs_per_cell_velocity << std::endl
      << "  number of dofs (velocity):\t" << std::setw(10) << dof_handler.n_dofs()*dim << std::endl
      << "Pressure:" << std::endl
      << "  degree of 1D polynomials:\t" << std::setw(10) << fe_degree_p << std::endl
      << "  number of dofs per cell:\t" << std::setw(10) << ndofs_per_cell_pressure << std::endl
      << "  number of dofs (pressure):\t" << std::setw(10) << dof_handler_p.n_dofs() << std::endl;
  }

  template<int dim>
  void NavierStokesProblem<dim>::
  write_output(std::vector<parallel::distributed::Vector<value_type>>   &solution_n,
          std::vector<parallel::distributed::Vector<value_type>>   &vorticity,
          XWall<dim,fe_degree>* xwall,
          const unsigned int                     output_number)
  {

    // velocity
    const FESystem<dim> joint_fe (fe, dim);
  DoFHandler<dim> joint_dof_handler (dof_handler.get_tria());
  joint_dof_handler.distribute_dofs (joint_fe);
  Vector<double> joint_velocity (joint_dof_handler.n_dofs());
  std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
  loc_vel_dof_indices (fe.dofs_per_cell);
  typename DoFHandler<dim>::active_cell_iterator joint_cell = joint_dof_handler.begin_active(), joint_endc = joint_dof_handler.end(), vel_cell = dof_handler.begin_active();
  for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell)
  {
    joint_cell->get_dof_indices (loc_joint_dof_indices);
    vel_cell->get_dof_indices (loc_vel_dof_indices);
    for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
    switch (joint_fe.system_to_base_index(i).first.first)
      {
      case 0:
      Assert (joint_fe.system_to_base_index(i).first.second < dim,
          ExcInternalError());
      joint_velocity (loc_joint_dof_indices[i]) =
        solution_n[ joint_fe.system_to_base_index(i).first.second ]
        (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
      break;
      default:
      Assert (false, ExcInternalError());
      }
  }

  DataOut<dim> data_out;

  std::vector<std::string> velocity_name (dim, "velocity");
    std::vector< DataComponentInterpretation::DataComponentInterpretation > component_interpretation(dim,DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (joint_dof_handler,joint_velocity, velocity_name, component_interpretation);

    // vorticity
  Vector<double> joint_vorticity (joint_dof_handler.n_dofs());
    if (dim==2)
    {  data_out.add_data_vector (dof_handler,vorticity[0], "vorticity"); }
    else if (dim==3)
    {
//      for (unsigned int d=0; d<dim; ++d)
//        data_out.add_data_vector (dof_handler,vorticity[d], "vorticity_" + Utilities::int_to_string(d+1));

    std::vector<types::global_dof_index> loc_joint_dof_indices (joint_fe.dofs_per_cell),
    loc_vel_dof_indices (fe.dofs_per_cell);
    typename DoFHandler<dim>::active_cell_iterator joint_cell = joint_dof_handler.begin_active(), joint_endc = joint_dof_handler.end(), vel_cell = dof_handler.begin_active();
    for (; joint_cell != joint_endc; ++joint_cell, ++vel_cell)
    {
      joint_cell->get_dof_indices (loc_joint_dof_indices);
      vel_cell->get_dof_indices (loc_vel_dof_indices);
      for (unsigned int i=0; i<joint_fe.dofs_per_cell; ++i)
      switch (joint_fe.system_to_base_index(i).first.first)
        {
        case 0:
        Assert (joint_fe.system_to_base_index(i).first.second < dim,
            ExcInternalError());
        joint_vorticity (loc_joint_dof_indices[i]) =
          vorticity[ joint_fe.system_to_base_index(i).first.second ]
          (loc_vel_dof_indices[ joint_fe.system_to_base_index(i).second ]);
        break;
        default:
        Assert (false, ExcInternalError());
        }
    }
    std::vector<std::string> vorticity_name (dim, "vorticity");
    std::vector< DataComponentInterpretation::DataComponentInterpretation > component_interpretation(dim,DataComponentInterpretation::component_is_part_of_vector);
    data_out.add_data_vector (joint_dof_handler,joint_vorticity, vorticity_name, component_interpretation);
    }
    data_out.add_data_vector (dof_handler_p,solution_n[dim], "pressure");
    data_out.add_data_vector (*(*xwall).ReturnDofHandlerWallDistance(),(*(*xwall).ReturnWDist()), "wdist");
    data_out.build_patches (10);
    std::ostringstream filename;
    filename << "solution_"
             << output_number
             << ".vtu";

    std::ofstream output (filename.str().c_str());
    data_out.write_vtu (output);
  }

  template<int dim>
  void NavierStokesProblem<dim>::
  calculate_error(std::vector<parallel::distributed::Vector<value_type>>   &solution_n,
              const double                         delta_t)
  {
  for(unsigned int d=0;d<dim;++d)
  {
    Vector<double> norm_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
                       solution_n[d],
                       AnalyticalSolution<dim>(d,time+delta_t),
                       norm_per_cell,
                       QGauss<dim>(fe.degree+2),
                       VectorTools::L2_norm);
    double solution_norm =
      std::sqrt(Utilities::MPI::sum (norm_per_cell.norm_sqr(), MPI_COMM_WORLD));
    pcout << "error (L2-norm) velocity u" << d+1 << ":"
        << std::setprecision(5) << std::setw(10) << solution_norm
        << std::endl;
  }
  Vector<double> norm_per_cell (triangulation.n_active_cells());
  VectorTools::integrate_difference (dof_handler_p,
                     solution_n[dim],
                     AnalyticalSolution<dim>(dim,time+delta_t),
                     norm_per_cell,
                     QGauss<dim>(fe.degree+2),
                     VectorTools::L2_norm);
  double solution_norm =
    std::sqrt(Utilities::MPI::sum (norm_per_cell.norm_sqr(), MPI_COMM_WORLD));
  pcout << "error (L2-norm) pressure p:"
      << std::setprecision(5) << std::setw(10) << solution_norm
      << std::endl;
  }

  template<int dim>
  void NavierStokesProblem<dim>::calculate_time_step()
  {
    typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active(),
                                                          endc = triangulation.end();

    double diameter = 0.0, min_cell_diameter = std::numeric_limits<double>::max();
    Tensor<1,dim, value_type> velocity;
//    double max_cell_a = std::numeric_limits<double>::min();
//    double a;
    for (; cell!=endc; ++cell)
    {
    // calculate minimum diameter
    diameter = cell->diameter()/std::sqrt(dim); // diameter is the largest diagonal -> divide by sqrt(dim)
    //diameter = cell->minimum_vertex_distance();
    if (diameter < min_cell_diameter)
      min_cell_diameter = diameter;

    // calculate maximum velocity a
    /*Point<dim> point = cell->center();
    velocity = get_velocity(point,time);
    a = velocity.norm();
    if (a > max_cell_a)
      max_cell_a = a;*/
    }
    const double global_min_cell_diameter = -Utilities::MPI::max(-min_cell_diameter, MPI_COMM_WORLD);
    pcout << std::endl << "min cell diameter:\t" << std::setw(10) << global_min_cell_diameter;

    /*const double global_max_cell_a = Utilities::MPI::max(max_cell_a, MPI_COMM_WORLD);*/
    /*pcout << std::endl << "maximum velocity:\t" << std::setw(10) << global_max_cell_a;*/
    pcout << std::endl << "maximum velocity:\t" << std::setw(10) << MAX_VELOCITY;

    // cfl = a * time_step / d_min
    //time_step = cfl * global_min_cell_diameter / global_max_cell_a;
    time_step = cfl * global_min_cell_diameter / MAX_VELOCITY;

    // decrease time_step in order to exactly hit END_TIME
    time_step = (END_TIME-START_TIME)/(1+int((END_TIME-START_TIME)/time_step));

//    time_step = 2.e-4;// 0.1/pow(2.0,8);

    pcout << std::endl << "time step size:\t" << std::setw(10) << time_step << std::endl;
  }

  template<int dim>
  void NavierStokesProblem<dim>::run()
  {
  make_grid_and_dofs();

  calculate_time_step();

  NavierStokesOperation<dim, fe_degree, fe_degree_p>  navier_stokes_operation(dof_handler, dof_handler_p, time_step);

  // prescribe initial conditions
  for(unsigned int d=0;d<dim;++d)
    VectorTools::interpolate(dof_handler, AnalyticalSolution<dim>(d,time), navier_stokes_operation.solution_n[d]);
  VectorTools::interpolate(dof_handler_p, AnalyticalSolution<dim>(dim,time), navier_stokes_operation.solution_n[dim]);
  navier_stokes_operation.solution_nm = navier_stokes_operation.solution_n;

  // compute vorticity from initial data at time t = START_TIME
  navier_stokes_operation.compute_vorticity(navier_stokes_operation.solution_n,navier_stokes_operation.vorticity_n);
  navier_stokes_operation.vorticity_nm = navier_stokes_operation.vorticity_n;

  unsigned int output_number = 0;
  write_output(navier_stokes_operation.solution_n,
          navier_stokes_operation.vorticity_n,
          navier_stokes_operation.ReturnXWall(),
          output_number++);
    pcout << std::endl << "Write output at START_TIME t = " << START_TIME << std::endl;
  calculate_error(navier_stokes_operation.solution_n);

  const double EPSILON = 1.0e-10;
  unsigned int time_step_number = 1;

  for(;time<(END_TIME-EPSILON);time+=time_step,++time_step_number)
  {
    navier_stokes_operation.do_timestep(time,time_step,time_step_number);
    pcout << "Step = " << time_step_number << "  t = " << time << std::endl;
    if( (time+time_step) > (output_number*output_interval_time-EPSILON) )
    {
    write_output(navier_stokes_operation.solution_n,
            navier_stokes_operation.solution_n,
            navier_stokes_operation.ReturnXWall(),
            output_number++);
      pcout << std::endl << "Write output at TIME t = " << time+time_step << std::endl;
      calculate_error(navier_stokes_operation.solution_n,time_step);
    }
  }
  navier_stokes_operation.analyse_computing_times();
  }
}

int main (int argc, char** argv)
{
  try
    {
      using namespace DG_NavierStokes;
      Utilities::MPI::MPI_InitFinalize mpi(argc, argv, -1);

      deallog.depth_console(0);

      for(unsigned int refine_steps = refine_steps_min;refine_steps <= refine_steps_max;++refine_steps)
      {
        NavierStokesProblem<dimension> navier_stokes_problem(refine_steps);
        navier_stokes_problem.run();
      }
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }
  return 0;
}
