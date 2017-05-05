/*
 * laplace_operator_matrix_free.cc
 *
 *  Created on: May 2, 2017
 *      Author: fehn
 */


/**************************************************************************************/
/*                                                                                    */
/*                                        HEADER                                      */
/*                                                                                    */
/**************************************************************************************/

// C++
#include <iostream>

// deal.ii

// output
#include <deal.II/base/conditional_ostream.h>

// triangulation, grid, DG element, mapping, matrix-free evaluation
#include <deal.II/distributed/tria.h>
#include <deal.II/grid/grid_tools.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/fe/fe_dgq.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/fe/mapping_q.h>
#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/lac/parallel_vector.h>

// timer
#include <deal.II/base/timer.h>

using namespace dealii;



/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 3;

// set the polynomial degree of the shape functions k = 1,...,10
unsigned int const FE_DEGREE_MIN = 1;
unsigned int const FE_DEGREE_MAX = 10;

// set the number of refinement levels
unsigned int const REFINE_STEPS_SPACE_MIN = 3;
unsigned int const REFINE_STEPS_SPACE_MAX = 5;

// mesh type: currently, a cartesian mesh is implemented (hyper cube)
// and a non-cartesian mesh (hyper shell)
enum class MeshType{
  Cartesian,
  NonCartesian
};

MeshType MESH_TYPE = MeshType::Cartesian; //Cartesian; //NonCartesian;

// compute cell integrals (default: true)
bool const COMPUTE_CELL_INTEGRALS = true;
// compute face integrals (default: true)
bool const COMPUTE_FACE_INTEGRALS = true;

// float or double precision
//typedef float VALUE_TYPE;
typedef double VALUE_TYPE;

// number of repetitions used to determine the average/minimum wall time required
// to compute the matrix-vector product
unsigned int const N_REPETITIONS = 100;

// Type of wall time calculation used to measure efficiency
enum class WallTimeCalculation{
  Average,
  Minimum
};

WallTimeCalculation const WALL_TIME_CALCULATION = WallTimeCalculation::Average; //Average; //Minimum;

// global variable used to store the wall times for different polynomial degrees
std::vector<std::pair<unsigned int, double> > wall_times;

/**************************************************************************************/
/*                                                                                    */
/*         GENERATE GRID, SET BOUNDARY INDICATORS AND FILL BOUNDARY DESCRIPTOR        */
/*                                                                                    */
/**************************************************************************************/

#include <deal.II/base/function.h>
#include <deal.II/base/types.h>

template<int dim>
struct BoundaryDescriptorLaplace
{
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > dirichlet;
  std::map<types::boundary_id,std::shared_ptr<Function<dim> > > neumann;
};

template<int dim>
void create_grid(parallel::distributed::Triangulation<dim>        &triangulation,
                 unsigned int const                               n_refine_space,
                 std::shared_ptr<BoundaryDescriptorLaplace<dim> > boundary_descriptor)
{
  if(MESH_TYPE == MeshType::Cartesian)
  {
    GridGenerator::hyper_cube(triangulation);
  }
  else if(MESH_TYPE == MeshType::NonCartesian)
  {
    GridGenerator::hyper_ball(triangulation);
  }
  else
  {
    AssertThrow(MESH_TYPE == MeshType::Cartesian || MESH_TYPE == MeshType::NonCartesian,
                ExcMessage("Not implemented."));
  }

  // boundary ID's are 0 by default -> pure Dirichlet BC's
  std::shared_ptr<Function<dim> > function;
  function.reset(new ZeroFunction<dim>());
  boundary_descriptor->dirichlet.insert(
      std::pair<types::boundary_id,std::shared_ptr<Function<dim> > >(0,function));

  // refinements
  triangulation.refine_global(n_refine_space);
}

template<int dim>
void print_grid_data(ConditionalOStream                        &pcout,
                     parallel::distributed::Triangulation<dim> &triangulation,
                     unsigned int const                        n_refine_space)
{
  pcout << std::endl
        << "Generating grid for " << dim << "-dimensional problem:" << std::endl
        << std::endl;

  pcout << "Number of refinements: " << n_refine_space << std::endl;
  pcout << "Number of cells:       " << triangulation.n_global_active_cells() << std::endl;
  pcout << "Number of faces:       " << triangulation.n_active_faces() << std::endl;
  pcout << "Number of vertices:    " << triangulation.n_vertices() << std::endl;
}



/**************************************************************************************/
/*                                                                                    */
/*       LAPLACE OPERATOR (copied from /include/poisson/laplace_operator.h!!!)        */
/*                                                                                    */
/**************************************************************************************/

template<int dim>
struct LaplaceOperatorData
{
  LaplaceOperatorData ()
    :
    laplace_dof_index(0),
    laplace_quad_index(0),
    penalty_factor(1.),
    compute_cell_integrals(true),
    compute_face_integrals(true)
  {}

  // If an external MatrixFree object is given which can contain other
  // components than the variable for which the Poisson equation should be
  // solved, this selects the correct DoFHandler component
  unsigned int laplace_dof_index;

  // If an external MatrixFree object is given which can contain other
  // quadrature formulas than the quadrature formula which should be used by
  // the Poisson solver, this selects the correct quadrature index
  unsigned int laplace_quad_index;

  // The penalty parameter for the symmetric interior penalty method is
  // computed as penalty_factor * (fe_degree+1)^2 /
  // characteristic_element_length. This variable gives the scaling factor
  double penalty_factor;

  // boundary descriptor
  std::shared_ptr<BoundaryDescriptorLaplace<dim> >  bc;

  // If periodic boundaries are present, this variable collects matching faces
  // on the two sides of the domain
  std::vector<GridTools::PeriodicFacePair<
    typename Triangulation<dim>::cell_iterator> > periodic_face_pairs_level0;

  // compute cell integrals
  bool compute_cell_integrals;

  // compute face integrals
  bool compute_face_integrals;
};

template <int dim, int degree, typename Number=double>
class LaplaceOperator
{
public:
  // Constructor.
  LaplaceOperator ();

  // Initialization with given MatrixFree object
  void reinit(const MatrixFree<dim,Number>       &mf_data,
              const Mapping<dim>                 &mapping,
              const LaplaceOperatorData<dim>     &operator_data);


  // Performs matrix-vector multiplication
  void vmult(parallel::distributed::Vector<Number>       &dst,
             const parallel::distributed::Vector<Number> &src) const;

  // Performs a matrix-vector multiplication, adding the result in
  // the previous content of dst
  void vmult_add(parallel::distributed::Vector<Number>       &dst,
                 const parallel::distributed::Vector<Number> &src) const;

  // Returns a reference to the ratio between the element surface and the
  // element volume for the symmetric interior penalty method (only available
  // in the DG case).
  AlignedVector<VectorizedArray<Number> > const & get_array_penalty_parameter() const;

  // Returns the current factor by which array_penalty_parameter() is
  // multiplied in the definition of the interior penalty parameter through
  // get_array_penalty_parameter()[cell] * get_penalty_factor().
  Number get_penalty_factor() const;

private:
  // Computes the array penalty parameter for later use of the symmetric
  // interior penalty method. Called in reinit().
  void compute_array_penalty_parameter(const Mapping<dim> &mapping);

  // Runs the loop over all cells and faces for use in matrix-vector
  // multiplication, adding the result in the previous content of dst
  void run_vmult_loop(parallel::distributed::Vector<Number>       &dst,
                      const parallel::distributed::Vector<Number> &src) const;

  // cell loop: volume integrals
  void cell_loop (const MatrixFree<dim,Number>                &data,
                  parallel::distributed::Vector<Number>       &dst,
                  const parallel::distributed::Vector<Number> &src,
                  const std::pair<unsigned int,unsigned int>  &cell_range) const;

  // face loop: face integrals for interior faces
  void face_loop (const MatrixFree<dim,Number>                &data,
                  parallel::distributed::Vector<Number>       &dst,
                  const parallel::distributed::Vector<Number> &src,
                  const std::pair<unsigned int,unsigned int>  &face_range) const;

  // boundary face loop: face integrals for boundary faces
  void boundary_face_loop (const MatrixFree<dim,Number>                &data,
                           parallel::distributed::Vector<Number>       &dst,
                           const parallel::distributed::Vector<Number> &src,
                           const std::pair<unsigned int,unsigned int>  &face_range) const;

  // cell integral
  template<typename FEEvaluation>
  inline void do_cell_integral(FEEvaluation &fe_eval) const;

  MatrixFree<dim,Number> const *data;

  LaplaceOperatorData<dim> operator_data;

  AlignedVector<VectorizedArray<Number> > array_penalty_parameter;
};

template <int dim, int degree, typename Number>
LaplaceOperator<dim,degree,Number>::LaplaceOperator ()
  :
  data (0)
{}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
reinit (MatrixFree<dim,Number> const   &mf_data,
        Mapping<dim> const             &mapping,
        LaplaceOperatorData<dim> const &operator_data)
{
  this->data = &mf_data;
  this->operator_data = operator_data;

  AssertThrow (Utilities::fixed_power<dim>((unsigned int)degree+1) ==
               mf_data.get_n_q_points(operator_data.laplace_quad_index),
               ExcMessage("Expected fe_degree+1 quadrature points"));

  compute_array_penalty_parameter(mapping);
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
vmult(parallel::distributed::Vector<Number>       &dst,
      parallel::distributed::Vector<Number> const &src) const
{
  dst = 0;
  vmult_add(dst, src);
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
vmult_add(parallel::distributed::Vector<Number>       &dst,
          parallel::distributed::Vector<Number> const &src) const
{
  run_vmult_loop(dst, src);
}

template <int dim, int degree, typename Number>
AlignedVector<VectorizedArray<Number> > const & LaplaceOperator<dim,degree,Number>::
get_array_penalty_parameter() const
{
  return array_penalty_parameter;
}

template <int dim, int degree, typename Number>
Number LaplaceOperator<dim,degree,Number>::get_penalty_factor() const
{
  return operator_data.penalty_factor * (degree + 1.0) * (degree + 1.0);
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
compute_array_penalty_parameter(const Mapping<dim> &mapping)
{
  std::set<types::boundary_id> periodic_boundary_ids;
  for (unsigned int i=0; i<operator_data.periodic_face_pairs_level0.size(); ++i)
  {
    AssertThrow(operator_data.periodic_face_pairs_level0[i].cell[0]->level() == 0,
                ExcMessage("Received periodic cell pairs on non-zero level"));
    periodic_boundary_ids.insert(operator_data.periodic_face_pairs_level0[i].cell[0]->face(operator_data.periodic_face_pairs_level0[i].face_idx[0])->boundary_id());
    periodic_boundary_ids.insert(operator_data.periodic_face_pairs_level0[i].cell[1]->face(operator_data.periodic_face_pairs_level0[i].face_idx[1])->boundary_id());
  }

  // Compute penalty parameter for each cell
  array_penalty_parameter.resize(data->n_macro_cells()+data->n_macro_ghost_cells());
  QGauss<dim> quadrature(degree+1);
  FEValues<dim> fe_values(mapping,
                          data->get_dof_handler(operator_data.laplace_dof_index).get_fe(),
                          quadrature, update_JxW_values);
  QGauss<dim-1> face_quadrature(degree+1);
  FEFaceValues<dim> fe_face_values(mapping,
                                   data->get_dof_handler(operator_data.laplace_dof_index).get_fe(),
                                   face_quadrature,
                                   update_JxW_values);

  for (unsigned int i=0; i<data->n_macro_cells()+data->n_macro_ghost_cells(); ++i)
  {
    for (unsigned int v=0; v<data->n_components_filled(i); ++v)
    {
      typename DoFHandler<dim>::cell_iterator cell = data->get_cell_iterator(i,v,operator_data.laplace_dof_index);
      fe_values.reinit(cell);
      double volume = 0;
      for (unsigned int q=0; q<quadrature.size(); ++q)
        volume += fe_values.JxW(q);
      double surface_area = 0;
      for (unsigned int f=0; f<GeometryInfo<dim>::faces_per_cell; ++f)
      {
        fe_face_values.reinit(cell, f);
        const double factor = (cell->at_boundary(f) &&
                               periodic_boundary_ids.find(cell->face(f)->boundary_id()) ==
                               periodic_boundary_ids.end()) ? 1. : 0.5;
        for (unsigned int q=0; q<face_quadrature.size(); ++q)
          surface_area += fe_face_values.JxW(q) * factor;
      }
      array_penalty_parameter[i][v] = surface_area / volume;
    }
  }
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
run_vmult_loop(parallel::distributed::Vector<Number>       &dst,
               parallel::distributed::Vector<Number> const &src) const
{
  if (this->operator_data.compute_face_integrals == false)
    data->cell_loop (&LaplaceOperator<dim, degree, Number>::cell_loop,
                     this, dst, src);
  else
    data->loop (&LaplaceOperator<dim, degree, Number>::cell_loop,
                &LaplaceOperator<dim, degree, Number>::face_loop,
                &LaplaceOperator<dim, degree, Number>::boundary_face_loop,
                this, dst, src,
                MatrixFree<dim,Number>::values_and_gradients,
                MatrixFree<dim,Number>::values_and_gradients);
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
cell_loop (const MatrixFree<dim,Number>                &data,
           parallel::distributed::Vector<Number>       &dst,
           const parallel::distributed::Vector<Number> &src,
           const std::pair<unsigned int,unsigned int>  &cell_range) const
{
  if(this->operator_data.compute_cell_integrals == true)
  {
    FEEvaluation<dim,degree,degree+1,1,Number> phi (data,
                                                    operator_data.laplace_dof_index,
                                                    operator_data.laplace_quad_index);

    for (unsigned int cell=cell_range.first; cell<cell_range.second; ++cell)
    {
      phi.reinit (cell);
      phi.read_dof_values(src);

      do_cell_integral(phi);

      phi.distribute_local_to_global (dst);
    }
  }
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
face_loop (const MatrixFree<dim,Number>                &data,
           parallel::distributed::Vector<Number>       &dst,
           const parallel::distributed::Vector<Number> &src,
           const std::pair<unsigned int,unsigned int>  &face_range) const
{
  if(this->operator_data.compute_face_integrals == true)
  {
    FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval(data,true,
                                                           operator_data.laplace_dof_index,
                                                           operator_data.laplace_quad_index);
    FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval_neighbor(data,false,
                                                                    operator_data.laplace_dof_index,
                                                                    operator_data.laplace_quad_index);

    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval_neighbor.reinit (face);

      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);
      fe_eval_neighbor.read_dof_values(src);
      fe_eval_neighbor.evaluate(true,true);

      VectorizedArray<Number> sigmaF =
        std::max(fe_eval.read_cell_data(array_penalty_parameter),
                 fe_eval_neighbor.read_cell_data(array_penalty_parameter)) *
        get_penalty_factor();

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        VectorizedArray<Number> valueM = fe_eval.get_value(q);
        VectorizedArray<Number> valueP = fe_eval_neighbor.get_value(q);

        VectorizedArray<Number> jump_value = valueM - valueP;
        VectorizedArray<Number> average_gradient =
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
}

template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
boundary_face_loop (const MatrixFree<dim,Number>                &data,
                    parallel::distributed::Vector<Number>       &dst,
                    const parallel::distributed::Vector<Number> &src,
                    const std::pair<unsigned int,unsigned int>  &face_range) const
{
  if(this->operator_data.compute_face_integrals == true)
  {
    FEFaceEvaluation<dim,degree,degree+1,1,Number> fe_eval(data, true,
                                                           operator_data.laplace_dof_index,
                                                           operator_data.laplace_quad_index);
    for(unsigned int face=face_range.first; face<face_range.second; face++)
    {
      fe_eval.reinit (face);
      fe_eval.read_dof_values(src);
      fe_eval.evaluate(true,true);
      VectorizedArray<Number> sigmaF =
        fe_eval.read_cell_data(array_penalty_parameter) *
        get_penalty_factor();

      typename std::map<types::boundary_id,std::shared_ptr<Function<dim> > >::iterator it;
      types::boundary_id boundary_id = data.get_boundary_id(face);

      for(unsigned int q=0;q<fe_eval.n_q_points;++q)
      {
        it = operator_data.bc->dirichlet.find(boundary_id);
        if(it != operator_data.bc->dirichlet.end())
        {
          //set value to zero, i.e. u+ = - u- , grad+ = grad-
          VectorizedArray<Number> valueM = fe_eval.get_value(q);

          VectorizedArray<Number> jump_value = 2.0*valueM;
          VectorizedArray<Number> average_gradient = fe_eval.get_normal_gradient(q);
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval.submit_normal_gradient(-0.5*jump_value,q);
          fe_eval.submit_value(-average_gradient,q);
        }
        it = operator_data.bc->neumann.find(boundary_id);
        if (it != operator_data.bc->neumann.end())
        {
          //set gradient in normal direction to zero, i.e. u+ = u-, grad+ = -grad-
          VectorizedArray<Number> jump_value = make_vectorized_array<Number>(0.0);
          VectorizedArray<Number> average_gradient = make_vectorized_array<Number>(0.0);
          average_gradient = average_gradient - jump_value * sigmaF;

          fe_eval.submit_normal_gradient(-0.5*jump_value,q);
          fe_eval.submit_value(-average_gradient,q);
        }
      }

      fe_eval.integrate(true,true);
      fe_eval.distribute_local_to_global(dst);
    }
  }
}

template <int dim, int degree, typename Number>
template<typename FEEvaluation>
inline void LaplaceOperator<dim,degree,Number>::
do_cell_integral(FEEvaluation &fe_eval) const
{
  fe_eval.evaluate (false,true,false);

  for (unsigned int q=0; q<fe_eval.n_q_points; ++q)
  {
    fe_eval.submit_gradient (fe_eval.get_gradient(q), q);
  }

  fe_eval.integrate (false,true);
}



/**************************************************************************************/
/*                                                                                    */
/*                                LAPLACE PROBLEM                                     */
/*                                                                                    */
/**************************************************************************************/

template<int dim, int fe_degree, typename Number=double>
class LaplaceProblem
{
public:
  /*
   *  Constructor.
   */
  LaplaceProblem(unsigned int const refine_steps_space)
    :
  pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD),
  fe(fe_degree),
  mapping(fe_degree),
  dof_handler(triangulation),
  n_refinements(refine_steps_space),
  n_repetitions(N_REPETITIONS),
  wall_time_calculation(WALL_TIME_CALCULATION)
  {
    pcout << std::endl << std::endl << std::endl
    << "_________________________________________________________________________________" << std::endl
    << "                                                                                 " << std::endl
    << "                High-order discontinuous Galerkin discretization                 " << std::endl
    << "           of Laplace operator based on a matrix-free implementation             " << std::endl
    << "_________________________________________________________________________________" << std::endl
    << std::endl;

    boundary_descriptor.reset(new BoundaryDescriptorLaplace<dim>());
  }

  /*
   *  Setup of grid, dofs, matrix-free object and Laplace operator.
   */
  void setup()
  {
    // create grid
    create_grid(triangulation,n_refinements,boundary_descriptor);
    print_grid_data(pcout,triangulation,n_refinements);

    // create dofs (enumerate degrees of freedom)
    setup_dofs();

    // setup matrix free object
    setup_matrix_free();

    // setup Laplace operator
    setup_laplace_operator();
  }

  /*
   *  This function applies the matrix-vector product several times
   *  and computes the required average wall time as well as the
   *  average wall time per degree of freedom.
   */
  void apply_laplace_operator() const
  {
    pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

    // initialize vectors
    parallel::distributed::Vector<Number> dst, src;
    matrix_free_data.initialize_dof_vector(src);
    matrix_free_data.initialize_dof_vector(dst);
    src = 1.0;

    // Timer and wall times
    Timer timer;

    double wall_time = 0.0;

    if(wall_time_calculation == WallTimeCalculation::Minimum)
      wall_time = std::numeric_limits<double>::max();


    // apply matrix-vector product several times
    for(unsigned int i=0; i<n_repetitions; ++i)
    {
      timer.restart();

      laplace_operator.vmult(dst,src);

      double const current_wall_time = timer.wall_time();

      if(wall_time_calculation == WallTimeCalculation::Average)
        wall_time += current_wall_time;
      else if(wall_time_calculation == WallTimeCalculation::Minimum)
        wall_time = std::min(wall_time,current_wall_time);
    }


    // compute wall times
    if(wall_time_calculation == WallTimeCalculation::Average)
      wall_time /= (double)n_repetitions;

    unsigned int dofs = dof_handler.n_dofs();

    wall_time /= (double) dofs;

    pcout << std::endl << std::scientific << std::setprecision(4)
          << "Wall time / dofs [s]: " << wall_time << std::endl;

    pcout << std::endl << " ... done." << std::endl << std::endl;

    wall_times.push_back(std::pair<unsigned int,double>(fe_degree,wall_time));
  }

private:
  /*
   *  Setup dofs
   */
  void setup_dofs()
  {
    dof_handler.distribute_dofs(fe);

    pcout << std::endl
          << "Discontinuous Galerkin finite element discretization:" << std::endl;

    pcout << std::endl << std::fixed
          << "degree of 1D polynomials: " << fe_degree << std::endl
          << "number of dofs per cell:  " << Utilities::fixed_int_power<fe_degree+1,dim>::value << std::endl
          << "number of dofs (total):   " << dof_handler.n_dofs() << std::endl;
  }

  /*
   *  Setup matrix free data
   */
  void setup_matrix_free()
  {
    pcout << std::endl << "Setup matrix-free object ...";

    // quadrature formula used to perform integrals
    QGauss<1> quadrature (fe_degree+1);

    // initialize matrix_free_data
    typename MatrixFree<dim,Number>::AdditionalData additional_data;
    additional_data.tasks_parallel_scheme = MatrixFree<dim,Number>::AdditionalData::partition_partition;
    additional_data.build_face_info = true;
    additional_data.mapping_update_flags = (update_gradients | update_JxW_values |
                                            update_quadrature_points | update_normal_vectors |
                                            update_values);

    // constraints
    ConstraintMatrix dummy;
    dummy.close();

    matrix_free_data.reinit (mapping, dof_handler, dummy, quadrature, additional_data);

    pcout << " done." << std::endl;
  }

  /*
   *  Setup of Laplace operator
   */
  void setup_laplace_operator()
  {
    pcout << std::endl << "Setup Laplace operator ...";

    LaplaceOperatorData<dim> operator_data;
    operator_data.bc = boundary_descriptor;
    operator_data.compute_cell_integrals = COMPUTE_CELL_INTEGRALS;
    operator_data.compute_face_integrals = COMPUTE_FACE_INTEGRALS;

    laplace_operator.reinit(matrix_free_data,mapping,operator_data);

    pcout << " done." << std::endl;
  }


  // output
  ConditionalOStream pcout;

  // Triangulation
  parallel::distributed::Triangulation<dim> triangulation;

  // BC's
  std::shared_ptr<BoundaryDescriptorLaplace<dim> > boundary_descriptor;

  // Discontinuous Galerkin finite element
  FE_DGQ<dim> fe;

  // Mapping
  MappingQGeneric<dim> mapping;

  // Dof handler
  DoFHandler<dim> dof_handler;

  // Matrix-free object
  MatrixFree<dim,Number> matrix_free_data;

  // Laplace operator
  LaplaceOperator<dim, fe_degree, Number> laplace_operator;

  // number of refinement levels
  const unsigned int n_refinements;

  // number of matrix-vector products
  const unsigned int n_repetitions;

  // wall time calculation
  WallTimeCalculation wall_time_calculation;
};



/**************************************************************************************/
/*                                                                                    */
/*                                         MAIN                                       */
/*                                                                                    */
/**************************************************************************************/

template<int dim, int degree, int max_degree, typename value_type>
class LaplaceRunTime
{
public:
  static void run(unsigned int const refine_steps_space)
  {
    LaplaceRunTime<dim,degree,degree,value_type>::run(refine_steps_space);
    LaplaceRunTime<dim,degree+1,max_degree,value_type>::run(refine_steps_space);
  }
};

template <int dim, int degree,typename value_type>
class LaplaceRunTime<dim,degree,degree,value_type>
{
public:
  static void run(unsigned int const refine_steps_space)
  {
    LaplaceProblem<dim,degree,value_type> laplace_problem(refine_steps_space);
    laplace_problem.setup();
    laplace_problem.apply_laplace_operator();
  }
};

void print_wall_times(std::vector<std::pair<unsigned int, double> > const &wall_times,
                      unsigned int const                                  refine_steps_space)
{
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << std::endl
              << "_________________________________________________________________________________"
              << std::endl << std::endl
              << "Wall times for refine level l = " << refine_steps_space << ":"
              << std::endl << std::endl
              << "  k    " << "wall time / dofs [s]" << std::endl;

    typedef typename std::vector<std::pair<unsigned int, double> >::const_iterator ITERATOR;
    for(ITERATOR it=wall_times.begin(); it != wall_times.end(); ++it)
    {
      std::cout << "  " << std::setw(5) << std::left << it->first
                << std::setw(2) << std::left << std::scientific << std::setprecision(4) << it->second
                << std::endl;
    }

    std::cout << "_________________________________________________________________________________"
              << std::endl << std::endl;
  }
}

int main (int argc, char** argv)
{
  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    deallog.depth_console(0);

    //mesh refinements
    for(unsigned int refine_steps_space = REFINE_STEPS_SPACE_MIN;
        refine_steps_space <= REFINE_STEPS_SPACE_MAX; ++refine_steps_space)
    {
      // increasing polynomial degrees
      LaplaceRunTime<DIMENSION,FE_DEGREE_MIN,FE_DEGREE_MAX,VALUE_TYPE>::run(refine_steps_space);

      print_wall_times(wall_times, refine_steps_space);
      wall_times.clear();
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

