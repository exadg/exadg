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
#include <deal.II/matrix_free/tensor_product_kernels.h>
#include <deal.II/lac/parallel_vector.h>

// timer
#include <deal.II/base/timer.h>

using namespace dealii;

#ifdef LIKWID_PERFMON
  #include <likwid.h>
#endif


/**************************************************************************************/
/*                                                                                    */
/*                                 INPUT PARAMETERS                                   */
/*                                                                                    */
/**************************************************************************************/

// set the number of space dimensions: dimension = 2, 3
unsigned int const DIMENSION = 3;

// set the polynomial degree of the shape functions k
unsigned int const FE_DEGREE_MIN = 1;
unsigned int const FE_DEGREE_MAX = 15;
bool const RUN_EQUAL_SIZE = true;

// set the number of refinement levels
unsigned int const REFINE_STEPS_SPACE_MIN = 6;
unsigned int const REFINE_STEPS_SPACE_MAX = 6;

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
bool const COMPUTE_FACE_INTEGRALS = false;

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
std::vector<std::pair<unsigned int, std::array<double,4> > > wall_times;

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

  // Performs cell loop manually
  void cell_loop_manual_1(parallel::distributed::Vector<Number>       &dst,
                          const parallel::distributed::Vector<Number> &src) const;
  // Performs cell loop manually
  void cell_loop_manual_2(parallel::distributed::Vector<Number>       &dst,
                          const parallel::distributed::Vector<Number> &src) const;
  // Performs cell loop manually
  void cell_loop_manual_3(parallel::distributed::Vector<Number>       &dst,
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
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START(("cell_loop_basic_p" + std::to_string(degree)).c_str());
#endif
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
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP(("cell_loop_basic_p" + std::to_string(degree)).c_str());
#endif
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

      phi.evaluate (false,true,false);

      for (unsigned int q=0; q<phi.n_q_points; ++q)
        {
          phi.submit_gradient (phi.get_gradient(q), q);
        }

      phi.integrate (false,true);

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
void LaplaceOperator<dim,degree,Number>::
cell_loop_manual_1(parallel::distributed::Vector<Number> &dst,
                   const parallel::distributed::Vector<Number> &src) const
{
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START(("cell_loop_1_p" + std::to_string(degree)).c_str());
#endif
  // do not exchange data or zero out, assume DG operator does not need to
  // exchange and that the loop below takes care of the zeroing

  // global data structures
  const unsigned int dofs_per_cell = Utilities::fixed_int_power<degree+1,dim>::value;
  AlignedVector<VectorizedArray<Number> > *scratch_data_array = data->acquire_scratch_data();
  scratch_data_array->resize_fast((dim+2)*dofs_per_cell);
  VectorizedArray<Number> *__restrict data_ptr = scratch_data_array->begin();

  typedef internal::EvaluatorTensorProduct<internal::evaluate_evenodd, dim, degree+1, degree+1,
                                           VectorizedArray<Number> > Eval;
  Eval eval_val (data->get_shape_info().shape_values_eo);
  Eval eval(AlignedVector<VectorizedArray<Number> >(),
            data->get_shape_info().shape_gradients_collocation_eo,
            data->get_shape_info().shape_hessians_collocation_eo);
  const Number*__restrict quadrature_weights =
    data->get_mapping_info().cell_data[0].descriptor[0].quadrature_weights.begin();

  for (unsigned int cell=0; cell<data->n_macro_cells(); ++cell)
    {
      // read from src vector
      const unsigned int *dof_indices =
        &data->get_dof_info().dof_indices_contiguous[2][cell*VectorizedArray<Number>::n_array_elements];
      const unsigned int vectorization_populated =
        data->n_components_filled(cell);
      // transform array-of-struct to struct-of-array
      if (vectorization_populated == VectorizedArray<Number>::n_array_elements)
        vectorized_load_and_transpose(dofs_per_cell, src.begin(),
                                      dof_indices, data_ptr);
      else
        {
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            data_ptr[i] = VectorizedArray<Number>();
          for (unsigned int v=0; v<vectorization_populated; ++v)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              data_ptr[i][v] = src.local_element(dof_indices[v]+i);
        }

      // apply tensor product kernels
      if (dim == 1)
        eval_val.template values<0,true,false>(data_ptr, data_ptr+dofs_per_cell);
      else if (dim == 2)
        {
          eval_val.template values<0,true,false>(data_ptr, data_ptr+2*dofs_per_cell);
          eval_val.template values<1,true,false>(data_ptr+2*dofs_per_cell, data_ptr+dofs_per_cell);
        }
      else if (dim == 3)
        {
          eval_val.template values<0,true,false>(data_ptr, data_ptr+dofs_per_cell);
          eval_val.template values<1,true,false>(data_ptr+dofs_per_cell, data_ptr+2*dofs_per_cell);
          eval_val.template values<2,true,false>(data_ptr+2*dofs_per_cell, data_ptr+dofs_per_cell);
        }
      eval.template gradients<0,true,false>(data_ptr+dofs_per_cell, data_ptr+2*dofs_per_cell);
      if (dim > 1)
        eval.template gradients<1,true,false>(data_ptr+dofs_per_cell, data_ptr+3*dofs_per_cell);
      if (dim > 2)
        eval.template gradients<2,true,false>(data_ptr+dofs_per_cell, data_ptr+4*dofs_per_cell);

      // loop over quadrature points. depends on the data layout in
      // MappingInfo
      const VectorizedArray<Number> *__restrict jxw_ptr =
        data->get_mapping_info().cell_data[0].JxW_values.begin() +
        data->get_mapping_info().cell_data[0].data_index_offsets[cell];
      const Tensor<2,dim,VectorizedArray<Number>> *__restrict jac_ptr =
        data->get_mapping_info().cell_data[0].jacobians[0].begin() +
        data->get_mapping_info().cell_data[0].data_index_offsets[cell];
      // Cartesian cell case
      if (data->get_mapping_info().cell_type[cell] == internal::MatrixFreeFunctions::cartesian)
        for (unsigned int q=0; q<dofs_per_cell; ++q)
          for (unsigned int d=0; d<dim; ++d)
            data_ptr[(2+d)*dofs_per_cell+q] *= jac_ptr[0][d][d] *
              jac_ptr[0][d][d] * (jxw_ptr[0] * quadrature_weights[q]);
      // affine cell case
      else if (data->get_mapping_info().cell_type[cell] == internal::MatrixFreeFunctions::affine)
        for (unsigned int q=0; q<dofs_per_cell; ++q)
          {
            VectorizedArray<Number> grad[dim];
            // get gradient
            for (unsigned int d=0; d<dim; ++d)
              {
                grad[d] = jac_ptr[0][d][0] * data_ptr[2*dofs_per_cell+q];
                for (unsigned int e=1; e<dim; ++e)
                  grad[d] += jac_ptr[0][d][e] * data_ptr[(2+e)*dofs_per_cell+q];
              }
            // apply gradient of test function
            for (unsigned int d=0; d<dim; ++d)
              {
                data_ptr[(2+d)*dofs_per_cell+q] = jac_ptr[0][0][d] * grad[0];
                for (unsigned int e=1; e<dim; ++e)
                  data_ptr[(2+d)*dofs_per_cell+q] += jac_ptr[0][e][d] * grad[e];
                // multiply by quadrature weight
                data_ptr[(2+d)*dofs_per_cell+q] *= jxw_ptr[0] * quadrature_weights[q];
              }
          }
      else
        // non-affine cell case
        for (unsigned int q=0; q<dofs_per_cell; ++q)
          {
            VectorizedArray<Number> grad[dim];
            // get gradient
            for (unsigned int d=0; d<dim; ++d)
              {
                grad[d] = jac_ptr[q][d][0] * data_ptr[2*dofs_per_cell+q];
                for (unsigned int e=1; e<dim; ++e)
                  grad[d] += jac_ptr[q][d][e] * data_ptr[(2+e)*dofs_per_cell+q];
              }
            // apply gradient of test function
            for (unsigned int d=0; d<dim; ++d)
              {
                data_ptr[(2+d)*dofs_per_cell+q] = jac_ptr[q][0][d] * grad[0];
                for (unsigned int e=1; e<dim; ++e)
                  data_ptr[(2+d)*dofs_per_cell+q] += jac_ptr[q][e][d] * grad[e];
                // multiply by quadrature weight
                data_ptr[(2+d)*dofs_per_cell+q] *= jxw_ptr[q];
              }
          }

      // apply tensor product kernels
      eval.template gradients<0,false,false>(data_ptr+2*dofs_per_cell, data_ptr+dofs_per_cell);
      if (dim > 1)
        eval.template gradients<1,false,true>(data_ptr+3*dofs_per_cell, data_ptr+dofs_per_cell);
      if (dim > 2)
        eval.template gradients<2,false,true>(data_ptr+4*dofs_per_cell, data_ptr+dofs_per_cell);

      if (dim == 1)
        eval_val.template values<0,false,false>(data_ptr+dofs_per_cell, data_ptr);
      else if (dim == 2)
        {
          eval_val.template values<0,false,false>(data_ptr+dofs_per_cell, data_ptr+2*dofs_per_cell);
          eval_val.template values<1,false,false>(data_ptr+2*dofs_per_cell, data_ptr);
        }
      else if (dim == 3)
        {
          eval_val.template values<0,false,false>(data_ptr+dofs_per_cell, data_ptr+2*dofs_per_cell);
          eval_val.template values<1,false,false>(data_ptr+2*dofs_per_cell, data_ptr+dofs_per_cell);
          eval_val.template values<2,false,false>(data_ptr+dofs_per_cell, data_ptr);
        }
      // write into the solution vector, overwrite rather than sum-into
      if (vectorization_populated == VectorizedArray<Number>::n_array_elements)
        // transform struct-of-array to array-of-struct
        vectorized_transpose_and_store(false, dofs_per_cell, data_ptr,
                                       dof_indices, dst.begin());
      else
        {
          for (unsigned int v=0; v<vectorization_populated; ++v)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              dst.local_element(dof_indices[v]+i) = data_ptr[i][v];
        }
    }
  data->release_scratch_data(scratch_data_array);
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP(("cell_loop_1_p" + std::to_string(degree)).c_str());
#endif
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
cell_loop_manual_2(parallel::distributed::Vector<Number> &dst,
                   const parallel::distributed::Vector<Number> &src) const
{
  if (dim != 3 || degree < 1)
    return;
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START(("cell_loop_2_p" + std::to_string(degree)).c_str());
#endif
  // simlar to cell_loop_manual_1 but expanding the loops of eval_val.value
  // and eval.gradient

  // do not exchange data or zero out, assume DG operator does not need to
  // exchange and that the loop below takes care of the zeroing

  // global data structures
  const unsigned int dofs_per_cell = Utilities::fixed_int_power<degree+1,dim>::value;
  AlignedVector<VectorizedArray<Number> > *scratch_data_array = data->acquire_scratch_data();
  scratch_data_array->resize((dim+1)*dofs_per_cell);
  VectorizedArray<Number> *__restrict data_ptr = scratch_data_array->begin();

  const unsigned int nn = degree+1;
  const unsigned int mid = nn/2;
  const unsigned int offset = (nn+1)/2;
  const VectorizedArray<Number> *__restrict shape_vals = data->get_shape_info().shape_values_eo.begin();
  const VectorizedArray<Number> *__restrict shape_grads = data->get_shape_info().shape_gradients_collocation_eo.begin();
  typedef internal::EvaluatorTensorProduct<internal::evaluate_evenodd, dim, degree, degree+1,
                                           VectorizedArray<Number> > Eval;
  Eval eval_val (data->get_shape_info().shape_values_eo);
  Eval eval(AlignedVector<VectorizedArray<Number> >(),
            data->get_shape_info().shape_gradients_collocation_eo,
            data->get_shape_info().shape_hessians_collocation_eo);

  const Number*__restrict quadrature_weights =
    data->get_mapping_info().cell_data[0].descriptor[0].quadrature_weights.begin();

  for (unsigned int cell=0; cell<data->n_macro_cells(); ++cell)
    {
      // read from src vector
      const unsigned int *dof_indices =
        &data->get_dof_info().dof_indices_contiguous[2][cell*VectorizedArray<Number>::n_array_elements];
      const unsigned int vectorization_populated =
        data->n_components_filled(cell);
      // transform array-of-struct to struct-of-array
      if (vectorization_populated == VectorizedArray<Number>::n_array_elements)
        vectorized_load_and_transpose(dofs_per_cell, src.begin(),
                                      dof_indices, data_ptr);
      else
        {
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            data_ptr[i] = VectorizedArray<Number>();
          for (unsigned int v=0; v<vectorization_populated; ++v)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              data_ptr[i][v] = src.local_element(dof_indices[v]+i);
        }

      // apply tensor product kernels
      for (unsigned int i2=0; i2<nn; ++i2)
        {
          // x-direction
          VectorizedArray<Number> *__restrict in = data_ptr + i2*nn*nn;
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i+i1*nn] + in[nn-1-i+i1*nn];
                  xm[i] = in[i+i1*nn] - in[nn-1-i+i1*nn];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_vals[col]                 * xp[0];
                  r1 = shape_vals[degree*offset + col] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_vals[ind*offset+col]          * xp[ind];
                      r1 += shape_vals[(degree-ind)*offset+col] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r0 += shape_vals[mid*offset+col] * in[i1*nn+mid];

                  in[i1*nn+col]      = r0 + r1;
                  in[i1*nn+nn-1-col] = r0 - r1;
                }
            }
          // y-direction
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i*nn+i1] + in[(nn-1-i)*nn+i1];
                  xm[i] = in[i*nn+i1] - in[(nn-1-i)*nn+i1];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_vals[col]                 * xp[0];
                  r1 = shape_vals[degree*offset + col] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_vals[ind*offset+col]          * xp[ind];
                      r1 += shape_vals[(degree-ind)*offset+col] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r0 += shape_vals[mid*offset+col] * in[i1+mid*nn];

                  in[i1+col*nn]        = r0 + r1;
                  in[i1+(nn-1-col)*nn] = r0 - r1;
                }
            }
        }

      // z direction
      for (unsigned int i1 = 0; i1<nn*nn; ++i1)
        {
          VectorizedArray<Number> *__restrict in = data_ptr + i1;
          VectorizedArray<Number> *__restrict outz = data_ptr + i1 + 3*dofs_per_cell;
          const unsigned int stride = nn*nn;
          VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
          VectorizedArray<Number> xxp[mid>0?mid:1], xxm[mid>0?mid:1];
          for (unsigned int i=0; i<mid; ++i)
            {
              xp[i] = in[i*stride] + in[(nn-1-i)*stride];
              xm[i] = in[i*stride] - in[(nn-1-i)*stride];
            }
          for (unsigned int col=0; col<mid; ++col)
            {
              xxp[col] = shape_vals[col]                 * xp[0];
              xxm[col] = shape_vals[degree*offset + col] * xm[0];
              for (unsigned int ind=1; ind<mid; ++ind)
                {
                  xxp[col] += shape_vals[ind*offset+col]          * xp[ind];
                  xxm[col] += shape_vals[(degree-ind)*offset+col] * xm[ind];
                }
              if (nn % 2 == 1)
                xxp[col] += shape_vals[mid*offset+col] * in[mid*stride];

              in[col*stride]        = xxp[col] + xxm[col];
              in[(nn-1-col)*stride] = xxp[col] - xxm[col];
            }
          // z-derivative
          for (unsigned int col=0; col<mid; ++col)
            {
              VectorizedArray<Number> r0, r1;
              r0 = shape_grads[col]                 * xxm[0];
              r1 = shape_grads[degree*offset + col] * xxp[0];
              for (unsigned int ind=1; ind<mid; ++ind)
                {
                  r0 += shape_grads[ind*offset+col]          * xxm[ind];
                  r1 += shape_grads[(degree-ind)*offset+col] * xxp[ind];
                }
              r0 += r0;
              r1 += r1;
              if (nn % 2 == 1)
                r1 += shape_grads[mid*offset+col] * in[mid*stride];

              outz[col*stride]        = r0 + r1;
              outz[(nn-1-col)*stride] = r0 - r1;
            }
          if (nn%2==1)
            {
              VectorizedArray<Number> r0 = shape_grads[mid] * xxm[0];
              for (unsigned int ind=1; ind<mid; ++ind)
                r0 += shape_grads[ind*offset+mid] * xxm[ind];
              outz[stride*mid] = 2.*r0;
            }
        }

      for (unsigned int i2=0; i2<nn; ++i2)
        {
          VectorizedArray<Number> *__restrict in = data_ptr + i2*nn*nn;
          VectorizedArray<Number> *__restrict outy = data_ptr + i2*nn*nn + 2*dofs_per_cell;
          VectorizedArray<Number> *__restrict outx = data_ptr + i2*nn*nn + dofs_per_cell;
          // y-derivative
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i*nn+i1] - in[(nn-1-i)*nn+i1];
                  xm[i] = in[i*nn+i1] + in[(nn-1-i)*nn+i1];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_grads[col]                 * xp[0];
                  r1 = shape_grads[degree*offset + col] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_grads[ind*offset+col]          * xp[ind];
                      r1 += shape_grads[(degree-ind)*offset+col] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r1 += shape_grads[mid*offset+col] * in[i1+mid*nn];

                  outy[i1+col*nn]        = r0 + r1;
                  outy[i1+(nn-1-col)*nn] = r0 - r1;
                }
              if (nn%2 == 1)
                {
                  VectorizedArray<Number> r0 = shape_grads[mid] * xp[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    r0 += shape_grads[ind*offset+mid] * xp[ind];
                  outy[i1+nn*mid] = r0;
                }
            }
          // x-derivative
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i+i1*nn] - in[nn-1-i+i1*nn];
                  xm[i] = in[i+i1*nn] + in[nn-1-i+i1*nn];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_grads[col]                 * xp[0];
                  r1 = shape_grads[degree*offset + col] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_grads[ind*offset+col]          * xp[ind];
                      r1 += shape_grads[(degree-ind)*offset+col] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r1 += shape_grads[mid*offset+col] * in[i1*nn+mid];

                  outx[i1*nn+col]      = r0 + r1;
                  outx[i1*nn+nn-1-col] = r0 - r1;
                }
              if (nn%2 == 1)
                {
                  VectorizedArray<Number> r0 = shape_grads[mid] * xp[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    r0 += shape_grads[ind*offset+mid] * xp[ind];
                  outx[i1*nn+mid] = r0;
                }
            }
        }


      // loop over quadrature points. depends on the data layout in
      // MappingInfo
      const VectorizedArray<Number> *__restrict jxw_ptr =
        data->get_mapping_info().cell_data[0].JxW_values.begin() +
        data->get_mapping_info().cell_data[0].data_index_offsets[cell];
      const Tensor<2,dim,VectorizedArray<Number>> *__restrict jac_ptr =
        data->get_mapping_info().cell_data[0].jacobians[0].begin() +
        data->get_mapping_info().cell_data[0].data_index_offsets[cell];
      // Cartesian cell case
      if (data->get_mapping_info().cell_type[cell] == internal::MatrixFreeFunctions::cartesian)
        for (unsigned int q=0; q<dofs_per_cell; ++q)
          for (unsigned int d=0; d<dim; ++d)
            data_ptr[(1+d)*dofs_per_cell+q] *= jac_ptr[0][d][d] *
              jac_ptr[0][d][d] * (jxw_ptr[0] * quadrature_weights[q]);
      // affine cell case
      else if (data->get_mapping_info().cell_type[cell] == internal::MatrixFreeFunctions::affine)
        for (unsigned int q=0; q<dofs_per_cell; ++q)
          {
            VectorizedArray<Number> grad[dim];
            // get gradient
            for (unsigned int d=0; d<dim; ++d)
              {
                grad[d] = jac_ptr[0][d][0] * data_ptr[1*dofs_per_cell+q];
                for (unsigned int e=1; e<dim; ++e)
                  grad[d] += jac_ptr[0][d][e] * data_ptr[(1+e)*dofs_per_cell+q];
              }
            // apply gradient of test function
            for (unsigned int d=0; d<dim; ++d)
              {
                data_ptr[(1+d)*dofs_per_cell+q] = jac_ptr[0][0][d] * grad[0];
                for (unsigned int e=1; e<dim; ++e)
                  data_ptr[(1+d)*dofs_per_cell+q] += jac_ptr[0][e][d] * grad[e];
                // multiply by quadrature weight
                data_ptr[(1+d)*dofs_per_cell+q] *= jxw_ptr[0] * quadrature_weights[q];
              }
          }
      else
        // non-affine cell case
        for (unsigned int q=0; q<dofs_per_cell; ++q)
          {
            VectorizedArray<Number> grad[dim];
            // get gradient
            for (unsigned int d=0; d<dim; ++d)
              {
                grad[d] = jac_ptr[q][d][0] * data_ptr[1*dofs_per_cell+q];
                for (unsigned int e=1; e<dim; ++e)
                  grad[d] += jac_ptr[q][d][e] * data_ptr[(1+e)*dofs_per_cell+q];
              }
            // apply gradient of test function
            for (unsigned int d=0; d<dim; ++d)
              {
                data_ptr[(1+d)*dofs_per_cell+q] = jac_ptr[q][0][d] * grad[0];
                for (unsigned int e=1; e<dim; ++e)
                  data_ptr[(1+d)*dofs_per_cell+q] += jac_ptr[q][e][d] * grad[e];
                // multiply by quadrature weight
                data_ptr[(1+d)*dofs_per_cell+q] *= jxw_ptr[q];
              }
          }

      // apply tensor product kernels
      for (unsigned int i2=0; i2<nn; ++i2)
        {
          VectorizedArray<Number> *__restrict iny = data_ptr + i2*nn*nn + 2*dofs_per_cell;
          VectorizedArray<Number> *__restrict inx = data_ptr + i2*nn*nn + dofs_per_cell;
          VectorizedArray<Number> *__restrict out = data_ptr + i2*nn*nn;
          // x-derivative
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = inx[i+i1*nn] + inx[nn-1-i+i1*nn];
                  xm[i] = inx[i+i1*nn] - inx[nn-1-i+i1*nn];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_grads[col*offset]          * xp[0];
                  r1 = shape_grads[(degree-col)*offset] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_grads[col*offset+ind]          * xp[ind];
                      r1 += shape_grads[(degree-col)*offset+ind] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r0 += shape_grads[col*offset+mid] * inx[i1*nn+mid];

                  out[i1*nn+col]      = r0 + r1;
                  out[i1*nn+nn-1-col] = r1 - r0;
                }
              if (nn%2 == 1)
                {
                  VectorizedArray<Number> r0 = shape_grads[mid*offset] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    r0 += shape_grads[ind+mid*offset] * xm[ind];
                  out[i1*nn+mid] = r0;
                }
            }
          // y-derivative
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = iny[i*nn+i1] + iny[(nn-1-i)*nn+i1];
                  xm[i] = iny[i*nn+i1] - iny[(nn-1-i)*nn+i1];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_grads[col*offset]          * xp[0];
                  r1 = shape_grads[(degree-col)*offset] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_grads[col*offset+ind]          * xp[ind];
                      r1 += shape_grads[(degree-col)*offset+ind] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r0 += shape_grads[col*offset+mid] * iny[i1+mid*nn];

                  out[i1+col*nn]        += r0 + r1;
                  out[i1+(nn-1-col)*nn] += r1 - r0;
                }
              if (nn%2 == 1)
                {
                  VectorizedArray<Number> r0 = out[i1+nn*mid];
                  for (unsigned int ind=0; ind<mid; ++ind)
                    r0 += shape_grads[ind+mid*offset] * xm[ind];
                  out[i1+nn*mid] = r0;
                }
            }
        }

      // z direction
      for (unsigned int i1 = 0; i1<nn*nn; ++i1)
        {
          // z-derivative
          VectorizedArray<Number> *__restrict inz = data_ptr + i1 + 3*dofs_per_cell;
          VectorizedArray<Number> *__restrict out = data_ptr + i1;
          const unsigned int stride = nn*nn;
          VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
          for (unsigned int i=0; i<mid; ++i)
            {
              xp[i] = inz[i*stride] + inz[(nn-1-i)*stride];
              xm[i] = inz[i*stride] - inz[(nn-1-i)*stride];
            }
          for (unsigned int col=0; col<mid; ++col)
            {
              VectorizedArray<Number> r0, r1;
              r0 = shape_grads[col*offset]          * xp[0];
              r1 = shape_grads[(degree-col)*offset] * xm[0];
              for (unsigned int ind=1; ind<mid; ++ind)
                {
                  r0 += shape_grads[col*offset+ind]          * xp[ind];
                  r1 += shape_grads[(degree-col)*offset+ind] * xm[ind];
                }
              if (nn % 2 == 1)
                r0 += shape_grads[col*offset+mid] * inz[mid*stride];

              out[col*stride]        += r0 + r1;
              out[(nn-1-col)*stride] += r1 - r0;
            }
          if (nn%2 == 1)
            {
              VectorizedArray<Number> r0 = out[mid*stride];
              for (unsigned int ind=0; ind<mid; ++ind)
                r0 += shape_grads[ind+mid*offset] * xm[ind];
              out[mid*stride] = r0;
            }
          // z-values
          for (unsigned int i=0; i<mid; ++i)
            {
              xp[i] = out[i*stride] + out[(nn-1-i)*stride];
              xm[i] = out[i*stride] - out[(nn-1-i)*stride];
            }
          for (unsigned int col=0; col<mid; ++col)
            {
              VectorizedArray<Number> r0, r1;
              r0 = shape_vals[col*offset]          * xp[0];
              r1 = shape_vals[(degree-col)*offset] * xm[0];
              for (unsigned int ind=1; ind<mid; ++ind)
                {
                  r0 += shape_vals[col*offset+ind]          * xp[ind];
                  r1 += shape_vals[(degree-col)*offset+ind] * xm[ind];
                }

              out[col*stride]        = r0 + r1;
              out[(nn-1-col)*stride] = r0 - r1;
            }
          if (nn%2==1)
            {
              // sum into because shape value is one in the middle
              VectorizedArray<Number> r0 = out[stride*mid];
              for (unsigned int ind=0; ind<mid; ++ind)
                r0 += shape_vals[ind+mid*offset] * xp[ind];
              out[stride*mid] = r0;
            }
        }
      for (unsigned int i2=0; i2<nn; ++i2)
        {
          VectorizedArray<Number> *__restrict in = data_ptr + i2*nn*nn;
          // y-direction
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i*nn+i1] + in[(nn-1-i)*nn+i1];
                  xm[i] = in[i*nn+i1] - in[(nn-1-i)*nn+i1];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_vals[col*offset]          * xp[0];
                  r1 = shape_vals[(degree-col)*offset] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_vals[col*offset+ind]          * xp[ind];
                      r1 += shape_vals[(degree-col)*offset+ind] * xm[ind];
                    }

                  in[i1+col*nn]        = r0 + r1;
                  in[i1+(nn-1-col)*nn] = r0 - r1;
                }
              if (nn%2==1)
                {
                  // sum into because shape value is one in the middle
                  VectorizedArray<Number> r0 = in[i1+mid*nn];
                  for (unsigned int ind=0; ind<mid; ++ind)
                    r0 += shape_vals[ind+mid*offset] * xp[ind];
                  in[i1+mid*nn] = r0;
                }
            }
          // x-direction
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i+i1*nn] + in[nn-1-i+i1*nn];
                  xm[i] = in[i+i1*nn] - in[nn-1-i+i1*nn];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_vals[col*offset]          * xp[0];
                  r1 = shape_vals[(degree-col)*offset] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_vals[col*offset+ind]          * xp[ind];
                      r1 += shape_vals[(degree-col)*offset+ind] * xm[ind];
                    }

                  in[col+i1*nn]        = r0 + r1;
                  in[(nn-1-col)+i1*nn] = r0 - r1;
                }
              if (nn%2==1)
                {
                  // sum into because shape value is one in the middle
                  VectorizedArray<Number> r0 = in[i1*nn+mid];
                  for (unsigned int ind=0; ind<mid; ++ind)
                    r0 += shape_vals[ind+mid*offset] * xp[ind];
                  in[i1*nn+mid] = r0;
                }
            }
        }

      // write into the solution vector, overwrite rather than sum-into
      if (vectorization_populated == VectorizedArray<Number>::n_array_elements)
        // transform struct-of-array to array-of-struct
        vectorized_transpose_and_store(false, dofs_per_cell, data_ptr,
                                       dof_indices, dst.begin());
      else
        {
          for (unsigned int v=0; v<vectorization_populated; ++v)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              dst.local_element(dof_indices[v]+i) = data_ptr[i][v];
        }
    }
  data->release_scratch_data(scratch_data_array);
#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP(("cell_loop_2_p" + std::to_string(degree)).c_str());
#endif
}



template <int dim, int degree, typename Number>
void LaplaceOperator<dim,degree,Number>::
cell_loop_manual_3(parallel::distributed::Vector<Number> &dst,
                   const parallel::distributed::Vector<Number> &src) const
{
  if (dim != 3 || degree < 1)
    return;
  // simlar to cell_loop_manual_1 but expanding the loops of eval_val.value
  // and eval.gradient

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_START(("cell_loop_3_p" + std::to_string(degree)).c_str());
#endif

  // do not exchange data or zero out, assume DG operator does not need to
  // exchange and that the loop below takes care of the zeroing

  // global data structures
  const unsigned int dofs_per_cell = Utilities::fixed_int_power<degree+1,dim>::value;
  AlignedVector<VectorizedArray<Number> > *scratch_data_array = data->acquire_scratch_data();
  scratch_data_array->resize(2*dofs_per_cell);
  VectorizedArray<Number> my_array[degree < 9 ? 2*dofs_per_cell : 1];
  VectorizedArray<Number> *__restrict data_ptr = degree < 9 ? my_array : scratch_data_array->begin();
  VectorizedArray<Number> merged_array[dim*(dim+1)/2];
  for (unsigned int d=0; d<dim*(dim+1)/2; ++d)
    merged_array[d] = VectorizedArray<Number>();

  const unsigned int nn = degree+1;
  const unsigned int mid = nn/2;
  const unsigned int offset = (nn+1)/2;
  const VectorizedArray<Number> *__restrict shape_vals = data->get_shape_info().shape_values_eo.begin();
  const VectorizedArray<Number> *__restrict shape_grads = data->get_shape_info().shape_gradients_collocation_eo.begin();
  typedef internal::EvaluatorTensorProduct<internal::evaluate_evenodd, dim, degree, degree+1,
                                           VectorizedArray<Number> > Eval;
  Eval eval_val (data->get_shape_info().shape_values_eo);
  Eval eval(AlignedVector<VectorizedArray<Number> >(),
            data->get_shape_info().shape_gradients_collocation_eo,
            data->get_shape_info().shape_hessians_collocation_eo);

  const Number*__restrict quadrature_weights =
    data->get_mapping_info().cell_data[0].descriptor[0].quadrature_weights.begin();

  for (unsigned int cell=0; cell<data->n_macro_cells(); ++cell)
    {
      // read from src vector
      const unsigned int *dof_indices =
        &data->get_dof_info().dof_indices_contiguous[2][cell*VectorizedArray<Number>::n_array_elements];
      const unsigned int vectorization_populated =
        data->n_components_filled(cell);
      // transform array-of-struct to struct-of-array
      if (vectorization_populated == VectorizedArray<Number>::n_array_elements)
        vectorized_load_and_transpose(dofs_per_cell, src.begin(),
                                      dof_indices, data_ptr);
      else
        {
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            data_ptr[i] = VectorizedArray<Number>();
          for (unsigned int v=0; v<vectorization_populated; ++v)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              data_ptr[i][v] = src.local_element(dof_indices[v]+i);
        }

      // --------------------------------------------------------------------
      // apply tensor product kernels
      for (unsigned int i2=0; i2<nn; ++i2)
        {
          // x-direction
          VectorizedArray<Number> *__restrict in = data_ptr + i2*nn*nn;
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i+i1*nn] + in[nn-1-i+i1*nn];
                  xm[i] = in[i+i1*nn] - in[nn-1-i+i1*nn];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_vals[col]                 * xp[0];
                  r1 = shape_vals[degree*offset + col] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_vals[ind*offset+col]          * xp[ind];
                      r1 += shape_vals[(degree-ind)*offset+col] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r0 += shape_vals[mid*offset+col] * in[i1*nn+mid];

                  in[i1*nn+col]      = r0 + r1;
                  in[i1*nn+nn-1-col] = r0 - r1;
                }
            }
          // y-direction
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i*nn+i1] + in[(nn-1-i)*nn+i1];
                  xm[i] = in[i*nn+i1] - in[(nn-1-i)*nn+i1];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_vals[col]                 * xp[0];
                  r1 = shape_vals[degree*offset + col] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_vals[ind*offset+col]          * xp[ind];
                      r1 += shape_vals[(degree-ind)*offset+col] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r0 += shape_vals[mid*offset+col] * in[i1+mid*nn];

                  in[i1+col*nn]        = r0 + r1;
                  in[i1+(nn-1-col)*nn] = r0 - r1;
                }
            }
        }

      // z direction
      for (unsigned int i1 = 0; i1<nn*nn; ++i1)
        {
          VectorizedArray<Number> *__restrict in = data_ptr + i1;
          VectorizedArray<Number> *__restrict outz = data_ptr + i1 + dofs_per_cell;
          const unsigned int stride = nn*nn;
          VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
          VectorizedArray<Number> xxp[mid>0?mid:1], xxm[mid>0?mid:1];
          for (unsigned int i=0; i<mid; ++i)
            {
              xp[i] = in[i*stride] + in[(nn-1-i)*stride];
              xm[i] = in[i*stride] - in[(nn-1-i)*stride];
            }
          for (unsigned int col=0; col<mid; ++col)
            {
              xxp[col] = shape_vals[col]                 * xp[0];
              xxm[col] = shape_vals[degree*offset + col] * xm[0];
              for (unsigned int ind=1; ind<mid; ++ind)
                {
                  xxp[col] += shape_vals[ind*offset+col]          * xp[ind];
                  xxm[col] += shape_vals[(degree-ind)*offset+col] * xm[ind];
                }
              if (nn % 2 == 1)
                xxp[col] += shape_vals[mid*offset+col] * in[mid*stride];

              in[col*stride]        = xxp[col] + xxm[col];
              in[(nn-1-col)*stride] = xxp[col] - xxm[col];
            }
          // z-derivative
          for (unsigned int col=0; col<mid; ++col)
            {
              VectorizedArray<Number> r0, r1;
              r0 = shape_grads[col]                 * xxm[0];
              r1 = shape_grads[degree*offset + col] * xxp[0];
              for (unsigned int ind=1; ind<mid; ++ind)
                {
                  r0 += shape_grads[ind*offset+col]          * xxm[ind];
                  r1 += shape_grads[(degree-ind)*offset+col] * xxp[ind];
                }
              r0 += r0;
              r1 += r1;
              if (nn % 2 == 1)
                r1 += shape_grads[mid*offset+col] * in[mid*stride];

              outz[col*stride]        = r0 + r1;
              outz[(nn-1-col)*stride] = r0 - r1;
            }
          if (nn%2==1)
            {
              VectorizedArray<Number> r0 = shape_grads[mid] * xxm[0];
              for (unsigned int ind=1; ind<mid; ++ind)
                r0 += shape_grads[ind*offset+mid] * xxm[ind];
              outz[stride*mid] = 2.*r0;
            }
        }

      // --------------------------------------------------------------------
      // mix with loop over quadrature points. depends on the data layout in
      // MappingInfo
      const VectorizedArray<Number> *__restrict jxw_ptr =
        data->get_mapping_info().cell_data[0].JxW_values.begin() +
        data->get_mapping_info().cell_data[0].data_index_offsets[cell];
      const Tensor<2,dim,VectorizedArray<Number>> *__restrict jac_ptr =
        data->get_mapping_info().cell_data[0].jacobians[0].begin() +
        data->get_mapping_info().cell_data[0].data_index_offsets[cell];
      const internal::MatrixFreeFunctions::CellType cell_type =
        data->get_mapping_info().cell_type[cell];
      if (cell_type == internal::MatrixFreeFunctions::cartesian)
        for (unsigned int d=0; d<dim; ++d)
          merged_array[d] = jxw_ptr[0] * jac_ptr[0][d][d] * jac_ptr[0][d][d];
      else if (cell_type == internal::MatrixFreeFunctions::affine)
        {
          for (unsigned int d=0, c=0; d<dim; ++d)
            for (unsigned int e=d; e<dim; ++e, ++c)
              {
                VectorizedArray<Number> sum = jac_ptr[0][d] * jac_ptr[0][e];
                for (unsigned int f=1; f<dim; ++f)
                  sum += jac_ptr[f][d] * jac_ptr[f][e];
                merged_array[c] = jxw_ptr[0] * sum;
              }
        }

      for (unsigned int i2=0; i2<nn; ++i2)  // loop over z layers
        {
          VectorizedArray<Number> *__restrict in = data_ptr + i2*nn*nn;
          VectorizedArray<Number> *__restrict outz = data_ptr + dofs_per_cell;
          VectorizedArray<Number> outy[nn*nn];
          // y-derivative
          for (unsigned int i1=0; i1<nn; ++i1) // loop over x layers
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i*nn+i1] - in[(nn-1-i)*nn+i1];
                  xm[i] = in[i*nn+i1] + in[(nn-1-i)*nn+i1];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_grads[col]                 * xp[0];
                  r1 = shape_grads[degree*offset + col] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_grads[ind*offset+col]          * xp[ind];
                      r1 += shape_grads[(degree-ind)*offset+col] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r1 += shape_grads[mid*offset+col] * in[i1+mid*nn];

                  outy[i1+col*nn]        = r0 + r1;
                  outy[i1+(nn-1-col)*nn] = r0 - r1;
                }
              if (nn%2 == 1)
                {
                  VectorizedArray<Number> r0 = shape_grads[mid] * xp[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    r0 += shape_grads[ind*offset+mid] * xp[ind];
                  outy[i1+nn*mid] = r0;
                }
            }

          // x-derivative
          for (unsigned int i1=0; i1<nn; ++i1) // loop over y layers
            {
              VectorizedArray<Number> outx[nn];
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i+i1*nn] - in[nn-1-i+i1*nn];
                  xm[i] = in[i+i1*nn] + in[nn-1-i+i1*nn];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_grads[col]                 * xp[0];
                  r1 = shape_grads[degree*offset + col] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_grads[ind*offset+col]          * xp[ind];
                      r1 += shape_grads[(degree-ind)*offset+col] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r1 += shape_grads[mid*offset+col] * in[i1*nn+mid];

                  outx[col]      = r0 + r1;
                  outx[nn-1-col] = r0 - r1;
                }
              if (nn%2 == 1)
                {
                  VectorizedArray<Number> r0 = shape_grads[mid] * xp[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    r0 += shape_grads[ind*offset+mid] * xp[ind];
                  outx[mid] = r0;
                }

              // operations on quadrature points
              // Cartesian cell case
              if (cell_type == internal::MatrixFreeFunctions::cartesian)
                for (unsigned int i=0; i<nn; ++i)
                  {
                    const VectorizedArray<Number> weight =
                      make_vectorized_array(quadrature_weights[i2*nn*nn+i1*nn+i]);
                    outx[i] *= weight * merged_array[0];
                    outy[i1*nn+i] *= weight * merged_array[1];
                    outz[i2*nn*nn+i1*nn+i] *= weight * merged_array[2];
                  }
              else if (cell_type == internal::MatrixFreeFunctions::affine)
                for (unsigned int i=0; i<nn; ++i)
                  {
                    const VectorizedArray<Number> weight =
                      make_vectorized_array(quadrature_weights[i2*nn*nn+i1*nn+i]);
                    const VectorizedArray<Number> tmpx =
                      weight * (merged_array[0] * outx[i] + merged_array[1] * outy[i1*nn+i]
                                + merged_array[2] * outz[i2*nn*nn+i1*nn+i]);
                    const VectorizedArray<Number> tmpy =
                      weight * (merged_array[1] * outx[i] + merged_array[3] * outy[i1*nn+i]
                                + merged_array[4] * outz[i2*nn*nn+i1*nn+i]);
                    outz[i2*nn*nn+i1*nn+i] =
                      weight * (merged_array[2] * outx[i] + merged_array[4] * outy[i1*nn+i]
                                + merged_array[5] * outz[i2*nn*nn+i1*nn+i]);
                    outy[i1*nn+i] = tmpy;
                    outx[i] = tmpx;
                  }
              else
                for (unsigned int i=0; i<nn; ++i)
                  {
                    const unsigned int q=i2*nn*nn+i1*nn+i;
                    VectorizedArray<Number> grad[dim];
                    // get gradient
                    for (unsigned int d=0; d<dim; ++d)
                      grad[d] = jac_ptr[q][d][0] * outx[i] + jac_ptr[q][d][1] * outy[i1*nn+i]
                        + jac_ptr[q][d][2] * outz[q];

                    // apply gradient of test function
                    outx[i] = jxw_ptr[q] * (jac_ptr[q][0][0] * grad[0] + jac_ptr[q][1][0] * grad[1]
                                            + jac_ptr[q][2][0] * grad[2]);
                    outy[i1*nn+i] = jxw_ptr[q] * (jac_ptr[q][0][1] * grad[0] +
                                                  jac_ptr[q][1][1] * grad[1] +
                                                  jac_ptr[q][2][1] * grad[2]);
                    outz[q] = jxw_ptr[q] * (jac_ptr[q][0][2] * grad[0] + jac_ptr[q][1][2] * grad[1]
                                            + jac_ptr[q][2][2] * grad[2]);
                  }

              // x-derivative
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = outx[i] + outx[nn-1-i];
                  xm[i] = outx[i] - outx[nn-1-i];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_grads[col*offset]          * xp[0];
                  r1 = shape_grads[(degree-col)*offset] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_grads[col*offset+ind]          * xp[ind];
                      r1 += shape_grads[(degree-col)*offset+ind] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r0 += shape_grads[col*offset+mid] * outx[mid];

                  in[i1*nn+col]      = r0 + r1;
                  in[i1*nn+nn-1-col] = r1 - r0;
                }
              if (nn%2 == 1)
                {
                  VectorizedArray<Number> r0 = shape_grads[mid*offset] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    r0 += shape_grads[ind+mid*offset] * xm[ind];
                  in[i1*nn+mid] = r0;
                }
            } // end of loop over y layers

          // y-derivative
          for (unsigned int i1=0; i1<nn; ++i1) // loop over x layers
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = outy[i*nn+i1] + outy[(nn-1-i)*nn+i1];
                  xm[i] = outy[i*nn+i1] - outy[(nn-1-i)*nn+i1];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_grads[col*offset]          * xp[0];
                  r1 = shape_grads[(degree-col)*offset] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_grads[col*offset+ind]          * xp[ind];
                      r1 += shape_grads[(degree-col)*offset+ind] * xm[ind];
                    }
                  if (nn % 2 == 1)
                    r0 += shape_grads[col*offset+mid] * outy[i1+mid*nn];

                  in[i1+col*nn]        += r0 + r1;
                  in[i1+(nn-1-col)*nn] += r1 - r0;
                }
              if (nn%2 == 1)
                {
                  VectorizedArray<Number> r0 = in[i1+nn*mid];
                  for (unsigned int ind=0; ind<mid; ++ind)
                    r0 += shape_grads[ind+mid*offset] * xm[ind];
                  in[i1+nn*mid] = r0;
                }
            }
        } // end of loop over z layers

      // z direction
      for (unsigned int i1 = 0; i1<nn*nn; ++i1)
        {
          // z-derivative
          VectorizedArray<Number> *__restrict inz = data_ptr + i1 + dofs_per_cell;
          VectorizedArray<Number> *__restrict out = data_ptr + i1;
          const unsigned int stride = nn*nn;
          VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
          for (unsigned int i=0; i<mid; ++i)
            {
              xp[i] = inz[i*stride] + inz[(nn-1-i)*stride];
              xm[i] = inz[i*stride] - inz[(nn-1-i)*stride];
            }
          for (unsigned int col=0; col<mid; ++col)
            {
              VectorizedArray<Number> r0, r1;
              r0 = shape_grads[col*offset]          * xp[0];
              r1 = shape_grads[(degree-col)*offset] * xm[0];
              for (unsigned int ind=1; ind<mid; ++ind)
                {
                  r0 += shape_grads[col*offset+ind]          * xp[ind];
                  r1 += shape_grads[(degree-col)*offset+ind] * xm[ind];
                }
              if (nn % 2 == 1)
                r0 += shape_grads[col*offset+mid] * inz[mid*stride];

              out[col*stride]        += r0 + r1;
              out[(nn-1-col)*stride] += r1 - r0;
            }
          if (nn%2 == 1)
            {
              VectorizedArray<Number> r0 = out[mid*stride];
              for (unsigned int ind=0; ind<mid; ++ind)
                r0 += shape_grads[ind+mid*offset] * xm[ind];
              out[mid*stride] = r0;
            }
          // z-values
          for (unsigned int i=0; i<mid; ++i)
            {
              xp[i] = out[i*stride] + out[(nn-1-i)*stride];
              xm[i] = out[i*stride] - out[(nn-1-i)*stride];
            }
          for (unsigned int col=0; col<mid; ++col)
            {
              VectorizedArray<Number> r0, r1;
              r0 = shape_vals[col*offset]          * xp[0];
              r1 = shape_vals[(degree-col)*offset] * xm[0];
              for (unsigned int ind=1; ind<mid; ++ind)
                {
                  r0 += shape_vals[col*offset+ind]          * xp[ind];
                  r1 += shape_vals[(degree-col)*offset+ind] * xm[ind];
                }

              out[col*stride]        = r0 + r1;
              out[(nn-1-col)*stride] = r0 - r1;
            }
          if (nn%2==1)
            {
              // sum into because shape value is one in the middle
              VectorizedArray<Number> r0 = out[stride*mid];
              for (unsigned int ind=0; ind<mid; ++ind)
                r0 += shape_vals[ind+mid*offset] * xp[ind];
              out[stride*mid] = r0;
            }
        }
      for (unsigned int i2=0; i2<nn; ++i2)
        {
          VectorizedArray<Number> *__restrict in = data_ptr + i2*nn*nn;
          // y-direction
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i*nn+i1] + in[(nn-1-i)*nn+i1];
                  xm[i] = in[i*nn+i1] - in[(nn-1-i)*nn+i1];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_vals[col*offset]          * xp[0];
                  r1 = shape_vals[(degree-col)*offset] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_vals[col*offset+ind]          * xp[ind];
                      r1 += shape_vals[(degree-col)*offset+ind] * xm[ind];
                    }

                  in[i1+col*nn]        = r0 + r1;
                  in[i1+(nn-1-col)*nn] = r0 - r1;
                }
              if (nn%2==1)
                {
                  // sum into because shape value is one in the middle
                  VectorizedArray<Number> r0 = in[i1+mid*nn];
                  for (unsigned int ind=0; ind<mid; ++ind)
                    r0 += shape_vals[ind+mid*offset] * xp[ind];
                  in[i1+mid*nn] = r0;
                }
            }
          // x-direction
          for (unsigned int i1=0; i1<nn; ++i1)
            {
              VectorizedArray<Number> xp[mid>0?mid:1], xm[mid>0?mid:1];
              for (unsigned int i=0; i<mid; ++i)
                {
                  xp[i] = in[i+i1*nn] + in[nn-1-i+i1*nn];
                  xm[i] = in[i+i1*nn] - in[nn-1-i+i1*nn];
                }
              for (unsigned int col=0; col<mid; ++col)
                {
                  VectorizedArray<Number> r0, r1;
                  r0 = shape_vals[col*offset]          * xp[0];
                  r1 = shape_vals[(degree-col)*offset] * xm[0];
                  for (unsigned int ind=1; ind<mid; ++ind)
                    {
                      r0 += shape_vals[col*offset+ind]          * xp[ind];
                      r1 += shape_vals[(degree-col)*offset+ind] * xm[ind];
                    }

                  in[col+i1*nn]        = r0 + r1;
                  in[(nn-1-col)+i1*nn] = r0 - r1;
                }
              if (nn%2==1)
                {
                  // sum into because shape value is one in the middle
                  VectorizedArray<Number> r0 = in[i1*nn+mid];
                  for (unsigned int ind=0; ind<mid; ++ind)
                    r0 += shape_vals[ind+mid*offset] * xp[ind];
                  in[i1*nn+mid] = r0;
                }
            }
        }

      // write into the solution vector, overwrite rather than sum-into
      if (vectorization_populated == VectorizedArray<Number>::n_array_elements)
        // transform struct-of-array to array-of-struct
        vectorized_transpose_and_store(false, dofs_per_cell, data_ptr,
                                       dof_indices, dst.begin());
      else
        {
          for (unsigned int v=0; v<vectorization_populated; ++v)
            for (unsigned int i=0; i<dofs_per_cell; ++i)
              dst.local_element(dof_indices[v]+i) = data_ptr[i][v];
        }
    }
  data->release_scratch_data(scratch_data_array);

#ifdef LIKWID_PERFMON
  LIKWID_MARKER_STOP(("cell_loop_3_p" + std::to_string(degree)).c_str());
#endif
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

  // Constructor.
  LaplaceProblem(unsigned int const refine_steps_space)
    :
  pcout (std::cout, Utilities::MPI::this_mpi_process(MPI_COMM_WORLD)==0),
  triangulation(MPI_COMM_WORLD),
  fe(fe_degree),
  mapping(3),
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

  // Setup of grid, dofs, matrix-free object and Laplace operator.
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

  // This function applies the matrix-vector product several times
  // and computes the required average wall time per degree of freedom.
  void apply_laplace_operator() const
  {
    pcout << std::endl << "Computing matrix-vector product ..." << std::endl;

    // initialize vectors
    parallel::distributed::Vector<Number> dst, src;
    matrix_free_data.initialize_dof_vector(src);
    matrix_free_data.initialize_dof_vector(dst);
    for (unsigned int i=0; i<src.local_size(); ++i)
      src.local_element(i) = i%17;

    // Timer and wall times
    Timer timer;

    double wall_time = 0.0;

    if(wall_time_calculation == WallTimeCalculation::Minimum)
      wall_time = std::numeric_limits<double>::max();
    MPI_Barrier(MPI_COMM_WORLD);

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

    types::global_dof_index dofs = dof_handler.n_dofs();

    wall_time /= (double) dofs;

    pcout << std::endl << std::scientific << std::setprecision(4)
          << "Wall time / dof [s]: " << wall_time << std::endl;

    std::array<double,4> times;
    times[0] = wall_time;

    if (COMPUTE_FACE_INTEGRALS == false)
    {
      parallel::distributed::Vector<Number> dst2;
      matrix_free_data.initialize_dof_vector(dst2);
      wall_time = 0.0;

      if(wall_time_calculation == WallTimeCalculation::Minimum)
        wall_time = std::numeric_limits<double>::max();
      MPI_Barrier(MPI_COMM_WORLD);

      // apply matrix-vector product several times
      for(unsigned int i=0; i<n_repetitions; ++i)
      {
        timer.restart();

        laplace_operator.cell_loop_manual_1(dst2,src);

        double const current_wall_time = timer.wall_time();

        if(wall_time_calculation == WallTimeCalculation::Average)
          wall_time += current_wall_time;
        else if(wall_time_calculation == WallTimeCalculation::Minimum)
          wall_time = std::min(wall_time,current_wall_time);
      }

      // compute wall times
      if(wall_time_calculation == WallTimeCalculation::Average)
        wall_time /= (double)n_repetitions;
      wall_time /= (double) dofs;

      dst2 -= dst;
      pcout << std::scientific << std::setprecision(4)
            << "Wall time / dof [s]: " << wall_time << ", error to nice code: "
            << dst2.linfty_norm() << std::endl;
      times[1] = wall_time;




      if(wall_time_calculation == WallTimeCalculation::Minimum)
        wall_time = std::numeric_limits<double>::max();
      MPI_Barrier(MPI_COMM_WORLD);

      // apply matrix-vector product several times
      for(unsigned int i=0; i<n_repetitions; ++i)
      {
        timer.restart();

        laplace_operator.cell_loop_manual_2(dst2,src);

        double const current_wall_time = timer.wall_time();

        if(wall_time_calculation == WallTimeCalculation::Average)
          wall_time += current_wall_time;
        else if(wall_time_calculation == WallTimeCalculation::Minimum)
          wall_time = std::min(wall_time,current_wall_time);
      }


      // compute wall times
      if(wall_time_calculation == WallTimeCalculation::Average)
        wall_time /= (double)n_repetitions;
      wall_time /= (double) dofs;

      dst2 -= dst;
      pcout << std::scientific << std::setprecision(4)
            << "Wall time / dof [s]: " << wall_time << ", error to nice code: "
            << dst2.linfty_norm() << std::endl;
      times[2] = wall_time;

      if(wall_time_calculation == WallTimeCalculation::Minimum)
        wall_time = std::numeric_limits<double>::max();
      MPI_Barrier(MPI_COMM_WORLD);

      // apply matrix-vector product several times
      for(unsigned int i=0; i<n_repetitions; ++i)
      {
        timer.restart();

        laplace_operator.cell_loop_manual_3(dst2,src);

        double const current_wall_time = timer.wall_time();

        if(wall_time_calculation == WallTimeCalculation::Average)
          wall_time += current_wall_time;
        else if(wall_time_calculation == WallTimeCalculation::Minimum)
          wall_time = std::min(wall_time,current_wall_time);
      }


      // compute wall times
      if(wall_time_calculation == WallTimeCalculation::Average)
        wall_time /= (double)n_repetitions;
      wall_time /= (double) dofs;

      dst2 -= dst;
      pcout << std::scientific << std::setprecision(4)
            << "Wall time / dof [s]: " << wall_time << ", error to nice code: "
            << dst2.linfty_norm() << std::endl;
      times[3] = wall_time;
    }

    wall_times.emplace_back(fe_degree,times);

    pcout << std::endl << " ... done." << std::endl << std::endl;
  }

private:
  // Setup dofs
  void setup_dofs()
  {
    dof_handler.distribute_dofs(fe);

    pcout << std::endl
          << "Discontinuous Galerkin finite element discretization:" << std::endl;

    const unsigned int dofs_per_cell = Utilities::fixed_int_power<fe_degree+1,dim>::value;
    pcout << std::endl << std::fixed
          << "degree of 1D polynomials: " << fe_degree << std::endl
          << "number of dofs per cell:  " << dofs_per_cell << std::endl
          << "number of dofs (total):   " << dof_handler.n_dofs() << std::endl;
  }

  // Setup matrix free data
  void setup_matrix_free()
  {
    pcout << std::endl << "Setup matrix-free object ..." << std::flush;

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

  // Setup of Laplace operator
  void setup_laplace_operator()
  {
    pcout << std::endl << "Setup Laplace operator ..." << std::flush;

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
  unsigned int const n_refinements;

  // number of matrix-vector products
  unsigned int const n_repetitions;

  // wall time calculation
  WallTimeCalculation const wall_time_calculation;
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
    const unsigned int next_degree =
      RUN_EQUAL_SIZE ? ((2*(degree+1)-1)>max_degree?max_degree:(2*(degree+1)-1)) :
      degree+1;
    LaplaceRunTime<dim,next_degree,max_degree,value_type>::run(refine_steps_space-
                                                               (RUN_EQUAL_SIZE ? 1 : 0));
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

void print_wall_times(std::vector<std::pair<unsigned int, std::array<double,4> > > const &wall_times,
                      unsigned int const                                  refine_steps_space)
{
  if(Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
  {
    std::cout << std::endl
              << "_________________________________________________________________________________"
              << std::endl << std::endl
              << "Wall times for refine level l = " << refine_steps_space << ":"
              << std::endl << std::endl
              << "       " << "wall time / dof [s]" << std::endl
              << "  k    " << "standard    manual_1    manual_2    manual_3" << std::endl;

    typedef typename std::vector<std::pair<unsigned int, std::array<double,4> > >::const_iterator ITERATOR;
    for(ITERATOR it = wall_times.begin(); it != wall_times.end(); ++it)
    {
      std::cout << "  " << std::setw(5) << std::left << it->first;
      for (unsigned int i=0; i<4; ++i)
        std::cout << std::setw(12) << std::left << std::scientific << std::setprecision(4) << it->second[i];
      std::cout << std::endl;
    }

    std::cout << "_________________________________________________________________________________"
              << std::endl << std::endl;
  }
}

int main (int argc, char** argv)
{
#ifdef LIKWID_PERFMON
LIKWID_MARKER_INIT;
#pragma omp parallel
{
  LIKWID_MARKER_THREADINIT;
}
// On the first MARKER_START likwid forks all threads. They inherit the parents
// memory usage, which is the full allocate amount for the domain. This can easily
// become too much and likwid crashes. As a workaround the forks are called here,
// where hardly any memory is reserved.
#pragma omp parallel
{
  LIKWID_MARKER_START("zzzdummy");
  LIKWID_MARKER_STOP("zzzdummy");
}
#endif

  try
  {
    Utilities::MPI::MPI_InitFinalize mpi(argc, argv, 1);

    if (Utilities::MPI::this_mpi_process(MPI_COMM_WORLD) == 0)
    {
      std::cout << "deal.II git version " << DEAL_II_GIT_SHORTREV << " on branch "
                << DEAL_II_GIT_BRANCH << std::endl << std::endl;
    }

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

#ifdef LIKWID_PERFMON
LIKWID_MARKER_CLOSE;
#endif

  return 0;
}
