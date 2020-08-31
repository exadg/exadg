#ifndef INCLUDE_MATRIX_FREE_INTEGRATORS_H_
#define INCLUDE_MATRIX_FREE_INTEGRATORS_H_

#include <deal.II/base/config.h>

#include <deal.II/matrix_free/fe_evaluation.h>
#include <deal.II/matrix_free/operators.h>

#include "evaluation_template_factory.h"

DEAL_II_NAMESPACE_OPEN

/**
 * This class provides an equivalent interface to FEEvaluation, but without the
 * template parameters on the polynomial degree `fe_degree` and the number of
 * 1D quadrature points `n_q_points_1d`.
 */
template<int dim,
         int n_components_,
         typename Number,
         typename VectorizedArrayType = VectorizedArray<Number>>
class CellIntegrator
  : public FEEvaluationAccess<dim, n_components_, Number, false, VectorizedArrayType>
{
public:
  /**
   * An alias to the base class.
   */
  using BaseClass = FEEvaluationAccess<dim, n_components_, Number, false, VectorizedArrayType>;

  /**
   * An underlying number type specified as template argument.
   */
  using number_type = Number;

  /**
   * The type of function values, e.g. `VectorizedArrayType` for
   * `n_components=1` or `Tensor<1,dim,VectorizedArrayType >` for
   * `n_components=dim`.
   */
  using value_type = typename BaseClass::value_type;

  /**
   * The type of gradients, e.g. `Tensor<1,dim,VectorizedArrayType>` for
   * `n_components=1` or `Tensor<2,dim,VectorizedArrayType >` for
   * `n_components=dim`.
   */
  using gradient_type = typename BaseClass::gradient_type;

  /**
   * The dimension given as template argument.
   */
  static constexpr unsigned int dimension = dim;

  /**
   * The number of solution components of the evaluator given as template
   * argument.
   */
  static constexpr unsigned int n_components = n_components_;

  /**
   * Constructor. Takes all data stored in MatrixFree. If applied to problems
   * with more than one finite element or more than one quadrature formula
   * selected during construction of @p matrix_free, the appropriate component
   * can be selected by the optional arguments.
   *
   * @param matrix_free Data object that contains all data
   *
   * @param dof_no If matrix_free was set up with multiple DoFHandler
   * objects, this parameter selects to which DoFHandler/AffineConstraints pair
   * the given evaluator should be attached to.
   *
   * @param quad_no If matrix_free was set up with multiple Quadrature
   * objects, this parameter selects the appropriate number of the quadrature
   * formula.
   *
   * @param first_selected_component If the dof_handler selected by dof_no
   * uses an FESystem consisting of more than one component, this parameter
   * allows for selecting the component where the current evaluation routine
   * should start. Note that one evaluator does not support combining
   * different shape functions in different components. In other words, the
   * same base element of a FESystem needs to be set for the components
   * between @p first_selected_component and
   * <code>first_selected_component+n_components_</code>.
   */
  CellIntegrator(const MatrixFree<dim, Number, VectorizedArrayType> & matrix_free,
                 const unsigned int                                   dof_no                   = 0,
                 const unsigned int                                   quad_no                  = 0,
                 const unsigned int                                   first_selected_component = 0);

  /**
   * Constructor that comes with reduced functionality and works similar as
   * FEValues. The arguments are similar to the ones passed to the constructor
   * of FEValues, with the notable difference that FEEvaluation expects a one-
   * dimensional quadrature formula, Quadrature<1>, instead of a @p dim
   * dimensional one. The finite element can be both scalar or vector valued,
   * but this method always only selects a scalar base element at a time (with
   * @p n_components copies as specified by the class template). For vector-
   * valued elements, the optional argument @p first_selected_component allows
   * to specify the index of the base element to be used for evaluation. Note
   * that the internal data structures always assume that the base element is
   * primitive, non-primitive are not supported currently.
   *
   * As known from FEValues, a call to the reinit method with a
   * Triangulation<dim>::cell_iterator is necessary to make the geometry and
   * degrees of freedom of the current class known. If the iterator includes
   * DoFHandler information (i.e., it is a DoFHandler<dim>::cell_iterator or
   * similar), the initialization allows to also read from or write to vectors
   * in the standard way for DoFHandler<dim>::active_cell_iterator types for
   * one cell at a time. However, this approach is much slower than the path
   * with MatrixFree with MPI since index translation has to be done. As only
   * one cell at a time is used, this method does not vectorize over several
   * elements (which is most efficient for vector operations), but only
   * possibly within the element if the evaluate/integrate routines are
   * combined inside user code (e.g. for computing cell matrices).
   */
  CellIntegrator(const Mapping<dim> &       mapping,
                 const FiniteElement<dim> & fe,
                 const Quadrature<1> &      quadrature,
                 const UpdateFlags          update_flags,
                 const unsigned int         first_selected_component = 0);

  /**
   * Constructor for the reduced functionality. This constructor is equivalent
   * to the other one except that it makes the object use a $Q_1$ mapping
   * (i.e., an object of type MappingQGeneric(1)) implicitly.
   */
  CellIntegrator(const FiniteElement<dim> & fe,
                 const Quadrature<1> &      quadrature,
                 const UpdateFlags          update_flags,
                 const unsigned int         first_selected_component = 0);

  /**
   * Constructor for the reduced functionality. Similar to the other
   * constructor with FiniteElement argument but using another
   * FEEvaluationBase object to provide information about the geometry. This
   * allows several FEEvaluation objects to share the geometry evaluation, i.e.,
   * the underlying mapping and quadrature points do only need to be evaluated
   * once. Make sure to not pass an optional object around when you intend to
   * use the FEEvaluation object in %parallel to the given one because
   * otherwise the intended sharing may create race conditions.
   */
  template<int n_components_other>
  CellIntegrator(
    const FiniteElement<dim> &                                                            fe,
    const FEEvaluationBase<dim, n_components_other, Number, false, VectorizedArrayType> & other,
    const unsigned int first_selected_component = 0);

  /**
   * Copy constructor. If FEEvaluationBase was constructed from a mapping, fe,
   * quadrature, and update flags, the underlying geometry evaluation based on
   * FEValues will be deep-copied in order to allow for using in parallel with
   * threads.
   */
  CellIntegrator(const CellIntegrator & other);

  /**
   * Copy assignment operator. If FEEvaluationBase was constructed from a
   * mapping, fe, quadrature, and update flags, the underlying geometry
   * evaluation based on FEValues will be deep-copied in order to allow for
   * using in parallel with threads.
   */
  CellIntegrator &
  operator=(const CellIntegrator & other);

  /**
   * Initialize the operation pointer to the current cell batch index. Unlike
   * the reinit functions taking a cell iterator as argument below and the
   * FEValues::reinit() methods, where the information related to a particular
   * cell is generated in the reinit call, this function is very cheap since
   * all data is pre-computed in @p matrix_free, and only a few indices have
   * to be set appropriately.
   */
  void
  reinit(const unsigned int cell_batch_index);

  /**
   * Initialize the data to the current cell using a TriaIterator object as
   * usual in FEValues. The argument is either of type
   * DoFHandler::active_cell_iterator or DoFHandler::level_cell_iterator. This
   * option is only available if the FEEvaluation object was created with a
   * finite element, quadrature formula and correct update flags and
   * <b>without</b> a MatrixFree object. This initialization method loses the
   * ability to use vectorization, see also the description of the
   * FEEvaluation class. When this reinit method is used, FEEvaluation can
   * also read from vectors (but less efficient than with data coming from
   * MatrixFree).
   */
  template<bool level_dof_access>
  void
  reinit(const TriaIterator<DoFCellAccessor<dim, dim, level_dof_access>> & cell);

  /**
   * Initialize the data to the current cell using a TriaIterator object as
   * usual in FEValues. This option is only available if the FEEvaluation
   * object was created with a finite element, quadrature formula and correct
   * update flags and <b>without</b> a MatrixFree object. This initialization
   * method loses the ability to use vectorization, see also the description
   * of the FEEvaluation class. When this reinit method is used, FEEvaluation
   * can <b>not</b> read from vectors because no DoFHandler information is
   * available.
   */
  void
  reinit(const typename Triangulation<dim>::cell_iterator & cell);

  /**
   * Return the cell index set upon reinit;
   */
  unsigned int
  get_cell_index() const;

  /**
   * Evaluate the function values, the gradients, and the Hessians of the
   * polynomial interpolation from the DoF values in the input vector to the
   * quadrature points on the unit cell.  The function arguments specify which
   * parts shall actually be computed. This function has to be called first so
   * that the access functions @p get_value(), @p get_gradient() or @p
   * get_laplacian give useful information (unless these values have been set
   * manually).
   */
  void
  evaluate(const bool evaluate_values,
           const bool evaluate_gradients,
           const bool evaluate_hessians = false);

  /**
   * Evaluate the function values, the gradients, and the Hessians of the
   * polynomial interpolation from the DoF values in the input array @p
   * values_array to the quadrature points on the unit cell. If multiple
   * components are involved in the current FEEvaluation object, the sorting
   * in @p values_array is such that all degrees of freedom for the first
   * component come first, then all degrees of freedom for the second, and so
   * on. The function arguments specify which parts shall actually be
   * computed. This function has to be called first so that the access
   * functions @p get_value(), @p get_gradient() or @p get_laplacian give
   * useful information (unless these values have been set manually).
   */
  void
  evaluate(const VectorizedArrayType * values_array,
           const bool                  evaluate_values,
           const bool                  evaluate_gradients,
           const bool                  evaluate_hessians = false);

  /**
   * Read from the input vector and evaluates the function values, the
   * gradients, and the Hessians of the polynomial interpolation of the vector
   * entries from @p input_vector associated with the current cell to the
   * quadrature points on the unit cell. The function arguments specify which
   * parts shall actually be computed. This function has to be called first so
   * that the access functions @p get_value(), @p get_gradient() or @p
   * get_laplacian give useful information (unless these values have been set
   * manually).
   *
   * This call is equivalent to calling read_dof_values() followed by
   * evaluate(), but might internally use some additional optimizations.
   */
  template<typename VectorType>
  void
  gather_evaluate(const VectorType & input_vector,
                  const bool         evaluate_values,
                  const bool         evaluate_gradients,
                  const bool         evaluate_hessians = false);

  /**
   * This function takes the values and/or gradients that are stored on
   * quadrature points, tests them by all the basis functions/gradients on the
   * cell and performs the cell integration. The two function arguments
   * @p integrate_values and @p integrate_gradients are used to enable/disable
   * summation of the contributions submitted to the values or gradients slots,
   * respectively. The result is written into the internal data field
   * @p dof_values (that is usually written into the result vector by the
   * distribute_local_to_global() or set_dof_values() methods).
   */
  void
  integrate(const bool integrate_values, const bool integrate_gradients);

  /**
   * This function takes the values and/or gradients that are stored on
   * quadrature points, tests them by all the basis functions/gradients on the
   * cell and performs the cell integration. The two function arguments @p
   * integrate_values and @p integrate_gradients are used to enable/disable
   * summation of the contributions submitted to the values or gradients
   * slots, respectively. As opposed to the other integrate() method, this
   * call stores the result of the testing in the given array @p values_array,
   * whose previous results is overwritten, rather than writing it on the
   * internal data structures behind begin_dof_values().
   */
  void
  integrate(const bool            integrate_values,
            const bool            integrate_gradients,
            VectorizedArrayType * values_array);

  /**
   * This function takes the values and/or gradients that are stored on
   * quadrature points, tests them by all the basis functions/gradients on the
   * cell, performs the cell integration, and adds the result into the global
   * vector @p output_vector on the degrees of freedom associated with the
   * present cell index. The two function arguments @p integrate_values and
   * @p integrate_gradients are used to enable/disable summation of the
   * contributions submitted to the values or gradients slots, respectively.
   *
   * This call is equivalent to calling integrate() followed by
   * distribute_local_to_global(), but might internally use
   * some additional optimizations.
   */
  template<typename VectorType>
  void
  integrate_scatter(const bool   integrate_values,
                    const bool   integrate_gradients,
                    VectorType & output_vector);

  /**
   * Return the q-th quadrature point in real coordinates stored in
   * MappingInfo.
   */
  Point<dim, VectorizedArrayType>
  quadrature_point(const unsigned int q_point) const;

  /**
   * The number of degrees of freedom of a single component on the cell for
   * the underlying evaluation object. Usually close to
   * static_dofs_per_component, but the number depends on the actual element
   * selected and is thus not static.
   */
  const unsigned int dofs_per_component;

  /**
   * The number of degrees of freedom on the cell accumulated over all
   * components in the current evaluation object. Usually close to
   * static_dofs_per_cell = static_dofs_per_component*n_components, but the
   * number depends on the actual element selected and is thus not static.
   */
  const unsigned int dofs_per_cell;

  /**
   * The number of quadrature points in use. If the number of quadrature
   * points in 1d is given as a template, this number is simply the
   * <tt>dim</tt>-th power of that value. If the element degree is set to -1
   * (dynamic selection of element degree), the static value of quadrature
   * points is inaccurate and this value must be used instead.
   */
  const unsigned int n_q_points;
};



/**
 * This class provides an equivalent interface to FEFaceEvaluation, but without
 * the template parameters on the polynomial degree `fe_degree` and the number
 * of 1D quadrature points `n_q_points_1d`.
 */
template<int dim,
         int n_components_            = 1,
         typename Number              = double,
         typename VectorizedArrayType = VectorizedArray<Number>>
class FaceIntegrator
  : public FEEvaluationAccess<dim, n_components_, Number, true, VectorizedArrayType>
{
public:
  /**
   * An alias to the base class.
   */
  using BaseClass = FEEvaluationAccess<dim, n_components_, Number, true, VectorizedArrayType>;

  /**
   * A underlying number type specified as template argument.
   */
  using number_type = Number;

  /**
   * The type of function values, e.g. `VectorizedArrayType` for
   * `n_components=1` or `Tensor<1,dim,VectorizedArrayType >` for
   * `n_components=dim`.
   */
  using value_type = typename BaseClass::value_type;

  /**
   * The type of gradients, e.g. `Tensor<1,dim,VectorizedArrayType>` for
   * `n_components=1` or `Tensor<2,dim,VectorizedArrayType >` for
   * `n_components=dim`.
   */
  using gradient_type = typename BaseClass::gradient_type;

  /**
   * The dimension given as template argument.
   */
  static constexpr unsigned int dimension = dim;

  /**
   * The number of solution components of the evaluator given as template
   * argument.
   */
  static constexpr unsigned int n_components = n_components_;

  /**
   * Constructor. Takes all data stored in MatrixFree. If applied to problems
   * with more than one finite element or more than one quadrature formula
   * selected during construction of @p matrix_free, the appropriate component
   * can be selected by the optional arguments.
   *
   * @param matrix_free Data object that contains all data
   *
   * @param is_interior_face This selects which of the two cells of an
   * internal face the current evaluator will be based upon. The interior face
   * is the main face along which the normal vectors are oriented. The
   * exterior face coming from the other side provides the same normal vector
   * as the interior side, so if the outer normal vector to that side is
   * desired, it must be multiplied by -1.
   *
   * @param dof_no If matrix_free was set up with multiple DoFHandler
   * objects, this parameter selects to which DoFHandler/AffineConstraints pair
   * the given evaluator should be attached to.
   *
   * @param quad_no If matrix_free was set up with multiple Quadrature
   * objects, this parameter selects the appropriate number of the quadrature
   * formula.
   *
   * @param first_selected_component If the dof_handler selected by dof_no
   * uses an FESystem consisting of more than one base element, this parameter
   * selects the number of the base element in FESystem. Note that this does
   * not directly relate to the component of the respective element due to the
   * possibility for a multiplicity in the element.
   */
  FaceIntegrator(const MatrixFree<dim, Number, VectorizedArrayType> & matrix_free,
                 const bool                                           is_interior_face = true,
                 const unsigned int                                   dof_no           = 0,
                 const unsigned int                                   quad_no          = 0,
                 const unsigned int                                   first_selected_component = 0);

  /**
   * Initializes the operation pointer to the current face. This method is the
   * default choice for face integration as the data stored in MappingInfo is
   * stored according to this numbering. Unlike the reinit functions taking a
   * cell iterator as argument below and the FEValues::reinit() methods, where
   * the information related to a particular cell is generated in the reinit
   * call, this function is very cheap since all data is pre-computed in
   * @p matrix_free, and only a few indices and pointers have to be set
   * appropriately.
   */
  void
  reinit(const unsigned int face_batch_number);

  /**
   * As opposed to the reinit() method from the base class, this reinit()
   * method initializes for a given number of cells and a face number. This
   * method is less efficient than the other reinit() method taking a
   * numbering of the faces because it needs to copy the data associated with
   * the faces to the cells in this call.
   */
  void
  reinit(const unsigned int cell_batch_number, const unsigned int face_number);

  /**
   * Return the effective face index set upon reinit; for the face-based variant
   * with a single index, this simply returns this index, for the cell-based
   * variant the given number is cell_batch_number * faces_per_cell +
   * face_number.
   */
  unsigned int
  get_face_index() const;

  /**
   * Return the quadrature formula index FaceIntegrator was
   * constructed with by the third/fourth argument passed to the constructor.
   */
  unsigned int
  quadrature_formula_index() const;

  /**
   * Return the finite element index within a DoFHandler the current
   * FaceIntegrator object was constructed with by the
   * fourth / fifth argument passed to the constructor.
   */
  unsigned int
  finite_element_index() const;

  /**
   * Same as FEEvaluationBase::set_dof_values, but without resolving
   * the constraints.
   */
  template<typename VectorType>
  void
  set_dof_values_plain(const VectorType & dst, const unsigned int first_index = 0);

  /**
   * Evaluates the function values, the gradients, and the Laplacians of the
   * FE function given at the DoF values stored in the internal data field
   * `dof_values` (that is usually filled by the read_dof_values() method) at
   * the quadrature points on the unit cell.  The function arguments specify
   * which parts shall actually be computed. Needs to be called before the
   * functions get_value(), get_gradient() or get_normal_derivative() give
   * useful information (unless these values have been set manually by
   * accessing the internal data pointers).
   */
  void
  evaluate(const bool evaluate_values, const bool evaluate_gradients);

  /**
   * Evaluates the function values, the gradients, and the Laplacians of the
   * FE function given at the DoF values in the input array `values_array` at
   * the quadrature points on the unit cell. If multiple components are
   * involved in the current FEEvaluation object, the sorting in values_array
   * is such that all degrees of freedom for the first component come first,
   * then all degrees of freedom for the second, and so on. The function
   * arguments specify which parts shall actually be computed. Needs to be
   * called before the functions get_value(), get_gradient(), or
   * get_normal_derivative() give useful information (unless these values have
   * been set manually).
   */
  void
  evaluate(const VectorizedArrayType * values_array,
           const bool                  evaluate_values,
           const bool                  evaluate_gradients);

  /**
   * Reads from the input vector and evaluates the function values, the
   * gradients, and the Laplacians of the FE function at the quadrature points
   * on the unit cell. The function arguments specify which parts shall
   * actually be computed. Needs to be called before the functions
   * get_value(), get_gradient(), or get_normal_derivative() give useful
   * information.
   *
   * This call is equivalent to calling read_dof_values() followed by
   * evaluate(), but might internally use some additional optimizations.
   */
  template<typename VectorType>
  void
  gather_evaluate(const VectorType & input_vector,
                  const bool         evaluate_values,
                  const bool         evaluate_gradients);

  /**
   * This function takes the values and/or gradients that are stored on
   * quadrature points, tests them by all the basis functions/gradients on the
   * cell and performs the cell integration. The two function arguments
   * `integrate_val` and `integrate_grad` are used to enable/disable some of
   * values or gradients. The result is written into the internal data field
   * `dof_values` (that is usually written into the result vector by the
   * distribute_local_to_global() or set_dof_values() methods).
   */
  void
  integrate(const bool integrate_values, const bool integrate_gradients);

  /**
   * This function takes the values and/or gradients that are stored on
   * quadrature points, tests them by all the basis functions/gradients on the
   * cell and performs the cell integration. The two function arguments
   * `integrate_val` and `integrate_grad` are used to enable/disable some of
   * values or gradients. As opposed to the other integrate() method, this
   * call stores the result of the testing in the given array `values_array`.
   */
  void
  integrate(const bool            integrate_values,
            const bool            integrate_gradients,
            VectorizedArrayType * values_array);

  /**
   * This function takes the values and/or gradients that are stored on
   * quadrature points, tests them by all the basis functions/gradients on the
   * cell and performs the cell integration. The two function arguments
   * `integrate_val` and `integrate_grad` are used to enable/disable some of
   * values or gradients.
   *
   * This call is equivalent to calling integrate() followed by
   * distribute_local_to_global(), but might internally use some additional
   * optimizations.
   */
  template<typename VectorType>
  void
  integrate_scatter(const bool   integrate_values,
                    const bool   integrate_gradients,
                    VectorType & output_vector);

  /**
   * Returns the q-th quadrature point on the face in real coordinates stored
   * in MappingInfo.
   */
  Point<dim, VectorizedArrayType>
  quadrature_point(const unsigned int q_point) const;

  /**
   * The number of degrees of freedom of a single component on the cell for
   * the underlying evaluation object. Usually close to
   * static_dofs_per_component, but the number depends on the actual element
   * selected and is thus not static.
   */
  const unsigned int dofs_per_component;

  /**
   * The number of degrees of freedom on the cell accumulated over all
   * components in the current evaluation object. Usually close to
   * static_dofs_per_cell = static_dofs_per_component*n_components, but the
   * number depends on the actual element selected and is thus not static.
   */
  const unsigned int dofs_per_cell;

  /**
   * The number of quadrature points in use. If the number of quadrature
   * points in 1d is given as a template, this number is simply the
   * <tt>dim-1</tt>-th power of that value. If the element degree is set to -1
   * (dynamic selection of element degree), the static value of quadrature
   * points is inaccurate and this value must be used instead.
   */
  const unsigned int n_q_points;
};



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::CellIntegrator(
  const MatrixFree<dim, Number, VectorizedArrayType> & data_in,
  const unsigned int                                   fe_no,
  const unsigned int                                   quad_no,
  const unsigned int                                   first_selected_component)
  : BaseClass(data_in, fe_no, first_selected_component, quad_no, numbers::invalid_unsigned_int, 0),
    dofs_per_component(this->data->dofs_per_component_on_cell),
    dofs_per_cell(this->data->dofs_per_component_on_cell * n_components_),
    n_q_points(this->data->n_q_points)
{
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::CellIntegrator(
  const Mapping<dim> &       mapping,
  const FiniteElement<dim> & fe,
  const Quadrature<1> &      quadrature,
  const UpdateFlags          update_flags,
  const unsigned int         first_selected_component)
  : BaseClass(mapping,
              fe,
              quadrature,
              update_flags,
              first_selected_component,
              static_cast<FEEvaluationBase<dim, 1, Number, false, VectorizedArrayType> *>(nullptr)),
    dofs_per_component(this->data->dofs_per_component_on_cell),
    dofs_per_cell(this->data->dofs_per_component_on_cell * n_components_),
    n_q_points(this->data->n_q_points)
{
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::CellIntegrator(
  const FiniteElement<dim> & fe,
  const Quadrature<1> &      quadrature,
  const UpdateFlags          update_flags,
  const unsigned int         first_selected_component)
  : BaseClass(StaticMappingQ1<dim>::mapping,
              fe,
              quadrature,
              update_flags,
              first_selected_component,
              static_cast<FEEvaluationBase<dim, 1, Number, false, VectorizedArrayType> *>(nullptr)),
    dofs_per_component(this->data->dofs_per_component_on_cell),
    dofs_per_cell(this->data->dofs_per_component_on_cell * n_components_),
    n_q_points(this->data->n_q_points)
{
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
template<int n_components_other>
inline CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::CellIntegrator(
  const FiniteElement<dim> &                                                            fe,
  const FEEvaluationBase<dim, n_components_other, Number, false, VectorizedArrayType> & other,
  const unsigned int first_selected_component)
  : BaseClass(other.mapped_geometry->get_fe_values().get_mapping(),
              fe,
              other.mapped_geometry->get_quadrature(),
              other.mapped_geometry->get_fe_values().get_update_flags(),
              first_selected_component,
              &other),
    dofs_per_component(this->data->dofs_per_component_on_cell),
    dofs_per_cell(this->data->dofs_per_component_on_cell * n_components_),
    n_q_points(this->data->n_q_points)
{
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::CellIntegrator(
  const CellIntegrator & other)
  : BaseClass(other),
    dofs_per_component(this->data->dofs_per_component_on_cell),
    dofs_per_cell(this->data->dofs_per_component_on_cell * n_components_),
    n_q_points(this->data->n_q_points)
{
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline CellIntegrator<dim, n_components_, Number, VectorizedArrayType> &
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::
operator=(const CellIntegrator & other)
{
  BaseClass::operator=(other);
  return *this;
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline void
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::reinit(
  const unsigned int cell_index)
{
  Assert(this->mapped_geometry == nullptr,
         ExcMessage("CellIntegrator was initialized without a matrix-free object."
                    " Integer indexing is not possible"));
  if(this->mapped_geometry != nullptr)
    return;

  Assert(this->dof_info != nullptr, ExcNotInitialized());
  Assert(this->mapping_data != nullptr, ExcNotInitialized());
  this->cell      = cell_index;
  this->cell_type = this->matrix_info->get_mapping_info().get_cell_type(cell_index);

  const unsigned int offsets = this->mapping_data->data_index_offsets[cell_index];
  this->jacobian             = &this->mapping_data->jacobians[0][offsets];
  this->J_value              = &this->mapping_data->JxW_values[offsets];

#ifdef DEBUG
  this->dof_values_initialized     = false;
  this->values_quad_initialized    = false;
  this->gradients_quad_initialized = false;
  this->hessians_quad_initialized  = false;
#endif
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
template<bool level_dof_access>
inline void
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::reinit(
  const TriaIterator<DoFCellAccessor<dim, dim, level_dof_access>> & cell)
{
  Assert(this->matrix_info == nullptr,
         ExcMessage("Cannot use initialization from cell iterator if "
                    "initialized from MatrixFree object. Use variant for "
                    "on the fly computation with arguments as for FEValues "
                    "instead"));
  Assert(this->mapped_geometry.get() != nullptr, ExcNotInitialized());
  this->mapped_geometry->reinit(static_cast<typename Triangulation<dim>::cell_iterator>(cell));
  this->local_dof_indices.resize(cell->get_fe().n_dofs_per_cell());
  if(level_dof_access)
    cell->get_mg_dof_indices(this->local_dof_indices);
  else
    cell->get_dof_indices(this->local_dof_indices);
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline void
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::reinit(
  const typename Triangulation<dim>::cell_iterator & cell)
{
  Assert(this->matrix_info == 0,
         ExcMessage("Cannot use initialization from cell iterator if "
                    "initialized from MatrixFree object. Use variant for "
                    "on the fly computation with arguments as for FEValues "
                    "instead"));
  Assert(this->mapped_geometry.get() != 0, ExcNotInitialized());
  this->mapped_geometry->reinit(cell);
}


template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline unsigned int
CellIntegrator<dim, n_components, Number, VectorizedArrayType>::get_cell_index() const
{
  return this->cell;
}


template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline Point<dim, VectorizedArrayType>
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::quadrature_point(
  const unsigned int q) const
{
  if(this->matrix_info == nullptr)
  {
    Assert((this->mapped_geometry->get_fe_values().get_update_flags() | update_quadrature_points),
           ExcNotInitialized());
  }
  else
  {
    Assert(this->mapping_data->quadrature_point_offsets.empty() == false, ExcNotInitialized());
  }

  AssertIndexRange(q, n_q_points);

  const Point<dim, VectorizedArrayType> * quadrature_points =
    &this->mapping_data
       ->quadrature_points[this->mapping_data->quadrature_point_offsets[this->cell]];

  // Cartesian/affine mesh: only first quadrature point is stored, we must
  // compute it through the Jacobian
  if(this->cell_type < internal::MatrixFreeFunctions::general)
  {
    Assert(this->jacobian != nullptr, ExcNotInitialized());
    Point<dim, VectorizedArrayType>           point = quadrature_points[0];
    const Tensor<2, dim, VectorizedArrayType> jac   = invert(this->jacobian[0]);
    for(unsigned int d = 0; d < dim; ++d)
      for(unsigned int e = 0; e < dim; ++e)
        point[d] += jac[e][d] *
                    this->mapping_data->descriptor[this->active_quad_index].quadrature.point(q)[e];
    return point;
  }
  else
    return quadrature_points[q];
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline void
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::evaluate(
  const bool evaluate_values,
  const bool evaluate_gradients,
  const bool evaluate_hessians)
{
  Assert(this->dof_values_initialized == true, internal::ExcAccessToUninitializedField());
  evaluate(this->values_dofs[0], evaluate_values, evaluate_gradients, evaluate_hessians);
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline void
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::evaluate(
  const VectorizedArrayType * values_array,
  const bool                  evaluate_values,
  const bool                  evaluate_gradients,
  const bool                  evaluate_hessians)
{
  const EvaluationFlags::EvaluationFlags flag =
    ((evaluate_values) ? EvaluationFlags::values : EvaluationFlags::nothing) |
    ((evaluate_gradients) ? EvaluationFlags::gradients : EvaluationFlags::nothing) |
    ((evaluate_hessians) ? EvaluationFlags::hessians : EvaluationFlags::nothing);

  internal::FEEvaluationFactory<dim, Number, VectorizedArrayType>::evaluate(
    n_components,
    flag,
    *this->data,
    const_cast<VectorizedArrayType *>(values_array),
    this->values_quad,
    this->gradients_quad,
    this->hessians_quad,
    this->scratch_data);

#ifdef DEBUG
  if(evaluate_values == true)
    this->values_quad_initialized = true;
  if(evaluate_gradients == true)
    this->gradients_quad_initialized = true;
  if(evaluate_hessians == true)
    this->hessians_quad_initialized = true;
#endif
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
template<typename VectorType>
inline void
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::gather_evaluate(
  const VectorType & input_vector,
  const bool         evaluate_values,
  const bool         evaluate_gradients,
  const bool         evaluate_hessians)
{
  // If the index storage is interleaved and contiguous and the vector storage
  // has the correct alignment, we can directly pass the pointer into the
  // vector to the evaluate() call, without reading the vector entries into a
  // separate data field. This saves some operations.
  if(std::is_same<typename VectorType::value_type, Number>::value &&
     this->dof_info->index_storage_variants[internal::MatrixFreeFunctions::DoFInfo::dof_access_cell]
                                           [this->cell] ==
       internal::MatrixFreeFunctions::DoFInfo::IndexStorageVariants::interleaved_contiguous &&
     reinterpret_cast<std::size_t>(
       input_vector.begin() +
       this->dof_info
         ->dof_indices_contiguous[internal::MatrixFreeFunctions::DoFInfo::dof_access_cell]
                                 [this->cell * VectorizedArrayType::size()]) %
         sizeof(VectorizedArrayType) ==
       0)
  {
    const VectorizedArrayType * vec_values = reinterpret_cast<const VectorizedArrayType *>(
      input_vector.begin() +
      this->dof_info
        ->dof_indices_contiguous[internal::MatrixFreeFunctions::DoFInfo::dof_access_cell]
                                [this->cell * VectorizedArrayType::size()] +
      this->dof_info
          ->component_dof_indices_offset[this->active_fe_index][this->first_selected_component] *
        VectorizedArrayType::size());

    evaluate(vec_values, evaluate_values, evaluate_gradients, evaluate_hessians);
  }
  else
  {
    this->read_dof_values(input_vector);
    evaluate(this->begin_dof_values(), evaluate_values, evaluate_gradients, evaluate_hessians);
  }
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline void
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::integrate(
  const bool integrate_values,
  const bool integrate_gradients)
{
  integrate(integrate_values, integrate_gradients, this->values_dofs[0]);

#ifdef DEBUG
  this->dof_values_initialized = true;
#endif
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline void
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::integrate(
  const bool            integrate_values,
  const bool            integrate_gradients,
  VectorizedArrayType * values_array)
{
  if(integrate_values == true)
    Assert(this->values_quad_submitted == true, internal::ExcAccessToUninitializedField());
  if(integrate_gradients == true)
    Assert(this->gradients_quad_submitted == true, internal::ExcAccessToUninitializedField());
  Assert(this->matrix_info != nullptr || this->mapped_geometry->is_initialized(),
         ExcNotInitialized());

  EvaluationFlags::EvaluationFlags flag =
    (integrate_values ? EvaluationFlags::values : EvaluationFlags::nothing) |
    (integrate_gradients ? EvaluationFlags::gradients : EvaluationFlags::nothing);

  internal::FEEvaluationFactory<dim, Number, VectorizedArrayType>::integrate(n_components,
                                                                             flag,
                                                                             *this->data,
                                                                             values_array,
                                                                             this->values_quad,
                                                                             this->gradients_quad,
                                                                             this->scratch_data,
                                                                             false);

#ifdef DEBUG
  this->dof_values_initialized = true;
#endif
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
template<typename VectorType>
inline void
CellIntegrator<dim, n_components_, Number, VectorizedArrayType>::integrate_scatter(
  const bool   integrate_values,
  const bool   integrate_gradients,
  VectorType & destination)
{
  // If the index storage is interleaved and contiguous and the vector storage
  // has the correct alignment, we can directly pass the pointer into the
  // vector to the integrate() call, without writing temporary results into a
  // separate data field that will later be added into the vector. This saves
  // some operations.
  if(std::is_same<typename VectorType::value_type, Number>::value &&
     this->dof_info->index_storage_variants[internal::MatrixFreeFunctions::DoFInfo::dof_access_cell]
                                           [this->cell] ==
       internal::MatrixFreeFunctions::DoFInfo::IndexStorageVariants::interleaved_contiguous &&
     reinterpret_cast<std::size_t>(
       destination.begin() +
       this->dof_info
         ->dof_indices_contiguous[internal::MatrixFreeFunctions::DoFInfo::dof_access_cell]
                                 [this->cell * VectorizedArrayType::size()]) %
         sizeof(VectorizedArrayType) ==
       0)
  {
    EvaluationFlags::EvaluationFlags flag =
      (integrate_values ? EvaluationFlags::values : EvaluationFlags::nothing) |
      (integrate_gradients ? EvaluationFlags::gradients : EvaluationFlags::nothing);

    VectorizedArrayType * vec_values = reinterpret_cast<VectorizedArrayType *>(
      destination.begin() +
      this->dof_info
        ->dof_indices_contiguous[internal::MatrixFreeFunctions::DoFInfo::dof_access_cell]
                                [this->cell * VectorizedArrayType::size()] +
      this->dof_info
          ->component_dof_indices_offset[this->active_fe_index][this->first_selected_component] *
        VectorizedArrayType::size());
    internal::FEEvaluationFactory<dim, Number, VectorizedArrayType>::integrate(n_components,
                                                                               flag,
                                                                               *this->data,
                                                                               vec_values,
                                                                               this->values_quad,
                                                                               this->gradients_quad,
                                                                               this->scratch_data,
                                                                               true);
  }
  else
  {
    integrate(integrate_values, integrate_gradients, this->begin_dof_values());
    this->distribute_local_to_global(destination);
  }
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline FaceIntegrator<dim, n_components_, Number, VectorizedArrayType>::FaceIntegrator(
  const MatrixFree<dim, Number, VectorizedArrayType> & matrix_free,
  const bool                                           is_interior_face,
  const unsigned int                                   dof_no,
  const unsigned int                                   quad_no,
  const unsigned int                                   first_selected_component)
  : BaseClass(matrix_free,
              dof_no,
              first_selected_component,
              quad_no,
              numbers::invalid_unsigned_int,
              0,
              is_interior_face),
    dofs_per_component(this->data->dofs_per_component_on_cell),
    dofs_per_cell(this->data->dofs_per_component_on_cell * n_components_),
    n_q_points(this->data->n_q_points_face)
{
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline void
FaceIntegrator<dim, n_components_, Number, VectorizedArrayType>::reinit(
  const unsigned int face_index)
{
  Assert(this->mapped_geometry == nullptr,
         ExcMessage("FEEvaluation was initialized without a matrix-free object."
                    " Integer indexing is not possible"));
  if(this->mapped_geometry != nullptr)
    return;

  this->cell             = face_index;
  this->dof_access_index = this->is_interior_face ?
                             internal::MatrixFreeFunctions::DoFInfo::dof_access_face_interior :
                             internal::MatrixFreeFunctions::DoFInfo::dof_access_face_exterior;
  Assert(this->mapping_data != nullptr, ExcNotInitialized());
  const unsigned int n_vectors = VectorizedArrayType::size();
  const internal::MatrixFreeFunctions::FaceToCellTopology<n_vectors> & faces =
    this->matrix_info->get_face_info(face_index);
  if(face_index >= this->matrix_info->get_task_info().face_partition_data.back() &&
     face_index < this->matrix_info->get_task_info().boundary_partition_data.back())
    Assert(this->is_interior_face, ExcMessage("Boundary faces do not have a neighbor"));

  this->face_no       = (this->is_interior_face ? faces.interior_face_no : faces.exterior_face_no);
  this->subface_index = faces.subface_index;
  if(this->is_interior_face == true)
  {
    this->subface_index = GeometryInfo<dim>::max_children_per_cell;
    if(faces.face_orientation > 8)
      this->face_orientation = faces.face_orientation - 8;
    else
      this->face_orientation = 0;
  }
  else
  {
    if(faces.face_orientation < 8)
      this->face_orientation = faces.face_orientation;
    else
      this->face_orientation = 0;
  }

  this->values_quad_submitted = false;

  this->cell_type            = this->matrix_info->get_mapping_info().face_type[face_index];
  const unsigned int offsets = this->mapping_data->data_index_offsets[face_index];
  this->J_value              = &this->mapping_data->JxW_values[offsets];
  this->normal_vectors       = &this->mapping_data->normal_vectors[offsets];
  this->jacobian             = &this->mapping_data->jacobians[!this->is_interior_face][offsets];
  this->normal_x_jacobian =
    &this->mapping_data->normals_times_jacobians[!this->is_interior_face][offsets];

#ifdef DEBUG
  this->dof_values_initialized     = false;
  this->values_quad_initialized    = false;
  this->gradients_quad_initialized = false;
  this->hessians_quad_initialized  = false;
#endif
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline void
FaceIntegrator<dim, n_components_, Number, VectorizedArrayType>::reinit(
  const unsigned int cell_index,
  const unsigned int face_number)
{
  Assert(
    this->quad_no < this->matrix_info->get_mapping_info().face_data_by_cells.size(),
    ExcMessage(
      "You must set MatrixFree::AdditionalData::mapping_update_flags_faces_by_cells to use the present reinit method."));
  AssertIndexRange(face_number, GeometryInfo<dim>::faces_per_cell);
  AssertIndexRange(cell_index, this->matrix_info->get_mapping_info().cell_type.size());
  Assert(this->mapped_geometry == nullptr,
         ExcMessage("FEEvaluation was initialized without a matrix-free object."
                    " Integer indexing is not possible"));
  if(this->mapped_geometry != nullptr)
    return;
  Assert(this->matrix_info != nullptr, ExcNotInitialized());

  if(this->is_interior_face == false)
  {
    this->face_no          = face_number;
    this->cell             = cell_index;
    this->dof_access_index = internal::MatrixFreeFunctions::DoFInfo::dof_access_cell;
    return;
  }

  this->cell_type        = this->matrix_info->get_mapping_info().cell_type[cell_index];
  this->cell             = cell_index;
  this->face_orientation = 0;
  this->subface_index    = GeometryInfo<dim>::max_children_per_cell;
  this->face_no          = face_number;
  this->dof_access_index = internal::MatrixFreeFunctions::DoFInfo::dof_access_cell;

  const unsigned int offsets =
    this->matrix_info->get_mapping_info()
      .face_data_by_cells[this->quad_no]
      .data_index_offsets[cell_index * GeometryInfo<dim>::faces_per_cell + face_number];
  AssertIndexRange(
    offsets,
    this->matrix_info->get_mapping_info().face_data_by_cells[this->quad_no].JxW_values.size());
  this->J_value =
    &this->matrix_info->get_mapping_info().face_data_by_cells[this->quad_no].JxW_values[offsets];
  this->normal_vectors = &this->matrix_info->get_mapping_info()
                            .face_data_by_cells[this->quad_no]
                            .normal_vectors[offsets];
  this->jacobian =
    &this->matrix_info->get_mapping_info().face_data_by_cells[this->quad_no].jacobians[0][offsets];
  this->normal_x_jacobian = &this->matrix_info->get_mapping_info()
                               .face_data_by_cells[this->quad_no]
                               .normals_times_jacobians[0][offsets];

#ifdef DEBUG
  this->dof_values_initialized     = false;
  this->values_quad_initialized    = false;
  this->gradients_quad_initialized = false;
  this->hessians_quad_initialized  = false;
#endif
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline unsigned int
FaceIntegrator<dim, n_components, Number, VectorizedArrayType>::get_face_index() const
{
  if(this->dof_access_index == internal::MatrixFreeFunctions::DoFInfo::dof_access_cell)
    return this->cell * GeometryInfo<dim>::faces_per_cell + this->face_no;
  else
    return this->cell;
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline unsigned int
FaceIntegrator<dim, n_components_, Number, VectorizedArrayType>::quadrature_formula_index() const
{
  return this->quad_no;
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline unsigned int
FaceIntegrator<dim, n_components_, Number, VectorizedArrayType>::finite_element_index() const
{
  return this->first_selected_component;
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
template<typename VectorType>
inline void
FaceIntegrator<dim, n_components_, Number, VectorizedArrayType>::set_dof_values_plain(
  const VectorType & dst,
  const unsigned int first_index)
{
#ifdef DEBUG
  Assert(this->dof_values_initialized == true, internal::ExcAccessToUninitializedField());
#endif

  // select between block vectors and non-block vectors. Note that the number
  // of components is checked in the internal data
  typename internal::BlockVectorSelector<VectorType,
                                         IsBlockVector<VectorType>::value>::BaseVectorType *
    dst_data[n_components];
  for(unsigned int d = 0; d < n_components; ++d)
    dst_data[d] = internal::BlockVectorSelector<VectorType, IsBlockVector<VectorType>::value>::
      get_vector_component(const_cast<VectorType &>(dst), d + first_index);

  internal::VectorSetter<Number, VectorizedArrayType> setter;
  this->read_write_operation(setter,
                             dst_data,
                             std::bitset<VectorizedArrayType::size()>().flip(),
                             false);
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline void
FaceIntegrator<dim, n_components, Number, VectorizedArrayType>::evaluate(
  const bool evaluate_values,
  const bool evaluate_gradients)
{
  Assert(this->dof_values_initialized, ExcNotInitialized());

  evaluate(this->values_dofs[0], evaluate_values, evaluate_gradients);
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline void
FaceIntegrator<dim, n_components, Number, VectorizedArrayType>::evaluate(
  const VectorizedArrayType * values_array,
  const bool                  evaluate_values,
  const bool                  evaluate_gradients)
{
  if(!(evaluate_values + evaluate_gradients))
    return;

  internal::FEFaceEvaluationFactory<dim, n_components, Number, VectorizedArrayType>::evaluate(
    *this->data,
    values_array,
    this->begin_values(),
    this->begin_gradients(),
    this->scratch_data,
    evaluate_values,
    evaluate_gradients,
    this->face_no,
    this->subface_index,
    this->face_orientation,
    this->mapping_data->descriptor[this->active_fe_index].face_orientations);

#ifdef DEBUG
  if(evaluate_values == true)
    this->values_quad_initialized = true;
  if(evaluate_gradients == true)
    this->gradients_quad_initialized = true;
#endif
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline void
FaceIntegrator<dim, n_components, Number, VectorizedArrayType>::integrate(
  const bool integrate_values,
  const bool integrate_gradients)
{
  integrate(integrate_values, integrate_gradients, this->values_dofs[0]);

#ifdef DEBUG
  this->dof_values_initialized = true;
#endif
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline void
FaceIntegrator<dim, n_components, Number, VectorizedArrayType>::integrate(
  const bool            integrate_values,
  const bool            integrate_gradients,
  VectorizedArrayType * values_array)
{
  if(!(integrate_values + integrate_gradients))
    return;

  internal::FEFaceEvaluationFactory<dim, n_components, Number, VectorizedArrayType>::integrate(
    *this->data,
    values_array,
    this->begin_values(),
    this->begin_gradients(),
    this->scratch_data,
    integrate_values,
    integrate_gradients,
    this->face_no,
    this->subface_index,
    this->face_orientation,
    this->mapping_data->descriptor[this->active_fe_index].face_orientations);
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
template<typename VectorType>
inline void
FaceIntegrator<dim, n_components_, Number, VectorizedArrayType>::gather_evaluate(
  const VectorType & input_vector,
  const bool         evaluate_values,
  const bool         evaluate_gradients)
{
  static_assert(internal::has_begin<VectorType>::value &&
                  (std::is_same<decltype(std::declval<VectorType>().begin()), double *>::value ||
                   std::is_same<decltype(std::declval<VectorType>().begin()), float *>::value),
                "This function requires a vector type with begin() function "
                "evaluating to a pointer to basic number (float,double). "
                "Use read_dof_values() followed by evaluate() instead.");

  if(!internal::FEFaceEvaluationFactory<dim, n_components, Number, VectorizedArrayType>::
       gather_evaluate(input_vector.begin(),
                       *this->data,
                       *this->dof_info,
                       this->begin_values(),
                       this->begin_gradients(),
                       this->scratch_data,
                       evaluate_values,
                       evaluate_gradients,
                       this->active_fe_index,
                       this->first_selected_component,
                       this->cell,
                       this->face_no,
                       this->subface_index,
                       this->dof_access_index,
                       this->face_orientation,
                       this->mapping_data->descriptor[this->active_fe_index].face_orientations))
  {
    this->read_dof_values(input_vector);
    this->evaluate(evaluate_values, evaluate_gradients);
  }

#ifdef DEBUG
  if(evaluate_values == true)
    this->values_quad_initialized = true;
  if(evaluate_gradients == true)
    this->gradients_quad_initialized = true;
#endif
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
template<typename VectorType>
inline void
FaceIntegrator<dim, n_components_, Number, VectorizedArrayType>::integrate_scatter(
  const bool   integrate_values,
  const bool   integrate_gradients,
  VectorType & destination)
{
  static_assert(internal::has_begin<VectorType>::value &&
                  (std::is_same<decltype(std::declval<VectorType>().begin()), double *>::value ||
                   std::is_same<decltype(std::declval<VectorType>().begin()), float *>::value),
                "This function requires a vector type with begin() function "
                "evaluating to a pointer to basic number (float,double). "
                "Use integrate() followed by distribute_local_to_global() "
                "instead.");

  if(!internal::FEFaceEvaluationFactory<dim, n_components, Number, VectorizedArrayType>::
       integrate_scatter(destination.begin(),
                         *this->data,
                         *this->dof_info,
                         this->begin_dof_values(),
                         this->begin_values(),
                         this->begin_gradients(),
                         this->scratch_data,
                         integrate_values,
                         integrate_gradients,
                         this->active_fe_index,
                         this->first_selected_component,
                         this->cell,
                         this->face_no,
                         this->subface_index,
                         this->dof_access_index,
                         this->face_orientation,
                         this->mapping_data->descriptor[this->active_fe_index].face_orientations))
  {
    // if we arrive here, writing into the destination vector did not succeed
    // because some of the assumptions in integrate_scatter were not
    // fulfilled (e.g. an element or degree that does not support direct
    // writing), so we must do it here
    this->distribute_local_to_global(destination);
  }
}



template<int dim, int n_components_, typename Number, typename VectorizedArrayType>
inline Point<dim, VectorizedArrayType>
FaceIntegrator<dim, n_components_, Number, VectorizedArrayType>::quadrature_point(
  const unsigned int q) const
{
  AssertIndexRange(q, n_q_points);
  if(this->dof_access_index < 2)
  {
    Assert(this->mapping_data->quadrature_point_offsets.empty() == false, ExcNotImplemented());
    AssertIndexRange(this->cell, this->mapping_data->quadrature_point_offsets.size());
    return this->mapping_data
      ->quadrature_points[this->mapping_data->quadrature_point_offsets[this->cell] + q];
  }
  else
  {
    Assert(this->matrix_info->get_mapping_info()
               .face_data_by_cells[this->quad_no]
               .quadrature_point_offsets.empty() == false,
           ExcNotImplemented());
    const unsigned int index = this->cell * GeometryInfo<dim>::faces_per_cell + this->face_no;
    AssertIndexRange(index,
                     this->matrix_info->get_mapping_info()
                       .face_data_by_cells[this->quad_no]
                       .quadrature_point_offsets.size());
    return this->matrix_info->get_mapping_info()
      .face_data_by_cells[this->quad_no]
      .quadrature_points[this->matrix_info->get_mapping_info()
                           .face_data_by_cells[this->quad_no]
                           .quadrature_point_offsets[index] +
                         q];
  }
}


namespace MatrixFreeOperators
{
/**
 * This is a specialization of the CellwiseInverseMassMatrix class for the
 * case when the template argument `fe_degree` is set to -1, which means that
 * the implementation is taken from a pre-compiled variant that is controlled
 * via `evaluation_template_factory.templates.h`.
 *
 * Refer to the class CellwiseInverseMassMatrix for information about this
 * class.
 *
 * @author Martin Kronbichler, 2014
 */
template<int dim, int n_components, typename Number, typename VectorizedArrayType>
class CellwiseInverseMassMatrix<dim, -1, n_components, Number, VectorizedArrayType>
{
public:
  /**
   * Constructor. Initializes the shape information from the ShapeInfo field
   * in the class FEEval.
   */
  CellwiseInverseMassMatrix(
    const FEEvaluationBase<dim, n_components, Number, false, VectorizedArrayType> & fe_eval);

  /**
   * Applies the inverse mass matrix operation on an input array. It is
   * assumed that the passed input and output arrays are of correct size,
   * namely FEEval::dofs_per_cell * n_components long. The inverse of the
   * local coefficient (also containing the inverse JxW values) must be
   * passed as first argument. Passing more than one component in the
   * coefficient is allowed.
   */
  void
  apply(const AlignedVector<VectorizedArrayType> & inverse_coefficient,
        const unsigned int                         n_actual_components,
        const VectorizedArrayType *                in_array,
        VectorizedArrayType *                      out_array) const;

  /**
   * Applies the inverse mass matrix operation on an input array, using the
   * inverse of the JxW values provided by the `fe_eval` argument passed to
   * the constructor of this class. Note that the user code must call
   * FEEvaluation::reinit() on the underlying evaluator to make the
   * FEEvaluationBase::JxW() method return the information of the correct
   * cell. It is assumed that the pointers of the input and output arrays
   * are valid over the length FEEvaluation::dofs_per_cell, which is the
   * number of entries processed by this function. The `in_array` and
   * `out_array` arguments may point to the same memory position.
   */
  void
  apply(const VectorizedArrayType * in_array, VectorizedArrayType * out_array) const;

  /**
   * This operation performs a projection from the data given in quadrature
   * points to the actual basis underlying this object. This projection can
   * also be interpreted as a change of the basis from the Lagrange
   * interpolation polynomials in the quadrature points to the
   * basis underlying the current `fe_eval` object.
   */
  void
  transform_from_q_points_to_basis(const unsigned int          n_actual_components,
                                   const VectorizedArrayType * in_array,
                                   VectorizedArrayType *       out_array) const;

  /**
   * Fills the given array with the inverse of the JxW values, i.e., a mass
   * matrix with coefficient 1. Non-unit coefficients must be multiplied (in
   * inverse form) to this array.
   */
  void
  fill_inverse_JxW_values(AlignedVector<VectorizedArrayType> & inverse_jxw) const;

private:
  /**
   * A reference to the FEEvaluation object for getting the JxW_values.
   */
  const FEEvaluationBase<dim, n_components, Number, false, VectorizedArrayType> & fe_eval;
};



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline CellwiseInverseMassMatrix<dim, -1, n_components, Number, VectorizedArrayType>::
  CellwiseInverseMassMatrix(
    const FEEvaluationBase<dim, n_components, Number, false, VectorizedArrayType> & fe_eval)
  : fe_eval(fe_eval)
{
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline void
CellwiseInverseMassMatrix<dim, -1, n_components, Number, VectorizedArrayType>::
  fill_inverse_JxW_values(AlignedVector<VectorizedArrayType> & inverse_jxw) const
{
  const unsigned int fe_degree                  = fe_eval.get_shape_info().data[0].fe_degree;
  const unsigned int dofs_per_component_on_cell = Utilities::pow(fe_degree + 1, dim);
  Assert(inverse_jxw.size() > 0 && inverse_jxw.size() % dofs_per_component_on_cell == 0,
         ExcMessage("Expected diagonal to be a multiple of scalar dof per cells"));

  // compute values for the first component
  for(unsigned int q = 0; q < dofs_per_component_on_cell; ++q)
    inverse_jxw[q] = 1. / fe_eval.JxW(q);
  // copy values to rest of vector
  for(unsigned int q = dofs_per_component_on_cell; q < inverse_jxw.size();)
    for(unsigned int i = 0; i < dofs_per_component_on_cell; ++i, ++q)
      inverse_jxw[q] = inverse_jxw[i];
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline void
CellwiseInverseMassMatrix<dim, -1, n_components, Number, VectorizedArrayType>::apply(
  const AlignedVector<VectorizedArrayType> & inverse_coefficients,
  const unsigned int                         n_actual_components,
  const VectorizedArrayType *                in_array,
  VectorizedArrayType *                      out_array) const
{
  internal::CellwiseInverseMassFactory<dim, n_components, Number, VectorizedArrayType>::apply(
    n_actual_components,
    fe_eval.get_shape_info().data[0].fe_degree,
    fe_eval.get_shape_info().data[0].inverse_shape_values_eo,
    inverse_coefficients,
    in_array,
    out_array);
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline void
CellwiseInverseMassMatrix<dim, -1, n_components, Number, VectorizedArrayType>::apply(
  const VectorizedArrayType * in_array,
  VectorizedArrayType *       out_array) const
{
  internal::CellwiseInverseMassFactory<dim, n_components, Number, VectorizedArrayType>::apply(
    fe_eval.get_shape_info().data[0].fe_degree, fe_eval, in_array, out_array);
}



template<int dim, int n_components, typename Number, typename VectorizedArrayType>
inline void
CellwiseInverseMassMatrix<dim, -1, n_components, Number, VectorizedArrayType>::
  transform_from_q_points_to_basis(const unsigned int          n_actual_components,
                                   const VectorizedArrayType * in_array,
                                   VectorizedArrayType *       out_array) const
{
  internal::CellwiseInverseMassFactory<dim, n_components, Number, VectorizedArrayType>::
    transform_from_q_points_to_basis(n_actual_components,
                                     fe_eval.get_shape_info().data[0].fe_degree,
                                     fe_eval.get_shape_info().data[0].inverse_shape_values_eo,
                                     in_array,
                                     out_array);
}
} // namespace MatrixFreeOperators



DEAL_II_NAMESPACE_CLOSE

#endif
