#ifndef MESH_WORKER_WRAPPER
#define MESH_WORKER_WRAPPER

template <class Operator>
class MeshWorkerWrapper
    : public MeshWorker::LocalIntegrator<Operator::DIM, Operator::DIM, double> {
public:
  static const int dim = Operator::DIM;

  MeshWorkerWrapper(const Operator &t, bool use_cell = true,
                    bool use_boundary = true, bool use_face = true)
      : MeshWorker::LocalIntegrator<dim, dim, double>(use_cell, use_boundary,
                                                      use_face),
        t(t) {}

  inline DEAL_II_ALWAYS_INLINE void
  cell(MeshWorker::DoFInfo<dim, dim, double> &dinfo,
       typename MeshWorker::IntegrationInfo<dim> &info) const {
    t.cell(dinfo, info);
  }

  inline DEAL_II_ALWAYS_INLINE void
  boundary(MeshWorker::DoFInfo<dim, dim, double> &dinfo,
           typename MeshWorker::IntegrationInfo<dim> &info) const {
    t.boundary(dinfo, info);
  }

  inline DEAL_II_ALWAYS_INLINE void
  face(MeshWorker::DoFInfo<dim, dim, double> &dinfo1,
       MeshWorker::DoFInfo<dim, dim, double> &dinfo2,
       typename MeshWorker::IntegrationInfo<dim> &info1,
       typename MeshWorker::IntegrationInfo<dim> &info2) const {
    t.face(dinfo1, dinfo2, info1, info2);
  }

  const Operator &t;
};

#endif