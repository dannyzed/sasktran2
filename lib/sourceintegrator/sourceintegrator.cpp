
#include <sasktran2/source_integrator.h>
#include <sasktran2/autodiff/operation.h>

namespace sasktran2 {
    template <int NSTOKES>
    struct AttenuationExpression : autodiff::UnaryExpression {
        sasktran2::SparseODDualView m_od;
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>& m_rad;

        AttenuationExpression(
            autodiff::ExprPtr parent,
            sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>&
                rad,
            const sasktran2::SparseODDualView& od)
            : UnaryExpression(parent), m_od(od), m_rad(rad) {}

        void forward_impl() override {
            if (m_rad.deriv.size() > 0) {
                for (auto it = m_od.deriv_iter; it; ++it) {
                    m_rad.deriv(Eigen::all, it.index()) -=
                        it.value() * m_rad.value;
                }
            }

            m_rad.value *= m_od.exp_minus_od;
            m_rad.deriv *= m_od.exp_minus_od;
        }

        void backward_impl() override {}
    };

    template <int NSTOKES>
    SourceIntegrator<NSTOKES>::SourceIntegrator(bool calculate_derivatives)
        : m_calculate_derivatives(calculate_derivatives) {}

    template <int NSTOKES>
    void SourceIntegrator<NSTOKES>::initialize_geometry(
        const std::vector<sasktran2::raytracing::TracedRay>& traced_rays,
        const Geometry& geometry) {
        // Construct the optical depth matrices.
        // This is the matrix so that matrix @ extinction = layer od, one matrix
        // for each ray Calculating this matrix beforehand makes calculating
        // derivatives easier, and removes excess computation for every
        // wavelength
        m_traced_ray_od_matrix.resize(traced_rays.size());
        for (int i = 0; i < traced_rays.size(); ++i) {
            sasktran2::raytracing::construct_od_matrix(
                traced_rays[i], geometry, m_traced_ray_od_matrix[i]);
        }

        m_shell_od.resize(traced_rays.size());
        m_exp_minus_shell_od.resize(traced_rays.size());

        m_traced_rays = &traced_rays;
    }

    template <int NSTOKES>
    void SourceIntegrator<NSTOKES>::initialize_atmosphere(
        const sasktran2::atmosphere::Atmosphere<NSTOKES>& atmo) {
// Multithread over LOS? or wavelength? Or just let Eigen do it?
#pragma omp parallel for
        for (int i = 0; i < m_traced_ray_od_matrix.size(); ++i) {
            m_shell_od[i].noalias() =
                m_traced_ray_od_matrix[i] * atmo.storage().total_extinction;

#ifdef SASKTRAN_DEBUG_ASSERTS
            if (!m_shell_od[i].allFinite()) {
                spdlog::error("Error calculating Layer OD for ray: ", i);
            }
#endif
        }

        m_atmosphere = &atmo;

        if (atmo.num_deriv() == 0) {
            m_calculate_derivatives = false;
        }
    }

    template <int NSTOKES>
    void SourceIntegrator<NSTOKES>::generate_integrate_expression(
        sasktran2::autodiff::ExprPtr& expr,
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>&
            radiance,
        std::vector<SourceTermInterface<NSTOKES>*> source_terms, int wavelidx,
        int rayidx, int wavel_threadidx, int threadidx) {
        const auto& ray = (*m_traced_rays)[rayidx];
        // TOOD: Add source at the end of the ray

        // Iterate through each layer from the end of the ray to the observer
        for (int j = 0; j < ray.layers.size(); ++j) {
            const sasktran2::raytracing::SphericalLayer& layer = ray.layers[j];

            sasktran2::SparseODDualView local_shell_od(
                m_shell_od[rayidx](j, wavelidx),
                std::exp(-m_shell_od[rayidx](j, wavelidx)),
                m_traced_ray_od_matrix[rayidx], j);

            // expr = std::make_shared<AttenuationExpression<NSTOKES>>(expr,
            // radiance, local_shell_od);

            // Calculate all of the layer sources
            for (const auto& source : source_terms) {
                source->integrated_source_expression(
                    wavelidx, rayidx, j, wavel_threadidx, threadidx, layer,
                    local_shell_od, radiance, expr);
            }
        }
    }

    template <int NSTOKES>
    void SourceIntegrator<NSTOKES>::integrate(
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>&
            radiance,
        std::vector<SourceTermInterface<NSTOKES>*> source_terms, int wavelidx,
        int rayidx, int wavel_threadidx, int threadidx) {
        const auto& ray = (*m_traced_rays)[rayidx];
        // Add source at the end of the ray
        for (const auto& source : source_terms) {
            source->end_of_ray_source(wavelidx, rayidx, wavel_threadidx,
                                      threadidx, radiance);
        }

        // Iterate through each layer from the end of the ray to the observer
        for (int j = 0; j < ray.layers.size(); ++j) {
            const sasktran2::raytracing::SphericalLayer& layer = ray.layers[j];

            sasktran2::SparseODDualView local_shell_od(
                m_shell_od[rayidx](j, wavelidx),
                std::exp(-m_shell_od[rayidx](j, wavelidx)),
                m_traced_ray_od_matrix[rayidx], j);

            // Attenuate the radiance by the layer OD
            // rad = rad * atten, drad = drad * atten + rad * datten
            // Atten effects all derivative, datten only affects the extinction
            // derivatives

            if (m_calculate_derivatives) {
                for (auto it = local_shell_od.deriv_iter; it; ++it) {
                    radiance.deriv(Eigen::all, it.index()) -=
                        it.value() * radiance.value;
                }
            }

            radiance.value *= local_shell_od.exp_minus_od;
            if (m_calculate_derivatives) {
                radiance.deriv *= local_shell_od.exp_minus_od;
            }

            // Calculate all of the layer sources
            for (const auto& source : source_terms) {
                source->integrated_source(wavelidx, rayidx, j, wavel_threadidx,
                                          threadidx, layer, local_shell_od,
                                          radiance);
            }

#ifdef SASKTRAN_DEBUG_ASSERTS
            if (radiance.value.hasNaN()) {
                static bool message = false;
                if (!message) {
                    spdlog::error("One of the sources was  NaN Ray: {} layer: "
                                  "{} Layer od: {} Layer Atten Factor: {}",
                                  rayidx, j, local_shell_od.od,
                                  local_shell_od.exp_minus_od);
                    message = true;
                }
            }
#endif
        }
    }

    template <int NSTOKES>
    void SourceIntegrator<NSTOKES>::integrate_optical_depth(
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>&
            radiance,
        int wavelidx, int rayidx, int wavel_threadidx, int threadidx) {
        radiance.value(0) =
            m_shell_od[rayidx](Eigen::all, wavelidx).array().sum();
    }

    template <int NSTOKES>
    void SourceIntegrator<NSTOKES>::integrate_and_emplace_accumulation_triplets(
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>&
            radiance,
        std::vector<SourceTermInterface<NSTOKES>*> source_terms, int wavelidx,
        int rayidx, int wavel_threadidx, int threadidx,
        const SInterpolator& source_interpolator,
        Eigen::VectorXd& accumulation_values) {
        const auto& ray = (*m_traced_rays)[rayidx];
        const auto& interpolator = source_interpolator[rayidx];

        // If we don't have to calculate derivatives then it is faster to
        // iterate over the ray backwards, i.e., from the observer to the end of
        // the atmosphere
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>
            layer_source(NSTOKES, 0);

        double current_od = 0;
        for (int j = (int)ray.layers.size() - 1; j >= 0; --j) {
            const sasktran2::raytracing::SphericalLayer& layer = ray.layers[j];

            sasktran2::SparseODDualView local_shell_od(
                m_shell_od[rayidx](j, wavelidx),
                std::exp(-m_shell_od[rayidx](j, wavelidx)),
                m_traced_ray_od_matrix[rayidx], j);
            const auto& layer_interpolator = interpolator.interior_weights[j];
            // Calculate and add the layer source to the radiance
            double atten_factor = std::exp(-current_od);

            // Calculate all of the layer sources
            layer_source.value.setZero();
            for (const auto& source : source_terms) {
                source->integrated_source(wavelidx, rayidx, j, wavel_threadidx,
                                          threadidx, layer, local_shell_od,
                                          layer_source);
            }

            radiance.value += layer_source.value * atten_factor;

            // Assign the accumulation weights
            double omega = 0;
            for (int i = 0; i < layer_interpolator.first.size(); ++i) {
                auto& index_weight = layer_interpolator.first[i];
                omega +=
                    m_atmosphere->storage().ssa(index_weight.first, wavelidx) *
                    index_weight.second;
            }
            double source_factor =
                omega * (1 - local_shell_od.exp_minus_od) * atten_factor;

            for (const auto& ele : layer_interpolator.second) {
                for (int s = 0; s < NSTOKES; ++s) {
                    accumulation_values(std::get<2>(ele)[s]) +=
                        std::get<1>(ele) * source_factor;
                }
            }

            current_od += local_shell_od.od;
        }

        // Add source at the end of the ray
        layer_source.value.setZero();
        for (const auto& source : source_terms) {
            source->end_of_ray_source(wavelidx, rayidx, wavel_threadidx,
                                      threadidx, layer_source);
        }

        radiance.value += layer_source.value * std::exp(-1 * current_od);

        // Add ground interpolation triplets
        if (ray.ground_is_hit) {
            const auto& ground_interpolator = interpolator.ground_weights;

            for (const auto& ele : ground_interpolator) {
                for (int s = 0; s < NSTOKES; ++s) {
                    accumulation_values(std::get<2>(ele)[s]) +=
                        std::get<1>(ele) * std::exp(-1 * current_od);
                }
            }
        }
    }

    template class SourceIntegrator<1>;
    template class SourceIntegrator<3>;
} // namespace sasktran2
