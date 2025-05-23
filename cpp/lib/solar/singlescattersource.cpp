#include "sasktran2/raytracing.h"
#include "sasktran2/source_algorithms.h"
#include "sasktran2/source_interface.h"
#include <sasktran2/solartransmission.h>
#include <sasktran2/dual.h>
#include <sasktran2/config.h>
#include <sasktran2/atmosphere/atmosphere.h>
#include <sasktran2/math/trig.h>

namespace sasktran2::solartransmission {
    template <typename S, int NSTOKES>
    void SingleScatterSource<S, NSTOKES>::initialize_atmosphere(
        const sasktran2::atmosphere::Atmosphere<NSTOKES>& atmosphere) {
        // Store the atmosphere for later
        m_atmosphere = &atmosphere;
        this->m_phase_handler.initialize_atmosphere(atmosphere);

        // Initialize some local memory storage
        for (int i = 0; i < m_start_source_cache.size(); ++i) {
            m_start_source_cache[i].resize(NSTOKES, atmosphere.num_deriv(),
                                           false);
            m_end_source_cache[i].resize(NSTOKES, atmosphere.num_deriv(),
                                         false);
        }
    };

    template <typename S, int NSTOKES>
    void SingleScatterSource<S, NSTOKES>::initialize_config(
        const sasktran2::Config& config) {
        m_config = &config;

        this->m_solar_transmission.initialize_config(config);
        this->m_phase_handler.initialize_config(config);

        // Set up storage for each thread
        // m_solar_trans.resize(config.num_threads());
        m_thread_index_cache_one.resize(config.num_threads());
        m_thread_index_cache_two.resize(config.num_threads());

        m_solar_trans.resize(config.num_wavelength_threads());

        m_start_source_cache.resize(config.num_threads());
        m_end_source_cache.resize(config.num_threads());
    }

    template <typename S, int NSTOKES>
    void SingleScatterSource<S, NSTOKES>::calculate(int wavelidx,
                                                    int threadidx) {
        ZoneScopedN("Single Scatter Source Calculation");
        // Don't have to do anything here
        m_phase_handler.calculate(wavelidx, threadidx);

        // Calculate the solar transmission at each cell
        if constexpr (std::is_same_v<S, SolarTransmissionExact>) {
            // Faster to use the dense matrix if most of the elements are
            // nonzero
            // TODO: Is 0.25 a good number?
            if (double(m_geometry_sparse.nonZeros()) /
                    double(m_geometry_matrix.size()) >
                1) {
                m_solar_trans[threadidx].noalias() =
                    m_geometry_matrix *
                    m_atmosphere->storage().total_extinction(Eigen::all,
                                                             wavelidx);
            } else {
                m_solar_trans[threadidx].noalias() =
                    m_geometry_sparse *
                    m_atmosphere->storage().total_extinction(Eigen::all,
                                                             wavelidx);
            }
        }

        if constexpr (std::is_same_v<S, SolarTransmissionTable>) {
            m_solar_trans[threadidx].noalias() =
                m_geometry_sparse * (m_solar_transmission.geometry_matrix() *
                                     m_atmosphere->storage().total_extinction(
                                         Eigen::all, wavelidx));
        }

        m_solar_trans[threadidx] =
            exp(-m_solar_trans[threadidx].array()) *
            m_atmosphere->storage().solar_irradiance(wavelidx);
        for (int i = 0; i < m_ground_hit_flag.size(); ++i) {
            if (m_ground_hit_flag[i]) {
                m_solar_trans[threadidx][i] = 0;
            }
        }
    }

    template <typename S, int NSTOKES>
    void SingleScatterSource<S, NSTOKES>::end_of_ray_source(
        int wavelidx, int losidx, int wavel_threadidx, int threadidx,
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>& source)
        const {
        if (m_los_rays->at(losidx).ground_is_hit) {
            // Single scatter ground source is solar_trans * cos(th) * brdf

            // Cosine of direction to the sun at the surface
            // TODO: This does not account for refraction?
            double mu_in =
                m_los_rays->at(losidx).layers[0].exit.cos_zenith_angle(
                    m_geometry.coordinates().sun_unit());

            // Cosine of direction to LOS at the surface
            double mu_out =
                -1.0 * m_los_rays->at(losidx).layers[0].exit.cos_zenith_angle(
                           m_los_rays->at(losidx).layers[0].average_look_away);

            // We already have the azimuthal difference
            double phi_diff = m_los_rays->at(losidx).layers[0].saz_exit;

            Eigen::Matrix<double, NSTOKES, NSTOKES> brdf =
                m_atmosphere->surface().brdf(wavelidx, mu_in, mu_out, phi_diff);

            int exit_index = m_index_map[losidx][0];

            double solar_trans = m_solar_trans[wavel_threadidx](exit_index);

            Eigen::Vector<double, NSTOKES> source_value =
                solar_trans * brdf(Eigen::all, 0) * mu_in;

            source.value.array() += source_value.array();
            if (source.deriv.size() > 0) {
                // Add on the solar transmission derivative factors
                if constexpr (std::is_same_v<S, SolarTransmissionExact>) {
                    if (m_config->wf_precision() !=
                        sasktran2::Config::WeightingFunctionPrecision::
                            limited) {
                        // Have to apply the solar transmission derivative
                        // factors
                        for (Eigen::SparseMatrix<double,
                                                 Eigen::RowMajor>::InnerIterator
                                 it(m_geometry_sparse, exit_index);
                             it; ++it) {
                            source.deriv(Eigen::all, it.index()) -=
                                it.value() * source_value;
                        }
                    }
                }

                for (int k = 0; k < m_atmosphere->surface().num_deriv(); ++k) {
                    // And then the surface derivative factors
                    Eigen::Matrix<double, NSTOKES, NSTOKES> brdf_deriv =
                        m_atmosphere->surface().d_brdf(wavelidx, mu_in, mu_out,
                                                       phi_diff, k);

                    source.deriv(Eigen::all,
                                 m_atmosphere->surface_deriv_start_index() +
                                     k) +=
                        solar_trans * mu_in * brdf_deriv(Eigen::all, 0);
                }
            }
        }
    }

    template <typename S, int NSTOKES>
    void SingleScatterSource<S, NSTOKES>::initialize_geometry(
        const std::vector<sasktran2::raytracing::TracedRay>& los_rays) {
        ZoneScopedN("Initialize Single Scatter Source Geometry");
        this->m_solar_transmission.initialize_geometry(los_rays);

        if constexpr (std::is_same_v<S, SolarTransmissionExact>) {
            // Generates the geometry matrix so that matrix * extinction = solar
            // od at grid points
            this->m_solar_transmission.generate_geometry_matrix(
                los_rays, m_geometry_matrix, m_ground_hit_flag);

            // Usually faster to calculate the matrix densely and then convert
            // to sparse
            m_geometry_sparse = m_geometry_matrix.sparseView();
        }
        if constexpr (std::is_same_v<S, SolarTransmissionTable>) {
            this->m_solar_transmission.generate_interpolation_matrix(
                los_rays, m_geometry_sparse, m_ground_hit_flag);
        }

        // We need some mapping between the layers inside each ray to our
        // calculated solar transmission
        m_index_map.resize(los_rays.size());
        m_num_cells = 0;
        int c = 0;
        for (int i = 0; i < los_rays.size(); ++i) {
            m_index_map[i].resize(los_rays[i].layers.size());

            for (int j = 0; j < m_index_map[i].size(); ++j) {
                m_index_map[i][j] = c;
                ++c;
            }
            // Final exit layer
            ++c;

            m_num_cells += (int)los_rays[i].layers.size();
        }
        this->m_phase_handler.initialize_geometry(los_rays, m_index_map);

        // Store the rays for later
        m_los_rays = &los_rays;
    }

    template <typename S, int NSTOKES>
    void SingleScatterSource<S, NSTOKES>::integrated_source_quadrature(
        int wavelidx, int losidx, int layeridx, int threadidx,
        const sasktran2::raytracing::SphericalLayer& layer,
        const sasktran2::SparseODDualView& shell_od,
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>& source)
        const {

        // TODO: Retry quadrature calculation?
    }

    template <typename S, int NSTOKES>
    void SingleScatterSource<S, NSTOKES>::integrated_source_constant(
        int wavelidx, int losidx, int layeridx, int wavel_threadidx,
        int threadidx, const sasktran2::raytracing::SphericalLayer& layer,
        const sasktran2::SparseODDualView& shell_od,
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>& source,
        typename SourceTermInterface<NSTOKES>::IntegrationDirection direction)
        const {
        ZoneScopedN("Single Scatter Source Constant Calculation");

        bool calculate_derivatives = source.derivative_size() > 0;

        // Integrates assuming the source is constant in the layer and
        // determined by the average of the layer boundaries
        int exit_index = m_index_map[losidx][layeridx];
        int entrance_index = m_index_map[losidx][layeridx] + 1;

        double solar_trans_exit = m_solar_trans[wavel_threadidx](exit_index);
        double solar_trans_entrance =
            m_solar_trans[wavel_threadidx](entrance_index);

        auto& start_phase = m_start_source_cache[threadidx];
        auto& end_phase = m_end_source_cache[threadidx];

        double ssa_start = 0;
        double ssa_end = 0;

        double k_start = 0;
        double k_end = 0;

        scattering_source(
            m_phase_handler, wavel_threadidx, losidx, layeridx, wavelidx,
            layer.entrance.interpolation_weights, true, solar_trans_entrance,
            *m_atmosphere,
            Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator(
                m_geometry_sparse, entrance_index),
            calculate_derivatives, start_phase, ssa_start, k_start);

        scattering_source(
            m_phase_handler, wavel_threadidx, losidx, layeridx, wavelidx,
            layer.exit.interpolation_weights, false, solar_trans_exit,
            *m_atmosphere,
            Eigen::SparseMatrix<double, Eigen::RowMajor>::InnerIterator(
                m_geometry_sparse, exit_index),
            calculate_derivatives, end_phase, ssa_end, k_end);

        double source_factor1 = (1 - shell_od.exp_minus_od) / shell_od.od;
        // Note dsource_factor = d_od * (1/od - source_factor * (1 + 1/od))

        // Get the phase matrix and add on the sources
        // The source factor term will only have extinction derivatives, the
        // phase term will have local SSA/scattering derivatives and is ~dense
        // in a 1D atmosphere

        source.value.array() +=
            source_factor1 * (start_phase.value.array() * layer.od_quad_start +
                              end_phase.value.array() * layer.od_quad_end);

        if (calculate_derivatives) {
            // Now for the derivatives, start with dsource_factor which is
            // sparse
            for (auto it = shell_od.deriv_iter; it; ++it) {
                source.deriv(Eigen::all, it.index()).array() +=
                    it.value() *
                    (1 / shell_od.od - source_factor1 * (1 + 1 / shell_od.od)) *
                    (start_phase.value.array() * layer.od_quad_start +
                     end_phase.value.array() * layer.od_quad_end);
            }
            // And add on d_phase
            source.deriv.array() +=
                source_factor1 * start_phase.deriv.array() *
                    layer.od_quad_start +
                source_factor1 * end_phase.deriv.array() * layer.od_quad_end;
        }

#ifdef SASKTRAN_DEBUG_ASSERTS
        if (source.value.hasNaN()) {
            static bool message = false;
            if (!message) {
                spdlog::error("SS Source NaN {} {}", source_factor1,
                              layer.od_quad_start_fraction);
                message = true;
            }
        }
#endif
    }

    template <typename S, int NSTOKES>
    void SingleScatterSource<S, NSTOKES>::integrated_source_linear(
        int wavelidx, int losidx, int layeridx, int wavel_threadidx,
        int threadidx, const sasktran2::raytracing::SphericalLayer& layer,
        const sasktran2::SparseODDualView& shell_od,
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>& source)
        const {
        // TODO: Go back to this?
    }

    template <typename S, int NSTOKES>
    void SingleScatterSource<S, NSTOKES>::integrated_source(
        int wavelidx, int losidx, int layeridx, int wavel_threadidx,
        int threadidx, const sasktran2::raytracing::SphericalLayer& layer,
        const sasktran2::SparseODDualView& shell_od,
        sasktran2::Dual<double, sasktran2::dualstorage::dense, NSTOKES>& source,
        typename SourceTermInterface<NSTOKES>::IntegrationDirection direction)
        const {
        if (layer.layer_distance < MINIMUM_SHELL_SIZE_M) {
            // Essentially an empty shell from rounding, don't have to do
            // anything
            return;
        }

        integrated_source_constant(wavelidx, losidx, layeridx, wavel_threadidx,
                                   threadidx, layer, shell_od, source,
                                   direction);
    }

    template class SingleScatterSource<SolarTransmissionExact, 1>;
    template class SingleScatterSource<SolarTransmissionExact, 3>;

    template class SingleScatterSource<SolarTransmissionTable, 1>;
    template class SingleScatterSource<SolarTransmissionTable, 3>;
} // namespace sasktran2::solartransmission
