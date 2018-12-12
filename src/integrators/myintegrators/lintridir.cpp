#include <mitsuba/render/scene.h>
#include <mitsuba/core/convertaether.h>
#include <mitsuba/render/unidist.h>
#include <mitsuba/render/renderqueue.h>
#include <mitsuba/core/fresolver.h>
#include <mitsuba/core/fstream.h>
#include <mutex>
#include "integrand.h"
#include "samplingutils.h"
#include "appendsensor.h"
#include "appendbsdf.h"
#include "appendemitter.h"
#include "appendkeyhole.h"
#include "parallel.h"
#include "occlusioncache.h"
#include "progressreporter.h"

using namespace aether;

// Necessary for PDF caches
// TODO: make a macro for this or get rid of this totally
std::atomic<int> aether::Object::next_id{0};

MTS_NAMESPACE_BEGIN

class LinTriDirIntegrator : public Integrator {
public:
    LinTriDirIntegrator(const Properties &props) : Integrator(props) {
        /* Depth to begin using russian roulette */
        m_rrDepth = props.getInteger("rrDepth", 5);

        /* Longest visualized path depth (\c -1 = infinite).
           A value of \c 1 will visualize only directly visible light sources.
           \c 2 will lead to single-bounce (direct-only) illumination, and so on. */
        m_maxDepth = props.getInteger("maxDepth", 6);
        SAssert(m_maxDepth != -1);

        /* When this flag is set to true, contributions from directly
         * visible emitters will not be included in the rendered image */
        m_hideEmitters = props.getBoolean("hideEmitters", false);
    }

    /// Unserialize from a binary data stream
    LinTriDirIntegrator(Stream *stream, InstanceManager *manager)
     : Integrator(stream, manager) {
        m_hideEmitters = stream->readBool();
        configure();
    }

    void serialize(Stream *stream, InstanceManager *manager) const {
        Integrator::serialize(stream, manager);
        stream->writeBool(m_hideEmitters);
    }

    void configure() {
        Integrator::configure();
    }

    bool preprocess(const Scene *scene, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID) {
        Integrator::preprocess(scene, queue, job, sceneResID, sensorResID, samplerResID);
        m_raycaster = std::make_unique<Raycast>(scene);
        return true;
    }

    void cancel() {
        m_running = false;
    }

    void configureSampler(const Scene *scene, Sampler *sampler) {
        /* Prepare the sampler for tile-based rendering */
        sampler->setFilmResolution(scene->getFilm()->getCropSize(), true);
    }

    bool render(Scene *scene, RenderQueue *queue, const RenderJob *job,
            int sceneResID, int sensorResID, int samplerResID) {
        ref<Scheduler> sched = Scheduler::getInstance();
        ref<Sensor> sensor = scene->getSensor();
        ref<Film> film = sensor->getFilm();
        ref<Sampler> sampler = scene->getSampler();
        size_t nCores = sched->getCoreCount();
        Log(EInfo, "Starting render job (%ix%i, " SIZE_T_FMT " %s, " SSE_STR ") ..",
            film->getCropSize().x, film->getCropSize().y,
            nCores, nCores == 1 ? "core" : "cores");

        // TODO: take cropOffset into account
        //Point2i cropOffset = film->getCropOffset();
        Vector2i cropSize = film->getCropSize();
        ref<ImageBlock> imageBlock = new ImageBlock(Bitmap::ESpectrum,
            cropSize, film->getReconstructionFilter());
        imageBlock->setOffset(Point2i(imageBlock->getBorderSize(), imageBlock->getBorderSize()));
        ref<Bitmap> bitmap = new Bitmap(Bitmap::ESpectrum, Bitmap::EFloat, film->getSize());
        imageBlock->clear();
        bitmap->clear();
        film->clear();
        m_running = true;

        const int tileSize = scene->getBlockSize();
        const int nXTiles = (cropSize.x + tileSize - 1) / tileSize;
        const int nYTiles = (cropSize.y + tileSize - 1) / tileSize;
        MyProgressReporter reporter(nXTiles * nYTiles);
        bool unfinished = false;
        std::mutex mutex;

        ParallelFor([&](const Vector2i tile) {
            ref<Sampler> clonedSampler = sampler->clone();
            const int x0 = tile.x * tileSize;
            const int x1 = std::min(x0 + tileSize, cropSize.x);
            const int y0 = tile.y * tileSize;
            const int y1 = std::min(y0 + tileSize, cropSize.y);
            // For each pixel
            for (int y = y0; y < y1; y++) {
                for (int x = x0; x < x1; x++) {
                    if (!m_running) {
                        unfinished = true;
                        return;
                    }
                    // For each sample
                    for (int sampleId = 0; sampleId < sampler->getSampleCount(); sampleId++) {
                        render(scene, clonedSampler, imageBlock.get(), x, y, mutex);
                    }
                }
            }
            std::lock_guard<std::mutex> lock(mutex);
            bitmap->copyFrom(imageBlock->getBitmap());
            film->setBitmap(bitmap, Float(1) / sampler->getSampleCount());
            queue->signalRefresh(job);
            reporter.Update(1);
        }, Vector2i(nXTiles, nYTiles));
        TerminateWorkerThreads();

        if (unfinished) {
            return false;
        }

        reporter.Done();
        return true;
    }

    struct SplatElement {
        Point2 position;
        Spectrum contribution;
        int pathLength;
        int sensorSubpathIndex;
    };

    void render(const Scene *scene, Sampler *sampler, ImageBlock *imageBlock, const int x, const int y,
                std::mutex &mutex) {
        UniDist uniDist(sampler);

        // TODO: move triangle co-ordinates to scene file
        // cbox_keyhole
        //aether::Vector3 tri1_v0{-0.173878, 1.990000, -0.189689};
        //aether::Vector3 tri1_v1{ 0.153878, 1.990000, 0.139690};
        //aether::Vector3 tri1_v2{-0.173878, 1.990000,  0.139690};

        //aether::Vector3 tri2_v0{-0.173878, 1.990000, -0.189689};
        //aether::Vector3 tri2_v1{ 0.153878, 1.990000, -0.189689};
        //aether::Vector3 tri2_v2{ 0.153878, 1.990000, 0.139690};

        std::vector<std::array<aether::Vector3, 3>> keyhole_tris;
        // veach_door
        // aether::Vector3 tri1_v0{121.9446, 0, -73.0101};
        // aether::Vector3 tri1_v1{119.1694, 121.5088, -56.649};
        // aether::Vector3 tri1_v2{119.1694, 0.5088, -56.649};
        // keyhole_tris.push_back({{tri1_v0, tri1_v1, tri1_v2}});

        // aether::Vector3 tri2_v0{121.9446, 0, -73.0101};
        // aether::Vector3 tri2_v1{121.9446, 123.3378, -73.0101};
        // aether::Vector3 tri2_v2{119.1694, 121.5088, -56.649};
        // keyhole_tris.push_back({{tri2_v0, tri2_v1, tri2_v2}});

        // Top of opening
        aether::Vector3 tri3_v0{121.9446, 123.3378, -73.0101};
        aether::Vector3 tri3_v1{62.1593, 121.5088, -74.4293};
        aether::Vector3 tri3_v2{119.1694, 121.5088, -56.649};
        keyhole_tris.push_back({{tri3_v0, tri3_v1, tri3_v2}});

        // // Back of door
        // aether::Vector3 tri4_v0{121.9446, 123.3378, -77.0547};
        // aether::Vector3 tri4_v1{58.7111, 123.3378, -77.0547};
        // aether::Vector3 tri4_v2{58.7111, 0, -77.0547};
        // keyhole_tris.push_back({{tri4_v0, tri4_v1, tri4_v2}});

        // aether::Vector3 tri5_v0{121.9446, 123.3378, -77.0547};
        // aether::Vector3 tri5_v1{58.7111, 0, -77.0547 };
        // aether::Vector3 tri5_v2{121.9446, 0, -77.0547};
        // keyhole_tris.push_back({{tri5_v0, tri5_v1, tri5_v2}});

        // auto GetArea = [&](std::array<aether::Vector3, 3> &tri) {
        //     auto a = to_vector(tri[0] - tri[1]);
        //     auto b = to_vector(tri[0] - tri[2]);
        //     return 2.f / (a[1] * b[2] + a[2] * b[0] + a[0] * b[1] - a[0] * b[2] - a[1] * b[0] - a[2] * b[1]);
        // };
        // for(auto &tri: keyhole_tris)
        //     cout << GetArea(tri) << endl;

        RandomSequence<Vertex> keyholePath;
        AppendKeyhole(keyholePath, uniDist, *m_raycaster, keyhole_tris);
        keyholePath.Sample();

        for(int i = 0; i < 3; i++) {
            aether::variable<0> u0;
            aether::variable<1> u1;
            auto r = sqrt(aether::one - sq(u0));
            auto phi = two * get_pi() * u1;
            auto x = make_random_var(r * cos(phi));
            auto y = make_random_var(r * sin(phi));
            auto z = make_random_var(u0);
            auto dir_local = make_random_vector(x, y, z).Sample(0.5f, 0.5f);
            cout << "dir_local type:\n\t" << typeid(dir_local).name() << endl;
            cout << dir_local.Pdf() << endl;
            cout << dir_local.Pdf(Vector(0.1f, 0.1f, 0.1f)) << endl;

            { // output type
                aether::variable<0> u0;
                aether::variable<1> u1;
                auto r = sqrt(aether::one - sq(u0));
                auto phi = two * get_pi() * u1;
                auto x = r * cos(phi);
                cout << "x type:\n\t" << typeid(x).name() << endl;
                auto x_jac = aether::jacobian(x);
                cout << "x_jac type:\n\t" << typeid(x_jac).name() << endl;
                // constexpr auto x_det = abs(rcp(make_expr(determinant_tag, x_jac)));
                // cout << "x_det type:\n\t" << typeid(x_det).name() << endl;

                auto y = r * sin(phi);
                auto z = u0;
                auto dir = make_vector_expr(x, y, z);
                auto dir_jac = aether::jacobian(dir);
                cout << "dir_jac type:\n\t" << typeid(dir_jac).name() << endl;
                // auto dir_det = abs(rcp(make_expr(determinant_tag, dir_jac)));
                // cout << "dir_det type:\n\t" << typeid(dir_det).name() << endl;
            }

            { // test
                aether::variable<1> u1;
                aether::variable<2> u2;
                auto x = make_random_var(sin(u1));
                auto y = make_random_var(sq(u2));
                auto z = make_random_var(u1 * u2);
                auto dir_local = make_random_vector(x, y, z).Sample(0.5f, 0.5f);
                cout << "dir_local type:\n\t" << typeid(dir_local).name() << endl;
                cout << dir_local.Pdf() << endl;
                cout << dir_local.Pdf(Vector(0.1f, 0.1f, 0.1f)) << endl;
            }
        }

        while(1) {
            RandomSequence<Vertex> keyholePath;
            AppendKeyhole(keyholePath, uniDist, *m_raycaster, keyhole_tris);
            cout << "--------------------" << endl;
            keyholePath.Sample();
            exportPath(keyholePath);
            auto p0 = to_point(keyholePath[0].Value());
            auto p1 = to_point(keyholePath[1].Value());
            cout << p0.toString() << endl;
            cout << p1.toString() << endl;
            cout << "-----" << endl;
            Intersection its = keyholePath[0].Get(intersection_);
            Intersection nextIts = keyholePath[1].Get(intersection_);
            const mitsuba::Vector toNext = directionToNext(keyholePath[0], keyholePath[1]);
            // cout << its.geoFrame.n.toString() << endl;
            // cout << toNext.toString() << endl;
            // cout << nextIts.geoFrame.n.toString() << endl;
            const Float nextCosine = nextIts.isValid() ?
                fabs(dot(-toNext, nextIts.geoFrame.n)) : Float(1.f);
            const Float distSq = distanceSquared(
                to_point(keyholePath[0].Value()),
                to_point(keyholePath[1].Value()));
            cout << fabs(dot(toNext, its.geoFrame.n)) << " " << 1.f / distSq << " " << nextCosine << endl;
            Float geometryTerm = fabs(dot(toNext, its.geoFrame.n)) / distSq * nextCosine;
            cout << geometryTerm << endl;
            cout << geometryTerm * INV_TWOPI << endl;
            cout << "----------" << endl;
            cout << keyholePath.Pdf() << endl;
            getchar();
        }

        const ref_vector<Emitter> &emitters = scene->getEmitters();

        // Sample time
        // TODO: make this optional
        const Float time = sampler->next1D();
        OcclusionCache occlusionCache(scene, time);

        // Sample camera subpath
        RandomSequence<Vertex> sensorSubpath;
        AppendPositionOnSensor(sensorSubpath, uniDist, scene->getSensor(), time);
        AppendDirectionFromSensor(sensorSubpath, uniDist, *m_raycaster, x, y);
        sensorSubpath.Sample();

        // Loop until reaching specified maximum depth
        // e.g. if maxDepth == 2, we don't need sensorSubpath.Size() > 3
        for(;sensorSubpath.Size() <= m_maxDepth;) {
            AppendBSDF(sensorSubpath, uniDist, *m_raycaster);
            sensorSubpath.Sample();
        }

        // Sample emitter subpath
        RandomSequence<Vertex> emitterSubpath;
        AppendPositionOnEmitter(emitterSubpath, uniDist, emitters, time);
        AppendDirectionFromEmitter(emitterSubpath, uniDist, *m_raycaster);
        emitterSubpath.Sample();
        // Loop until reaching specified maximum depth
        // e.g. if maxDepth == 2, we don't need sensorSubpath.Size() > 3
        for(;emitterSubpath.Size() <= m_maxDepth;) {
            AppendBSDF(emitterSubpath, uniDist, *m_raycaster);
            emitterSubpath.Sample();
        }

        // Combine
        std::vector<SplatElement> splats;
        for (int pathLength = 2; pathLength <= m_maxDepth; pathLength++) {
            std::vector<RandomSequence<Vertex>> paths;
            for (int sensorSubpathSize = 1; sensorSubpathSize <= pathLength + 1; sensorSubpathSize++) {
                if(sensorSubpathSize != 3) continue;

                const int emitterSubpathSize = pathLength + 1 - sensorSubpathSize;
                auto sensorSubpathSlice = sensorSubpath.Slice(0, sensorSubpathSize);
                RandomSequence<Vertex> path;

                // Tri-directional path
                // - Shorten the sensor and emitter subpaths by 1
                // - Replace the removed vertices with the two keyhole vertices
                if (sensorSubpathSize > 1 && emitterSubpathSize >= 1) {
                    auto sensorTriDirSubpathSlice = sensorSubpath.Slice(0, sensorSubpathSize - 1);
                    auto emitterTriDirSubpathSlice = reverse_(emitterSubpath.Slice(0, emitterSubpathSize - 1));
                    paths.push_back(sensorTriDirSubpathSlice.Concat(keyholePath).Concat(emitterTriDirSubpathSlice));
                }

                // // - Shorten the sensor subpaths by 2
                // if (sensorSubpathSize > 2) {
                //     auto sensorTriDirSubpathSlice = sensorSubpath.Slice(0, sensorSubpathSize - 2);
                //     auto emitterTriDirSubpathSlice = reverse_(emitterSubpath.Slice(0, emitterSubpathSize));
                //     paths.push_back(sensorTriDirSubpathSlice.Concat(keyholePath).Concat(emitterTriDirSubpathSlice));
                // }
                // // - Shorten the emitter subpaths by 2
                // if (emitterSubpathSize >= 2) {
                //     auto sensorTriDirSubpathSlice = sensorSubpath.Slice(0, sensorSubpathSize);
                //     auto emitterTriDirSubpathSlice = reverse_(emitterSubpath.Slice(0, emitterSubpathSize - 2));
                //     paths.push_back(sensorTriDirSubpathSlice.Concat(keyholePath).Concat(emitterTriDirSubpathSlice));
                // }


                // Bi-directional paths
                // if (emitterSubpathSize != 1) {
                //     auto emitterSubpathSlice = reverse_(emitterSubpath.Slice(0, emitterSubpathSize));
                //     path = sensorSubpathSlice.Concat(emitterSubpathSlice);
                // } else {
                //     // Special case: we want to do specialized direct importance sampling here
                //     AppendDirectSampleEmitter(sensorSubpathSlice, uniDist, scene->getEmitters());
                //     sensorSubpathSlice.Sample();
                //     path = sensorSubpathSlice;
                // }
                // SAssert(path.Size() == pathLength + 1);
                // paths.push_back(path);
            }
            estimate(scene, imageBlock, paths, time, splats, occlusionCache);
        }

        std::lock_guard<std::mutex> lock(mutex);

        if (!m_hideEmitters) {
            auto path = sensorSubpath.Slice(0, 2);
            if (path.AllValid()) {
                Spectrum contribution = estimateBidir(scene, make_view(path));
                Point2 position;
                if (project(scene, sensorSubpath, position)) {
                    imageBlock->put(position, &contribution[0]);
                }
            }
        }

        for (const auto &splatElement : splats) {
            Point2 position = splatElement.position;
            Spectrum contribution = splatElement.contribution;
            imageBlock->put(position, &contribution[0]);
        }
    }

	void exportPath(const RandomSequence<Vertex> &path) const {
		std::ofstream ofs("/media/lin/MintSpace/program/CG/PortalFinding/scene/veach_door/path.obj");
		int k = 0;
		auto exportSegment = [&](const Vertex &v0, const Vertex &v1) {
			const Point3 p0 = to_point(v0.Value());
			const Point3 p1 = to_point(v1.Value());
			ofs << "v " << p0.x << " " << -p0.z << " " << p0.y << endl;
			ofs << "v " << p0.x << " " << -p0.z << " " << p0.y << endl;
			ofs << "v " << p1.x << " " << -p1.z << " " << p1.y << endl;
			ofs << "f " << k * 3 + 1 << " " << k * 3 + 2 << " " << k * 3 + 3 << endl;
			k++;
		};

		for(int i = 0; i < path.Size() - 1; i++)
			exportSegment(path[i], path[i + 1]);
	}

inline mitsuba::Vector directionToPrevious(const Vertex &previousVertex, const Vertex &vertex) const {
    return to_vector((previousVertex.Value() - vertex.Value()).normalized());
}

inline mitsuba::Vector directionToNext(const Vertex &vertex, const Vertex &nextVertex) const {
    return to_vector((nextVertex.Value() - vertex.Value()).normalized());
}

Spectrum Le(const Scene *scene, const Vertex &previousVertex, const Vertex &vertex) const {
    const Emitter *emitter = vertex.Get(emitter_);
    const Intersection &its = vertex.Get(intersection_);
    const mitsuba::Vector directionToLight = -directionToPrevious(previousVertex, vertex);
    if (emitter != nullptr) {
        // Sampled emitter
        return emitter->eval(its, -directionToLight);
    }
    if (its.isValid()) {
        // if its is an emitter, emitter shouldn't be nullptr
        SAssert(!its.isEmitter());
        if (its.isEmitter()) {
            // Hit an area light
            return its.Le(-directionToLight);
        }
        // Hit a non-emitter object
        return Spectrum(0.f);
    }
    // Hit background
    // It might seem not making sense using its.p, but evalEnvironment doesn't use ray.o anyway
    return scene->evalEnvironment(RayDifferential(its.p, directionToLight, its.time));
}

Spectrum Eval(const Scene *scene, const RandomSequenceView<Vertex> &path, const int index) const {
    SAssert(path.Size() > 1);
    SAssert(index >= 1);
    if (index == path.Size() - 1) {
        return Le(scene, path[index - 1], path[index]);
    }

    SAssert(index + 1 < path.Size());
    Intersection its = path[index].Get(intersection_);
    if (!its.isValid()) {
        // Not a valid surface
        return Spectrum(0.f);
    }
    SAssert(its.shape != nullptr);
    its.wi = its.toLocal(directionToPrevious(path[index - 1], path[index]));
    const mitsuba::Vector toNext = directionToNext(path[index], path[index + 1]);
    BSDFSamplingRecord bRec(its, its.toLocal(toNext));
    bRec.framePerturbed = true;
    // TODO: support ray differentials
    const BSDF *bsdf = its.getBSDF();
    SAssert(bsdf != nullptr);
    const Spectrum bsdfValue = bsdf->eval(bRec);
    if (bsdfValue.isZero()) {
        return Spectrum(0.f);
    }
    const Intersection &nextIts = path[index + 1].Get(intersection_);
    const Float nextCosine = nextIts.isValid() ?
        fabs(dot(-toNext, nextIts.geoFrame.n)) : Float(1.f);
    const Float distSq =
        distanceSquared(to_point(path[index].Value()),
                        to_point(path[index + 1].Value()));
    if (distSq == Float(0)) {
        return Spectrum(0.f);
    }
    // bsdf->eval already includes one of the cosines
    const Float geometryTerm = nextCosine / distSq;
    if (!std::isfinite(geometryTerm) || !bsdfValue.isValid()) {
        return Spectrum(0.f);
    }

    cout << "    " << index << " " << fabs(dot(toNext, its.geoFrame.n))
         << " " << 1.f / distSq << " " << nextCosine << endl;
    return geometryTerm * bsdfValue * Eval(scene, path, index + 1);
}

Spectrum EvalSensor(const Scene *scene, const RandomSequenceView<Vertex> &path) const {
    // Should we obtain sensor from the path?
    const Sensor *sensor = scene->getSensor();
    SAssert(path[0].Get(sensor_) != nullptr);
    PositionSamplingRecord pRec(path[1].Get(intersection_).time);
    pRec.p = to_point(path[0].Value());
    DirectionSamplingRecord dRec;
    dRec.d = to_vector((path[1].Value() - path[0].Value()).normalized());
    dRec.measure = ESolidAngle;
    Spectrum We = sensor->evalDirection(dRec, pRec);
    const Intersection &nextIts = path[1].Get(intersection_);
    const Float nextCosine = nextIts.isValid() ?
        fabs(dot(-dRec.d, nextIts.geoFrame.n)) : Float(1.f);
    // bsdf->eval already includes one of the cosines
    const Float geometryTerm = nextCosine /
        distanceSquared(to_point(path[0].Value()),
                        to_point(path[1].Value()));

    cout << "    " << 0 << " " << geometryTerm / nextCosine << " " << nextCosine << endl;
    return geometryTerm * We;
}

TSpectrum<double, SPECTRUM_SAMPLES> EvalBidir(const Scene *scene, const RandomSequenceView<Vertex> &path) const {
    SAssert(path.AllValid());
    if (path.Size() <= 1) {
        // Need to at least have a sensor and an emitter
        return TSpectrum<double, SPECTRUM_SAMPLES>(0.f);
    }

    return TSpectrum<double, SPECTRUM_SAMPLES>(EvalSensor(scene, path) * Eval(scene, path, 1));
}

    void estimate(const Scene *scene,
                  ImageBlock *imageBlock,
                  const std::vector<RandomSequence<Vertex>> &paths,
                  const Float time,
                  std::vector<SplatElement> &splats,
                  OcclusionCache& occlusionCache) const {
        std::vector<double> pdfs;
        for (size_t i = 0; i < paths.size(); i++) {
            const RandomSequence<Vertex> &path = paths[i];
            if (!path.AllValid()) {
                continue;
            }
            Point2 position;
            if (!project(scene, path, position)) {
                continue;
            }
            // TODO: improve this; check only the connection edges
            bool occ = false;
            for (size_t j = 0, N = path.Size() - 1; j < N; j++) {
                if (occlusionCache.Query(std::make_pair(path[j].Value(), path[j + 1].Value()))) {
                    occ = true;
                    break;
                }
            }
            if (occ) {
                continue;
            }
            Spectrum contribution = estimateBidir(scene, make_view(path));
            if (contribution.isZero()) {
                continue;
            }

            pdfs.clear();
            for (size_t j = 0; j < paths.size(); j++) {
                double pdf = paths[j].Pdf(path);
                pdfs.push_back(pdf);
            }



            #define OUTPUT_VERTEX_PDF
            #if defined OUTPUT_VERTEX_PDF
            cout << "--------------------" << endl;
            cout << paths.size() << " " << i << endl;

            SourceDistributionHelper source_distribution_helper;

            Real p = 1;
            using T = typename RandomSequence<Vertex>::sample_t::optional_element_t;

            std::size_t sample_index = 0;
            for (std::size_t strategy_index = 0, N = paths[0].strategies.Size(); strategy_index < N; ++strategy_index) {
              const auto& strategy = paths[0].strategies[strategy_index];
              std::vector<T> query_samples;
              query_samples.reserve(strategy->output_size);
              for (std::size_t i = 0; i < strategy->output_size; ++i) {
                query_samples.push_back(path.store.at(sample_index + i).get().rv);
              }

              auto pdf = strategy->Pdf(path, query_samples, sample_index,
                                       paths[0].reversed[sample_index], source_distribution_helper);
              if (pdf == 0 || std::isnan(pdf)) {
                break;
              }
              // if(sample_index == 1)
              cout << "    " << sample_index << " " << pdf << endl;
              p *= pdf;
              sample_index += strategy->output_size;
            }
            cout << p << endl;
            exportPath(path);
            auto Li = EvalBidir(scene, make_view(path));
            getchar();
            #endif



            double weight = misWeight(i, pdfs);
            Spectrum weightedContribution = weight * contribution;
            int pathLength = path.Size() - 1;
            splats.push_back(SplatElement{position, weightedContribution, pathLength, (int)i});
        }
    }

    std::string toString() const {
        std::ostringstream oss;
        oss << "LinTriDirIntegrator[" << endl
            << "  hideEmitters = " << m_hideEmitters << endl
            << "]";
        return oss.str();
    }

    MTS_DECLARE_CLASS()
private:
    bool m_hideEmitters;
    int m_rrDepth;
    int m_maxDepth;

    std::unique_ptr<Raycast> m_raycaster;
    bool m_running;
};

MTS_IMPLEMENT_CLASS_S(LinTriDirIntegrator, false, SamplingIntegrator)
MTS_EXPORT_PLUGIN(LinTriDirIntegrator, "Lin's tridir integrator");
MTS_NAMESPACE_END
