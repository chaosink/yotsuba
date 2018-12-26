# Integrators
if (NOT MTS_VERSION)
  message(FATAL_ERROR "Use the top level configuration file")
endif()

macro(add_integrator)
  add_mts_plugin(${ARGN} TYPE integrator)
endmacro()

macro(add_bidir)
  add_integrator(${ARGN} MTS_BIDIR)
endmacro()

add_integrator(ao       direct/ao.cpp)
add_integrator(direct   direct/direct.cpp)
add_integrator(path     path/path.cpp)
add_integrator(volpath  path/volpath.cpp)
add_integrator(volpath_simple path/volpath_simple.cpp)
add_integrator(ptracer  ptracer/ptracer.cpp
                        ptracer/ptracer_proc.h ptracer/ptracer_proc.cpp)

# Photon mapping-based techniques
add_integrator(photonmapper photonmapper/photonmapper.cpp photonmapper/bre.cpp)
add_integrator(ppm          photonmapper/ppm.cpp)
add_integrator(sppm         photonmapper/sppm.cpp)

# Miscellaneous
add_integrator(vpl          vpl/vpl.cpp MTS_HW)
add_integrator(adaptive     misc/adaptive.cpp)
add_integrator(irrcache     misc/irrcache.cpp
                            misc/irrcache_proc.h misc/irrcache_proc.cpp)
add_integrator(multichannel misc/multichannel.cpp)
add_integrator(field        misc/field.cpp)
add_integrator(motion       misc/motion.cpp)

set(common
	myintegrators/samplingutils.cpp
	myintegrators/integrand.cpp
	myintegrators/parallel.cpp
	myintegrators/appendsensor.cpp
	myintegrators/appendbsdf.cpp
	myintegrators/appendemitter.cpp
	myintegrators/sampleseedpaths.cpp
	myintegrators/classification.cpp
	myintegrators/occlusioncache.cpp)

#add_integrator(mydirect
#	myintegrators/mydirect.cpp
#	${common})
#add_integrator(mypath
#	myintegrators/mypath.cpp
#	${common})
## add_integrator(myptracer
## 	myintegrators/myptracer.cpp
## 	${common})
#add_integrator(mybidir
#	myintegrators/mybidir.cpp
#	${common})
#add_integrator(mymlt
#	myintegrators/mymlt.cpp
#	myintegrators/bidirmutation.cpp
#	myintegrators/lensperturbation.cpp
#	myintegrators/causticsperturbation.cpp
#	myintegrators/multichainperturbation.cpp
#	myintegrators/mutation.cpp
#	${common})
#add_integrator(mytridir
#	myintegrators/mytridir.cpp
#	myintegrators/appendkeyhole.cpp
#	${common})
#add_integrator(mygpt
#	myintegrators/mygpt.cpp
#	myintegrators/poissonsolver.cpp
#	${common})

add_integrator(lintridir
	myintegrators/lintridir.cpp
	myintegrators/appendkeyhole.cpp
	${common})

# Bidirectional techniques
add_bidir(bdpt          bdpt/bdpt.h      bdpt/bdpt.cpp
                        bdpt/bdpt_proc.h bdpt/bdpt_proc.cpp
                        bdpt/bdpt_wr.h   bdpt/bdpt_wr.cpp)

add_bidir(pssmlt        pssmlt/pssmlt.h         pssmlt/pssmlt.cpp
                        pssmlt/pssmlt_proc.h    pssmlt/pssmlt_proc.cpp
                        pssmlt/pssmlt_sampler.h pssmlt/pssmlt_sampler.cpp)

add_bidir(mlt           mlt/mlt.h      mlt/mlt.cpp
                        mlt/mlt_proc.h mlt/mlt_proc.cpp)

add_bidir(erpt          erpt/erpt.h      erpt/erpt.cpp
                        erpt/erpt_proc.h erpt/erpt_proc.cpp)
