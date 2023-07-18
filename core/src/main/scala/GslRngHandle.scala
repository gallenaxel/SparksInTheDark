package co.wiklund.disthist

import com.github.sbt.jni.syntax.NativeLoader

/**
 * GslRngHandle - Wrapper class around GNU Scientific Library's random number generator. This forces us to manually handle
 * 	any memory allocated in the C code. Thus, before discarding any GslRngHandle Object, you must call .free() on the
 * 	object to release any memory used.
 *
 *  NOTE: given library name in "sbt nativeInit libname" must correspond to NativeLoader("libname0")
 */
class GslRngHandle(seed : Long) extends NativeLoader("gsl_wrapper0") {
	
	/* gslRngAddress - Handle to be used when sampling using gsl_rng  */
	val gslRngAddress = wrap_gsl_rng_alloc
	wrap_gsl_rng_set(gslRngAddress, seed)	
  
        /**
         * free - Frees any resources used by the random number generator
         */
	def free = {
		wrap_gsl_rng_free(gslRngAddress)
	}
        
	/**
	 * Wrapper of gsl_rng_alloc - Returns address of allocated random number generator
	 */
	@native private def wrap_gsl_rng_alloc : Long

	/**
	 * Wrapper of gsl_rng_set - Set the seed of the generator at the given address
	 */
	@native private def wrap_gsl_rng_set(gslRngAddress : Long, seed : Long)
	
	/**
	 * Wrapper of gsl_rng_free - Free the underlying memory of the random number generator 
	 */
	@native private def wrap_gsl_rng_free(gslRngAddress : Long)
}

object GslRngFunctions extends NativeLoader("gsl_wrapper0") {

        /**
         *  gsl_ran_discrete_fill_buffer() - Returns buffer with indices i correponding to events e_i with prob. probabilities[i].
         *    Why not a wrapper around gsl_ran_discrete()? Calling native code from Scala/Java is costly, so we should batch all
         *    wanted samples together into an array and generate them all at once instead.
         *
         * @param gslRngAddress - Address of gsl RNG 
         * @param probabilities - Array of probabilities corresponding to discrete outcomes 0,1,2,3...
         * @param sampleSize - Size of wanted sample
         * @return Buffer filled with sample from given discrete distribution
         *
         */
        @native def gsl_ran_discrete_fill_buffer(gslRngAddress : Long, probabilities : Array[Double], sampleSize : Int) : Array[Int] 
        
        /**
         *  gsl_ran_flat_fill_buffer() - Fills buffer with Uniform[low,high] sample.
         *
         * @param gslRngAddress - Address of gsl RNG 
         * @param min - Minimum value from distribution
         * @param max - Maximum value from distribution
         * @return Buffer filled with sample from given Uniform distribution
         */
        @native def gsl_ran_flat_fill_buffer(gslRngAddress : Long, min : Double, max : Double, sampleSize : Int) : Array[Double] 
}
