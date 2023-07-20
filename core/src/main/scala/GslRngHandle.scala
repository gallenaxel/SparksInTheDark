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
	val gslRngAddress = gsl_rng_alloc
	gsl_rng_set(gslRngAddress, seed)	
  
        /**
         * free - Frees any resources used by the random number generator
         */
	def free = {
		gsl_rng_free(gslRngAddress)
	}
        
	/**
	 * Wrapper of gsl_rng_alloc - Returns address of allocated random number generator
	 */
	@native private def gsl_rng_alloc : Long

	/**
	 * Wrapper of gsl_rng_set - Set the seed of the generator at the given address
	 */
	@native private def gsl_rng_set(gslRngAddress : Long, seed : Long)
	
	/**
	 * Wrapper of gsl_rng_free - Free the underlying memory of the random number generator 
	 */
	@native private def gsl_rng_free(gslRngAddress : Long)
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
         */
        @native def gsl_ran_discrete_fill_buffer(gslRngAddress : Long, probabilities : Array[Double], sampleSize : Int) : Array[Int] 
        /**
         *  gsl_ran_discrete() - Returns sampled int set to some index i correponding to events e_i with prob. probabilities[i].
         *    Use this very sparingly, the setup for sampling from such a distribution is costly, in addition to the added cost
         *    of calling native functions through scala.
         *
         * @param gslRngAddress - Address of gsl RNG 
         * @param probabilities - Array of probabilities corresponding to discrete outcomes 0,1,2,3...
         * @return index from given discrete distribution
         */
        @native def gsl_ran_discrete(gslRngAddress : Long, probabilities : Array[Double]) : Int
        
        /**
         *  gsl_ran_flat_fill_buffer() - Fills buffer with Uniform[low,high] sample.
         *
         * @param gslRngAddress - Address of gsl RNG 
         * @param min - Minimum value from distribution
         * @param max - Maximum value from distribution
         * @return Buffer filled with sample from given Uniform distribution
         */
        @native def gsl_ran_flat_fill_buffer(gslRngAddress : Long, min : Double, max : Double, sampleSize : Int) : Array[Double] 

        /**
         *  gsl_ran_flat() - Sample point from Uniform[min, max]. If a large sample is needed, it is more appropriate to use
         *    gsl_ran_flat_fill_buffer to avoid any overhead of calling native functions in scala.
         *
         * @param gslRngAddress - Address of gsl RNG 
         * @param min - Minimum value from distribution
         * @param max - Maximum value from distribution
         * @return Buffer filled with sample from given Uniform distribution
         */
        @native def gsl_ran_flat(gslRngAddress : Long, min : Double, max : Double) : Double 
}
