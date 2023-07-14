package co.wiklund.disthist

/**
 * GslRngHandle - Wrapper class around GNU Scientific Library's random number generator. This forces us to manually handle
 * 	any memory allocated in the C code. Thus, before discarding any GslRngHandle Object, you must call .free() on the
 * 	object to release any memory used.
 */
class GslRngHandle(seed : Long) {
	
	/* gslRngAddress - Handle to be used when sampling using gsl_rng  */
	val gslRngAddress : Long = wrap_gsl_rng_alloc()
	wrap_gsl_rng_set(gslRngAddress, seed)	
  
	def free = {
		wrap_gsl_rng_free(gslRngAddress)
	}

	/**
	 * Wrapper of gsl_rng_alloc - Returns address of allocated random number generator
	 */
	private def wrap_gsl_rng_alloc = {

	}

	/**
	 * Wrapper of gsl_rng_set - Set the seed of the generator at the given address
	 */
	private def wrap_gsl_rng_set = {

	}
	
	/**
	 * Wrapper of gsl_rng_free - Free the underlying memory of the random number generator 
	 */
	private def wrap_gsl_rng_free = {

	}
}
