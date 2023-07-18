#include <gsl/gsl_rng.h>
#include <gsl/gsl_randist.h>

#include "gsl_wrapper.h"

JNIEXPORT jlong JNICALL Java_co_wiklund_disthist_GslRngHandle_wrap_1gsl_1rng_1alloc(JNIEnv *env, jobject obj)
{
	/*TODO: Setup type choice */
	const gsl_rng_type *type = gsl_rng_default;
	return (jlong) gsl_rng_alloc(type);
}

JNIEXPORT void JNICALL Java_co_wiklund_disthist_GslRngHandle_wrap_1gsl_1rng_1set(JNIEnv *env, jobject obj, jlong gslRngAddress, jlong seed)
{
	gsl_rng_set((gsl_rng *) gslRngAddress, seed);
}

JNIEXPORT void JNICALL Java_co_wiklund_disthist_GslRngHandle_wrap_1gsl_1rng_1free(JNIEnv *env, jobject obj, jlong gslRngAddress)
{
	gsl_rng_free((void *) gslRngAddress);		
}

JNIEXPORT jintArray JNICALL Java_co_wiklund_disthist_GslRngFunctions_00024_gsl_1ran_1discrete_1fill_1buffer(JNIEnv *env, jobject obj, jlong gsl_rng_address, jdoubleArray j_probabilities, jint sample_size)
{
	const jsize probabilities_len = (*env)->GetArrayLength(env, j_probabilities);
	const jdouble *probabilities = (*env)->GetDoubleArrayElements(env, j_probabilities, 0);

	jintArray j_buf = (*env)->NewIntArray(env, sample_size);
	jint *buf = malloc(sample_size * sizeof(jint));
	const gsl_ran_discrete_t *table = gsl_ran_discrete_preproc(probabilities_len, probabilities);

	for (int i = 0; i < sample_size; ++i)
	{
		buf[i] = gsl_ran_discrete((gsl_rng *) gsl_rng_address, table); 
	}

	const jsize start = 0;
	const jsize len = sample_size;
	(*env)->SetIntArrayRegion(env, j_buf, start, len, buf);

	(*env)->ReleaseDoubleArrayElements(env, j_probabilities, probabilities, 0);
	gsl_ran_discrete_free(table);
	free(buf);

	return j_buf;
}

JNIEXPORT jdoubleArray JNICALL Java_co_wiklund_disthist_GslRngFunctions_00024_gsl_1ran_1flat_1fill_1buffer
  (JNIEnv *env, jobject obj, jlong gsl_rng_address, jdouble min, jdouble max, jint sample_size)
{

	jdoubleArray j_buf = (*env)->NewDoubleArray(env, sample_size);
	jdouble *buf = malloc(sample_size * sizeof(jdouble));

	for (int i = 0; i < sample_size; ++i)
	{
		buf[i] = gsl_ran_flat((gsl_rng *) gsl_rng_address, min, max); 
	}

	const jsize start = 0;
	const jsize len = sample_size;
	(*env)->SetDoubleArrayRegion(env, j_buf, start, len, buf);

	free(buf);

	return j_buf;
}
