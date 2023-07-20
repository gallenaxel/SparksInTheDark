/* DO NOT EDIT THIS FILE - it is machine generated */
#include <jni.h>
/* Header for class co_wiklund_disthist_GslRngHandle */

#ifndef _Included_co_wiklund_disthist_GslRngHandle
#define _Included_co_wiklund_disthist_GslRngHandle
#ifdef __cplusplus
extern "C" {
#endif
/*
 * Class:      co_wiklund_disthist_GslRngHandle
 * Method:     gsl_1rng_1alloc
 * Signature:  ()J
 */
JNIEXPORT jlong JNICALL Java_co_wiklund_disthist_GslRngHandle_gsl_1rng_1alloc
  (JNIEnv *, jobject);

/*
 * Class:      co_wiklund_disthist_GslRngHandle
 * Method:     gsl_1rng_1set
 * Signature:  (JJ)V
 */
JNIEXPORT void JNICALL Java_co_wiklund_disthist_GslRngHandle_gsl_1rng_1set
  (JNIEnv *, jobject, jlong, jlong);

/*
 * Class:      co_wiklund_disthist_GslRngHandle
 * Method:     gsl_1rng_1free
 * Signature:  (J)V
 */
JNIEXPORT void JNICALL Java_co_wiklund_disthist_GslRngHandle_gsl_1rng_1free
  (JNIEnv *, jobject, jlong);
/*
 * Class:      co_wiklund_disthist_GslRngFunctions_00024
 * Method:     gsl_1ran_1discrete_1fill_1buffer
 * Signature:  (J[DI)[I
 */
JNIEXPORT jintArray JNICALL Java_co_wiklund_disthist_GslRngFunctions_00024_gsl_1ran_1discrete_1fill_1buffer
  (JNIEnv *, jobject, jlong, jdoubleArray, jint);

/*
 * Class:      co_wiklund_disthist_GslRngFunctions_00024
 * Method:     gsl_1ran_1discrete
 * Signature:  (J[D)I
 */
JNIEXPORT jint JNICALL Java_co_wiklund_disthist_GslRngFunctions_00024_gsl_1ran_1discrete
  (JNIEnv *, jobject, jlong, jdoubleArray);

/*
 * Class:      co_wiklund_disthist_GslRngFunctions_00024
 * Method:     gsl_1ran_1flat_1fill_1buffer
 * Signature:  (JDDI)[D
 */
JNIEXPORT jdoubleArray JNICALL Java_co_wiklund_disthist_GslRngFunctions_00024_gsl_1ran_1flat_1fill_1buffer
  (JNIEnv *, jobject, jlong, jdouble, jdouble, jint);

/*
 * Class:      co_wiklund_disthist_GslRngFunctions_00024
 * Method:     gsl_1ran_1flat
 * Signature:  (JDD)D
 */
JNIEXPORT jdouble JNICALL Java_co_wiklund_disthist_GslRngFunctions_00024_gsl_1ran_1flat
  (JNIEnv *, jobject, jlong, jdouble, jdouble);
#ifdef __cplusplus
}
#endif
#endif
