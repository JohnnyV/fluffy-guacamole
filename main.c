#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <gsl/gsl_blas.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_rng.h>

uint32_t msbFirstToLsbFirst( uint32_t n )
{
    unsigned char a[4], *pn;

    pn = (unsigned char*)&n;
    a[3] = *(pn+0);
    a[2] = *(pn+1);
    a[1] = *(pn+2);
    a[0] = *(pn+3);
    return *(uint32_t *)&a;

    return n;
}

float sigmoid(float z)
{
    return 1.0f/(1.0f+exp(-z));
}

float sigmoid_derivative(float z)
{
    return sigmoid(z)*(1-sigmoid(z));
}

void randomizeMatrix(gsl_matrix * m, int x, int y) {
  const gsl_rng_type * T;
  gsl_rng * r;

  int i, j = 0;

  gsl_rng_env_setup();

  T = gsl_rng_default;
  r = gsl_rng_alloc (T);

    for (i = 0; i < x; i++)
    {
        for (j = 0; j < y; j++)
        {
            float u = gsl_rng_uniform (r);
            gsl_matrix_set (m, i, j, u);
            printf ("%.5f\n", u);
        }
    }

    gsl_rng_free (r);
}


float error_func_total(gsl_vector * actual, gsl_vector * result) {
    float err = 0;
    int i = 0;
    for(i=0; i<actual->size; i++) {
        err+=powf(gsl_vector_get(actual,i)-gsl_vector_get(result,i), 2);
    }
    return err/2.0f;
}

int main()
{
    int MATRIX_X = 28;
    int MATRIX_Y = 28;
    int INPUT_VECTOR_SIZE = 3;
    int HIDDEN_VECTOR_SIZE = 2;
    int OUTPUT_VECTOR_SIZE = 3;
    float LEARNING_RATE = 0.01f;

    char inputs[] = {2, 3, 0};
    float matrixW1[] = {0.1,0.9, 0.3,0.4, 0.8,0.5};

    float matrixW2[] = {0.3,0.4,0.1, 0.8,0.7,0.5};

    gsl_vector * input = gsl_vector_alloc(INPUT_VECTOR_SIZE);
    gsl_matrix * w1 = gsl_matrix_alloc(INPUT_VECTOR_SIZE, HIDDEN_VECTOR_SIZE);

    gsl_vector * hidden = gsl_vector_alloc(HIDDEN_VECTOR_SIZE);
    gsl_matrix * w2 = gsl_matrix_alloc (HIDDEN_VECTOR_SIZE, OUTPUT_VECTOR_SIZE);
    gsl_vector * output = gsl_vector_alloc(OUTPUT_VECTOR_SIZE);
    gsl_vector * expected_output = gsl_vector_alloc(OUTPUT_VECTOR_SIZE);

    gsl_vector_set(input, 0, 2);
    gsl_vector_set(input, 1, 3);
    gsl_vector_set(input, 2, 0);

    gsl_vector_set(expected_output, 0, 3);
    gsl_vector_set(expected_output, 1, 2.5);
    gsl_vector_set(expected_output, 2, 1.5);

    // init weights
    gsl_matrix_set(w1, 0, 0, matrixW1[0]);
    gsl_matrix_set(w1, 0, 1, matrixW1[1]);
    gsl_matrix_set(w1, 1, 0, matrixW1[2]);
    gsl_matrix_set(w1, 1, 1, matrixW1[3]);
    gsl_matrix_set(w1, 2, 0, matrixW1[4]);
    gsl_matrix_set(w1, 2, 1, matrixW1[5]);

    gsl_matrix_set(w2, 0, 0, matrixW2[0]);
    gsl_matrix_set(w2, 0, 1, matrixW2[1]);
    gsl_matrix_set(w2, 0, 2, matrixW2[2]);
    gsl_matrix_set(w2, 1, 0, matrixW2[3]);
    gsl_matrix_set(w2, 1, 1, matrixW2[4]);
    gsl_matrix_set(w2, 1, 2, matrixW2[5]);

    printf("\n");
    printf("%.2f\n", gsl_matrix_get(w1, 0, 0));
    printf("%.2f\n", gsl_matrix_get(w1, 0, 1));
    printf("%.2f\n", gsl_matrix_get(w1, 1, 0));
    printf("%.2f\n", gsl_matrix_get(w1, 1, 1));
    printf("%.2f\n", gsl_matrix_get(w1, 2, 0));
    printf("%.2f\n", gsl_matrix_get(w1, 2, 1));

    printf("%.2f\n", gsl_matrix_get(w2, 0, 0));
    printf("%.2f\n", gsl_matrix_get(w2, 0, 1));
    printf("%.2f\n", gsl_matrix_get(w2, 0, 2));
    printf("%.2f\n", gsl_matrix_get(w2, 1, 0));
    printf("%.2f\n", gsl_matrix_get(w2, 1, 1));
    printf("%.2f\n", gsl_matrix_get(w2, 1, 2));

    // input read, feed forward
    gsl_blas_dgemv(CblasTrans, 1.0, w1, input, 0.0, hidden);
    //printf("\nhidden layer result: %d\n", );
    printf("\nHidden vector: [%.2f, %.2f]\n",
    gsl_vector_get(hidden, 0),
    gsl_vector_get(hidden, 1));

    gsl_blas_dgemv(CblasTrans, 1.0, w2, hidden, 0.0, output);
    printf("\nOutput vector: [%.2f, %.2f, %.2f]",
    gsl_vector_get(output, 0),
    gsl_vector_get(output, 1),
    gsl_vector_get(output, 2));

    printf("\nError: %.2f\n", error_func_total(expected_output, output));

    float diff = 0;
    //diff = sigmoid_derivative(error_func_total(expected_output, output));
    diff = gsl_vector_get(expected_output, 0) - gsl_vector_get(output, 0);
    printf("\ndiff %.2f\n", diff);
    printf("\nDiff: %.2f\n", diff*gsl_matrix_get(w2, 0, 0));
    printf("\nDiff: %.2f\n", diff*gsl_matrix_get(w2, 1, 0));
    diff = gsl_vector_get(expected_output, 1) - gsl_vector_get(output, 1);
    printf("\ndiff %.2f\n", diff);
    printf("\nDiff: %.2f\n", diff*gsl_matrix_get(w2, 0, 1));
    printf("\nDiff: %.2f\n", diff*gsl_matrix_get(w2, 1, 1));
    diff = gsl_vector_get(expected_output, 2) - gsl_vector_get(output, 2);
    printf("\ndiff %.2f\n", diff);
    printf("\nDiff: %.2f\n", diff*gsl_matrix_get(w2, 0, 2));
    printf("\nDiff: %.2f\n", diff*gsl_matrix_get(w2, 1, 2));

    return 0;
}



