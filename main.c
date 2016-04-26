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

void randomizeMatrix(gsl_matrix_float * m, int x, int y) {
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
            gsl_matrix_float_set (m, i, j, u);
            printf ("%.5f\n", u);
        }
    }

    gsl_rng_free (r);
}

int main()
{
    int MATRIX_X = 28;
    int MATRIX_Y = 28;
    int INPUT_VECTOR_SIZE = 28*28;
    int HIDDEN_VECTOR_SIZE = 15;
    int OUTPUT_VECTOR_SIZE = 10;

    FILE *trainingImagesp;
    FILE *imageLabelsp;

    unsigned char c;

    //trainingImagesp = fopen("train-images-idx3-ubyte", "r");
    //imageLabelsp = fopen("train-labels-idx1-ubyte", "r");
    trainingImagesp = fopen("t10k-images-idx3-ubyte", "r");
    imageLabelsp = fopen("t10k-labels-idx1-ubyte", "r");

    sleep(2);

    // Reading images header
    uint32_t op;
    int header = 0;
    while(1)
    {
        //fread(&c,sizeof(c),1,trainingImagesp);
        fread(&op, sizeof(op), 1, trainingImagesp);
        //c = fgetc(trainingImagesp);
        if( feof(trainingImagesp) )
        {
            break;
        }

        op = msbFirstToLsbFirst(op);
        printf("Label %u - %u ",header,op);
        /*
        d or i	Signed decimal integer	392
        u	Unsigned decimal integer	7235
        o	Unsigned octal	610
        x	Unsigned hexadecimal integer	7fa
        X	Unsigned hexadecimal integer (uppercase)	7FA
        f	Decimal floating point, lowercase	392.65
        F	Decimal floating point, uppercase	392.65
        e	Scientific notation (mantissa/exponent), lowercase	3.9265e+2
        E	Scientific notation (mantissa/exponent), uppercase	3.9265E+2
        g	Use the shortest representation: %e or %f	392.65
        G	Use the shortest representation: %E or %F	392.65
        a	Hexadecimal floating point, lowercase	-0xc.90fep-2
        A	Hexadecimal floating point, uppercase	-0XC.90FEP-2
        c	Character	a
        s	String of characters	sample
        p	Pointer address	b8000000
        /**/
        if( header == 3 )
        {
            break;
        }
        header++;
    }

    header = 0;

    // Reading labels header
    while(1)
    {
        fread(&op, sizeof(op), 1, imageLabelsp);
        //c = fgetc(imageLabelsp);
        if( feof(imageLabelsp) )
        {
            break;
        }

        op = msbFirstToLsbFirst(op);
        printf("\n\n");
        printf("Label %u - %u ",header,op);
        /*
        d or i	Signed decimal integer	392
        u	Unsigned decimal integer	7235
        o	Unsigned octal	610
        x	Unsigned hexadecimal integer	7fa
        X	Unsigned hexadecimal integer (uppercase)	7FA
        f	Decimal floating point, lowercase	392.65
        F	Decimal floating point, uppercase	392.65
        e	Scientific notation (mantissa/exponent), lowercase	3.9265e+2
        E	Scientific notation (mantissa/exponent), uppercase	3.9265E+2
        g	Use the shortest representation: %e or %f	392.65
        G	Use the shortest representation: %E or %F	392.65
        a	Hexadecimal floating point, lowercase	-0xc.90fep-2
        A	Hexadecimal floating point, uppercase	-0XC.90FEP-2
        c	Character	a
        s	String of characters	sample
        p	Pointer address	b8000000
        /**/
        if( header == 1 )
        {
            break;
        }
        header++;
    }

    printf("\n");
    int i=0, j=0, vec=0, matrixIdx=0;
    //gsl_matrix * m = gsl_matrix_alloc (MATRIX_X, MATRIX_Y);

    gsl_vector * input = gsl_vector_alloc(INPUT_VECTOR_SIZE);
    gsl_matrix_float * w1 = gsl_matrix_float_alloc(INPUT_VECTOR_SIZE, HIDDEN_VECTOR_SIZE);
    gsl_vector * hidden = gsl_vector_alloc(HIDDEN_VECTOR_SIZE);
    gsl_matrix_float * w2 = gsl_matrix_float_alloc (HIDDEN_VECTOR_SIZE, OUTPUT_VECTOR_SIZE);
    gsl_vector * output = gsl_vector_alloc(OUTPUT_VECTOR_SIZE);

    // init weights
    randomizeMatrix(w1, INPUT_VECTOR_SIZE, HIDDEN_VECTOR_SIZE);
    randomizeMatrix(w2, HIDDEN_VECTOR_SIZE, OUTPUT_VECTOR_SIZE);

    printf("\n%f\n", gsl_matrix_float_get (w1, INPUT_VECTOR_SIZE-1, HIDDEN_VECTOR_SIZE-1));
    printf("\n%f\n", gsl_matrix_float_get (w2, HIDDEN_VECTOR_SIZE-1, OUTPUT_VECTOR_SIZE-1));

    //*
    while(1)
    {
        fread(&c, sizeof(c), 1, trainingImagesp);
        if( feof(trainingImagesp) )
        {
            break;
        }

        //printf("i %d, j %d, mID %d\n", i,j, matrixIdx);

        //gsl_matrix_set (m, i, j, c);
        gsl_vector_set(input, vec, c);

        j++;
        vec++;
        vec%=INPUT_VECTOR_SIZE;

        if(j==28) i++;

        // print matrix
        if (i == 28)
        {
            fread(&c, sizeof(c), 1, imageLabelsp);
            if( feof(imageLabelsp) )
            {
                break;
            }

            /*
            //if(matrixIdx < 10) {
            printf("\nMatrix %u - %u\n", matrixIdx, c);
                for (i = 0; i < MATRIX_X; i++)
                {
                    for (j = 0; j < MATRIX_Y; j++)
                    {
                        if((char)gsl_matrix_get (m, i, j) > 0)
                            printf ("%03u", (char)gsl_matrix_get (m, i, j));
                        else
                            printf("   ");
                    }
                    printf ("\n");
                }

            //}
            /**/
            // input read, feed forward
            printf("\nhidden layer result: %d\n", gsl_blas_dgemv(CblasTrans, 1, w1, input, 1, hidden));
            printf("\noutput layer result: %d\n", gsl_blas_dgemv(CblasTrans, 1, w2, hidden, 1, output));

            printf("\n[%.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f, %.2f]",
            gsl_vector_float_get(output, 0),
            gsl_vector_float_get(output, 1),
            gsl_vector_float_get(output, 2),
            gsl_vector_float_get(output, 3),
            gsl_vector_float_get(output, 4),
            gsl_vector_float_get(output, 5),
            gsl_vector_float_get(output, 6),
            gsl_vector_float_get(output, 7),
            gsl_vector_float_get(output, 8),
            gsl_vector_float_get(output, 9));
            matrixIdx++;
        }

        i%=MATRIX_X;
        j%=MATRIX_Y;
    }

    printf("\ni %d, j %d, mID %d\n", i,j, matrixIdx);

    /**/
    fclose(trainingImagesp);

    //gsl_matrix_free (m);
    gsl_vector_free(input);

    printf("matrixIdx %d", matrixIdx);

/*
    int labelCount = 0;
    while(1)
    {
        c = fgetc(imageLabelsp);

        if( feof(imageLabelsp) )
        {
            break;
        }
        printf("%i", c);
        labelCount++;
    }
    printf ("\nlength of labels = %zu\n", labelCount);
    /**/
    fclose(imageLabelsp);


    //gsl_block * b = gsl_block_alloc (100);

    //printf ("length of block = %zu\n", b->size);
    //printf ("block data address = %p\n", b->data);
    //gsl_block_free (b);

    return 0;
}


