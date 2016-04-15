#include <stdio.h>
#include <stdlib.h>
#include <gsl/gsl_matrix.h>

int main()
{
    int MATRIX_X = 28;
    int MATRIX_Y = 28;

sleep(5);

    FILE *trainingImagesp;
    FILE *imageLabelsp;

    unsigned int c;

    //trainingImagesp = fopen("train-images-idx3-ubyte", "r");
    //imageLabelsp = fopen("train-labels-idx1-ubyte", "r");
    trainingImagesp = fopen("t10k-images-idx3-ubyte", "r");
    imageLabelsp = fopen("t10k-labels-idx1-ubyte", "r");

    int header = 0;
    while(1)
    {
        c = fgetc(trainingImagesp);
        if( feof(trainingImagesp) )
        {
            break;
        }
        printf("%u ",c);
        if( header > 7 )
        {
            break;
        }
        header++;
    }

    int i=0, j=0, matrixIdx=0;
    gsl_matrix * m = gsl_matrix_alloc (MATRIX_X, MATRIX_Y);
//*
    while(1)
    {
        c = fgetc(trainingImagesp);
        if( feof(trainingImagesp) )
        {
            break;
        }
        //printf("%i, %i, %u", i, j, c);
        gsl_matrix_set (m, i, j, c);
        i++;
        j++;

        // print matrix
        if (i == 28 && matrixIdx > 9997)
        {
            printf("Matrix %d\n", matrixIdx);
            for (i = 0; i < MATRIX_X; i++)
            {
                for (j = 0; j < MATRIX_Y; j++)
                {
                    //printf ("m(%d,%d) = %g\n", i, j, gsl_matrix_get (m, i, j));
                    printf ("%u ", gsl_matrix_get (m, i, j));
                }
                printf ("\n");
            }
            matrixIdx++;
        }

        i%=MATRIX_X;
        j%=MATRIX_Y;

        //printf("%i", c);
    }
    /**/
    fclose(trainingImagesp);

    gsl_matrix_free (m);

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
