#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <gsl/gsl_matrix.h>

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

int main()
{
    int MATRIX_X = 28;
    int MATRIX_Y = 28;

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
    int i=0, j=0, matrixIdx=0;
    gsl_matrix * m = gsl_matrix_alloc (MATRIX_X, MATRIX_Y);
    //*
    while(1)
    {
        fread(&c, sizeof(c), 1, trainingImagesp);
        if( feof(trainingImagesp) )
        {
            break;
        }

        //printf("i %d, j %d, mID %d\n", i,j, matrixIdx);

        gsl_matrix_set (m, i, j, c);
        j++;
        if(j==28) i++;

        // print matrix
        if (i == 28)
        {
            fread(&c, sizeof(c), 1, imageLabelsp);
            if( feof(imageLabelsp) )
            {
                break;
            }

            //*
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
            matrixIdx++;
        }

        i%=MATRIX_X;
        j%=MATRIX_Y;
    }

    printf("\ni %d, j %d, mID %d\n", i,j, matrixIdx);

    /**/
    fclose(trainingImagesp);

    gsl_matrix_free (m);

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


