package nn

import (
    "fmt"
    "math"
    "math/rand"
)

type Matrix struct {
    rows int
    cols int
    Stride int
    Samples []float64
}

func MatrixCreate(numRows *int, numCols *int) *Matrix {
    arr := make([]float64, *numRows * *numCols)
    return &Matrix{rows: *numRows, cols: *numCols, Stride: *numCols, Samples: arr}
}

func MatrixRandomize(mat *Matrix, low *float64, high *float64) *Matrix {
    for i := 0; i < mat.rows; i++ {
        for j := 0; j < mat.cols; j++ {
            rnd := rand.Float64() 
            mat.Samples[mat.cols * i + j] = rnd * (*high - *low) + *low;
        }
    }
    return mat;
}

func MatrixMult(dst *Matrix, a *Matrix, b *Matrix) (*Matrix, error) {
    if a.cols != b.rows {
        return nil, fmt.Errorf("mult error 1: for param2 and param3, the rows do not match")
    }
    if dst.rows != a.rows || dst.cols != b.cols {
        return nil, fmt.Errorf("mult error 2: for params, either the rows of param1 and param2 or cols of param1 and param3 do not match")
    }

    var n int = a.cols
    for i := 0; i < dst.rows; i++ {
        for j := 0; j < dst.cols; j++ {
            dst.Samples[dst.cols*i+j] = 0
            for k := 0; k < n; k++ {
                dst.Samples[dst.cols*i+j] += a.Samples[a.cols*i+k] * b.Samples[b.cols*k+j]
            }
        }
    }
    return dst, nil
}

func MatrixSum(org *Matrix, other *Matrix) (*Matrix, error) {
    if other.rows != org.rows || other.cols != org.cols {
        return nil, fmt.Errorf("matrix sum error: other and org must have the same size")
    }
    for i := 0; i < other.rows; i++ {
        for j := 0; j < other.cols; j++ {
            org.Samples[other.cols * i + j] += other.Samples[other.cols * i + j];
        }
    }
    return org, nil
}

func MatrixFill(mat *Matrix, num *float64) *Matrix {
    for i := 0; i < mat.rows; i++ {
        for j := 0; j < mat.cols; j++ {
            mat.Samples[mat.cols * i + j] = *num
        }
    }
    return mat
}

func Sigmoidf(x float64) float64 {
    return 1.0 / (1.0 + math.Exp(-x))
}

func MatrixSigmoidf(mat *Matrix) *Matrix {
    for i := 0; i < mat.rows; i++ {
        for j := 0; j < mat.cols; j++ {
            mat.Samples[mat.cols * i +j] = Sigmoidf(mat.Samples[mat.cols * i +j])
        }
    }
    return mat
}

func MatrixRow(mat *Matrix, row *int) *Matrix {
    startIndex := *row * mat.cols
    endIndex := startIndex + mat.cols
    rowSamples := mat.Samples[startIndex:endIndex]
    return &Matrix{rows: 1, cols: mat.cols, Stride: mat.cols, Samples: rowSamples} 
}

func MatrixCopy(dst *Matrix, src *Matrix) (*Matrix, error) {
    if dst.rows != src.rows || dst.cols != src.cols {
        return nil, fmt.Errorf("copy error: matrices don't match") 
    }
    dst.Samples = make([]float64, len(src.Samples))
    copy(dst.Samples, src.Samples)
    return dst, nil
}

func MatrixSlice(arr *[]float64, rows *int, cols *int, step *int, start *int) *[]float64 {
    // if (arr == undefined) throw new Error("array is undefined");
    temp := []float64{} 
    index := *start 
    for i := 0; i < *rows; i++ {
        for j := 0; j < *cols; j++ {
            if (index < len(*arr)) {
                temp = append(temp, (*arr)[index])
                index++;
            }
        }
        index += *step - *cols; 
    }
    return &temp;
}
