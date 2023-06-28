package nn

import (
    "fmt"
)

type Gate struct {
    X  Matrix  

    W1 Matrix
    B1 Matrix
    a1 Matrix
             
    W2 Matrix
    B2 Matrix
    A2 Matrix

    Expected []float64
}

func GateCreate(x *Matrix, w1 *Matrix, b1 *Matrix, a1 *Matrix, w2 *Matrix, b2 *Matrix, a2 *Matrix) *Gate {
    return &Gate{
        X: *x, W1: *w1, B1: *b1, a1: *a1, W2: *w2, B2: *b2, A2: *a2,
    }
}

func GateLoss(gate *Gate, ti *Matrix, to *Matrix) (*float64, error) {
    if ti.rows != to.rows || to.cols != gate.A2.cols {
        return nil, fmt.Errorf("loss error: ti.rows != to.rows || to.cols != gate.A2.cols")
    }
    var result float64 = 0 
    for i := 0; i < ti.rows; i++ {
        rowMatrix1 := MatrixRow(ti, &i) 
        rowMatrix2 := MatrixRow(to, &i) 

        val, err := MatrixCopy(&gate.X, rowMatrix1)
        if err != nil {
            fmt.Println("Error in GateLoss:", err)
            return nil, err
        }
        gate.X = *val
        gate = GateForward(gate) 

        for j := 0; j < to.cols; j++ {
            dist := gate.A2.Samples[gate.A2.cols * 0 + j] - rowMatrix2.Samples[rowMatrix2.cols * 0 + j] 
            result += (dist * dist) 
        }
    }
    result /= float64(ti.rows)
    return &result, nil
}

func GateLearn(gate *Gate, gradient *Gate, rate *float64){
    for i := 0; i < gate.W1.rows; i++ {
        for j := 0; j < gate.W1.cols; j++ {
            gate.W1.Samples[gate.W1.cols * i + j] -= gradient.W1.Samples[gradient.W1.cols * i + j] * *rate 
        }
    }

    for i := 0; i < gate.B1.rows; i++ {
        for j := 0; j < gate.B1.cols; j++ {
            gate.B1.Samples[gate.B1.cols * i + j] -= gradient.B1.Samples[gradient.B1.cols * i + j] * *rate 
        }
    }

    for i := 0; i < gate.W2.rows; i++ {
        for j := 0; j < gate.W2.cols; j++ {
            gate.W2.Samples[gate.W2.cols * i + j] -= gradient.W2.Samples[gradient.W2.cols * i + j] * *rate 
        }
    }

    for i := 0; i < gate.B2.rows; i++ {
        for j := 0; j < gate.B2.cols; j++ {
            gate.B2.Samples[gate.B2.cols * i + j] -= gradient.B2.Samples[gradient.B2.cols * i + j] * *rate 
        }
    }
}

func GateFiniteDiff(gate *Gate, gradient *Gate, epsilon *float64, ti *Matrix, to *Matrix) {
    var saved float64

    loss, err := GateLoss(gate, ti, to)
    if err != nil {
        fmt.Println("Error in GateLoss:", err)
        return
    }

    for i := 0; i < gate.W1.rows; i++ {
        for j := 0; j < gate.W1.cols; j++ {
            saved = gate.W1.Samples[gate.W1.cols * i + j]
            gate.W1.Samples[gate.W1.cols * i + j] += *epsilon
            newLoss, err := GateLoss(gate, ti, to)
            if err != nil {
                fmt.Println("Error in GateFiniteDiff:", err)
                return
            }
            gradient.W1.Samples[gradient.W1.cols * i + j] = (*newLoss - *loss) / *epsilon
            gate.W1.Samples[gate.W1.cols * i + j] = saved
        }
    }

    for i := 0; i < gate.B1.rows; i++ {
        for j := 0; j < gate.B1.cols; j++ {
            saved = gate.B1.Samples[gate.B1.cols * i + j]
            gate.B1.Samples[gate.B1.cols * i + j] += *epsilon
            newLoss, err := GateLoss(gate, ti, to)
            if err != nil {
                fmt.Println("Error in GateLoss:", err)
                return
            }
            gradient.B1.Samples[gradient.B1.cols * i + j] = (*newLoss - *loss) / *epsilon
            gate.B1.Samples[gate.B1.cols * i + j] = saved
        }
    }

    for i := 0; i < gate.W2.rows; i++ {
        for j := 0; j < gate.W2.cols; j++ {
            saved = gate.W2.Samples[gate.W2.cols * i + j]
            gate.W2.Samples[gate.W2.cols * i + j] += *epsilon
            newLoss, err := GateLoss(gate, ti, to)
            if err != nil {
                fmt.Println("Error in GateLoss:", err)
                return
            }
            gradient.W2.Samples[gradient.W2.cols * i + j] = (*newLoss - *loss) / *epsilon
            gate.W2.Samples[gate.W2.cols * i + j] = saved
        }
    }

    for i := 0; i < gate.B2.rows; i++ {
        for j := 0; j < gate.B2.cols; j++ {
            saved = gate.B2.Samples[gate.B2.cols * i + j]
            gate.B2.Samples[gate.B2.cols * i + j] += *epsilon
            newLoss, err := GateLoss(gate, ti, to)
            if err != nil {
                fmt.Println("Error in GateLoss:", err)
                return
            }
            gradient.B2.Samples[gradient.B2.cols * i + j] = (*newLoss - *loss) / *epsilon
            gate.B2.Samples[gate.B2.cols * i + j] = saved
        }
    }
}

func GateForward(gate *Gate) *Gate {
    val, err := MatrixMult(&gate.a1, &gate.X, &gate.W1)
    if err != nil {
        fmt.Println("Error in GateForward:", err)
        return nil
    }
    gate.a1 = *val
    val, err = MatrixSum(&gate.a1, &gate.B1)
    if err != nil {
        fmt.Println("Error in GateForward:", err)
        return nil
    }
    gate.a1 = *val
    val = MatrixSigmoidf(&gate.a1)
    gate.a1 = *val

    val, err = MatrixMult(&gate.A2, &gate.a1, &gate.W2)
    if err != nil {
        fmt.Println("Error in GateForward:", err)
        return nil
    }
    gate.A2 = *val
    val, err = MatrixSum(&gate.A2, &gate.B2)
    if err != nil {
        fmt.Println("Error in GateForward:", err)
        return nil
    }
    gate.A2 = *val
    val = MatrixSigmoidf(&gate.A2)
    gate.A2 = *val
    
    return gate
}

