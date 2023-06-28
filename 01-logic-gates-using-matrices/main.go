package main

import (
    "fmt" 
    "nn/main/nn"
)

func main() {
    var val0 int = 0
    var val1 int = 1
    var val2 int = 2
    gate := nn.GateCreate(
        nn.MatrixCreate(&val1, &val2),

        nn.MatrixCreate(&val2, &val2),
        nn.MatrixCreate(&val1, &val2),
        nn.MatrixCreate(&val1, &val2),

        nn.MatrixCreate(&val2, &val1),
        nn.MatrixCreate(&val1, &val1),
        nn.MatrixCreate(&val1, &val1),
    )

    gradient := nn.GateCreate(
        nn.MatrixCreate(&val1, &val2),

        nn.MatrixCreate(&val2, &val2),
        nn.MatrixCreate(&val1, &val2),
        nn.MatrixCreate(&val1, &val2),

        nn.MatrixCreate(&val2, &val1),
        nn.MatrixCreate(&val1, &val1),
        nn.MatrixCreate(&val1, &val1),
    )

    // or
    logic := []float64 {
        0, 0, 0,
        1, 0, 1,
        0, 1, 1,
        1, 1, 1,
    }

    // and
    logic = []float64 {
        0,0,0,
        1,0,0,
        0,1,0,
        1,1,1,
    }

    // nand
    logic = []float64 {
        0,0,1,
        1,0,1,
        0,1,1,
        1,1,0,
    }

    // nor
    logic = []float64 {
        0,0,1,
        1,0,0,
        0,1,0,
        1,1,0,
    }

    // xnor
    logic = []float64 {
        0, 0, 1,
        0, 1, 0,
        1, 0, 0,
        1, 1, 1,
    }

    // xor
    logic = []float64 {
        0, 0, 0,
        0, 1, 1,
        1, 0, 1,
        1, 1, 0,
    }

    gate.Expected = logic

    var stride int = 3
    var row int = int(len(gate.Expected)/stride)

    tiArr := nn.MatrixSlice(&gate.Expected,&row,&val2,&stride,&val0);
    toArr := nn.MatrixSlice(&gate.Expected,&row,&val1,&stride,&val2);

    ti := *(nn.MatrixCreate(&row, &val2))
    ti.Stride = stride
    ti.Samples = *tiArr
    to := *(nn.MatrixCreate(&row, &val1))
    to.Stride = stride
    to.Samples = *toArr

    var valf0 float64 = 0
    var valf1 float64 = 1

    gate.W1 = *nn.MatrixRandomize(&gate.W1, &valf0, &valf1)
    gate.B1 = *nn.MatrixRandomize(&gate.B1, &valf0, &valf1)
    gate.W2 = *nn.MatrixRandomize(&gate.W2, &valf0, &valf1)
    gate.B2 = *nn.MatrixRandomize(&gate.B2, &valf0, &valf1)

    var epsilon = 1e-1
    var rate = 1e-1

    for i := 0; i < 50*1000; i++ {
        nn.GateFiniteDiff(gate, gradient, &epsilon, &ti, &to);
        nn.GateLearn(gate, gradient, &rate);
    }

    for i := 0; i < 2; i++ {
        for j := 0; j < 2; j++ {
            gate.X.Samples[0] = float64(i) 
            gate.X.Samples[1] = float64(j) 
            gate = nn.GateForward(gate)
            y := gate.A2.Samples[0]
            fmt.Printf("%d ^ %d = %f\n", i, j, y)
        }
    }
}

