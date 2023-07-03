package main

import (
    "fmt" 
    "nn/main/nn"
)

func main() {
    gate := nn.GateCreate(
        nn.MatrixCreate(1, 2),

        nn.MatrixCreate(2, 2),
        nn.MatrixCreate(1, 2),
        nn.MatrixCreate(1, 2),

        nn.MatrixCreate(2, 1),
        nn.MatrixCreate(1, 1),
        nn.MatrixCreate(1, 1),
    )

    gradient := nn.GateCreate(
        nn.MatrixCreate(1, 2),

        nn.MatrixCreate(2, 2),
        nn.MatrixCreate(1, 2),
        nn.MatrixCreate(1, 2),

        nn.MatrixCreate(2, 1),
        nn.MatrixCreate(1, 1),
        nn.MatrixCreate(1, 1),
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

    tiArr := nn.MatrixSlice(gate.Expected,row,2,stride,0);
    toArr := nn.MatrixSlice(gate.Expected,row,1,stride,2);

    ti := nn.MatrixCreate(row, 2)
    ti.Stride = stride
    ti.Samples = tiArr
    to := nn.MatrixCreate(row, 1)
    to.Stride = stride
    to.Samples = toArr

    gate.W1 = *nn.MatrixRandomize(&gate.W1, 0.0, 1.0)
    gate.B1 = *nn.MatrixRandomize(&gate.B1, 0.0, 1.0)
    gate.W2 = *nn.MatrixRandomize(&gate.W2, 0.0, 1.0)
    gate.B2 = *nn.MatrixRandomize(&gate.B2, 0.0, 1.0)

    var epsilon = 1e-1
    var rate = 1e-1

    for i := 0; i < 50*1000; i++ {
        nn.GateFiniteDiff(&gate, &gradient, &epsilon, &ti, &to)
        nn.GateLearn(&gate, &gradient, &rate);
    }

    for i := 0; i < 2; i++ {
        for j := 0; j < 2; j++ {
            gate.X.Samples[0] = float64(i) 
            gate.X.Samples[1] = float64(j) 
            gate = *nn.GateForward(&gate)
            y := gate.A2.Samples[0]
            fmt.Printf("%d ^ %d = %f\n", i, j, y)
        }
    }
}


