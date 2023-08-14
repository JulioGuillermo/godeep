# GODEEP

This is a library to provide a tensor system to make complex operations through a computational graph.

It also provides a model and differents layers to make neural networks.

## Tensor system

In the tensor system you can store all kind of vectors or multidimensional matrix through the TensorMat type. And you can make many operations on them.

You can also specify the tensor data type, it support float32, float64, int16, int32 and int64.

### Vectors

There are different ways to create a vector, here are some examples:

```go
// Creating a vector with the given 4 values
vector := tensor.NewFromValues[float32]([]float32{1, 2, 3, 4}, 4)

// Creating a vector of 10 zeros
vector := tensor.NewZero[float32](10)

// Creating a vector of 10 ones
vector := tensor.NewOne[float32](10)
```

### Matrix

Creating a matrix is just like creating a vector, but with more dimensions:

```go
// Creating a 2x2 matrix with values
vector := tensor.NewFromValues[float32]([]float32{1, 2, 3, 4}, 2, 2)

// Creating a 1x2x3 matrix of zeros
vector := tensor.NewZero[float32](1, 2, 3)

// Creating a 3x2 matrix of ones
vector := tensor.NewOne[float32](3, 2)
```

### Tensor constructors

- `NewZeros(Shape) Tensor`
- `NewOnes(Shape) Tensor`
- `NewRand(Min, Max, Shape) Tensor`
- `NewNormRand(Shape) Tensor`
- `NewFromValues(Values, Shape) Tensor`

### Scalar constructors

- `NewScalar(Value) Scalar`
- `NewZero() Scalar`
- `NewOne() Scalar`
- `NewRand(Min, Max) Scalar`
- `NewNorm() Scalar`

### Tensor operations

With the tensor you can make a lot of operations in a very efficient way.

```go
// Make a 3x3 matrix with values
m1 := tensor.NewFromValues[float32]([]float32{
 1, 2, 3,
 9, 8, 7,
 4, 6, 5,
}, 3, 3)

// Make a 3x3 matrix with values
m2 := tensor.NewFromValues[float32]([]float32{
 9, 8, 7,
 4, 6, 5,
 1, 2, 3,
}, 3, 3)

// Let's perform some mathematical operations on the matrices
add := tensor.Add[float32](m1, m2)
// Note: this mul operation is not a matrix multiplication or dot product
// is a simple multiplication of each component of the matrices
// if you need dot product, please use tensor.Dot
mul := tensor.Mul[float32](m1, m2)
sub := tensor.Sub(mul, add)

// Build the computational graph to execute all the operations.
// Note: we just pass the result tensor of all operations (the last tensor)
g, err := graph.NewGraph(sub)
if err != nil {
 log.Fatal(err)
}

// Execute all the operations...
g.Exec()
// Print the result: <Mat (dims: 2, len: 9, shape: [3, 3]) => [-1, 6, 11, 23, 34, 23, -1, 4, 7]>
// 3x3 Matrix
// -1   6  11
// 23  34  23
// -1   4   7
log.Println(sub)
```

#### Implemented tensors operations

- `Abs(Tensor) Tensor`
- `Activate(Tensor, Function) Tensor`
- `Concat(Tensor, Tensor, Dim) Tensor`
- `CopyTo(Tensor, Tensor) Tensor`
- `Transfer(Context, Tensor, Tensor) error`
- `Dot(Tensor, Tensor) Tensor`
- `DotAt(Tensor, Tensor, DimA, DimB) Tensor`
- `FillWith(Tensor, Scalar) Tensor`
- `Fill(Context, Tensor, Scalar) error`
- `Add(Tensor, Tensor) Tensor`
- `Sub(Tensor, Tensor) Tensor`
- `Mul(Tensor, Tensor) Tensor`
- `Div(Tensor, Tensor) Tensor`
- `AddScalar(Tensor, Scalar) Tensor`
- `SubScalar(Tensor, Scalar) Tensor`
- `MulScalar(Tensor, Scalar) Tensor`
- `DivScalar(Tensor, Scalar) Tensor`
- `Sum(Tensor) Tensor`
- `Avg(Tensor) Tensor`
- `Max(Tensor) Tensor`
- `Min(Tensor) Tensor`
- `DSum(Tensor, Dim) Tensor`
- `DAvg(Tensor, Dim) Tensor`
- `DMax(Tensor, Dim) Tensor`
- `DMin(Tensor, Dim) Tensor`
- `Neg(Tensor) Tensor`
- `Reshape(Tensor, Shape) Tensor`
- `Flatten(Tensor) Tensor`
- `SoftMax(Tensor) Tensor`
- `SubTensor(Tensor, Dim, From, To) Tensor`
- `SubExtendedTensor(Tensor, Dim, From, To) Tensor`
- `Transpose(Tensor) Tensor`

## Neural Network

You can also create neural networks with this library.

```go
// Select an activation
// you can use different activation functions for different layers
act := &activation.Tanh[float32]{}

// Make a model and set layers
m := model.NewModel[float32]().
 Push(layer.NewInDense[float32](2, 5, act)).
 Push(layer.NewDense[float32](5, act)).
 Push(layer.NewDense[float32](1, act))

// Get inputs and outputs
inputs := []tensor.Tensor[float32]{
 tensor.NewFromValues[float32]([]float32{0, 0}, 2),
 tensor.NewFromValues[float32]([]float32{0, 1}, 2),
 tensor.NewFromValues[float32]([]float32{1, 0}, 2),
 tensor.NewFromValues[float32]([]float32{1, 1}, 2),
}
outputs := []tensor.Tensor[float32]{
 tensor.NewFromValues[float32]([]float32{0}, 1),
 tensor.NewFromValues[float32]([]float32{1}, 1),
 tensor.NewFromValues[float32]([]float32{1}, 1),
 tensor.NewFromValues[float32]([]float32{0}, 1),
}

// Test the model before training
for i, in := range inputs {
 o, err := m.Predict(in)
 if err != nil {
  panic(err)
 }
 fmt.Println(o, outputs[i])
}

// Training the model
fmt.Println("###########")
start := time.Now() // Let's see how long it take
err := m.Train(inputs, outputs, 100_000, 0, 0.001, 0.4)
if err != nil {
 panic(err)
}
fmt.Println(time.Since(start)) // It must take about a second
fmt.Println("###########")

// Test the model after training
for i, in := range inputs {
 o, err := m.Predict(in)
 if err != nil {
  panic(err)
 }
 fmt.Println(o, outputs[i])
}

```

#### Implemented layers type

- Concat
- Conv2D
- Deconv2D
- Dense
- Flatten
- Input
- MaxPool
- Norm
- ENorm
- SoftMax
- SubTensor

##### Other layer types to be implemented in the future

This list can change.

- UpSampling
- Recurrent
- LSTM
- Attention
- Embedding
