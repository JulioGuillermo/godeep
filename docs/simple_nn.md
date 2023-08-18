# Simple neural network

## Crating model

```go
// Defining activation function
act := &activation.Tanh[float32]{}
// Creating a model
m := model.NewModel[float32]().
	Push(layer.Input[float32](2)).
	Push(layer.NewDense[float32](5, act)).
	Push(layer.NewDense[float32](5, act)).
	Push(layer.NewDense[float32](1, act))
```

## Creating a data set

We will teach the neural network the xor operation:

| A | B | xor(A, B) |
| :---: | :---: | :---: |
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

```go
// Creating inputs
inputs := []tensor.Tensor[float32]{
	tensor.NewFromValues[float32]([]float32{0, 0}, 2),
	tensor.NewFromValues[float32]([]float32{0, 1}, 2),
	tensor.NewFromValues[float32]([]float32{1, 0}, 2),
	tensor.NewFromValues[float32]([]float32{1, 1}, 2),
}
// Creating output
outputs := []tensor.Tensor[float32]{
	tensor.NewFromValues[float32]([]float32{0}, 1),
	tensor.NewFromValues[float32]([]float32{1}, 1),
	tensor.NewFromValues[float32]([]float32{1}, 1),
	tensor.NewFromValues[float32]([]float32{0}, 1),
}
```

## Test

We test the model before training

```go
// For each input
for i, in := range inputs {
	// Get the model output for the current input
	o, err := m.Predict(in)
	if err != nil {
		panic(err)
	}
	// print model output and target
	fmt.Println(o, outputs[i])
}
```

## Training

```go
fmt.Println("###########")
start := time.Now()
// Training the model
err := m.Train(inputs, outputs, 100_000, 0, 0.001, 0.4)
if err != nil {
	panic(err)
}
fmt.Println(time.Since(start))
fmt.Println("###########")
```

## Test

We test the model after training

```go
// For each input
for i, in := range inputs {
	// Get the model output for the current input
	o, err := m.Predict(in)
	if err != nil {
		panic(err)
	}
	// print model output and target
	fmt.Println(o, outputs[i])
}
```