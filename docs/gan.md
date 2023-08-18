# GAN

Lets create a gan neural network

## Data set

```go
// Function to load a dataset from a directory
func LoadDS(p string, W, H uint) []tensor.Tensor[float32] {
	// Read dataset directory
	res, err := os.ReadDir(p)
	if err != nil {
		panic(err)
	}

	// Make data set
	imgs := make([]tensor.Tensor[float32], len(res))
	for i, r := range res {
		// For each image in the data set directory
		// print it's name
		fmt.Println(p)
		// load the image
		imgs[i], err := img.LoadSmoothScaled[float32](path.Join(p, r.Name()), W, H)
		if err != nil {
			panic(err)
		}
	}

	return imgs
}
```

## Model

### Generator

```go
// Function to create generator model
func GetGenerator() *model.Model[float32] {
	// Define activation function
	var act = &activation.Sigmoid[float32]{}
	// Create the model
	m := model.NewModel[float32]().
		Push(layer.NewInput[float32](10, 1, 1)).
		// 10x1x1
		Push(layer.NewDeconv2D[float32](10, 5, 1, act)).
		// 10x5x5
		Push(layer.NewUpSampling2D[float32](2)).
		// 10x10x10
		Push(layer.NewDeconv2D[float32](10, 3, 2, act)).
		// 10x21x21
		Push(layer.NewDeconv2D[float32](4, 3, 2, act))
	// 4x43x43
	// Cal each Deconv output size:
	// out_size_width = (input_size_width - 1) * stride_width + kernel_width
	// out_size_height = (input_size_height - 1) * stride_height + kernel_height
	return m
}
```

### Discriminator

```go
// Function to create discriminator model
func GetDiscriminator() *model.Model[float32] {
	// Define activation function
	var act = &activation.Sigmoid[float32]{}
	// Create the model
	m := model.NewModel[float32]().
		Push(layer.NewInput[float32](4, 43, 43)).
		// 4x43x43 <- Generator model output
		Push(layer.NewConv2D[float32](10, 3, 2, act)).
		// 10x21x21
		Push(layer.NewConv2D[float32](10, 3, 2, act)).
		// 10x10x10
		Push(layer.NewMaxPool2D[float32](2)).
		// 10x5x5
		Push(layer.NewConv2D[float32](10, 5, 1, act)).
		// 10x1x1
		Push(layer.NewDense[float32](1, act))
	// 1
	// Cal each Conv output size:
	// out_size_width = (input_size_width - kernel_width) / stride_width + 1
	// out_size_height = (input_size_height - kernel_height) / stride_height + 1
	return m
}
```

### Model

```go
// Function to connect generator and discriminator into an unique model
func GetModels(g, d *model.Model[float32]) *model.Model[float32] {
	// Create a model connecting the generator and discriminator
	m, err := model.FromModels[float32](g, d)
	if err != nil {
		panic(err)
	}

	return m
}
```

## Save

### Save model

```go
// Function to save the model
func SaveModel(m *model.Model[float32]) {
	// Prepare the model to be saved
	m.ResetSave()
	// Create the file to save the model
	file, err := os.Create("gan")
	if err != nil {
		panic(err)
	}
	defer file.Close()

	// Save model weights to the created file
	m.Save(file)
}
```

### Load model

```go
// Function so load the model
func LoadModel(m *model.Model[float32]) {
	// Open the file with the saved weights
	file, err := os.Open("gan")
	if err != nil {
		log.Println(err)
		return
	}
	defer file.Close()

	// Prepare the model to load weights
	m.ResetLoad()
	// Load the weights to the model
	m.Load(file)
}
```

## Main

### Some parameters...

```go
const (
	TrainBatch   = 300
	Epochs       = 10
	LearningRate = 0.001
	Momentum     = 0.4
)
```

### Main function

```go
func main() {
	// Create generator and discriminator
	fmt.Println("Creating generator and discriminator")
	start := time.Now()
	g := GetGenerator()
	d := GetDiscriminator()
	fmt.Println("Duration:", time.Since(start))

	// Create the connected model
	fmt.Println("\nBuilding GAN")
	start = time.Now()
	m := GetModels(g, d)
	fmt.Println("Duration:", time.Since(start))

	// Load saved weights
	fmt.Println("\nLoading saved weights")
	start = time.Now()
	LoadModel(m)
	fmt.Println("Duration:", time.Since(start))

	inShape := g.GetInputs().GetShape()
	outShape := g.GetOutputs().GetShape()
	disShape := d.GetOutputs().GetShape()

	// Load data set
	fmt.Println("\nLoading images")
	start = time.Now()
	imgs := LoadDS("dataset_directory", outShape[1], outShape[2])
	numImgs := len(imgs)
	fmt.Println("Duration:", time.Since(start))

	dInputs := make([]tensor.Tensor[float32], TrainBatch*2)
	dTargets := make([]tensor.Tensor[float32], TrainBatch*2)

	gInputs := make([]tensor.Tensor[float32], TrainBatch)
	gTargets := make([]tensor.Tensor[float32], TrainBatch)

	// Training
	for {
		// Training discriminator
		fmt.Println("\nCreating discriminator dataset...")
		start = time.Now()
		// Getting inputs and target for discriminator
		for i := 0; i < TrainBatch; i++ {
			// Good input
			randimg := rand.Intn(numImgs)
			dInputs[i*2] = imgs[randimg]

			// Fake input
			randIn := tensor.NewRand[float32](0, 1, inShape...)
			out, err := g.Predict(randIn)
			if err != nil {
				panic(err)
			}
			dInputs[i*2+1] = out

			// Good target
			dTargets[i*2] = tensor.NewOnes[float32](disShape...)
			// Bad target for the fake input
			dTargets[i*2+1] = tensor.NewZeros[float32](disShape...)
		}
		fmt.Println("Duration:", time.Since(start))

		// Training discriminator
		fmt.Println("\nTraining discriminator...")
		start = time.Now()
		// Set trainable to true
		d.SetTrainable(true)
		// Training discriminator
		err := d.Train(dInputs, dTargets, Epochs, 0, LearningRate, Momentum)
		if err != nil {
			panic(err)
		}
		// Set trainable to false
		d.SetTrainable(false)
		fmt.Println("Duration:", time.Since(start))

		// Training generator
		fmt.Println("\nCreating generator dataset...")
		start = time.Now()
		// Getting input and targets for gan model
		for i := 0; i < TrainBatch; i++ {
			gInputs[i] = tensor.NewRand[float32](0, 1, inShape...)
			gTargets[i] = tensor.NewOnes[float32](disShape...)
		}
		fmt.Println("Duration:", time.Since(start))

		// Training generator
		fmt.Println("\nTraining generator...")
		start = time.Now()
		// The gan model do not fit the discriminator weights because
		// discriminator trainable is setted to false
		err = m.Train(gInputs, gTargets, Epochs*2, 0, LearningRate, Momentum)
		if err != nil {
			panic(err)
		}
		fmt.Println("Duration:", time.Since(start))

		// Save a backup
		fmt.Println("\nSaving...")
		start = time.Now()
		SaveModel(m)
		fmt.Println("Duration:", time.Since(start))

		// Save a generated image
		fmt.Println("\nSaving test img...")
		start = time.Now()
		// Generate a random input
		randIn := tensor.NewRand[float32](0, 1, inShape...)
		// Generate an image with the generator model
		out, err := g.Predict(randIn)
		if err != nil {
			panic(err)
		}
		// Save the image
		img.JpgSave(out, "test.jpg", 100)
		fmt.Println("Duration:", time.Since(start))
	}
}
```