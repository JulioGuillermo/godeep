package layer_test

import (
	"testing"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/layer"
	"github.com/julioguillermo/godeep/model"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestSoftMax(t *testing.T) {
	var err error
	act := &activation.Tanh[float32]{}
	m := model.NewModel[float32]().
		Push(layer.NewInDense[float32](3, 20, act)).
		Push(layer.NewDense[float32](20, act)).
		Push(layer.NewDense[float32](3, act)).
		Push(layer.NewSoftMax[float32]())

	err = m.Compile()
	if err != nil {
		t.Fatal(err)
	}

	inputs := []tensor.Tensor[float32]{
		tensor.NewFromValues[float32]([]float32{0, 0, 0}, 3),
		tensor.NewFromValues[float32]([]float32{0, 0, 1}, 3),
		tensor.NewFromValues[float32]([]float32{0, 1, 0}, 3),
		tensor.NewFromValues[float32]([]float32{0, 1, 1}, 3),
		tensor.NewFromValues[float32]([]float32{1, 0, 0}, 3),
		tensor.NewFromValues[float32]([]float32{1, 0, 1}, 3),
		tensor.NewFromValues[float32]([]float32{1, 1, 0}, 3),
		tensor.NewFromValues[float32]([]float32{1, 1, 1}, 3),
	}
	outputs := []tensor.Tensor[float32]{
		tensor.NewFromValues[float32]([]float32{1, 0, 0}, 3),
		tensor.NewFromValues[float32]([]float32{0, 1, 0}, 3),
		tensor.NewFromValues[float32]([]float32{0, 1, 0}, 3),
		tensor.NewFromValues[float32]([]float32{0, 0, 1}, 3),
		tensor.NewFromValues[float32]([]float32{0, 1, 0}, 3),
		tensor.NewFromValues[float32]([]float32{0, 0, 1}, 3),
		tensor.NewFromValues[float32]([]float32{0, 0, 1}, 3),
		tensor.NewFromValues[float32]([]float32{0, 0, 1}, 3),
	}

	err = m.Train(inputs, outputs, 5000, 0, 0.001, 0.4)
	if err != nil {
		t.Fatal(err)
	}

	for i, in := range inputs {
		y := outputs[i]
		o, err := m.Predict(in)
		if err != nil {
			t.Fatal(err)
		}
		d, err := tools.Distant(y.GetData(), o.GetData())
		if err != nil {
			t.Fatal(err)
		}
		if d > Threshold {
			t.Fatalf("Prediction to far from target:\n    T %s\n    O %s", y.String(), o.String())
		}
	}
}
