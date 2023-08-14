package layer_test

import (
	"testing"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/layer"
	"github.com/julioguillermo/godeep/model"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestDeconv2D(t *testing.T) {
	var err error
	act := &activation.Tanh[float32]{}
	m := model.NewModel[float32]().
		Push(layer.NewInput[float32](1, 1, 1)).
		// 1
		Push(layer.NewDeconv2D[float32](5, 3, 1, act)).
		// 3
		Push(layer.NewDeconv2D[float32](5, 3, 1, act)).
		// 5
		Push(layer.NewDeconv2D[float32](5, 3, 1, act)).
		// 7
		Push(layer.NewDeconv2D[float32](5, 3, 1, act)).
		// 9
		Push(layer.NewDeconv2D[float32](1, 2, 1, act))
	// 10

	err = m.Compile()
	if err != nil {
		t.Fatal(err)
	}

	outputs := []tensor.Tensor[float32]{
		tensor.NewFromValues[float32]([]float32{
			1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
		}, 1, 10, 10),
		tensor.NewFromValues[float32]([]float32{
			0, 0, 0, 0, 0, 0, 0, 0, 0, 1,
			0, 0, 0, 0, 0, 0, 0, 0, 1, 0,
			0, 0, 0, 0, 0, 0, 0, 1, 0, 0,
			0, 0, 0, 0, 0, 0, 1, 0, 0, 0,
			0, 0, 0, 0, 0, 1, 0, 0, 0, 0,
			0, 0, 0, 0, 1, 0, 0, 0, 0, 0,
			0, 0, 0, 1, 0, 0, 0, 0, 0, 0,
			0, 0, 1, 0, 0, 0, 0, 0, 0, 0,
			0, 1, 0, 0, 0, 0, 0, 0, 0, 0,
			1, 0, 0, 0, 0, 0, 0, 0, 0, 0,
		}, 1, 10, 10),
	}
	inputs := []tensor.Tensor[float32]{
		tensor.NewFromValues[float32]([]float32{0}, 1, 1, 1),
		tensor.NewFromValues[float32]([]float32{1}, 1, 1, 1),
	}

	err = m.Train(inputs, outputs, 10000, 0, 0.001, 0.4)
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
		if d > Threshold*10 {
			t.Fatalf("Prediction to far from target: %f", d)
		}
	}
}
