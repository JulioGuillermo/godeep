package layer_test

import (
	"testing"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/layer"
	"github.com/julioguillermo/godeep/model"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestConv(t *testing.T) {
	var err error
	act := &activation.Sin[float32]{}
	m := model.NewModel[float32]().
		Push(layer.NewInput[float32](1, 10, 10)).
		Push(layer.NewConv2D[float32](10, 3, 2, act)).
		Push(layer.NewConv2D[float32](20, 3, 2, act)).
		Push(layer.NewDense[float32](1, act))

	err = m.Compile()
	if err != nil {
		t.Fatal(err)
	}

	inputs := []tensor.Tensor[float32]{
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
	outputs := []tensor.Tensor[float32]{
		tensor.NewFromValues[float32]([]float32{0}, 1),
		tensor.NewFromValues[float32]([]float32{1}, 1),
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
