package layer_test

import (
	"testing"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/layer"
	"github.com/julioguillermo/godeep/model"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestConcat(t *testing.T) {
	var err error
	act := &activation.Sin[float32]{}

	inL := layer.NewInput[float32](2)

	x1 := layer.NewDense[float32](10, act)
	x1.Connect(inL)
	x2 := layer.NewDense[float32](10, act)
	x2.Connect(x1)
	x3 := layer.NewDense[float32](10, act)
	x3.Connect(x2)

	y1 := layer.NewDense[float32](10, act)
	y1.Connect(inL)
	y2 := layer.NewDense[float32](10, act)
	y2.Connect(y1)
	y3 := layer.NewDense[float32](10, act)
	y3.Connect(y2)

	c := layer.NewConcat[float32](x3, y3, 0)
	outL := layer.NewDense[float32](1, act)
	outL.Connect(c)

	m, err := model.FromInOut[float32](inL, outL)
	if err != nil {
		t.Fatal(err)
	}
	err = m.Compile()
	if err != nil {
		t.Fatal(err)
	}

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

	err = m.Train(inputs, outputs, 1000, 0, 0.001, 0.4)
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
