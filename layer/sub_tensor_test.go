package layer_test

import (
	"testing"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/layer"
	"github.com/julioguillermo/godeep/model"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
)

func TestSubTensor(t *testing.T) {
	var err error
	act := &activation.Tanh[float32]{}

	inL := layer.NewInput[float32](2)

	d1 := layer.NewDense[float32](20, act)
	d1.Connect(inL)

	s1 := layer.NewSubTensor[float32](0, 0, 10)
	s1.Connect(d1)
	s2 := layer.NewSubTensor[float32](0, 10, 20)
	s2.Connect(d1)

	x1 := layer.NewDense[float32](10, act)
	x1.Connect(s1)

	y1 := layer.NewDense[float32](10, act)
	y1.Connect(s2)

	c := layer.NewConcat[float32](x1, y1, 0)
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
