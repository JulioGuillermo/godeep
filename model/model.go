package model

import (
	"math/rand"

	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/graph"
	"github.com/julioguillermo/godeep/layer"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Model[T types.Number] struct {
	LastLayer layer.Layer[T]

	input    tensor.Tensor[T]
	output   tensor.Tensor[T]
	target   tensor.Tensor[T]
	alpha    *operation.Operand[T]
	momentum *operation.Operand[T]

	feedForward     *graph.Graph
	backPropagation *graph.Graph
}

func FromInOut[T types.Number](in, out layer.Layer[T]) (*Model[T], error) {
	err := out.Build()
	if err != nil {
		return nil, err
	}

	ctx := &context.Context{}
	err = out.BuildFeedforward(ctx)
	if err != nil {
		return nil, err
	}

	ff, err := graph.NewGraphFrom[T](ctx)
	if err != nil {
		return nil, err
	}
	input := in.GetInputs()
	output := out.GetOutputs()

	return &Model[T]{
		LastLayer: out,

		feedForward: ff,

		input:  input,
		output: output,
	}, nil
}

func (p *Model[T]) buildBackPropagation() error {
	ctx := &context.Context{}

	p.alpha = &operation.Operand[T]{}
	p.momentum = &operation.Operand[T]{}

	dif := p.LastLayer.GetDif()
	p.target = tensor.NewZeros[T](p.output.GetShape()...)
	out_ops := p.output.GetOperands()
	tar_ops := p.target.GetOperands()
	dif_ops := dif.GetOperands()
	for i := range tar_ops {
		ctx.Push(&operation.Sub[T]{
			Operand: dif_ops[i],
			A:       tar_ops[i],
			B:       out_ops[i],
		})
	}

	err := p.LastLayer.BuildBackpropagation(ctx, p.alpha, p.momentum)
	if err != nil {
		return err
	}

	bp, err := graph.NewGraphFrom[T](ctx)
	if err != nil {
		return err
	}
	p.backPropagation = bp
	return nil
}

func (p *Model[T]) Predict(t tensor.Tensor[T]) (tensor.Tensor[T], error) {
	err := p.input.LoadFromTensor(t)
	if err != nil {
		return nil, err
	}
	p.feedForward.Exec()
	return p.output.Copy(), nil
}

func (p *Model[T]) fit(x, y tensor.Tensor[T]) error {
	err := p.input.LoadFromTensor(x)
	if err != nil {
		return err
	}
	err = p.target.LoadFromTensor(y)
	if err != nil {
		return err
	}
	p.feedForward.Exec()
	p.backPropagation.Exec()
	return p.LastLayer.Fit()
}

func (p *Model[T]) Fit(
	inputs, targets []tensor.Tensor[T],
	epochs, batch uint,
	alpha, momentum T,
) error {
	if len(inputs) != len(targets) {
		return errors.FmtNeuralError(
			"Fail to fit the model, the number of inputs %d does not match with the number of targets %d",
			len(inputs),
			len(targets),
		)
	}
	if p.backPropagation == nil {
		err := p.buildBackPropagation()
		if err != nil {
			return err
		}
	}
	p.alpha.Value = alpha
	p.momentum.Value = momentum
	if batch == 0 {
		batch = uint(len(inputs))
	}
	for i := uint(0); i < epochs; i++ {
		for j := uint(0); j < batch; j++ {
			index := rand.Intn(len(inputs))
			err := p.fit(inputs[index], targets[index])
			if err != nil {
				return err
			}
		}
	}
	return nil
}
