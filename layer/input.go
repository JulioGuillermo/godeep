package layer

import (
	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Input[T types.Number] struct {
	Base[T]
}

func NewInput[T types.Number](shape ...uint) Layer[T] {
	in := &Input[T]{
		Base: Base[T]{
			Input:      tensor.NewZeros[T](shape...),
			Dif:        tensor.NewZeros[T](shape...),
			Activation: activation.Linear[T]{},
		},
	}
	return in
}

func (p *Input[T]) Build() (uint, error) {
	if p.CheckB() {
		return p.Index, nil
	}

	err := p.PreBuild()
	if err != nil {
		return 0, err
	}

	if p.PreLayer != nil {
		p.PreLayer.GetRef().Value += p.Ref.Value
		p.Ref = p.PreLayer.GetRef()
		// p.Activation = p.PreLayer.GetActivation()
	}

	return p.Index, nil
}

func (p *Input[T]) BuildFeedforward(ctx *context.Context) error {
	if p.CheckFF() {
		return nil
	}
	err := p.PreBuildFeedforward(ctx)
	if err != nil {
		return err
	}

	if p.PreLayer != nil {
		p.Input = p.PreLayer.GetOutputs()
		p.Output = p.PreLayer.GetOutputs()
		// p.Neta = p.PreLayer.GetNetas()
		p.Dif = p.PreLayer.GetDif()
		p.Ref = p.PreLayer.GetRef()
		// p.Activation = p.PreLayer.GetActivation()
		return nil
	}
	p.Neta = p.Input
	p.Output = p.Input
	return nil
}

func (p *Input[T]) BuildBackpropagation(
	ctx *context.Context,
	a *number.Scalar[T],
	m *number.Scalar[T],
) error {
	if p.CheckBP() {
		return nil
	}
	return p.PostBuildBackpropagation(ctx, a, m)
}

func (p *Input[T]) BuildDer(ctx *context.Context) (tensor.Tensor[T], error) {
	if p.Der == nil {
		if p.PreLayer != nil {
			der, err := p.PreLayer.BuildDer(ctx)
			if err != nil {
				return nil, err
			}
			p.Der = der
		} else {
			p.Der = tensor.NewZeros[T](p.Input.GetShape()...)
			err := p.Der.BuildGraph(ctx)
			if err != nil {
				return nil, err
			}
		}
	}
	return p.Der, nil
}
