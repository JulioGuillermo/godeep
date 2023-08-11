package layer

import (
	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/operation"
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

func (p *Input[T]) Build() error {
	if p.CheckB() {
		return nil
	}

	err := p.PreBuild()
	if err != nil {
		return err
	}

	return nil
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
		p.Output = p.Input
		p.Neta = p.PreLayer.GetNetas()
		p.Dif = p.PreLayer.GetDif()
		p.Activation = p.PreLayer.GetActivation()
		return nil
	}
	p.Neta = p.Input
	p.Output = p.Input
	return nil
}

func (p *Input[T]) BuildBackpropagation(
	ctx *context.Context,
	a *operation.Operand[T],
	m *operation.Operand[T],
) error {
	if p.CheckBP() {
		return nil
	}
	return p.PostBuildBackpropagation(ctx, a, m)
}
