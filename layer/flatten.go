package layer

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Flatten[T types.Number] struct {
	Base[T]
}

func NewFlatten[T types.Number]() Layer[T] {
	l := &Flatten[T]{}

	return l
}

func (p *Flatten[T]) Build() error {
	if p.CheckB() {
		return nil
	}

	err := p.PreBuild()
	if err != nil {
		return err
	}

	if p.PreLayer == nil {
		return errors.FmtNeuralError("Flatten layer can not be input layer")
	}
	return nil
}

func (p *Flatten[T]) BuildFeedforward(ctx *context.Context) error {
	if p.CheckFF() {
		return nil
	}
	err := p.PreBuildFeedforward(ctx)
	if err != nil {
		return err
	}
	p.Output = tensor.Flatten[T](p.PreLayer.GetOutputs())
	p.Neta = tensor.Flatten[T](p.PreLayer.GetNetas())
	p.Dif = tensor.Flatten[T](p.PreLayer.GetDif())
	p.Activation = p.PreLayer.GetActivation()

	err = p.Output.BuildGraph(ctx)
	if err != nil {
		return err
	}
	err = p.Neta.BuildGraph(ctx)
	if err != nil {
		return err
	}
	return nil
}

func (p *Flatten[T]) BuildBackpropagation(ctx *context.Context, a, m *operation.Operand[T]) error {
	if p.CheckBP() {
		return nil
	}

	err := p.Dif.BuildGraph(ctx)
	if err != nil {
		return err
	}
	return p.PostBuildBackpropagation(ctx, a, m)
}
