package layer

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Flatten[T types.Number] struct {
	Base[T]
}

func NewFlatten[T types.Number]() Layer[T] {
	l := &Flatten[T]{}
	l.Type = "Flatten"

	return l
}

func (p *Flatten[T]) Build() (uint, error) {
	if p.CheckB() {
		return p.Index, nil
	}

	err := p.PreBuild()
	if err != nil {
		return 0, err
	}

	if p.PreLayer == nil {
		return 0, p.Error("This layer can not be input layer")
	}
	return p.Index, nil
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

func (p *Flatten[T]) BuildBackpropagation(ctx *context.Context, a, m *number.Scalar[T]) error {
	if p.CheckBP() {
		return nil
	}

	p.Dif = tensor.NewZeros[T](p.Output.GetShape()...)
	dif := p.Dif
	if p.Ref.Value > 1 {
		dif = tensor.DivScalar[T](p.Dif, p.Ref)
	}
	preDif := tensor.Flatten(p.PreLayer.GetDif())
	dif = tensor.Add(preDif, dif)

	err := tensor.Transfer(ctx, dif, p.PreLayer.GetDif())
	if err != nil {
		return err
	}

	return p.PostBuildBackpropagation(ctx, a, m)
}
