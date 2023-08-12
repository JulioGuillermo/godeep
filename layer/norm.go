package layer

import (
	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Norm[T types.Number] struct {
	Base[T]
	min       T
	max       T
	maxScalar *number.Scalar[T]
}

func NewNorm[T types.Number](act activation.Activation[T]) Layer[T] {
	l := &Norm[T]{}
	l.Type = "Norm"
	l.min = 0
	l.max = 1
	l.Activation = act

	return l
}

func NewENorm[T types.Number](act activation.Activation[T]) Layer[T] {
	l := &Norm[T]{}
	l.Type = "Norm"
	l.min = -1
	l.max = 1
	l.Activation = act

	return l
}

func (p *Norm[T]) Build() (uint, error) {
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

func (p *Norm[T]) BuildFeedforward(ctx *context.Context) error {
	if p.CheckFF() {
		return nil
	}
	err := p.PreBuildFeedforward(ctx)
	if err != nil {
		return err
	}

	ops := p.Input.GetOperands()
	p.maxScalar = &number.Scalar[T]{}
	ctx.Push(&operation.Max[T]{
		Scalar: p.maxScalar,
		Args:   ops,
	})

	norm := tensor.DivScalar(p.Input, p.maxScalar)
	norm = tensor.MulScalar(norm, &number.Scalar[T]{Value: p.max - p.min})
	p.Neta = tensor.AddScalar(norm, &number.Scalar[T]{Value: p.min})
	p.Output = tensor.Activate(p.Neta, p.Activation.Activate)

	err = p.Output.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.Dif = tensor.NewZeros[T](p.Output.GetShape()...)

	return nil
}

func (p *Norm[T]) BuildBackpropagation(ctx *context.Context, a, m *number.Scalar[T]) error {
	if p.CheckBP() {
		return nil
	}

	Dif := p.Dif
	if p.Ref.Value > 1 {
		Dif = tensor.DivScalar(Dif, p.Ref)
	}

	// Best
	// Dif = tensor.AddScalar(Dif, &number.Scalar[T]{Value: p.min})
	Dif = tensor.MulScalar(Dif, &number.Scalar[T]{Value: p.max - p.min})
	Dif = tensor.DivScalar(Dif, p.maxScalar)

	der := tensor.Activate(p.PreLayer.GetNetas(), p.PreLayer.GetActivation().Derive)
	Dif = tensor.Mul(Dif, der)

	err := tensor.Transfer(ctx, Dif, p.PreLayer.GetDif())
	if err != nil {
		return err
	}

	return p.PostBuildBackpropagation(ctx, a, m)
}
