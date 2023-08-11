package layer

import (
	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type Concat[T types.Number] struct {
	Base[T]
	l1  Layer[T]
	l2  Layer[T]
	dim uint
}

func NewConcat[T types.Number](l1, l2 Layer[T], dim uint) Layer[T] {
	l := &Concat[T]{
		l1:  l1,
		l2:  l2,
		dim: dim,
	}

	return l
}

func (p *Concat[T]) Build() error {
	if p.CheckB() {
		return nil
	}
	if p.l1 == nil || p.l2 == nil {
		return errors.FmtNeuralError("Concat layer need to previous layers")
	}
	err := p.l1.Build()
	if err != nil {
		return err
	}
	return p.l2.Build()
}

func (p *Concat[T]) BuildFeedforward(ctx *context.Context) error {
	if p.CheckFF() {
		return nil
	}
	err := p.l1.BuildFeedforward(ctx)
	if err != nil {
		return err
	}
	err = p.l2.BuildFeedforward(ctx)
	if err != nil {
		return err
	}

	p.Output = tensor.Concat(p.l1.GetOutputs(), p.l2.GetOutputs(), p.dim)
	p.Neta = tensor.Concat(p.l1.GetNetas(), p.l2.GetNetas(), p.dim)
	p.Activation = &activation.Linear[T]{}

	p.Input = p.Output

	err = p.Output.BuildGraph(ctx)
	if err != nil {
		return err
	}
	err = p.Neta.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.Dif = tensor.NewZeros[T](p.Output.GetShape()...)

	return nil
}

func (p *Concat[T]) BuildBackpropagation(ctx *context.Context, a, m *operation.Operand[T]) error {
	if p.CheckBP() {
		return nil
	}

	s1 := p.l1.GetOutputs().GetShape()[p.dim]
	s2 := p.l2.GetOutputs().GetShape()[p.dim]

	dif1 := tensor.SubTensor(p.Dif, p.dim, 0, s1)
	dif2 := tensor.SubTensor(p.Dif, p.dim, s1, s1+s2)

	dif1 = tensor.Activate(dif1, p.l1.GetActivation().Derive)
	dif2 = tensor.Activate(dif2, p.l1.GetActivation().Derive)

	err := tensor.Transfer(ctx, dif1, p.l1.GetDif())
	if err != nil {
		return err
	}
	err = tensor.Transfer(ctx, dif2, p.l2.GetDif())
	if err != nil {
		return err
	}

	err = p.l1.BuildBackpropagation(ctx, a, m)
	if err != nil {
		return err
	}
	err = p.l2.BuildBackpropagation(ctx, a, m)
	if err != nil {
		return err
	}
	return p.PostBuildBackpropagation(ctx, a, m)
}
