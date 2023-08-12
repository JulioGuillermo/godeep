package layer

import (
	"fmt"
	"strings"

	"github.com/julioguillermo/godeep/activation"
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/tools"
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
	l.Type = "Concat"

	return l
}

func (p *Concat[T]) Build() (uint, error) {
	if p.CheckB() {
		return p.Index, nil
	}
	if p.l1 == nil || p.l2 == nil {
		return 0, p.Error("Need 2 previous layers")
	}
	p.l1.GetRef().Value++
	p.l2.GetRef().Value++

	c1, err := p.l1.Build()
	if err != nil {
		return 0, err
	}
	c2, err := p.l2.Build()
	if err != nil {
		return 0, err
	}
	if c1 > c2 {
		p.Index = c1 + 1
	} else {
		p.Index = c2 + 1
	}
	return p.Index, nil
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

func (p *Concat[T]) BuildBackpropagation(ctx *context.Context, a, m *number.Scalar[T]) error {
	if p.CheckBP() {
		return nil
	}

	s1 := p.l1.GetOutputs().GetShape()[p.dim]
	s2 := p.l2.GetOutputs().GetShape()[p.dim]

	Dif := p.Dif
	if p.Ref.Value > 1 {
		Dif = tensor.DivScalar(Dif, p.Ref)
	}

	dif1 := tensor.SubTensor(Dif, p.dim, 0, s1)
	dif2 := tensor.SubTensor(Dif, p.dim, s1, s1+s2)

	dif1 = tensor.Activate(dif1, p.l1.GetActivation().Derive)
	dif2 = tensor.Activate(dif2, p.l1.GetActivation().Derive)

	dif1 = tensor.Add(dif1, p.l1.GetDif())
	dif2 = tensor.Add(dif2, p.l2.GetDif())

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

func (p *Concat[T]) ResetPrinted() {
	p.printed = false

	p.l1.ResetPrinted()
	p.l2.ResetPrinted()
}

func (p *Concat[T]) PushToString(sb *strings.Builder) {
	p.l1.PushToString(sb)
	p.l2.PushToString(sb)

	sb.WriteString(
		fmt.Sprintf(
			"<Layer[%d] (%d, %d) => %s: I%s O%s>\n",
			p.Index,
			p.l1.GetIndex(),
			p.l2.GetIndex(),
			p.Type,
			tools.ShapeStr(p.Input.GetShape()),
			tools.ShapeStr(p.Output.GetShape()),
		),
	)
}
