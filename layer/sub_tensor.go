package layer

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tensor"
	"github.com/julioguillermo/godeep/types"
)

type SubTensor[T types.Number] struct {
	Base[T]
	dim  uint
	from uint
	to   uint
}

func NewSubTensor[T types.Number](dim, from, to uint) Layer[T] {
	l := &SubTensor[T]{
		dim:  dim,
		from: from,
		to:   to,
	}

	return l
}

func (p *SubTensor[T]) Build() (uint, error) {
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

func (p *SubTensor[T]) BuildFeedforward(ctx *context.Context) error {
	if p.CheckFF() {
		return nil
	}
	err := p.PreBuildFeedforward(ctx)
	if err != nil {
		return err
	}

	p.Output = tensor.SubTensor[T](p.PreLayer.GetOutputs(), p.dim, p.from, p.to)
	p.Neta = tensor.SubTensor[T](p.PreLayer.GetNetas(), p.dim, p.from, p.to)
	p.Activation = p.PreLayer.GetActivation()

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

func (p *SubTensor[T]) BuildBackpropagation(
	ctx *context.Context,
	a, m *number.Scalar[T],
) error {
	if p.CheckBP() {
		return nil
	}

	err := p.Dif.BuildGraph(ctx)
	if err != nil {
		return err
	}

	Dif := p.Dif
	if p.Ref.Value > 1 {
		Dif = tensor.DivScalar(Dif, p.Ref)
	}

	p_dif := tensor.SubTensor[T](p.PreLayer.GetDif(), p.dim, p.from, p.to)

	Dif = tensor.Add[T](Dif, p_dif)

	err = tensor.Transfer[T](ctx, Dif, p_dif)
	if err != nil {
		return err
	}
	return p.PostBuildBackpropagation(ctx, a, m)
}
