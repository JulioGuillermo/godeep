package layer

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/operation"
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

func (p *SubTensor[T]) Build() error {
	if p.CheckB() {
		return nil
	}

	err := p.PreBuild()
	if err != nil {
		return err
	}

	if p.PreLayer == nil {
		return errors.FmtNeuralError("SubTensor layer can not be input layer")
	}
	return nil
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
	a, m *operation.Operand[T],
) error {
	if p.CheckBP() {
		return nil
	}

	p_dif := tensor.SubTensor[T](p.PreLayer.GetDif(), p.dim, p.from, p.to)
	err := tensor.Transfer[T](ctx, p.Dif, p_dif)
	if err != nil {
		return err
	}
	return p.PostBuildBackpropagation(ctx, a, m)
}
