package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/types"
)

type TensorCopy[T types.Number] struct {
	TensorMat[T]
	F Tensor[T]
	T Tensor[T]
}

func CopyTo[T types.Number](from, to Tensor[T]) Tensor[T] {
	return &TensorCopy[T]{
		F: from,
		T: to,
	}
}

func Transfer[T types.Number](ctx *context.Context, from, to Tensor[T]) error {
	cp := CopyTo(from, to)
	return cp.BuildGraph(ctx)
}

func (p *TensorCopy[T]) BuildGraph(ctx *context.Context) error {
	if p.builded {
		return nil
	}
	p.builded = true

	err := p.F.BuildGraph(ctx)
	if err != nil {
		return err
	}
	err = p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

	if p.F.GetSize() != p.T.GetSize() {
		return errors.FmtNeuralError(
			"Fail to copy from tensor with size %d to tensor with size %d",
			p.F.GetSize(),
			p.T.GetSize(),
		)
	}

	p.Shape = p.T.GetShape()
	p.MulIndex = p.T.GetMulIndex()
	p.Operands = p.T.GetOperands()

	opsT := p.T.GetOperands()
	opsF := p.F.GetOperands()
	for i := range p.Operands {
		ctx.Push(&operation.Set[T]{Scalar: opsT[i], O: opsF[i]})
	}

	return nil
}
