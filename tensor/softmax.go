package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/types"
)

type TensorSoftMax[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
}

func SoftMax[T types.Number](t Tensor[T]) Tensor[T] {
	return &TensorSoftMax[T]{
		T: t,
	}
}

func (p *TensorSoftMax[T]) BuildGraph(ctx *context.Context) error {
	if p.builded {
		return nil
	}
	p.builded = true

	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.Shape = p.T.GetShape()
	p.MulIndex = p.T.GetMulIndex()

	min := &number.Scalar[T]{}
	ctx.Push(&operation.Min[T]{
		Scalar: min,
		Args:   p.T.GetOperands(),
	})

	p.Operands = make([]*number.Scalar[T], p.T.GetSize())

	ops := p.T.GetOperands()
	for i := range p.Operands {
		p.Operands[i] = &number.Scalar[T]{}
		ctx.Push(&operation.Sub[T]{
			Scalar: p.Operands[i],
			A:      ops[i],
			B:      min,
		})
	}

	sum := &number.Scalar[T]{}
	ctx.Push(&operation.Sum[T]{
		Scalar: sum,
		Args:   p.Operands,
	})

	for i := range p.Operands {
		ctx.Push(&operation.Div[T]{
			Scalar: p.Operands[i],
			A:      p.Operands[i],
			B:      sum,
		})
	}

	return nil
}
