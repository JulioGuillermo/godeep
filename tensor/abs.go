package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/types"
)

type TensorAbs[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
}

func Abs[T types.Number](t Tensor[T]) Tensor[T] {
	return &TensorAbs[T]{
		T: t,
	}
}

func (p *TensorAbs[T]) BuildGraph(ctx *context.Context) error {
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
	p.Operands = make([]*operation.Operand[T], p.T.GetSize())

	ops := p.T.GetOperands()
	for i := range p.Operands {
		o := &operation.Operand[T]{}
		p.Operands[i] = o
		ctx.Push(&operation.Abs[T]{Operand: o, O: ops[i]})
	}

	return nil
}
