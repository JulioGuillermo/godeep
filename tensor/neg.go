package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/types"
)

type TensorNeg[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
}

func Neg[T types.Number](t Tensor[T]) Tensor[T] {
	return &TensorNeg[T]{
		T: t,
	}
}

func (p *TensorNeg[T]) BuildGraph(ctx *context.Context) error {
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
	p.Operands = make([]*number.Scalar[T], p.T.GetSize())

	ops := p.T.GetOperands()
	for i := range p.Operands {
		o := &number.Scalar[T]{}
		p.Operands[i] = o
		ctx.Push(&operation.Neg[T]{Scalar: o, O: ops[i]})
	}

	return nil
}

func (p *TensorMat[T]) Neg() *TensorMat[T] {
	ops := make([]*number.Scalar[T], p.GetSize())
	for i := range ops {
		ops[i] = &number.Scalar[T]{
			Value: -p.Operands[i].Value,
		}
	}
	return &TensorMat[T]{
		Shape:    p.GetShape(),
		MulIndex: p.GetMulIndex(),
		Operands: ops,
	}
}
