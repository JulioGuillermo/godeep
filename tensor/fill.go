package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/types"
)

type TensorFill[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
	V *number.Scalar[T]
}

func FillWith[T types.Number](t Tensor[T], v *number.Scalar[T]) Tensor[T] {
	return &TensorFill[T]{
		T: t,
		V: v,
	}
}

func Fill[T types.Number](ctx *context.Context, t Tensor[T], v *number.Scalar[T]) error {
	cp := FillWith(t, v)
	return cp.BuildGraph(ctx)
}

func (p *TensorFill[T]) BuildGraph(ctx *context.Context) error {
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
	p.Operands = p.T.GetOperands()

	ops := p.T.GetOperands()
	for i := range p.Operands {
		ctx.Push(&operation.Set[T]{Scalar: ops[i], O: p.V})
	}

	return nil
}
