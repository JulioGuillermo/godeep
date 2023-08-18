package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/types"
)

type TensorActivate[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
	F operation.Function[T]
}

func Activate[T types.Number](t Tensor[T], f operation.Function[T]) Tensor[T] {
	return &TensorActivate[T]{
		T: t,
		F: f,
	}
}

func (p *TensorActivate[T]) BuildGraph(ctx *context.Context) error {
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
		ctx.Push(&operation.Func[T]{Scalar: o, O: ops[i], F: p.F})
	}

	return nil
}

func (p *TensorMat[T]) Activate(f operation.Function[T]) *TensorMat[T] {
	ops := make([]*number.Scalar[T], p.GetSize())
	for i := range ops {
		ops[i] = &number.Scalar[T]{
			Value: f(p.Operands[i].Value),
		}
	}
	return &TensorMat[T]{
		Shape:    p.GetShape(),
		MulIndex: p.GetMulIndex(),
		Operands: ops,
	}
}
