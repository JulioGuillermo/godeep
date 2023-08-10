package tensor

import (
	"github.com/julioguillermo/neuralnetwork/v2/context"
	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/types"
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
		ctx.Push(&operation.Func[T]{Operand: o, O: ops[i], F: p.F})
	}

	return nil
}
