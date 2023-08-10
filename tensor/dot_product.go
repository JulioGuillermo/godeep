package tensor

import (
	"github.com/julioguillermo/neuralnetwork/v2/context"
	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/tools"
	"github.com/julioguillermo/neuralnetwork/v2/types"
)

type TensorDotProduct[T types.Number] struct {
	TensorMat[T]
	A Tensor[T]
	B Tensor[T]
}

func DotProduct[T types.Number](a, b Tensor[T]) Tensor[T] {
	return &TensorDotProduct[T]{
		A: a,
		B: b,
	}
}

func (p *TensorDotProduct[T]) BuildGraph(ctx *context.Context) error {
	e := tools.GetEqShapeErr(p.A.GetShape(), p.B.GetShape())
	if e != nil {
		return e
	}
	err := p.A.BuildGraph(ctx)
	if err != nil {
		return err
	}
	err = p.B.BuildGraph(ctx)
	if err != nil {
		return err
	}

	op1 := p.A.GetOperands()
	op2 := p.B.GetOperands()
	ops := make([]*operation.Operand[T], len(op1))
	for i := range ops {
		o := &operation.Operand[T]{}
		ops[i] = o
		ctx.Push(&operation.Mul[T]{
			Operand: o,
			A:       op1[i],
			B:       op2[i],
		})
	}

	o := &operation.Operand[T]{}
	ctx.Push(&operation.Sum[T]{
		Operand: o,
		Args:    ops,
	})

	p.Operands = []*operation.Operand[T]{o}
	p.Shape = []uint{1}
	p.MulIndex = []uint{1}

	return nil
}
