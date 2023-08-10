package tensor

import (
	"github.com/julioguillermo/neuralnetwork/v2/context"
	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/tools"
	"github.com/julioguillermo/neuralnetwork/v2/types"
)

type Oper2 byte

const (
	Op2Add = Oper2(iota)
	Op2Sub
	Op2Mul
	Op2Div
)

type TensorOp2[T types.Number] struct {
	TensorMat[T]
	A Tensor[T]
	B Tensor[T]
	O Oper2
}

func NewOp2[T types.Number](a, b Tensor[T], o Oper2) Tensor[T] {
	return &TensorOp2[T]{
		A: a,
		B: b,
		O: o,
	}
}

func (p *TensorOp2[T]) BuildGraph(ctx *context.Context) error {
	e := tools.GetEqShapeErr(p.A.GetShape(), p.B.GetShape())
	if e != nil {
		return e
	}

	p.Shape = p.A.GetShape()
	p.MulIndex = p.A.GetMulIndex()

	err := p.A.BuildGraph(ctx)
	if err != nil {
		return err
	}
	err = p.B.BuildGraph(ctx)
	if err != nil {
		return err
	}

	opA := p.A.GetOperands()
	opB := p.B.GetOperands()

	p.Operands = make([]*operation.Operand[T], p.A.GetSize())
	for i := range p.Operands {
		o := &operation.Operand[T]{}
		p.Operands[i] = o
		switch p.O {
		case Op2Add:
			ctx.Push(&operation.Add[T]{Operand: o, A: opA[i], B: opB[i]})
		case Op2Sub:
			ctx.Push(&operation.Sub[T]{Operand: o, A: opA[i], B: opB[i]})
		case Op2Mul:
			ctx.Push(&operation.Mul[T]{Operand: o, A: opA[i], B: opB[i]})
		case Op2Div:
			ctx.Push(&operation.Div[T]{Operand: o, A: opA[i], B: opB[i]})
		}
	}

	return nil
}
