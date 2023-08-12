package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
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
	if p.builded {
		return nil
	}
	p.builded = true

	err := p.A.BuildGraph(ctx)
	if err != nil {
		return err
	}
	err = p.B.BuildGraph(ctx)
	if err != nil {
		return err
	}

	e := tools.GetEqShapeErr("Binary operation", p.A.GetShape(), p.B.GetShape())
	if e != nil {
		return e
	}

	p.Shape = p.A.GetShape()
	p.MulIndex = p.A.GetMulIndex()

	opA := p.A.GetOperands()
	opB := p.B.GetOperands()

	p.Operands = make([]*number.Scalar[T], p.A.GetSize())
	for i := range p.Operands {
		o := &number.Scalar[T]{}
		p.Operands[i] = o
		switch p.O {
		case Op2Add:
			ctx.Push(&operation.Add[T]{Scalar: o, A: opA[i], B: opB[i]})
		case Op2Sub:
			ctx.Push(&operation.Sub[T]{Scalar: o, A: opA[i], B: opB[i]})
		case Op2Mul:
			ctx.Push(&operation.Mul[T]{Scalar: o, A: opA[i], B: opB[i]})
		case Op2Div:
			ctx.Push(&operation.Div[T]{Scalar: o, A: opA[i], B: opB[i]})
		}
	}

	return nil
}
