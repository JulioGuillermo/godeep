package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type TensorTanspose[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
}

func Transpose[T types.Number](t Tensor[T]) Tensor[T] {
	return &TensorTanspose[T]{
		T: t,
	}
}

func (p *TensorTanspose[T]) BuildGraph(ctx *context.Context) error {
	if p.builded {
		return nil
	}
	p.builded = true

	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}
	TShape := p.T.GetShape()

	dims := len(TShape)
	p.Shape = make([]uint, dims)
	for i := range p.Shape {
		p.Shape[i] = TShape[dims-i-1]
	}
	p.MulIndex = tools.GetIndexMul(p.Shape)

	p.Operands = make([]*number.Scalar[T], p.T.GetSize())
	for i := range p.Operands {
		idx := tools.ReverseIndex(p.MulIndex, p.Shape, uint(i))
		tIdx := tools.GetInvertedIndex(idx)
		o, err := p.T.GetOperand(tIdx...)
		if err != nil {
			return err
		}
		p.Operands[i] = o
	}

	return nil
}

func (p *TensorMat[T]) Transpose() (*TensorMat[T], error) {
	dims := len(p.Shape)
	shape := make([]uint, dims)
	for i := range shape {
		shape[i] = p.Shape[dims-i-1]
	}
	mulIndex := tools.GetIndexMul(shape)

	ops := make([]*number.Scalar[T], p.GetSize())
	for i := range ops {
		idx := tools.ReverseIndex(mulIndex, shape, uint(i))
		tIdx := tools.GetInvertedIndex(idx)
		o, err := p.GetOperand(tIdx...)
		if err != nil {
			return nil, err
		}
		ops[i] = &number.Scalar[T]{
			Value: o.Value,
		}
	}

	return &TensorMat[T]{
		Shape:    shape,
		MulIndex: mulIndex,
		Operands: ops,
	}, nil
}
