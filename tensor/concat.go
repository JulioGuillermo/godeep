package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type TensorConcat[T types.Number] struct {
	TensorMat[T]
	A Tensor[T]
	B Tensor[T]
	D uint
}

func Concat[T types.Number](a, b Tensor[T], dim uint) Tensor[T] {
	return &TensorConcat[T]{
		A: a,
		B: b,
		D: dim,
	}
}

func (p *TensorConcat[T]) BuildGraph(ctx *context.Context) error {
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

	e := tools.Equals(p.A.GetShape(), p.B.GetShape())
	if e == -2 {
		return errors.FmtNeuralError(
			"Shapes with different dimensions %d and %d",
			len(p.A.GetShape()),
			len(p.B.GetShape()),
		)
	}
	if e >= 0 && uint(e) != p.D {
		return errors.FmtNeuralError(
			"Different shapes %d and %d at %d",
			p.A.GetShape()[e],
			p.B.GetShape()[e],
			e,
		)
	}

	p.Shape = p.A.GetShape()
	dimSize := p.Shape[p.D]
	p.Shape[p.D] += p.B.GetShape()[p.D]
	p.MulIndex = tools.GetIndexMul(p.Shape)

	p.Operands = make([]*number.Scalar[T], tools.GetDataSize(p.Shape))
	for i := range p.Operands {
		idx := tools.ReverseIndex(p.MulIndex, p.Shape, uint(i))
		err := p.concatLastDim(dimSize, uint(i), idx)
		if err != nil {
			return err
		}
	}
	return nil
	// return p.concatRecursive(0, 0, []uint{})
}

func (p *TensorConcat[_]) concatLastDim(dimSize, index uint, oIndex []uint) error {
	if oIndex[p.D] < dimSize {
		o, err := p.A.GetOperand(oIndex...)
		if err != nil {
			return err
		}
		p.Operands[index] = o
	} else {
		oIndex[p.D] -= dimSize
		o, err := p.B.GetOperand(oIndex...)
		if err != nil {
			return err
		}
		p.Operands[index] = o
	}
	return nil
}

func (p *TensorMat[T]) Concat(m *TensorMat[T], dim uint) (*TensorMat[T], error) {
	dimSize := p.Shape[dim]
	shape := p.GetShape()
	shape[dim] += m.GetShape()[dim]
	mulIndex := tools.GetIndexMul(shape)

	ops := make([]*number.Scalar[T], p.GetSize()+m.GetSize())
	for i := range ops {
		idx := tools.ReverseIndex(mulIndex, shape, uint(i))
		err := p.concatLastDim(ops, m, dim, dimSize, uint(i), idx)
		if err != nil {
			return nil, err
		}
	}
	return &TensorMat[T]{
		Shape:    shape,
		MulIndex: mulIndex,
		Operands: ops,
	}, nil
}

func (p *TensorMat[T]) concatLastDim(
	ops []*number.Scalar[T],
	m *TensorMat[T],
	dim, dimSize, index uint,
	oIndex []uint,
) error {
	if oIndex[dim] < dimSize {
		o, err := p.GetOperand(oIndex...)
		if err != nil {
			return err
		}
		ops[index] = o
	} else {
		oIndex[dim] -= dimSize
		o, err := m.GetOperand(oIndex...)
		if err != nil {
			return err
		}
		ops[index] = o
	}
	return nil
}
