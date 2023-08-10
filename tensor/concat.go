package tensor

import (
	"github.com/julioguillermo/neuralnetwork/v2/context"
	"github.com/julioguillermo/neuralnetwork/v2/errors"
	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/tools"
	"github.com/julioguillermo/neuralnetwork/v2/types"
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
	err := p.A.BuildGraph(ctx)
	if err != nil {
		return err
	}
	err = p.B.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.Shape = p.A.GetShape()
	p.Shape[p.D] += p.B.GetShape()[p.D]
	p.MulIndex = tools.GetIndexMul(p.Shape)

	p.Operands = make([]*operation.Operand[T], tools.GetDataSize(p.Shape))
	return p.concatRecursive(0, 0, []uint{})
}

// TODO better way...
func (p *TensorConcat[_]) concatRecursive(dim, index uint, oIndex []uint) error {
	if dim == uint(len(p.Shape)) {
		dimSize := p.A.GetShape()[p.D]
		dimIndex := oIndex[p.D]
		if dimIndex < dimSize {
			o, err := p.A.GetOperand(oIndex...)
			if err != nil {
				return err
			}
			p.Operands[index] = o
			return nil
		}
		oIndex[p.D] -= dimSize
		o, err := p.B.GetOperand(oIndex...)
		oIndex[p.D] += dimSize
		if err != nil {
			return err
		}
		p.Operands[index] = o
		return nil
	}
	for i := uint(0); i < p.Shape[dim]; i++ {
		err := p.concatRecursive(
			dim+1,
			index+i*p.MulIndex[dim],
			append(oIndex, i),
		)
		if err != nil {
			return err
		}
	}
	return nil
}
