package tensor

import (
	"github.com/julioguillermo/neuralnetwork/v2/context"
	"github.com/julioguillermo/neuralnetwork/v2/errors"
	"github.com/julioguillermo/neuralnetwork/v2/operation"
	"github.com/julioguillermo/neuralnetwork/v2/tools"
	"github.com/julioguillermo/neuralnetwork/v2/types"
)

type TensorSubTensor[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
	D uint
	S uint
	E uint
}

func SubTensor[T types.Number](t Tensor[T], dim, from, to uint) Tensor[T] {
	return &TensorSubTensor[T]{
		T: t,
		D: dim,
		S: from,
		E: to,
	}
}

func (p *TensorSubTensor[T]) BuildGraph(ctx *context.Context) error {
	if p.D >= uint(len(p.T.GetShape())) {
		return errors.FmtNeuralError(
			"Invalid subtensor dimension %d for a tensor with dimension %d",
			p.D,
			len(p.T.GetShape()),
		)
	}
	if p.E <= p.S {
		return errors.FmtNeuralError(
			"Invalid subtensor range from %d to %d",
			p.S,
			p.E,
		)
	}
	if p.E >= p.T.GetShape()[p.D] {
		return errors.FmtNeuralError(
			"Invalid subtensor range end %d for shape %d at dimension %d",
			p.E,
			p.T.GetShape()[p.D],
			p.D,
		)
	}
	if p.S >= p.T.GetShape()[p.D] {
		return errors.FmtNeuralError(
			"Invalid subtensor range start %d for shape %d at dimension %d",
			p.S,
			p.T.GetShape()[p.D],
			p.D,
		)
	}

	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

	p.Shape = p.T.GetShape()
	p.Shape[p.D] = p.E - p.S
	p.MulIndex = tools.GetIndexMul(p.Shape)

	p.Operands = make([]*operation.Operand[T], tools.GetDataSize(p.Shape))
	p.subRecursive(0, 0, []uint{})

	return nil
}

// TODO better way...
func (p *TensorSubTensor[_]) subRecursive(dim, index uint, oIndex []uint) error {
	if dim == uint(len(p.Shape)) {
		o, err := p.T.GetOperand(oIndex...)
		if err != nil {
			return err
		}
		p.Operands[index] = o
		return nil
	}
	if dim == p.D {
		to := p.E - p.S
		for i := uint(0); i < to; i++ {
			err := p.subRecursive(
				dim+1,
				index+i*p.MulIndex[dim],
				append(oIndex, i+p.S),
			)
			if err != nil {
				return err
			}
		}
		return nil
	}
	for i := uint(0); i < p.Shape[dim]; i++ {
		err := p.subRecursive(
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
