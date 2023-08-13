package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type TensorSubTensor[T types.Number] struct {
	TensorMat[T]
	T     Tensor[T]
	D     uint
	S     uint
	E     uint
	M     uint
	GoOut bool
}

func SubTensor[T types.Number](t Tensor[T], dim, from, to uint) Tensor[T] {
	return &TensorSubTensor[T]{
		T: t,
		D: dim,
		S: from,
		E: to,
	}
}

func SubExtendedTensor[T types.Number](t Tensor[T], dim, from, to uint) Tensor[T] {
	return &TensorSubTensor[T]{
		T:     t,
		D:     dim,
		S:     from,
		E:     to,
		GoOut: true,
	}
}

func (p *TensorSubTensor[T]) BuildGraph(ctx *context.Context) error {
	if p.builded {
		return nil
	}
	p.builded = true

	err := p.T.BuildGraph(ctx)
	if err != nil {
		return err
	}

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
	if !p.GoOut && p.E > p.T.GetShape()[p.D] {
		return errors.FmtNeuralError(
			"Invalid subtensor range end %d for shape %d at dimension %d",
			p.E,
			p.T.GetShape()[p.D],
			p.D,
		)
	}
	if !p.GoOut && p.S > p.T.GetShape()[p.D] {
		return errors.FmtNeuralError(
			"Invalid subtensor range start %d for shape %d at dimension %d",
			p.S,
			p.T.GetShape()[p.D],
			p.D,
		)
	}

	p.Shape = p.T.GetShape()
	p.M = p.Shape[p.D]
	p.Shape[p.D] = p.E - p.S
	p.MulIndex = tools.GetIndexMul(p.Shape)

	p.Operands = make([]*number.Scalar[T], tools.GetDataSize(p.Shape))
	for i := range p.Operands {
		idx := tools.ReverseIndex(p.MulIndex, p.Shape, uint(i))
		idx[p.D] += p.S
		if idx[p.D] > p.M && p.GoOut {
			p.Operands[i] = &number.Scalar[T]{}
		} else {
			o, err := p.T.GetOperand(idx...)
			if err != nil {
				return err
			}
			p.Operands[i] = o
		}
	}

	return nil
	// return p.subRecursive(0, 0, []uint{})
}

// TODO better way...
//func (p *TensorSubTensor[T]) subRecursive(dim, index uint, oIndex []uint) error {
//	if dim == uint(len(p.Shape)) {
//		if oIndex[p.D] > p.M && p.GoOut {
//			p.Operands[index] = &number.Scalar[T]{}
//			return nil
//		}
//		o, err := p.T.GetOperand(oIndex...)
//		if err != nil {
//			return err
//		}
//		p.Operands[index] = o
//		return nil
//	}
//	if dim == p.D {
//		to := p.E - p.S
//		for i := uint(0); i < to; i++ {
//			err := p.subRecursive(
//				dim+1,
//				index+i*p.MulIndex[dim],
//				append(oIndex, i+p.S),
//			)
//			if err != nil {
//				return err
//			}
//		}
//		return nil
//	}
//	for i := uint(0); i < p.Shape[dim]; i++ {
//		err := p.subRecursive(
//			dim+1,
//			index+i*p.MulIndex[dim],
//			append(oIndex, i),
//		)
//		if err != nil {
//			return err
//		}
//	}
//	return nil
//}
