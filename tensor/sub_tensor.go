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
}

func (p *TensorMat[T]) SubTensor(dim, start, end uint) (*TensorMat[T], error) {
	shape := p.GetShape()

	if dim >= uint(len(shape)) {
		return nil, errors.FmtNeuralError(
			"Invalid subtensor dimension %d for a tensor with dimension %d",
			dim,
			len(shape),
		)
	}
	if end <= start {
		return nil, errors.FmtNeuralError(
			"Invalid subtensor range from %d to %d",
			start,
			end,
		)
	}
	if end > shape[dim] {
		return nil, errors.FmtNeuralError(
			"Invalid subtensor range end %d for shape %d at dimension %d",
			end,
			shape[dim],
			dim,
		)
	}
	if start > shape[dim] {
		return nil, errors.FmtNeuralError(
			"Invalid subtensor range start %d for shape %d at dimension %d",
			start,
			shape[dim],
			dim,
		)
	}

	shape[dim] = end - start
	mulIndex := tools.GetIndexMul(shape)

	ops := make([]*number.Scalar[T], tools.GetDataSize(shape))

	for i := range ops {
		idx := tools.ReverseIndex(mulIndex, shape, uint(i))
		idx[dim] += start
		o, err := p.GetOperand(idx...)
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

func (p *TensorMat[T]) FOSubTensor(dim, start, end uint, fill T) (*TensorMat[T], error) {
	shape := p.GetShape()

	if dim >= uint(len(shape)) {
		return nil, errors.FmtNeuralError(
			"Invalid subtensor dimension %d for a tensor with dimension %d",
			dim,
			len(shape),
		)
	}
	if end <= start {
		return nil, errors.FmtNeuralError(
			"Invalid subtensor range from %d to %d",
			start,
			end,
		)
	}

	max := shape[dim]
	shape[dim] = end - start
	mulIndex := tools.GetIndexMul(shape)

	ops := make([]*number.Scalar[T], tools.GetDataSize(shape))

	for i := range ops {
		idx := tools.ReverseIndex(mulIndex, shape, uint(i))
		idx[dim] += start
		if idx[dim] > max {
			ops[i] = &number.Scalar[T]{
				Value: fill,
			}
		} else {
			o, err := p.GetOperand(idx...)
			if err != nil {
				return nil, err
			}
			ops[i] = &number.Scalar[T]{
				Value: o.Value,
			}
		}
	}

	return &TensorMat[T]{
		Shape:    shape,
		MulIndex: mulIndex,
		Operands: ops,
	}, nil
}
