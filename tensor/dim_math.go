package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type OperDim byte

const (
	DimSum = OperDim(iota)
	DimAvg
	DimMin
	DimMax
)

type TensorDimMath[T types.Number] struct {
	TensorMat[T]
	T Tensor[T]
	D uint
	O OperDim
}

func DimMath[T types.Number](t Tensor[T], dim uint, oper OperDim) Tensor[T] {
	return &TensorDimMath[T]{
		T: t,
		D: dim,
		O: oper,
	}
}

func (p *TensorDimMath[T]) BuildGraph(ctx *context.Context) error {
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

	p.Shape = p.T.GetShape()[:p.D+1]
	p.MulIndex = tools.GetIndexMul(p.Shape)
	p.Operands = make([]*number.Scalar[T], tools.GetDataSize(p.Shape))

	mi := p.T.GetMulIndex()[p.D]
	ops := p.T.GetOperands()
	for i := range p.Operands {
		offset := uint(i) * mi
		o := &number.Scalar[T]{}
		p.Operands[i] = o
		args := ops[offset : offset+mi]

		switch p.O {
		case DimSum:
			ctx.Push(&operation.Sum[T]{
				Scalar: o,
				Args:   args,
			})
		case DimAvg:
			ctx.Push(&operation.Avg[T]{
				Scalar: o,
				Args:   args,
			})
		case DimMin:
			ctx.Push(&operation.Min[T]{
				Scalar: o,
				Args:   args,
			})
		case DimMax:
			ctx.Push(&operation.Max[T]{
				Scalar: o,
				Args:   args,
			})
		}
	}

	return nil
}

func (p *TensorMat[T]) DSum(dim uint) (*TensorMat[T], error) {
	if dim >= uint(len(p.Shape)) {
		return nil, errors.FmtNeuralError(
			"Invalid subtensor dimension %d for a tensor with dimension %d",
			dim,
			len(p.Shape),
		)
	}

	shape := p.GetShape()[:dim+1]
	mulIndex := tools.GetIndexMul(shape)
	ops := make([]*number.Scalar[T], tools.GetDataSize(shape))
	mi := p.MulIndex[dim]

	for i := range ops {
		offset := uint(i) * mi
		args := p.Operands[offset : offset+mi]
		ops[i] = &number.Scalar[T]{
			Value: tools.Sum[T](args),
		}
	}

	return &TensorMat[T]{
		Shape:    shape,
		MulIndex: mulIndex,
		Operands: ops,
	}, nil
}

func (p *TensorMat[T]) DAvg(dim uint) (*TensorMat[T], error) {
	if dim >= uint(len(p.Shape)) {
		return nil, errors.FmtNeuralError(
			"Invalid subtensor dimension %d for a tensor with dimension %d",
			dim,
			len(p.Shape),
		)
	}

	shape := p.GetShape()[:dim+1]
	mulIndex := tools.GetIndexMul(shape)
	ops := make([]*number.Scalar[T], tools.GetDataSize(shape))
	mi := p.MulIndex[dim]

	for i := range ops {
		offset := uint(i) * mi
		args := p.Operands[offset : offset+mi]
		ops[i] = &number.Scalar[T]{
			Value: tools.Avg[T](args),
		}
	}

	return &TensorMat[T]{
		Shape:    shape,
		MulIndex: mulIndex,
		Operands: ops,
	}, nil
}

func (p *TensorMat[T]) DMax(dim uint) (*TensorMat[T], error) {
	if dim >= uint(len(p.Shape)) {
		return nil, errors.FmtNeuralError(
			"Invalid subtensor dimension %d for a tensor with dimension %d",
			dim,
			len(p.Shape),
		)
	}

	shape := p.GetShape()[:dim+1]
	mulIndex := tools.GetIndexMul(shape)
	ops := make([]*number.Scalar[T], tools.GetDataSize(shape))
	mi := p.MulIndex[dim]

	for i := range ops {
		offset := uint(i) * mi
		args := p.Operands[offset : offset+mi]
		ops[i] = &number.Scalar[T]{
			Value: tools.Max[T](args),
		}
	}

	return &TensorMat[T]{
		Shape:    shape,
		MulIndex: mulIndex,
		Operands: ops,
	}, nil
}

func (p *TensorMat[T]) DMin(dim uint) (*TensorMat[T], error) {
	if dim >= uint(len(p.Shape)) {
		return nil, errors.FmtNeuralError(
			"Invalid subtensor dimension %d for a tensor with dimension %d",
			dim,
			len(p.Shape),
		)
	}

	shape := p.GetShape()[:dim+1]
	mulIndex := tools.GetIndexMul(shape)
	ops := make([]*number.Scalar[T], tools.GetDataSize(shape))
	mi := p.MulIndex[dim]

	for i := range ops {
		offset := uint(i) * mi
		args := p.Operands[offset : offset+mi]
		ops[i] = &number.Scalar[T]{
			Value: tools.Min[T](args),
		}
	}

	return &TensorMat[T]{
		Shape:    shape,
		MulIndex: mulIndex,
		Operands: ops,
	}, nil
}
