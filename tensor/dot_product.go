package tensor

import (
	"github.com/julioguillermo/godeep/context"
	"github.com/julioguillermo/godeep/errors"
	"github.com/julioguillermo/godeep/number"
	"github.com/julioguillermo/godeep/operation"
	"github.com/julioguillermo/godeep/tools"
	"github.com/julioguillermo/godeep/types"
)

type TensorDotProduct[T types.Number] struct {
	TensorMat[T]
	A Tensor[T]
	B Tensor[T]

	Da    uint
	Db    uint
	D     uint
	undef bool
}

func Dot[T types.Number](a, b Tensor[T]) Tensor[T] {
	return &TensorDotProduct[T]{
		A:     a,
		B:     b,
		undef: true,
	}
}

func DotAt[T types.Number](a, b Tensor[T], dimA, dimB uint) Tensor[T] {
	return &TensorDotProduct[T]{
		A:  a,
		B:  b,
		Da: dimA,
		Db: dimB,
	}
}

func (p *TensorDotProduct[T]) BuildGraph(ctx *context.Context) error {
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

	sha := p.A.GetShape()
	shb := p.B.GetShape()

	if len(sha) == 0 || len(shb) == 0 {
		return errors.FmtNeuralError("Can not do dot to an empty tensor")
	}

	if p.undef {
		p.Da = uint(len(sha) - 1)
		p.Db = uint(len(shb) - 1)
		if p.Db > 0 {
			p.Db--
		}
	}

	if sha[p.Da] != shb[p.Db] {
		return errors.FmtNeuralError(
			"Invalid dot operation to tensors with shape[%d] %d and shape[%d] %d",
			p.Da,
			sha[p.Da],
			p.Db,
			shb[p.Db],
		)
	}

	if len(p.A.GetShape()) == 1 && len(p.B.GetShape()) == 1 {
		return p.buildVector(ctx)
	}

	p.Shape = append(sha[:p.Da], sha[p.Da+1:]...)
	p.D = uint(len(p.Shape))
	p.Shape = append(p.Shape, shb[:p.Db]...)
	p.Shape = append(p.Shape, shb[p.Db+1:]...)

	p.MulIndex = tools.GetIndexMul(p.Shape)

	p.Operands = make([]*number.Scalar[T], tools.GetDataSize(p.Shape))
	for i := range p.Operands {
		p.Operands[i] = &number.Scalar[T]{}
	}

	for i := range p.Operands {
		idx := tools.ReverseIndex(p.MulIndex, p.Shape, uint(i))
		idxA := idx[:p.D]
		idxB := idx[p.D:]
		err := p.buildLastDim(ctx, uint(i), idxA, idxB)
		if err != nil {
			return err
		}
	}

	return nil
}

func (p *TensorDotProduct[T]) buildLastDim(
	ctx *context.Context,
	index uint,
	indexA []uint,
	indexB []uint,
) error {
	sha := p.A.GetShape()
	ops := make([]*number.Scalar[T], sha[p.Da])

	indA := make([]uint, len(indexA)+1)
	indB := make([]uint, len(indexB)+1)
	for i := uint(0); i < uint(len(indexA)); i++ {
		if i < p.Da {
			indA[i] = indexA[i]
		} else {
			indA[i+1] = indexA[i]
		}
	}
	for i := uint(0); i < uint(len(indexB)); i++ {
		if i < p.Db {
			indB[i] = indexB[i]
		} else {
			indB[i+1] = indexB[i]
		}
	}

	for i := uint(0); i < sha[p.Da]; i++ {
		indA[p.Da] = i
		indB[p.Db] = i

		a, err := p.A.GetOperand(indA...)
		if err != nil {
			return nil
		}

		b, err := p.B.GetOperand(indB...)
		if err != nil {
			return err
		}

		o := &number.Scalar[T]{}
		ops[i] = o
		ctx.Push(&operation.Mul[T]{
			Scalar: o,
			A:      a,
			B:      b,
		})
	}
	o := p.Operands[index]
	ctx.Push(&operation.Sum[T]{
		Scalar: o,
		Args:   ops,
	})
	return nil
}

func (p *TensorDotProduct[T]) buildVector(ctx *context.Context) error {
	if p.A.GetSize() != p.B.GetSize() {
		return errors.FmtNeuralError(
			"Dot product fail on vectors with size %d and %d",
			p.A.GetSize(),
			p.B.GetSize(),
		)
	}
	op1 := p.A.GetOperands()
	op2 := p.B.GetOperands()
	ops := make([]*number.Scalar[T], len(op1))
	for i := range ops {
		o := &number.Scalar[T]{}
		ops[i] = o
		ctx.Push(&operation.Mul[T]{
			Scalar: o,
			A:      op1[i],
			B:      op2[i],
		})
	}

	o := &number.Scalar[T]{}
	ctx.Push(&operation.Sum[T]{
		Scalar: o,
		Args:   ops,
	})

	p.Operands = []*number.Scalar[T]{o}
	p.Shape = []uint{1}
	p.MulIndex = []uint{1}

	return nil
}

func (p *TensorMat[T]) Dot(m *TensorMat[T]) (*TensorMat[T], error) {
	d1 := uint(len(p.Shape) - 1)
	d2 := uint(len(m.Shape) - 1)
	if d2 > 0 {
		d2--
	}

	return p.DotAt(m, d1, d2)
}

func (p *TensorMat[T]) DotAt(m *TensorMat[T], dim1, dim2 uint) (*TensorMat[T], error) {
	sha := p.GetShape()
	shb := m.GetShape()

	if len(sha) == 0 || len(shb) == 0 {
		return nil, errors.FmtNeuralError("Can not do dot to an empty tensor")
	}

	if sha[dim1] != shb[dim2] {
		return nil, errors.FmtNeuralError(
			"Invalid dot operation to tensors with shape[%d] %d and shape[%d] %d",
			dim1,
			sha[dim1],
			dim2,
			shb[dim2],
		)
	}

	if len(sha) == 1 && len(shb) == 1 {
		return p.DotVector(m)
	}

	size := sha[dim1]
	shape := append(sha[:dim1], sha[dim1+1:]...)
	dim := uint(len(shape))
	shape = append(shape, shb[:dim2]...)
	shape = append(shape, shb[dim2+1:]...)

	mulIndex := tools.GetIndexMul(shape)

	ops := make([]*number.Scalar[T], tools.GetDataSize(shape))
	for i := range ops {
		ops[i] = &number.Scalar[T]{Value: 0}
	}

	for i := range ops {
		idx := tools.ReverseIndex(mulIndex, shape, uint(i))
		idxA := idx[:dim]
		idxB := idx[dim:]
		err := p.dotLastDim(m, ops, size, dim1, dim2, uint(i), idxA, idxB)
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

func (p *TensorMat[T]) dotLastDim(
	m *TensorMat[T],
	ops []*number.Scalar[T],
	size,
	dim1, dim2,
	index uint,
	indexA []uint,
	indexB []uint,
) error {
	indA := make([]uint, len(indexA)+1)
	indB := make([]uint, len(indexB)+1)
	for i := uint(0); i < uint(len(indexA)); i++ {
		if i < dim1 {
			indA[i] = indexA[i]
		} else {
			indA[i+1] = indexA[i]
		}
	}
	for i := uint(0); i < uint(len(indexB)); i++ {
		if i < dim2 {
			indB[i] = indexB[i]
		} else {
			indB[i+1] = indexB[i]
		}
	}

	for i := uint(0); i < size; i++ {
		indA[dim1] = i
		indB[dim2] = i

		a, err := p.GetOperand(indA...)
		if err != nil {
			return nil
		}

		b, err := m.GetOperand(indB...)
		if err != nil {
			return err
		}

		ops[index].Value += a.Value * b.Value
	}
	return nil
}

func (p *TensorMat[T]) DotVector(m *TensorMat[T]) (*TensorMat[T], error) {
	if p.GetSize() != m.GetSize() {
		return nil, errors.FmtNeuralError(
			"Dot product fail on vectors with size %d and %d",
			p.GetSize(),
			m.GetSize(),
		)
	}
	op1 := p.GetOperands()
	op2 := m.GetOperands()
	ops := make([]*number.Scalar[T], len(op1))
	for i := range ops {
		ops[i] = &number.Scalar[T]{
			Value: op1[i].Value * op2[i].Value,
		}
	}

	o := &number.Scalar[T]{
		Value: tools.Sum[T](ops),
	}

	return &TensorMat[T]{
		Shape:    []uint{1},
		MulIndex: []uint{1},
		Operands: []*number.Scalar[T]{o},
	}, nil
}
