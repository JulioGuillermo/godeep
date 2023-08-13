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

/*
func (p *TensorDotProduct[T]) buildMatVec(m, v Tensor[T], invert bool, ctx *context.Context) error {
	shape := m.GetShape()
	if invert {
		if shape[len(shape)-2] != v.GetSize() {
			return errors.FmtNeuralError(
				"Invalid dot product on matrix with shape[-2] %d and a vector of size %d",
				shape[len(shape)-2],
				v.GetSize(),
			)
		}
	} else {
		if shape[len(shape)-1] != v.GetSize() {
			return errors.FmtNeuralError(
				"Invalid dot product on matrix with shape[-1] %d and a vector of size %d",
				shape[len(shape)-1],
				v.GetSize(),
			)
		}
	}

	p.Shape = shape[:len(shape)-1]
	if invert {
		p.Shape[len(p.Shape)-1] = shape[len(shape)-1]
	}
	p.MulIndex = tools.GetIndexMul(p.Shape)
	p.Operands = make([]*number.Scalar[T], tools.GetDataSize(p.Shape))

	return p.buildMatVecRecursive(ctx, invert, m, v, 0, 0, []uint{})
}

// TODO better way...
func (p *TensorDotProduct[T]) buildMatVecRecursive(
	ctx *context.Context,
	invert bool,
	m, v Tensor[T],
	dim, index uint,
	oIndex []uint,
) error {
	if dim == uint(len(p.Shape)) {
		ops := make([]*number.Scalar[T], v.GetSize())
		for i, vo := range v.GetOperands() {
			var ind []uint
			if invert {
				ind = make([]uint, len(oIndex)+1)
				copy(ind, oIndex)
				ind[len(ind)-1] = ind[len(ind)-2]
				ind[len(ind)-2] = uint(i)
			} else {
				ind = append(oIndex, uint(i))
			}
			o, e := m.GetOperand(ind...)
			if e != nil {
				return e
			}
			mul := &number.Scalar[T]{}
			ops[i] = mul
			ctx.Push(&operation.Mul[T]{
				Scalar: mul,
				A:      vo,
				B:      o,
			})
		}

		sum := &number.Scalar[T]{}
		p.Operands[index] = sum
		ctx.Push(&operation.Sum[T]{
			Scalar: sum,
			Args:   ops,
		})
		return nil
	}
	for i := uint(0); i < p.Shape[dim]; i++ {
		err := p.buildMatVecRecursive(
			ctx,
			invert,
			m, v,
			dim+1,
			index+i*p.MulIndex[dim],
			append(oIndex, i),
		)
		if err != nil {
			return err
		}
	}
	return nil
}*/

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

	//if len(p.B.GetShape()) == 1 {
	//	return p.buildMatVec(p.A, p.B, false, ctx)
	//}

	//if len(p.A.GetShape()) == 1 {
	//	return p.buildMatVec(p.B, p.A, true, ctx)
	//}

	p.Shape = append(sha[:p.Da], sha[p.Da+1:]...)
	p.D = uint(len(p.Shape))
	p.Shape = append(p.Shape, shb[:p.Db]...)
	p.Shape = append(p.Shape, shb[p.Db+1:]...)

	p.MulIndex = tools.GetIndexMul(p.Shape)

	p.Operands = make([]*number.Scalar[T], tools.GetDataSize(p.Shape))
	for i := range p.Operands {
		p.Operands[i] = &number.Scalar[T]{}
	}

	return p.buildRecursiveDot(ctx, 0, []uint{}, []uint{}, []uint{})
}

func (p *TensorDotProduct[T]) buildRecursiveDot(
	ctx *context.Context,
	dim uint,
	index []uint,
	indexA []uint,
	indexB []uint,
) error {
	if dim == uint(len(p.Shape)) {
		return p.buildLastDim(ctx, index, indexA, indexB)
	}
	for i := uint(0); i < p.Shape[dim]; i++ {
		if dim < p.D {
			err := p.buildRecursiveDot(
				ctx,
				dim+1,
				append(index, i),
				append(indexA, i),
				indexB,
			)
			if err != nil {
				return err
			}
		} else {
			err := p.buildRecursiveDot(
				ctx,
				dim+1,
				append(index, i),
				indexA,
				append(indexB, i),
			)
			if err != nil {
				return err
			}
		}
	}
	return nil
}

func (p *TensorDotProduct[T]) buildLastDim(
	ctx *context.Context,
	index []uint,
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
	o, err := p.GetOperand(index...)
	if err != nil {
		return err
	}
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
